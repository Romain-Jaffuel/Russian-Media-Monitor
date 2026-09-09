"""Diagnostic de collecte : état en base, test du flux en direct, rattrapage.

Remplace diag_sources.py et check_source_freshness.py, qui posaient les mêmes
questions à deux endroits avec deux jeux d'en-têtes HTTP et deux façons de
calculer l'identifiant d'un article.

Sans argument, dresse l'état de toutes les sources du yaml et signale celles
qui n'ont rien publié depuis longtemps. Avec un ou plusieurs noms, détaille
chacune : ce que dit la base, ce que répond le flux, et quels articles du flux
manquent en base. `--refetch` collecte ces manquants.

Usage :
  python scripts/maintenance/check_sources.py
  python scripts/maintenance/check_sources.py "Meduza"
  python scripts/maintenance/check_sources.py "Meduza" --refetch
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import feedparser
import httpx
import yaml

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8
from src.extract import USER_AGENT

DB = Path("data/russia.duckdb")
CONFIG = Path("config/sources.yaml")

# Les mêmes en-têtes que la collecte, plus le refus de cache : un flux servi
# depuis un cache intermédiaire ferait conclure à tort que la source est morte.
ENTETES = {"User-Agent": USER_AGENT, "Cache-Control": "no-cache",
           "Pragma": "no-cache"}

# Au-delà, une source est signalée comme muette. Deux semaines laissent passer
# les hebdomadaires et les relâches d'été sans crier au loup.
JOURS_MUET = 14


def _sources():
    conf = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return conf.get("sources", []) if isinstance(conf, dict) else conf


def _date_entree(entree):
    for cle in ("published_parsed", "updated_parsed"):
        valeur = entree.get(cle)
        if valeur:
            try:
                return datetime(*valeur[:6])
            except (TypeError, ValueError):
                pass
    return None


def _panorama(conn):
    """Une ligne par source : volume, dernière parution, trous."""
    lignes = conn.execute(f"""
        SELECT source_name,
               COUNT(*),
               MAX(published_at),
               COUNT(*) FILTER (WHERE published_at IS NULL),
               COUNT(*) FILTER (WHERE content IS NULL OR TRIM(content) = ''),
               COUNT(*) FILTER (WHERE published_at >= CURRENT_DATE - INTERVAL {JOURS_MUET} DAY)
        FROM articles GROUP BY 1 ORDER BY 2 DESC""").fetchall()

    connues = {s["name"] for s in _sources()}
    print(f"{'SOURCE':32} {'TOTAL':>7} {'RÉCENTS':>8} {'SANS DATE':>10} "
          f"{'VIDES':>6}  DERNIÈRE PARUTION")
    print("-" * 92)
    muettes, orphelines = [], []
    for nom, total, dernier, sans_date, vides, recents in lignes:
        marque = " " if recents else "!"
        print(f"{marque}{nom[:31]:31} {total:7} {recents:8} {sans_date:10} "
              f"{vides:6}  {dernier}")
        if not recents:
            muettes.append(nom)
        if nom not in connues:
            orphelines.append(nom)

    if muettes:
        print(f"\n{len(muettes)} sources sans parution depuis {JOURS_MUET} jours :")
        for nom in muettes:
            print(f"   {nom}")
        print("   Détail d'une source : check_sources.py \"<nom>\"")
    if orphelines:
        print(f"\n{len(orphelines)} sources en base mais absentes du yaml :")
        for nom in orphelines:
            print(f"   {nom}")
    manquantes = connues - {r[0] for r in lignes}
    if manquantes:
        print(f"\n{len(manquantes)} sources du yaml sans aucun article :")
        for nom in sorted(manquantes):
            print(f"   {nom}")


def _flux(source):
    """Interroge la source et renvoie (réponse, entrées du flux)."""
    url = source["url"]
    print(f"  URL   : {url}")
    print(f"  Mode  : {source.get('type', 'rss')}")
    try:
        r = httpx.get(url, headers=ENTETES, timeout=20, follow_redirects=True)
    except httpx.HTTPError as ex:
        print(f"  ÉCHEC réseau : {type(ex).__name__} -- {ex}")
        print("  Un délai dépassé signale en général un blocage réseau.")
        return None, []
    print(f"  HTTP  : {r.status_code}, {len(r.content)} octets")
    entrees = feedparser.parse(r.content).entries
    if entrees:
        print(f"  Flux  : {len(entrees)} entrées")
    else:
        # Pas un flux : on compte les liens plausibles pour distinguer une page
        # vide d'une page dont la structure a changé sous le scraper.
        from urllib.parse import urlparse

        from bs4 import BeautifulSoup
        domaine = urlparse(url).netloc
        liens = BeautifulSoup(r.text, "html.parser").find_all("a", href=True)
        articles = list(dict.fromkeys(
            a["href"] for a in liens
            if domaine in a["href"] and a["href"].count("/") >= 4))
        print(f"  Page  : {len(liens)} liens, {len(articles)} liens d'articles "
              f"plausibles")
        if not articles:
            print("  Aucun lien d'article reconnu : la structure de la page a "
                  "changé, le scraper ne trouve plus rien.")
    return r, entrees


def _detail(conn, nom, source, entrees, refetch):
    """Compare le flux à la base pour une source, et rattrape si demandé."""
    en_base = {r[0] for r in conn.execute(
        "SELECT url FROM articles WHERE source_name = ?", [nom]).fetchall()}
    manquants = [(e, _date_entree(e)) for e in entrees
                 if e.get("link") and e["link"] not in en_base]

    jours = conn.execute("""
        SELECT CAST(published_at AS DATE), COUNT(*) FROM articles
        WHERE source_name = ? AND published_at IS NOT NULL
        GROUP BY 1 ORDER BY 1 DESC LIMIT 7""", [nom]).fetchall()
    if jours:
        print("  Derniers jours en base :")
        for jour, n in jours:
            print(f"     {jour}  {n:4} articles")
    print(f"  Flux {len(entrees)} entrées, {len(manquants)} absentes de la base.")

    if not manquants:
        return
    for entree, quand in manquants[:10]:
        print(f"     manquant  {str(quand)[:16]:16}  {entree['link'][:70]}")
    if len(manquants) > 10:
        print(f"     ... et {len(manquants) - 10} autres")
    if not refetch:
        print(f'  Pour les collecter : check_sources.py "{nom}" --refetch')
        return

    # Le rattrapage réutilise les fonctions de collecte plutôt que d'en
    # réécrire une version locale. C'est ce qui garantit le même identifiant
    # que la passe normale : l'ancien script hachait l'URL en md5 quand le
    # pipeline la hache en sha256, si bien qu'un article rattrapé ici pouvait
    # revenir en double sous deux identifiants différents.
    from src.collect import url_hash
    from src.extract import extract, fetch_html
    from src.pipeline import PAYS, SOURCE_KINDS

    print(f"  Collecte de {len(manquants)} articles...")
    ajoutes = 0
    for entree, quand in manquants:
        lien = entree["link"]
        contenu, langue = extract(fetch_html(lien))
        if not contenu:
            print(f"     sans contenu exploitable : {lien[:60]}")
            continue
        conn.execute(
            """INSERT INTO articles (id, source_name, url, title, content,
                   language, published_at, pays, type_media, statut_legal_ru,
                   source_kind)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            [url_hash(lien), nom, lien, (entree.get("title") or "")[:500],
             contenu, langue, quand, PAYS, source.get("media_type"),
             source.get("legal_status"),
             SOURCE_KINDS.get(source.get("type", "rss"), "press")])
        ajoutes += 1
    print(f"  {ajoutes} articles ajoutés. Analysez-les avec : "
          f"python update.py --skip-pipeline")


def run(noms, refetch=False):
    if not DB.exists():
        print(f"Base introuvable : {DB}")
        return 1
    try:
        conn = duckdb.connect(str(DB), read_only=not refetch)
    except duckdb.Error as ex:
        print(f"Ouverture impossible ({ex}). Fermez le tableau de bord et "
              f"réessayez.")
        return 1

    if not noms:
        _panorama(conn)
        conn.close()
        return 0

    par_nom = {s["name"]: s for s in _sources()}
    for nom in noms:
        print(f"\n=== {nom} ===")
        source = par_nom.get(nom)
        if not source:
            proches = [n for n in par_nom if nom.split()[0].lower() in n.lower()]
            print("  Absente de sources.yaml sous ce nom exact.")
            if proches:
                print("  Noms approchants :", ", ".join(proches[:5]))
            continue
        _, entrees = _flux(source)
        _detail(conn, nom, source, entrees, refetch)
    conn.close()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sources", nargs="*",
                   help="noms exacts à détailler ; sans argument, panorama")
    p.add_argument("--refetch", action="store_true",
                   help="collecte les articles du flux absents de la base")
    a = p.parse_args()
    sys.exit(run(a.sources, refetch=a.refetch))
