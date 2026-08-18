"""Procedes rhetoriques de persuasion, segment par segment.

Taxonomie de Da San Martino et al., « Fine-Grained Analysis of Propaganda in
News Articles » (EMNLP 2019), reduite aux procedes qui apparaissent
effectivement dans le corpus russophone. La grille de lecture des plateaux de
television vient de Gulenko, « Political discussion as a propaganda spectacle »
(Media, Culture & Society, 2021) : selectivite de l'information, illusion de
pluralite des invites, role du presentateur, discredit de l'adversaire.

Cette analyse repond a une question que ni le sentiment ni les themes ne
posent. Les themes disent DE QUOI on parle, le sentiment ENVERS QUI on penche,
les procedes COMMENT le texte cherche a convaincre.

Restreinte par defaut a la television et a YouTube : c'est la que le cadrage
est explicite, et cela borne le cout. Rien ici ne modifie les tables
existantes -- le tableau de bord fonctionne sans.

Usage :
  python scripts/analysis/analyze_techniques_mistral.py
  python scripts/analysis/analyze_techniques_mistral.py --kinds tv,youtube,telegram
  python scripts/analysis/analyze_techniques_mistral.py --limit 50 --reset
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from threading import Lock

from src.db import get_conn
from src.llm_mistral import complete_json, MODEL_SMALL
from src.logging_setup import setup_logging

log = setup_logging("techniques")

MIN_CONTENT_LEN = 300
MAX_CONTENT_CHARS = 3000
MAX_WORKERS = 8
KINDS_DEFAUT = ("tv", "youtube")

# Sous-ensemble des 18 procedes de la taxonomie d'origine. Les procedes
# ecartes (Repetition, Bandwagon, Obfuscation...) demandent le document entier
# ou sont trop rares pour etre mesures de facon fiable sur un segment.
TECHNIQUES = {
    "langage_charge": "mots a forte charge emotionnelle pour colorer un fait",
    "etiquetage": "coller a une personne ou un groupe un nom infamant ou elogieux",
    "appel_a_la_peur": "agiter une menace pour emporter l'adhesion",
    "exageration": "grossir ou minimiser demesurement",
    "doute": "insinuer sans affirmer, jeter la suspicion",
    "drapeau": "invoquer la patrie, le peuple, les valeurs nationales",
    "simplification_causale": "ramener un phenomene complexe a une cause unique",
    "slogan": "formule breve et frappante tenant lieu d'argument",
    "appel_a_l_autorite": "l'affirmation vaut parce qu'une figure la porte",
    "noir_ou_blanc": "presenter deux options comme les seules possibles",
    "cliche_final": "formule toute faite qui clot la discussion",
    "whataboutisme": "repondre a une critique en pointant les fautes de l'accusateur",
    "reductio_ad_hitlerum": "assimiler l'adversaire au nazisme ou au fascisme",
    "homme_de_paille": "deformer la position adverse pour la refuter plus aisement",
    "diversion": "deplacer l'attention vers un sujet sans rapport",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS article_techniques (
    article_id VARCHAR,
    technique  VARCHAR,
    confiance  DOUBLE,
    extrait    VARCHAR,   -- le fragment qui porte le procede
    run_date   DATE
);
CREATE INDEX IF NOT EXISTS idx_techniques_article ON article_techniques(article_id);
CREATE INDEX IF NOT EXISTS idx_techniques_nom ON article_techniques(technique);
"""

SYSTEM_PROMPT = """Vous analysez un extrait de media russophone et reperez les procedes rhetoriques de persuasion qu'il emploie.

Procedes possibles :
""" + "\n".join(f"- {k} : {v}" for k, v in TECHNIQUES.items()) + """

Regles :
- ne signalez un procede que s'il est manifeste, en citant le fragment russe qui le porte
- un extrait peut n'en contenir aucun : repondez alors une liste vide
- trois procedes au maximum, les plus nets
- confiance entre 0 et 1

Ce qui n'est PAS un procede :
- rapporter un fait, meme grave. Une alerte au tsunami, un bilan d'accident ou
  un fait divers violent informent ; l'appel a la peur, lui, agite une menace
  pour faire adopter une POSITION.
- mentionner un mot du champ patriotique en contexte factuel. Un athlete qui
  concourt sous son drapeau national est un fait ; le drapeau agite invoque la
  nation pour emporter l'adhesion.
- citer ou critiquer un procede employe par quelqu'un d'autre.
En cas de doute entre un fait rapporte et un procede, ne signalez rien.

Repondez en JSON : {"procedes": [{"technique": "...", "confiance": 0.8, "extrait": "..."}]}"""


def ensure_schema(conn, reset=False):
    if reset:
        conn.execute("DROP TABLE IF EXISTS article_techniques")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt + ";")


def _analyser(article_id, contenu):
    data = complete_json(SYSTEM_PROMPT, contenu[:MAX_CONTENT_CHARS],
                         model=MODEL_SMALL, max_tokens=500)
    if not data:
        return article_id, None
    out = []
    for p in (data.get("procedes") or [])[:3]:
        technique = str(p.get("technique", "")).strip().lower()
        if technique not in TECHNIQUES:
            continue          # le modele invente parfois un nom hors liste
        try:
            confiance = float(p.get("confiance", 0))
        except (TypeError, ValueError):
            confiance = 0.0
        out.append((article_id, technique, max(0.0, min(1.0, confiance)),
                    str(p.get("extrait", ""))[:400], date.today()))
    return article_id, out


def run(kinds=KINDS_DEFAUT, limit=None, reset=False):
    conn = get_conn()
    ensure_schema(conn, reset=reset)

    placeholders = ",".join(["?"] * len(kinds))
    sql = f"""SELECT a.id, a.content FROM articles a
              LEFT JOIN article_techniques t ON t.article_id = a.id
              WHERE a.source_kind IN ({placeholders})
                AND a.content IS NOT NULL AND LENGTH(a.content) >= ?
                AND a.language = 'ru' AND t.article_id IS NULL
              ORDER BY a.published_at DESC"""
    params = [*kinds, MIN_CONTENT_LEN]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    lignes = conn.execute(sql, params).fetchall()

    log.info("%d segments a analyser (%s). Cout estime : %.2f $",
             len(lignes), "/".join(kinds), len(lignes) * 0.0009)
    if not lignes:
        conn.close()
        return

    verrou = Lock()
    total_procedes = 0
    faits = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_analyser, aid, txt) for aid, txt in lignes]
        for f in as_completed(futures):
            try:
                article_id, resultats = f.result()
            except Exception as e:
                log.warning("analyse : %s", str(e)[:90])
                continue
            faits += 1
            if resultats:
                with verrou:
                    conn.executemany(
                        """INSERT INTO article_techniques
                           (article_id, technique, confiance, extrait, run_date)
                           VALUES (?, ?, ?, ?, ?)""", resultats)
                total_procedes += len(resultats)
            if faits % 100 == 0:
                log.info("  %d/%d segments, %d procedes releves",
                         faits, len(lignes), total_procedes)

    log.info("Termine. %d segments, %d procedes.", faits, total_procedes)
    for tech, n in conn.execute(
            """SELECT technique, COUNT(*) AS n FROM article_techniques
               GROUP BY 1 ORDER BY n DESC""").fetchall():
        log.info("  %-24s %d", tech, n)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(KINDS_DEFAUT),
                    help="natures de contenu, separees par des virgules")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    run(kinds=tuple(k.strip() for k in args.kinds.split(",") if k.strip()),
        limit=args.limit, reset=args.reset)
