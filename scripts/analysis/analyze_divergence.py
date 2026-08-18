"""Ce qui distingue un groupe de sources d'un autre, mot par mot.

Methode : divergence de Kullback-Leibler entre deux distributions de mots
(Vestel & Degaetano-Ortlieb, ICWSM 2025, applique au corpus WarMM-2022). Pour
chaque mot on calcule sa contribution a KL(A||B) : combien ce mot, a lui seul,
rend le groupe A reconnaissable face au groupe B.

L'interet ici est double. C'est interpretable -- pas de modele entraine, on
peut lire le resultat mot par mot et le verifier a la main. Et c'est
CONTRASTIF : on ne melange jamais les groupes, on les oppose. Les agregats du
tableau de bord souffrent de l'inverse -- la presse pro-Kremlin y est
surrepresentee, YouTube n'est que de l'opposition, la television que de l'Etat,
si bien qu'une moyenne globale ne decrit personne.

Comparaisons calculees :
    type_media    etat / para_etat / independant / exil
    source_kind   press / tv / telegram / youtube / vk

Chaque groupe est compare au reste du corpus.

Usage :
  python scripts/analysis/analyze_divergence.py
  python scripts/analysis/analyze_divergence.py --days 14 --reset
"""
import argparse
import math
import time
from collections import Counter
from datetime import date, timedelta

from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("divergence")

MIN_CONTENT_LEN = 200
# Un mot vu trois fois dans tout un corpus n'apprend rien et fait du bruit en
# tete de classement (sa probabilite dans l'autre groupe est quasi nulle, donc
# son rapport explose).
MIN_OCCURRENCES = 8
# Un mot doit servir a plusieurs sources du groupe. Sans ce garde-fou, le
# classement remontait surtout des signatures de chaine ("readovka", "shot",
# "meduzalive", "подписаться") : ce sont bien les mots les plus distinctifs,
# mais ils decrivent un compte, pas une famille de medias.
MIN_SOURCES = 3
TOP_PAR_GROUPE = 60
# Lissage de Laplace : sans lui, un mot absent du groupe B donne une division
# par zero et une divergence infinie.
ALPHA = 0.5

SCHEMA = """
CREATE TABLE IF NOT EXISTS lexical_divergence (
    axe          VARCHAR,   -- 'type_media' ou 'source_kind'
    groupe       VARCHAR,   -- le groupe decrit
    token        VARCHAR,   -- le lemme
    contribution DOUBLE,    -- part de ce mot dans KL(groupe || reste)
    freq_groupe  DOUBLE,    -- pour 10 000 mots
    freq_reste   DOUBLE,
    n_groupe     INTEGER,   -- occurrences brutes
    rang         INTEGER,
    run_date     DATE
);
"""


def ensure_schema(conn, reset=False):
    if reset:
        conn.execute("DROP TABLE IF EXISTS lexical_divergence")
    conn.execute(SCHEMA)


def _compter(documents, tokenizer):
    """Sac de lemmes d'un groupe, et nombre de sources distinctes par lemme.

    `documents` : liste de (source_name, texte).
    """
    total = Counter()
    sources = {}
    for source, texte in documents:
        vus = tokenizer(texte)
        total.update(vus)
        for mot in set(vus):
            sources.setdefault(mot, set()).add(source)
    return total, {m: len(s) for m, s in sources.items()}


def _divergence(compte_a, compte_b, sources_a, min_sources):
    """Contribution de chaque mot a KL(A||B), triee par ordre decroissant.

    On ne garde que les mots SURrepresentes dans A : une contribution peut etre
    negative (le mot caracterise B), mais ces mots-la ressortent quand on
    calcule la comparaison dans l'autre sens.
    """
    total_a = sum(compte_a.values())
    total_b = sum(compte_b.values())
    if not total_a or not total_b:
        return []

    vocab = set(compte_a) | set(compte_b)
    lissage_a = total_a + ALPHA * len(vocab)
    lissage_b = total_b + ALPHA * len(vocab)

    out = []
    for mot in compte_a:
        n_a = compte_a[mot]
        if n_a < MIN_OCCURRENCES or sources_a.get(mot, 0) < min_sources:
            continue
        p = (n_a + ALPHA) / lissage_a
        q = (compte_b.get(mot, 0) + ALPHA) / lissage_b
        contribution = p * math.log(p / q)
        if contribution <= 0:
            continue
        out.append((mot, contribution, p * 10_000, q * 10_000, n_a))
    out.sort(key=lambda r: -r[1])
    return out


def run(window_days=30, reset=False):
    # Le tokenizer lemmatisant du clustering : meme traitement du russe des
    # deux cotes, sinon les deux analyses ne parlent pas du meme vocabulaire.
    from scripts.analysis.analyze_topics import _lemmatizing_tokenizer

    conn = get_conn()
    ensure_schema(conn, reset=reset)
    debut = date.today() - timedelta(days=window_days)

    lignes = conn.execute(
        """SELECT type_media, source_kind, content, source_name FROM articles
           WHERE content IS NOT NULL AND LENGTH(content) >= ?
             AND language = 'ru' AND published_at >= ?""",
        [MIN_CONTENT_LEN, debut]).fetchall()
    log.info("Fenetre %dj : %d documents", window_days, len(lignes))
    if len(lignes) < 100:
        log.error("Pas assez de documents.")
        conn.close()
        return

    conn.execute("DELETE FROM lexical_divergence WHERE run_date = ?", [date.today()])

    for axe, position in (("type_media", 0), ("source_kind", 1)):
        textes_par_groupe = {}
        for ligne in lignes:
            groupe = ligne[position]
            if groupe:
                textes_par_groupe.setdefault(groupe, []).append((ligne[3], ligne[2]))

        log.info("Axe %s : %s", axe,
                 ", ".join(f"{g} ({len(t)})" for g, t in textes_par_groupe.items()))

        t0 = time.time()
        calcules = {g: _compter(d, _lemmatizing_tokenizer)
                    for g, d in textes_par_groupe.items()}
        comptes = {g: c for g, (c, _) in calcules.items()}
        sources = {g: s for g, (_, s) in calcules.items()}
        log.info("  lemmatisation en %.0f s", time.time() - t0)

        for groupe, compte in comptes.items():
            reste = Counter()
            for autre, c in comptes.items():
                if autre != groupe:
                    reste.update(c)
            if not reste:
                log.warning("  %s : aucun groupe de comparaison, ignore", groupe)
                continue

            # Le seuil s'adapte aux petits groupes : « independant » ne
            # compte que deux sources, exiger trois le viderait entierement.
            n_sources = len({s for s, _ in textes_par_groupe[groupe]})
            seuil = min(MIN_SOURCES, max(n_sources, 1))
            resultats = _divergence(compte, reste, sources[groupe],
                                    seuil)[:TOP_PAR_GROUPE]
            if not resultats:
                log.warning("  %-14s aucun mot distinctif (%d source(s))",
                            groupe, n_sources)
                continue
            conn.executemany(
                """INSERT INTO lexical_divergence
                   (axe, groupe, token, contribution, freq_groupe, freq_reste,
                    n_groupe, rang, run_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(axe, groupe, mot, contrib, f_g, f_r, n, i + 1, date.today())
                 for i, (mot, contrib, f_g, f_r, n) in enumerate(resultats)])
            apercu = ", ".join(m for m, *_ in resultats[:8])
            log.info("  %-14s %3d mots (>=%d sources sur %d) | %s",
                     groupe, len(resultats), seuil, n_sources, apercu)

    total = conn.execute("SELECT COUNT(*) FROM lexical_divergence").fetchone()[0]
    log.info("Termine. %d lignes en base.", total)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    run(window_days=args.days, reset=args.reset)
