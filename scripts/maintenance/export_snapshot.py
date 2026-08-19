"""Fabrique l'instantane publiable du tableau de bord.

La base complete pese plusieurs Go, dont l'essentiel est la colonne raw_html
que le tableau de bord ne lit jamais. En la retirant, ainsi que les tables des
analyses abandonnees, on tombe a une soixantaine de Mo -- treize une fois
compresse, ce qui se versionne sans faire enfler le depot.

C'est ce qui permet de garder la collecte en local (IP residentielle, GPU) et
de ne publier qu'une copie figee : le verrou DuckDB disparait de lui-meme,
puisque personne n'ecrit du cote heberge.

Usage :
  python scripts/maintenance/export_snapshot.py
  python scripts/maintenance/export_snapshot.py --sans-compression
"""
import argparse
import gzip
import shutil
import time
from pathlib import Path

import duckdb

from src.db import DB_PATH
from src.logging_setup import setup_logging

log = setup_logging("snapshot")

SORTIE = Path("data/snapshot.duckdb")

# Colonnes conservees : tout sauf raw_html. Enumerees plutot que « SELECT * EXCEPT »
# pour que l'ajout d'une colonne lourde en base ne se glisse pas dans
# l'instantane sans qu'on l'ait decide.
COLONNES = """id, source_name, feed_url, url, title, author, summary, content,
              language, published_at, fetched_at, pays, type_media,
              statut_legal_ru, source_kind, view_count"""

# Tables lues par le tableau de bord. Les tables des analyses retirees
# (entities, article_meta, article_events...) restent en local et ne sont pas
# publiees.
TABLES = ("article_target_sentiment", "article_topics", "topics",
          "lexical_divergence", "topic_quality", "article_techniques")


def run(compresser=True):
    if not DB_PATH.exists():
        log.error("Base introuvable : %s", DB_PATH)
        return
    for p in (SORTIE, Path(str(SORTIE) + ".wal")):
        p.unlink(missing_ok=True)

    t0 = time.time()
    conn = duckdb.connect(str(SORTIE))
    conn.execute(f"ATTACH '{DB_PATH.as_posix()}' AS src (READ_ONLY)")
    conn.execute(f"CREATE TABLE articles AS SELECT {COLONNES} FROM src.articles")
    n = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    presentes = {r[0] for r in conn.execute("SHOW TABLES FROM src").fetchall()}
    for t in TABLES:
        if t not in presentes:
            log.warning("table absente, ignoree : %s", t)
            continue
        conn.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
    conn.execute("DETACH src")
    conn.close()

    complet = DB_PATH.stat().st_size / 1e9
    brut = SORTIE.stat().st_size / 1e6
    log.info("%d lignes exportees | %.0f Mo (base complete : %.1f Go, "
             "rapport 1:%d)", n, brut, complet, complet * 1000 / max(brut, 1))

    if compresser:
        cible = Path(str(SORTIE) + ".gz")
        with open(SORTIE, "rb") as f, gzip.open(cible, "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
        # Seule l'archive est versionnee : le .duckdb decompresse serait
        # quatre fois plus lourd dans l'historique git.
        SORTIE.unlink()
        log.info("Compresse : %.0f Mo -> %s", cible.stat().st_size / 1e6, cible)

    log.info("Termine en %.0f s.", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sans-compression", action="store_true",
                    help="laisse le .duckdb tel quel, sans archive gzip")
    args = ap.parse_args()
    run(compresser=not args.sans_compression)
