"""Retire du corpus les segments des videos YouTube hors sujet politique.

Le filtre de src/youtube_scrape.py s'applique a la collecte : il n'agit donc
pas sur ce qui a deja ete collecte. Ce script rattrape l'existant, et sert
aussi apres une correction manuelle de data/youtube_video_topics.json (le
cache des decisions, editable : passer "politique" a true reintegre la video
a la prochaine collecte, a false la fait purger ici).

Pourquoi purger plutot que marquer : une video de 4 h donne ~80 segments qui
pesent autant qu'un mois de depeches. Laissee en base, elle forme son propre
theme dans le clustering et deforme toutes les proportions -- le cluster
« Cinema et emotions » venait a 67 % d'une seule video sur Hollywood. Les
segments sont recollectables a tout moment, la perte est nulle.

    python scripts/maintenance/purge_apolitical_videos.py            # simulation
    python scripts/maintenance/purge_apolitical_videos.py --apply    # execution
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb  # noqa: E402

from src import console_utf8  # noqa: F401,E402  -- sortie UTF-8 sous Windows
from src.youtube_scrape import is_political  # noqa: E402

DB = Path("data/russia.duckdb")

# Memes tables que scripts/maintenance/remove_source.py : toutes celles qui
# referencent un article par son id.
DEPENDENT_TABLES = ("entities", "article_target_sentiment",
                    "article_topics", "article_meta", "article_events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="execute reellement les suppressions")
    args = ap.parse_args()

    conn = duckdb.connect(str(DB), read_only=not args.apply)
    videos = conn.execute("""
        SELECT regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1) AS video_id,
               ANY_VALUE(source_name) AS source,
               regexp_replace(ANY_VALUE(title), ' \\[[0-9]+/[0-9]+\\]$', '') AS titre,
               COUNT(*) AS segments,
               SUM(LENGTH(content)) AS signes
        FROM articles
        WHERE source_kind = 'youtube'
        GROUP BY 1 ORDER BY signes DESC
    """).fetchall()

    if not videos:
        print("Aucune video en base.")
        return

    to_drop = [v for v in videos if not is_political(v[0], v[2])]

    print(f"{len(videos)} videos en base, {len(to_drop)} hors sujet politique.\n")
    if not to_drop:
        print("Rien a purger.")
        conn.close()
        return

    seg = sum(v[3] for v in to_drop)
    chars = sum(v[4] for v in to_drop)
    for v in to_drop:
        print(f"  {v[3]:4} seg  {v[4]:8} signes  {v[1][:22]:24} {v[2][:60]}")

    total_chars = conn.execute(
        "SELECT SUM(LENGTH(content)) FROM articles").fetchone()[0] or 1
    print(f"\nTotal : {seg} segments, {chars} signes "
          f"({100 * chars / total_chars:.1f} % du volume de texte du corpus)")

    if not args.apply:
        print("\nSimulation. Relancez avec --apply pour supprimer.")
        conn.close()
        return

    ids = [v[0] for v in to_drop]
    placeholders = ",".join(["?"] * len(ids))
    where_articles = (f"regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1) "
                      f"IN ({placeholders}) AND source_kind = 'youtube'")

    for tbl in DEPENDENT_TABLES:
        try:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {tbl} WHERE article_id IN "
                f"(SELECT id FROM articles WHERE {where_articles})", ids
            ).fetchone()[0]
            conn.execute(
                f"DELETE FROM {tbl} WHERE article_id IN "
                f"(SELECT id FROM articles WHERE {where_articles})", ids)
            print(f"  {tbl:28} -{n}")
        except duckdb.CatalogException:
            pass  # table absente : analyse jamais lancee

    conn.execute(f"DELETE FROM articles WHERE {where_articles}", ids)
    print(f"  {'articles':28} -{seg}")
    conn.close()

    print("\nPurge terminee. Relancez le clustering pour que les themes "
          "soient recalcules sans ces segments :")
    print("    python scripts/analysis/analyze_topics.py --reset")


if __name__ == "__main__":
    main()
