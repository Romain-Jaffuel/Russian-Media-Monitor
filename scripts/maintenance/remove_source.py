"""Supprime une source du yaml ET de la base (cascade sur toutes les analyses).

Usage : python scripts/maintenance/remove_source.py "The Bell"
"""
import re
import sys
from pathlib import Path

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

if len(sys.argv) < 2:
    print('Usage : python scripts/maintenance/remove_source.py "Nom de la source"')
    sys.exit(1)

name = sys.argv[1]

# 1. sources.yaml
SOURCES = Path("config/sources.yaml")
if SOURCES.exists():
    txt = SOURCES.read_text(encoding="utf-8")
    escaped = re.escape(name)
    pattern = re.compile(
        rf'^  - name: {escaped}\s*\n(?:^(?!  - name:).*\n)*',
        re.MULTILINE,
    )
    new_txt, n = pattern.subn('', txt)
    if n > 0:
        SOURCES.write_text(new_txt, encoding="utf-8")
        print(f"OK {name} retire de {SOURCES}")
    else:
        print(f"WARN {name} non trouve dans {SOURCES}")

# 2. base
import duckdb
DB = Path("data/russia.duckdb")
c = duckdb.connect(str(DB))
n_art = c.execute(
    "SELECT COUNT(*) FROM articles WHERE source_name = ?", [name]
).fetchone()[0]
if n_art:
    for tbl in ("entities", "article_target_sentiment",
                "article_topics", "article_meta", "article_events"):
        try:
            c.execute(
                f"DELETE FROM {tbl} WHERE article_id IN "
                f"(SELECT id FROM articles WHERE source_name = ?)", [name]
            )
        except Exception:
            pass
    c.execute("DELETE FROM articles WHERE source_name = ?", [name])
    print(f"OK {n_art} articles {name} supprimes (+ analyses)")
else:
    print(f"INFO aucun article {name} en base")
c.close()
