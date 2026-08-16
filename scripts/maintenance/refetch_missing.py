"""Re-fetch en ligne des articles dont le contenu est NULL.

A lancer apres `pip install brotli` pour que httpx decompresse correctement
les reponses en Content-Encoding: br (cause du probleme).

Usage :
    python scripts/maintenance/refetch_missing.py                 # toutes les sources
    python scripts/maintenance/refetch_missing.py "Meduza"        # une source
"""
import sys
import time
from pathlib import Path

import duckdb
import httpx
import trafilatura
from langdetect import detect, DetectorFactory, LangDetectException

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

DetectorFactory.seed = 42
MIN_LEN = 300
SLEEP = 0.4  # seconds entre fetchs

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def detect_lang(text):
    if not text or len(text) < 100:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def main():
    src_filter = sys.argv[1] if len(sys.argv) > 1 else None

    db = duckdb.connect("data/russia.duckdb")

    if src_filter:
        rows = db.execute("""
            SELECT id, url, source_name FROM articles
            WHERE (content IS NULL OR LENGTH(content) < ?)
              AND url IS NOT NULL AND source_name = ?
            ORDER BY published_at DESC NULLS LAST
        """, [MIN_LEN, src_filter]).fetchall()
    else:
        rows = db.execute("""
            SELECT id, url, source_name FROM articles
            WHERE (content IS NULL OR LENGTH(content) < ?)
              AND url IS NOT NULL
            ORDER BY source_name, published_at DESC NULLS LAST
        """, [MIN_LEN]).fetchall()

    if not rows:
        print("Aucun article a refetch.")
        return

    print(f"A re-fetch : {len(rows)} articles")
    print(f"Estimation : ~{len(rows) * (SLEEP + 1.5) / 60:.1f} minutes")
    print()

    stats = {}

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for i, (aid, url, src) in enumerate(rows, 1):
            if src not in stats:
                stats[src] = {"ok": 0, "fetch_ko": 0, "extract_ko": 0}

            try:
                r = client.get(url, headers=HEADERS)
                r.raise_for_status()
                html = r.text
            except Exception:
                stats[src]["fetch_ko"] += 1
                time.sleep(SLEEP)
                continue

            if not html or len(html) < 1000:
                stats[src]["fetch_ko"] += 1
                time.sleep(SLEEP)
                continue

            content = trafilatura.extract(html, url=url, favor_recall=True)
            if not content or len(content) < MIN_LEN:
                stats[src]["extract_ko"] += 1
                # Stocke quand meme le bon raw_html pour analyse ulterieure
                db.execute("UPDATE articles SET raw_html = ? WHERE id = ?",
                           [html, aid])
                time.sleep(SLEEP)
                continue

            lang = detect_lang(content)
            db.execute("""
                UPDATE articles SET raw_html = ?, content = ?, language = ?
                WHERE id = ?
            """, [html, content, lang, aid])
            stats[src]["ok"] += 1

            if i % 20 == 0:
                done = sum(s["ok"] for s in stats.values())
                print(f"  {i}/{len(rows)} ({done} OK)")
            time.sleep(SLEEP)

    db.close()

    print()
    print("=" * 70)
    print("RAPPORT")
    print("=" * 70)
    print(f"{'SOURCE':<35} {'OK':>5} {'FETCH_KO':>9} {'EXTR_KO':>8} {'%':>6}")
    print("-" * 70)
    for src in sorted(stats.keys(), key=lambda s: -stats[s]["ok"]):
        s = stats[src]
        tot = s["ok"] + s["fetch_ko"] + s["extract_ko"]
        pct = 100 * s["ok"] / max(tot, 1)
        print(f"  {src:<33} {s['ok']:>5} {s['fetch_ko']:>9} "
              f"{s['extract_ko']:>8} {pct:>5.0f}%")


if __name__ == "__main__":
    main()
