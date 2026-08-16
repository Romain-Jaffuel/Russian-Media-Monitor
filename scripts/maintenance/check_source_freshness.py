"""Compare le RSS d'une source avec ce qui est en base, identifie les manquants.

Usage :
  python scripts/maintenance/check_source_freshness.py "Meduza"
  python scripts/maintenance/check_source_freshness.py "Meduza" --refetch  # force la collecte
"""
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import feedparser
import httpx
import yaml

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def parse_date(entry):
    for k in ("published_parsed", "updated_parsed"):
        v = entry.get(k)
        if v:
            try:
                return datetime(*v[:6])
            except Exception:
                pass
    return None


def main():
    if len(sys.argv) < 2:
        print('Usage : python scripts/maintenance/check_source_freshness.py "Nom Source"')
        sys.exit(1)

    src_name = sys.argv[1]
    do_refetch = "--refetch" in sys.argv

    # Trouve la source dans le yaml
    sources = yaml.safe_load(
        Path("config/sources.yaml").read_text(encoding="utf-8")
    )
    sources = sources.get("sources", []) if isinstance(sources, dict) else sources
    src = next((s for s in sources if s["name"] == src_name), None)
    if not src:
        print(f"Source '{src_name}' introuvable dans sources.yaml")
        sys.exit(1)

    url = src["url"]
    print(f"Source : {src_name}")
    print(f"URL    : {url}")
    print(f"Mode   : {src.get('type', 'rss')}")
    print()

    # Fetch fresh (no cache)
    print("Fetch du flux (no-cache)...")
    r = httpx.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
    print(f"HTTP {r.status_code}, {len(r.content)} octets")

    feed = feedparser.parse(r.content)
    if not feed.entries:
        print("Aucune entree dans le flux !")
        sys.exit(1)

    print(f"{len(feed.entries)} entrees dans le flux\n")

    # Etat de la base
    c = duckdb.connect("data/russia.duckdb")
    db_urls = {r2[0] for r2 in c.execute(
        "SELECT url FROM articles WHERE source_name = ?", [src_name]
    ).fetchall()}

    # Compare
    print(f"{'DATE RSS':<20} {'EN BASE':<8} URL")
    print("-" * 100)
    missing = []
    for entry in feed.entries:
        url_e = entry.get("link", "")
        date_e = parse_date(entry)
        date_s = date_e.strftime("%Y-%m-%d %H:%M") if date_e else "(no date)"
        in_db = "OUI" if url_e in db_urls else "NON"
        print(f"  {date_s:<18} {in_db:<8} {url_e[:80]}")
        if url_e not in db_urls:
            missing.append((entry, date_e))

    print()
    print(f"Articles RSS  : {len(feed.entries)}")
    print(f"Manquants DB  : {len(missing)}")

    # Stats DB recente
    rows = c.execute("""
        SELECT DATE(published_at) AS d, COUNT(*) AS n
        FROM articles
        WHERE source_name = ? AND published_at IS NOT NULL
        GROUP BY 1 ORDER BY 1 DESC LIMIT 7
    """, [src_name]).fetchall()
    print(f"\nDerniers jours en base pour {src_name} :")
    for d, n in rows:
        print(f"  {d}  {n} articles")

    if not missing or not do_refetch:
        if missing and not do_refetch:
            print(f"\nPour forcer la collecte des {len(missing)} manquants :")
            print(f'  python scripts/maintenance/check_source_freshness.py "{src_name}" --refetch')
        c.close()
        return

    # Refetch
    import hashlib
    import trafilatura
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42

    print(f"\nRefetch de {len(missing)} articles...")
    added = 0
    for entry, date_e in missing:
        article_url = entry.get("link")
        if not article_url:
            continue
        try:
            rr = httpx.get(article_url, headers=HEADERS, timeout=20,
                           follow_redirects=True)
            if rr.status_code != 200:
                print(f"  HTTP {rr.status_code} : {article_url}")
                continue
            html = rr.text
            content = trafilatura.extract(html, url=article_url, favor_recall=True)
            if not content or len(content) < 300:
                print(f"  contenu trop court : {article_url}")
                continue
            try:
                lang = detect(content)
            except LangDetectException:
                lang = None
            aid = hashlib.md5(article_url.encode()).hexdigest()[:16]
            title = entry.get("title", "")[:500]
            c.execute("""
                INSERT INTO articles
                (id, url, title, source_name, content, raw_html, language,
                 published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
                ON CONFLICT DO NOTHING
            """, [aid, article_url, title, src_name, content, html, lang, date_e])
            added += 1
            print(f"  +1 {article_url}")
        except Exception as e:
            print(f"  ERREUR {type(e).__name__} : {article_url}")
    c.close()
    print(f"\n{added} articles ajoutes.")
    print("Lancez ensuite : python update.py --skip-pipeline")
    print("  (pour appliquer les analyses Mistral sur les nouveaux)")


if __name__ == "__main__":
    main()
