"""Recupere la date de publication depuis le raw_html pour les articles
ou published_at est NULL.

Strategies en cascade :
  1. JSON-LD datePublished / dateCreated
  2. Meta property="article:published_time"
  3. Meta itemprop="datePublished"
  4. Meta name="date" / "DC.date" / "pubdate"
  5. <time datetime="...">
  6. Pattern dans l'URL (/YYYY/MM/DD/)
  7. Texte en russe : "8 июня 2026", "Опубликовано 08.06.2026"
  8. Fallback : fetched_at (avec --fallback-fetched)

Usage :
  python scripts/maintenance/fix_missing_dates.py                       # diag + recup HTML
  python scripts/maintenance/fix_missing_dates.py --check               # diag seulement
  python scripts/maintenance/fix_missing_dates.py --fallback-fetched    # +fallback fetched_at
  python scripts/maintenance/fix_missing_dates.py "L'UNION"             # une source
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import duckdb
from bs4 import BeautifulSoup

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

DB = Path("data/russia.duckdb")

URL_DATE_RE = re.compile(r"/(\d{4})/(\d{1,2})/(\d{1,2})/")
ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2})?)")

# Russian date patterns (genitive month names, as used in running text).
MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
RU_DATE_RE = re.compile(
    r"(?:опубликовано\s+)?(\d{1,2})\s+"
    r"(" + "|".join(MONTHS_RU) + r")\s+(\d{4})",
    re.IGNORECASE,
)
NUM_DATE_RE = re.compile(r"(?:le\s+)?(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})",
                         re.IGNORECASE)


def parse_iso(s):
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"\.\d+", "", s)
    s = s.replace("Z", "+00:00")
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S",
    ]:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    m = ISO_DATE_RE.search(s)
    if m:
        try:
            return datetime.strptime(m.group(1)[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


def extract_date_from_jsonld(soup):
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if "@graph" in item and isinstance(item["@graph"], list):
                items.extend(item["@graph"])
                continue
            for key in ("datePublished", "dateCreated", "uploadDate"):
                if key in item:
                    dt = parse_iso(item[key])
                    if dt:
                        return dt, f"jsonld_{key}"
    return None, None


def extract_date_from_meta(soup):
    selectors = [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"property": "og:article:published_time"}, "content"),
        ("meta", {"itemprop": "datePublished"}, "content"),
        ("meta", {"name": "DC.date"}, "content"),
        ("meta", {"name": "DC.date.issued"}, "content"),
        ("meta", {"name": "date"}, "content"),
        ("meta", {"name": "pubdate"}, "content"),
        ("meta", {"name": "publish_date"}, "content"),
        ("meta", {"name": "publication_date"}, "content"),
    ]
    for tag, attrs, attr in selectors:
        el = soup.find(tag, attrs=attrs)
        if el and el.get(attr):
            dt = parse_iso(el.get(attr))
            if dt:
                return dt, f"meta_{list(attrs.values())[0]}"
    return None, None


def extract_date_from_time_tag(soup):
    for time_tag in soup.find_all("time"):
        for attr in ("datetime", "pubdate"):
            v = time_tag.get(attr)
            if v:
                dt = parse_iso(v)
                if dt:
                    return dt, f"time_{attr}"
    return None, None


def extract_date_from_url(url):
    if not url:
        return None, None
    m = URL_DATE_RE.search(url)
    if m:
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 2000 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
                return datetime(y, mo, d), "url_pattern"
        except ValueError:
            pass
    return None, None


def extract_date_from_russian_text(soup):
    """Cherche '8 июня 2026' ou '08.06.2026' dans les premiers paragraphes."""
    # Limite au debut de l'article (avant le corps texte long)
    candidates = []
    for tag in soup.find_all(["p", "div", "span", "time", "small", "header"], limit=30):
        txt = tag.get_text(" ", strip=True)
        if 4 < len(txt) < 200:
            candidates.append(txt)

    # Cherche en priorite "Опубликовано X" / "X"
    for txt in candidates:
        m = RU_DATE_RE.search(txt)
        if m:
            try:
                day = int(m.group(1))
                month = MONTHS_RU[m.group(2).lower()]
                year = int(m.group(3))
                if 2000 <= year <= 2030 and 1 <= day <= 31:
                    return datetime(year, month, day), "ru_text"
            except (ValueError, KeyError):
                continue
        m = NUM_DATE_RE.search(txt)
        if m:
            try:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                # Filtre les dates plausibles seulement
                if 2000 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    return datetime(year, month, day), "num_date"
            except ValueError:
                continue
    return None, None


def extract_date(html, url):
    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None
        if soup:
            for fn in (extract_date_from_jsonld, extract_date_from_meta,
                       extract_date_from_time_tag, extract_date_from_russian_text):
                dt, src = fn(soup)
                if dt:
                    return dt, src
    return extract_date_from_url(url)


def diagnostic(c):
    print("=" * 75)
    print("DIAGNOSTIC : articles sans published_at par source")
    print("=" * 75)
    rows = c.execute("""
        SELECT source_name,
               COUNT(*) AS total,
               SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END) AS sans_date,
               ROUND(100.0 * SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END)
                     / NULLIF(COUNT(*), 0), 0) AS pct_null
        FROM articles
        GROUP BY 1
        HAVING SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END) > 0
        ORDER BY pct_null DESC, sans_date DESC
    """).fetchall()
    if not rows:
        print("Aucun article sans date.")
        return False
    print(f"{'SOURCE':<35} {'TOTAL':>6} {'SANS DATE':>10} {'%':>5}")
    print("-" * 75)
    for src, tot, null, pct in rows:
        print(f"  {src:<33} {tot:>6} {null:>10} {pct:>4}%")
    return True


def recover(c, src_filter=None, fallback_fetched=False):
    where = "published_at IS NULL AND raw_html IS NOT NULL"
    params = []
    if src_filter:
        where += " AND source_name = ?"
        params.append(src_filter)

    rows = c.execute(
        f"SELECT id, url, raw_html, source_name, fetched_at FROM articles WHERE {where}",
        params
    ).fetchall()
    if not rows:
        print("Aucun article a retraiter (raw_html present).")
    else:
        print(f"\nExtraction depuis raw_html pour {len(rows)} articles...")
        stats = {}
        sources_stats = {}
        for i, (aid, url, html, src, fetched) in enumerate(rows, 1):
            dt, method = extract_date(html, url)
            sources_stats.setdefault(src, {"ok": 0, "ko": 0})
            if dt:
                c.execute("UPDATE articles SET published_at = ? WHERE id = ?",
                          [dt, aid])
                stats[method] = stats.get(method, 0) + 1
                sources_stats[src]["ok"] += 1
            else:
                sources_stats[src]["ko"] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}")

        print()
        print("Rapport extraction HTML :")
        print(f"{'SOURCE':<35} {'OK':>6} {'KO':>6} {'%':>5}")
        for src, s in sorted(sources_stats.items(), key=lambda x: -x[1]["ok"]):
            tot = s["ok"] + s["ko"]
            pct = 100 * s["ok"] / max(tot, 1)
            print(f"  {src:<33} {s['ok']:>6} {s['ko']:>6} {pct:>4.0f}%")
        print()
        print("Methodes :")
        for m, n in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {m:<30} {n}")

    # Fallback fetched_at
    if fallback_fetched:
        where2 = "published_at IS NULL AND fetched_at IS NOT NULL"
        params2 = []
        if src_filter:
            where2 += " AND source_name = ?"
            params2.append(src_filter)
        n = c.execute(
            f"SELECT COUNT(*) FROM articles WHERE {where2}", params2
        ).fetchone()[0]
        if n > 0:
            print(f"\nFallback fetched_at sur {n} articles restants...")
            c.execute(
                f"UPDATE articles SET published_at = fetched_at WHERE {where2}",
                params2,
            )
            print(f"  {n} dates renseignees avec fetched_at")


def main():
    if not DB.exists():
        print(f"DB introuvable : {DB}")
        sys.exit(1)

    check_only = "--check" in sys.argv
    fallback = "--fallback-fetched" in sys.argv
    src_filter = next(
        (a for a in sys.argv[1:] if not a.startswith("--")), None
    )

    c = duckdb.connect(str(DB))
    has_problem = diagnostic(c)
    if check_only or not has_problem:
        c.close()
        return
    recover(c, src_filter=src_filter, fallback_fetched=fallback)
    print()
    print("Diagnostic apres recuperation :")
    diagnostic(c)
    c.close()


if __name__ == "__main__":
    main()
