"""Diagnostic de collecte pour des sources problematiques.

Pour chaque source : etat en base (dernier article, volume recent, contenu
vide, dates manquantes) PUIS test de collecte en direct (le flux/page repond ?
combien de liens d'articles trouves ?). Separe ainsi un probleme reseau, un
scraper casse, ou une source simplement inactive.

Usage :
    python scripts/maintenance/diag_sources.py
    python scripts/maintenance/diag_sources.py --base   (lit la base principale ; defaut si pas de vue)
"""
import sys
from pathlib import Path

import duckdb

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

DB = Path("data/russia.duckdb")
VIEW_DIR = Path("data/view")
POINTER = VIEW_DIR / "current.txt"

# Sources a auditer par defaut ; passer d'autres noms en argv pour cibler
# une source precise.
SOURCES = [a for a in sys.argv[1:] if not a.startswith("--")] or [
    "TASS", "Meduza", "Novaya Gazeta (edition Moscou)",
]

# Resolution base
cible = DB
if "--base" not in sys.argv and POINTER.exists():
    v = VIEW_DIR / POINTER.read_text(encoding="utf-8").strip()
    if v.exists():
        cible = v

print("=" * 70)
print(f"ETAT EN BASE ({cible.name})")
print("=" * 70)

try:
    c = duckdb.connect(str(cible), read_only=True)
except Exception as ex:
    raise SystemExit(f"Ouverture impossible ({ex}). Fermez le dashboard et reessayez.")

# Nom exact des colonnes (souple sur lang/language)
cols = [r[0] for r in c.execute("DESCRIBE articles").fetchall()]
lang_col = "lang" if "lang" in cols else ("language" if "language" in cols else None)

for src in SOURCES:
    print(f"\n--- {src} ---")
    row = c.execute("""
        SELECT COUNT(*),
               MAX(published_at),
               SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN content IS NULL OR TRIM(content) = '' THEN 1 ELSE 0 END),
               SUM(CASE WHEN published_at >= CURRENT_DATE - INTERVAL 7 DAY THEN 1 ELSE 0 END),
               CAST(AVG(LENGTH(content)) AS INT)
        FROM articles WHERE source_name = ?
    """, [src]).fetchone()
    total, dernier, sans_date, sans_contenu, recent7, len_moy = row
    if total == 0:
        print("  AUCUN article en base sous ce nom exact.")
        # verifier un nom approchant
        approx = c.execute(
            "SELECT DISTINCT source_name FROM articles WHERE source_name ILIKE ?",
            [f"%{src.split()[0]}%"]).df()
        if not approx.empty:
            print("  Noms approchants en base :", ", ".join(approx["source_name"].tolist()[:5]))
        continue
    print(f"  Articles total        : {total}")
    print(f"  Dernier article       : {dernier}")
    print(f"  Publies (7 j)         : {recent7}")
    print(f"  Sans date             : {sans_date}")
    print(f"  Sans contenu          : {sans_contenu}")
    print(f"  Longueur contenu moy. : {len_moy}")
    if lang_col:
        langs = c.execute(
            f"SELECT {lang_col}, COUNT(*) FROM articles WHERE source_name = ? GROUP BY 1 ORDER BY 2 DESC",
            [src]).fetchall()
        print(f"  Langues               : {langs}")

c.close()

# --- Test de collecte en direct ---
print()
print("=" * 70)
print("TEST DE COLLECTE EN DIRECT")
print("=" * 70)

try:
    import yaml
    with open("config/sources.yaml", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    entries = {e["name"]: e for e in conf.get("sources", [])}
except Exception as ex:
    print(f"Impossible de lire sources.yaml : {ex}")
    entries = {}

try:
    import httpx
    from bs4 import BeautifulSoup
except ImportError as ex:
    raise SystemExit(f"Dependance manquante : {ex}")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

for src in SOURCES:
    print(f"\n--- {src} ---")
    e = entries.get(src)
    if not e:
        print("  Pas d'entree dans sources.yaml sous ce nom exact.")
        continue
    url = e.get("url", "")
    mode = e.get("type", "rss")
    print(f"  URL   : {url}")
    print(f"  Mode  : {mode}")
    try:
        r = httpx.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        print(f"  HTTP  : {r.status_code}, {len(r.content)} octets")
    except Exception as ex:
        print(f"  ECHEC reseau : {type(ex).__name__} - {ex}")
        print("  >>> Si timeout : source bloquee par le reseau (VPN requis).")
        continue

    ctype = r.headers.get("content-type", "")
    body = r.text
    if "xml" in ctype or "rss" in ctype or body.lstrip()[:5].lower().startswith("<?xml") or "<rss" in body[:500].lower():
        n_items = body.lower().count("<item")
        n_entry = body.lower().count("<entry")
        print(f"  Flux RSS/Atom : {n_items} <item>, {n_entry} <entry>")
        if n_items == 0 and n_entry == 0:
            print("  >>> Flux vide ou format inattendu.")
    else:
        soup = BeautifulSoup(body, "html.parser")
        links = soup.find_all("a", href=True)
        from urllib.parse import urlparse
        dom = urlparse(url).netloc
        arts = [a["href"] for a in links if dom in a["href"] and a["href"].count("/") >= 4]
        arts = list(dict.fromkeys(arts))
        print(f"  Page HTML : {len(links)} liens, {len(arts)} liens d'articles plausibles")
        for a in arts[:5]:
            print(f"    {a}")
        if len(arts) == 0:
            print("  >>> Aucun lien d'article reconnu : structure HTML changee,")
            print("      le scraper ne trouve plus les articles.")

print()
print("=" * 70)
print("Envoyez cette sortie complete pour le diagnostic.")
