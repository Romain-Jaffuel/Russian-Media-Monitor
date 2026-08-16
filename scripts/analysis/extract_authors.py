"""Extraction d'auteurs (V3).

Ameliorations vs V2 :
  - Filtre des handles Twitter (@xxx) qu'on prenait pour des auteurs (cas AGP)
  - Strip plus robuste des prefixes "By", "Par", "Auteur" colles sans espace
    (cas "ByRedac chef" / "AuteurLa Redaction")
  - Extracteur dedie pour les signatures AGP en fin d'article (ex "Roger MOUSSAVOU/AGP")
  - Patterns bad-author elargis

Usage : python scripts/analysis/extract_authors.py
"""
import json
import re

from bs4 import BeautifulSoup

from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("authors")

BAD_AUTHOR_PATTERNS = [
    re.compile(r"^(la\s+)?(redaction|rédaction|redac[\s\-]*chef|redac[\s\-]*ch?ef)\s*$", re.IGNORECASE),
    re.compile(r"modérateur|moderateur", re.IGNORECASE),
    re.compile(r"^admin$|^administrateur$", re.IGNORECASE),
    re.compile(r"^unknown\b", re.IGNORECASE),
    re.compile(r"noreply|no-reply", re.IGNORECASE),
    re.compile(r"^info\d+(\.com)?$", re.IGNORECASE),
    # Handles Twitter / reseaux sociaux
    re.compile(r"^@\w+$"),
    re.compile(r"^@[A-Za-z0-9_]+$"),
    # Generiques rédaction divers
    re.compile(r"^(notre|la)?\s*(team|equipe|équipe|staff)\s*$", re.IGNORECASE),
    re.compile(r"^auteur(s)?\s*$", re.IGNORECASE),
    re.compile(r"^webmaster$", re.IGNORECASE),
    # Signatures d'agence isolees
    re.compile(r"^tass$|^ria$|^interfax$|^reuters$|^ap$", re.IGNORECASE),
    re.compile(r"^редакция$", re.IGNORECASE),
    # Raison sociale plutot que personne (ex: "TV Rain, Inc." -- constate sur
    # un champ meta hors JSON-LD, donc pas couvert par le filtre @type)
    re.compile(r"\b(inc|llc|ltd|corp|gmbh|ооо|зао|пао)\.?\s*$", re.IGNORECASE),
]

# Auto-signatures d'organisation russes (nom du media lui-meme comme "auteur"
# JSON-LD, cf. extract_from_jsonld) constatees en base -- le fix a la source
# (filtre @type == "Organization") empeche toute nouvelle occurrence, mais ne
# nettoie pas ce qui a deja ete extrait avant ce fix.
KNOWN_ORG_SELFNAMES = {
    "ведомости", "риа новости", "риа новости спорт", "rt на русском",
    "медиазона", "телеканал дождь", "тасс", "известия", "коммерсантъ",
    "коммерсант", "рбк", "независимая газета", "комсомольская правда",
    "медуза", "meduza", "новая газета", "новая газета европа", "the bell",
    "проект", "спутник", "первый канал", "вести", "вгтрк", "the insider",
    "важные истории",
}

AUTHOR_SELECTORS = [
    ("meta", {"name": "author"}, "content"),
    ("meta", {"property": "article:author"}, "content"),
    # twitter:creator deplace en fin de priorite : c'est souvent un handle, pas un auteur
    ("span", {"itemprop": "author"}, None),
    ("div", {"itemprop": "author"}, None),
    ("a", {"itemprop": "author"}, None),
    ("a", {"rel": "author"}, None),
    ("a", {"class": re.compile(r"author", re.I)}, None),
    ("span", {"class": re.compile(r"author", re.I)}, None),
    ("p", {"class": re.compile(r"author", re.I)}, None),
    ("div", {"class": re.compile(r"author", re.I)}, None),
    ("span", {"class": re.compile(r"byline", re.I)}, None),
    ("div", {"class": re.compile(r"byline", re.I)}, None),
    ("p", {"class": re.compile(r"byline", re.I)}, None),
    ("span", {"class": re.compile(r"writer|signature|posted-by", re.I)}, None),
    ("div", {"class": re.compile(r"td-post-author-name", re.I)}, None),
    ("span", {"class": re.compile(r"td-post-author-name", re.I)}, None),
    ("span", {"class": re.compile(r"entry-author|posted-on", re.I)}, None),
    # twitter:creator en dernier (souvent juste un handle)
    ("meta", {"name": "twitter:creator"}, "content"),
    ("meta", {"name": "twitter:data1"}, "content"),
]


# Pattern AGP : "Prénom NOM / AGP" ou "Prénom NOM/AGP"
AGP_SIGNATURE_RE = re.compile(
    r"\b([A-ZÉÈÊÀÂÔÇ][A-ZÉÈÊÀÂÔÇa-zéèêàâôç\-']{2,}"
    r"(?:\s+[A-ZÉÈÊÀÂÔÇ][A-ZÉÈÊÀÂÔÇa-zéèêàâôç\-']{2,}){0,3})"
    r"\s*/\s*AGP\b"
)


def is_bad_author(author: str | None, source_name: str | None = None) -> bool:
    if not author or not author.strip():
        return True
    t = author.strip()
    for pat in BAD_AUTHOR_PATTERNS:
        if pat.search(t):
            return True
    if t.lower() in KNOWN_ORG_SELFNAMES:
        return True
    if source_name:
        s = source_name.lower().strip()
        a = t.lower()
        if a == s or a.replace(" ", "") == s.replace(" ", ""):
            return True
        # Match partiel : si l'auteur "est" la source (ex "Meduza.io" pour source "Meduza")
        if len(s) > 4 and s in a:
            return True
    if len(t) < 3 or len(t) > 60:
        return True
    return False


def clean_extracted(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    # Strip prefixes meme sans espace (par/by/auteur/author + separateur optionnel)
    t = re.sub(r"^(par|by|auteur|author|автор)[\s:.\-]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+(le|on)\s+\d.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+\d{4}-\d{2}-\d{2}.*$", "", t)
    t = re.sub(r"\s*\S+@\S+\s*$", "", t)
    if t and t == t.lower():
        t = t.title()
    if len(t) < 3 or len(t) > 60:
        return None
    return t


def extract_from_jsonld(soup) -> str | None:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text()
            if not raw:
                continue
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            if "@graph" in item and isinstance(item["@graph"], list):
                items.extend(item["@graph"])
                continue
            author = item.get("author")
            name = None
            if isinstance(author, dict):
                # Beaucoup de sites (Vedomosti, RIA, RT, Mediazona, TV Rain...)
                # renvoient l'auteur JSON-LD par defaut a "Organization" (le
                # media lui-meme) quand l'article n'a pas de byline humaine --
                # on ne veut que les auteurs "Person".
                if author.get("@type") == "Organization":
                    continue
                name = author.get("name")
            elif isinstance(author, list) and author:
                first = author[0]
                if isinstance(first, dict):
                    if first.get("@type") == "Organization":
                        continue
                    name = first.get("name")
                elif isinstance(first, str):
                    name = first
            elif isinstance(author, str):
                name = author
            if name:
                cleaned = clean_extracted(name)
                if cleaned:
                    return cleaned
    return None


def extract_par_pattern(soup) -> str | None:
    """Cherche un pattern 'Par X' ou 'By X' dans les premiers paragraphes."""
    pattern = re.compile(
        r"\b(?:par|by|автор)\s+([A-ZÉÈÊÀÂÔÇа-яёА-ЯЁa-zéèêàâôç]+"
        r"(?:[\s\-][A-ZÉÈÊÀÂÔÇа-яёА-ЯЁa-zéèêàâôç]+){0,4})",
        re.IGNORECASE,
    )
    candidates_tags = soup.find_all(["p", "div", "span", "small"], limit=20)
    for tag in candidates_tags:
        text = tag.get_text(" ", strip=True)
        if len(text) > 300 or len(text) < 4:
            continue
        m = pattern.search(text)
        if m:
            cand = m.group(1).strip()
            if 5 <= len(cand) <= 50 and " " in cand:
                cleaned = clean_extracted(cand)
                if cleaned:
                    return cleaned
    return None


def extract_agp_signature(soup) -> str | None:
    """Cherche une signature de type 'Roger MOUSSAVOU/AGP' en fin d'article."""
    # Recupere tout le texte, regarde la fin
    full_text = soup.get_text(" ", strip=True)
    # Cherche dans les derniers 1500 caracteres en priorite (fin d'article)
    tail = full_text[-1500:] if len(full_text) > 1500 else full_text
    m = AGP_SIGNATURE_RE.search(tail)
    if not m:
        m = AGP_SIGNATURE_RE.search(full_text)
    if m:
        cand = m.group(1).strip()
        cleaned = clean_extracted(cand)
        if cleaned:
            return cleaned
    return None


def extract_author_from_html(html: str | None, source_name: str | None = None) -> str | None:
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    # AGP : prioriser la signature en fin d'article AVANT tout autre extracteur,
    # car le meta twitter:creator donne @AgencepresseAGP, qui est pourri.
    if source_name and "agp" in source_name.lower():
        sig = extract_agp_signature(soup)
        if sig and not is_bad_author(sig, source_name):
            return sig

    # 1) JSON-LD (fiable sur WP moderne)
    author = extract_from_jsonld(soup)
    if author and not is_bad_author(author, source_name):
        return author

    # 2) Selecteurs HTML (twitter:creator est en fin de liste pour eviter @handles)
    for tag, attrs, content_attr in AUTHOR_SELECTORS:
        for el in soup.find_all(tag, attrs=attrs):
            raw = el.get(content_attr) if content_attr else el.get_text(strip=True)
            cleaned = clean_extracted(raw)
            if cleaned and not is_bad_author(cleaned, source_name):
                return cleaned

    # 3) Pattern "Par X" dans les premiers paragraphes
    author = extract_par_pattern(soup)
    if author and not is_bad_author(author, source_name):
        return author

    # 4) Signature AGP en dernier recours (pour sources non-AGP qui republient AGP)
    sig = extract_agp_signature(soup)
    if sig and not is_bad_author(sig, source_name):
        return sig

    return None


def run():
    conn = get_conn()

    log.info("Etape 1 : audit des auteurs existants...")
    rows = conn.execute(
        "SELECT id, author, source_name FROM articles WHERE author IS NOT NULL AND TRIM(author) != ''"
    ).fetchall()
    nullified = 0
    for art_id, author, source in rows:
        if is_bad_author(author, source):
            conn.execute("UPDATE articles SET author = NULL WHERE id = ?", [art_id])
            nullified += 1
    log.info("  %d auteurs invalides remis a NULL", nullified)

    log.info("Etape 2 : extraction depuis HTML brut...")
    rows = conn.execute(
        """
        SELECT id, raw_html, source_name FROM articles
        WHERE (author IS NULL OR TRIM(author) = '') AND raw_html IS NOT NULL
        """
    ).fetchall()
    log.info("  %d articles a analyser", len(rows))

    found = 0
    found_by_source = {}
    for i, (art_id, html, source) in enumerate(rows):
        author = extract_author_from_html(html, source)
        if author and not is_bad_author(author, source):
            conn.execute("UPDATE articles SET author = ? WHERE id = ?", [author, art_id])
            found += 1
            found_by_source[source] = found_by_source.get(source, 0) + 1
        if (i + 1) % 200 == 0:
            log.info("  ... %d/%d", i + 1, len(rows))
    log.info("  %d auteurs recuperes depuis le HTML", found)

    print("\n=== AUTEURS NOUVELLEMENT RECUPERES PAR SOURCE ===")
    for src, n in sorted(found_by_source.items(), key=lambda x: -x[1]):
        print(f"  {n:>3}  {src}")

    print("\n=== TOP 30 AUTEURS APRES NETTOYAGE ===")
    for src, auth, n in conn.execute(
        """
        SELECT source_name, author, COUNT(*) AS n FROM articles
        WHERE author IS NOT NULL AND TRIM(author) != ''
        GROUP BY 1, 2 ORDER BY n DESC LIMIT 30
        """
    ).fetchall():
        print(f"  {n:>3}  [{src}]  {auth}")

    unknown = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE author IS NULL OR TRIM(author) = ''"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\nCouverture : {total - unknown}/{total} ({100*(total-unknown)/total:.1f}%)")
    conn.close()


if __name__ == "__main__":
    run()
