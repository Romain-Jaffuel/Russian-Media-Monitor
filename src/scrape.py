"""Scraping de pages d'accueil pour sites sans flux RSS exploitable.

Heuristique : on récupère tous les liens internes de la page d'accueil,
on filtre ceux qui ressemblent à des articles (chemin avec tirets et
suffisamment long, pas de patterns de navigation/admin), et on retourne
des RawArticle prêts à être passés au pipeline standard.
"""
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.collect import RawArticle, url_hash
from src.extract import fetch_html

# Chemins à exclure (pages de nav, catégories, médias, etc.)
EXCLUDE_RE = re.compile(
    r"/(tag|tags|author|page|wp-content|wp-json|wp-admin|"
    r"wp-includes|feed|comments|search|login|register|contact|about|"
    r"mentions|privacy|cookies|rss|atom|sitemap|cdn-cgi)(/|$|\?)"
    r"|\.(jpg|jpeg|png|gif|svg|webp|pdf|zip|mp4|mp3|css|js)(\?|$)"
    r"|#",
    re.IGNORECASE,
)


def is_article_url(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != base_domain:
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    if EXCLUDE_RE.search(parsed.path):
        return False
    # Trop profond = probable page de pagination ou archive
    if path.count("/") > 6:
        return False
    # Rejeter les pages de rubrique /category|cat|rubrique se terminant SANS
    # slug d'article (ex: /cat/actualites/ ou /cat/actualites/economie/).
    seg = [s for s in path.split("/") if s]
    if seg and seg[0] in ("category", "cat", "rubrique", "rubriques"):
        last_seg = seg[-1]
        looks_like_article = last_seg.count("-") >= 2 or len(last_seg) >= 20
        if not looks_like_article:
            return False
    # Le dernier segment doit ressembler à un slug d'article, à un ID
    # numérique (courant sur les sites russes type TASS/RIA), ou le chemin
    # contient un motif de date (/2026/08/11/...).
    last = path.rsplit("/", 1)[-1]
    looks_like_slug = len(last) >= 20 or last.count("-") >= 2
    looks_like_numeric_id = last.isdigit() and len(last) >= 5
    has_date_path = bool(re.search(r"/\d{4}/\d{2}/\d{2}(/|$)", path))
    if not (looks_like_slug or looks_like_numeric_id or has_date_path):
        return False
    return True


def extract_article_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Renvoie [(url, titre), ...] des articles trouvés sur la page."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc

    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"]).split("#")[0].rstrip("/")
        if full in seen:
            continue
        if not is_article_url(full, base_domain):
            continue
        seen.add(full)
        title = (
            a.get_text(strip=True)
            or a.get("aria-label", "")
            or a.get("title", "")
        ).strip()[:300]
        out.append((full, title))

    return out


def scrape_homepage(source_name: str, homepage_url: str) -> list[RawArticle]:
    """Récupère l'accueil et produit une liste de RawArticle candidats."""
    html = fetch_html(homepage_url)
    if not html:
        return []
    articles: list[RawArticle] = []
    for url, title in extract_article_links(html, homepage_url):
        articles.append(
            RawArticle(
                id=url_hash(url),
                source_name=source_name,
                feed_url=homepage_url,
                url=url,
                title=title,
                author=None,
                summary=None,
                published_at=None,
            )
        )
    return articles
