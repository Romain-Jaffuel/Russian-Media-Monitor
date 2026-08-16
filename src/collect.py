"""Fetch RSS feeds and turn entries into RawArticle objects."""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import feedparser


@dataclass
class RawArticle:
    id: str
    source_name: str
    feed_url: str
    url: str
    title: str
    author: str | None
    summary: str | None
    published_at: datetime | None
    # Renseignes uniquement par les collecteurs qui recuperent le texte en
    # un seul appel (ex: src/telegram_scrape.py) -- le pipeline saute alors
    # le second fetch + extract() normalement fait pour RSS/scrape HTML.
    content: str | None = None
    language: str | None = None
    # Vues de la video/emission parente, quand la plateforme les publie.
    # Identique sur tous les segments d'une meme video.
    view_count: int | None = None


def url_hash(url: str) -> str:
    return sha256(url.encode("utf-8")).hexdigest()[:16]


def fetch_feed(source_name: str, feed_url: str) -> list[RawArticle]:
    parsed = feedparser.parse(feed_url)
    out: list[RawArticle] = []
    for entry in parsed.entries:
        url = entry.get("link")
        if not url:
            continue
        published: datetime | None = None
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6])
        elif entry.get("updated_parsed"):
            published = datetime(*entry.updated_parsed[:6])
        out.append(RawArticle(
            id=url_hash(url),
            source_name=source_name,
            feed_url=feed_url,
            url=url,
            title=entry.get("title", "").strip(),
            author=entry.get("author"),
            summary=entry.get("summary"),
            published_at=published,
        ))
    return out
