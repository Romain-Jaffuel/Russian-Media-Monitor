"""Scraping de canaux Telegram via leur apercu web public (t.me/s/<canal>).

Pas d'authentification necessaire : c'est la meme page que Telegram sert a
un navigateur non connecte. Certains gros canaux (rian_ru, rt_russian,
tvrussia1...) desactivent cet apercu -- ceux-la ne sont pas collectables par
ce module (il faudrait un client authentifie type telethon/MTProto).

A la difference de fetch_feed()/scrape_homepage(), un seul fetch donne a la
fois la liste des posts ET leur texte complet (RawArticle.content deja
rempli), donc pas de second aller-retour HTTP par article.
"""
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

from src.collect import RawArticle, url_hash

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 RussiaMonitor/0.1"
)
MIN_POST_LEN = 50  # sous ce seuil : probablement un post juste-image/repost, sans valeur textuelle
MAX_PAGES = 5       # ~20 messages/page ; suffisant pour un run incremental,
                     # rattrape un backlog de plusieurs jours sur les canaux actifs


def _fetch(url: str) -> str | None:
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True,
                           headers={"User-Agent": USER_AGENT}) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.text
    except Exception:
        return None


def _clean_text(message_text_div) -> str:
    """Convertit le HTML d'un message (liens, gras, <br>) en texte brut,
    en preservant les sauts de ligne."""
    for br in message_text_div.find_all("br"):
        br.replace_with("\n")
    return message_text_div.get_text().strip()


def _parse_page(html: str, channel: str) -> tuple[list[dict], str | None]:
    """Renvoie (posts, before_id_pour_page_suivante)."""
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    oldest_id = None
    for msg in soup.select(".tgme_widget_message"):
        post_id = msg.get("data-post")  # format "channel/12345"
        if not post_id:
            continue
        msg_num = post_id.split("/")[-1]
        if msg_num.isdigit():
            oldest_id = msg_num if oldest_id is None else min(oldest_id, msg_num, key=int)

        # ".tgme_widget_message_text" seul matche AUSSI l'aperçu de citation
        # d'un message auquel on repond (js-message_reply_text), tronque a
        # ~250 car. par Telegram -- select_one() prenait alors ce fragment
        # au lieu du vrai corps (js-message_text), quand la citation
        # apparait en premier dans le DOM.
        text_div = msg.select_one(".js-message_text")
        if not text_div:
            continue  # post media-only (photo/video sans legende) : rien a analyser
        text = _clean_text(text_div)
        if len(text) < MIN_POST_LEN:
            continue

        time_el = msg.select_one(".tgme_widget_message_date time")
        dt = None
        if time_el and time_el.get("datetime"):
            try:
                dt = datetime.fromisoformat(time_el["datetime"].replace("Z", "+00:00"))
            except ValueError:
                dt = None

        posts.append({
            "url": f"https://t.me/{post_id}",
            "text": text,
            "published_at": dt,
        })
    return posts, oldest_id


def _dedupe_posts(posts: list[dict]) -> list[dict]:
    """Un message cite/transfere imbrique dans un autre post partage le
    meme selecteur CSS que les messages de premier niveau et se fait donc
    parfois capter comme un post a part entiere -- avec un texte tronque
    identique au debut du vrai message. On ne garde que la version la plus
    longue par prefixe de texte."""
    best_by_prefix: dict[str, dict] = {}
    for p in posts:
        key = p["text"][:80]
        cur = best_by_prefix.get(key)
        if cur is None or len(p["text"]) > len(cur["text"]):
            best_by_prefix[key] = p
    return list(best_by_prefix.values())


def scrape_telegram_channel(source_name: str, channel_url: str) -> list[RawArticle]:
    """channel_url attendu : https://t.me/s/<canal> (voir config/sources.yaml)."""
    m = re.search(r"t\.me/s/([A-Za-z0-9_]+)", channel_url)
    if not m:
        return []
    channel = m.group(1)

    all_posts: list[dict] = []
    before = None
    for _ in range(MAX_PAGES):
        url = channel_url if before is None else f"{channel_url}?before={before}"
        html = _fetch(url)
        if not html:
            break
        posts, oldest_id = _parse_page(html, channel)
        if not posts:
            break
        all_posts.extend(posts)
        if oldest_id is None or oldest_id == before:
            break  # plus de pages precedentes (ou pagination bloquee)
        before = oldest_id
        time.sleep(0.3)

    all_posts = _dedupe_posts(all_posts)

    out: list[RawArticle] = []
    for p in all_posts:
        text = p["text"]
        try:
            lang = detect(text)
        except LangDetectException:
            lang = None
        title = text.split("\n", 1)[0].strip()[:120] or text[:120]
        out.append(RawArticle(
            id=url_hash(p["url"]),
            source_name=source_name,
            feed_url=channel_url,
            url=p["url"],
            title=title,
            author=None,
            summary=None,
            published_at=p["published_at"],
            content=text,
            language=lang,
        ))
    return out
