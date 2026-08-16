"""Pipeline parallélisée. Quatre types de sources, via `type:` dans
config/sources.yaml (sans clé `type`, c'est du RSS) :

    type: scrape      # site sans flux RSS, on parcourt la page d'accueil
    type: telegram    # canal public, via l'aperçu web t.me/s/<canal>
    type: youtube     # chaîne vidéo, via ses sous-titres découpés en segments

Exemple :

    - name: TV Rain / Dozhd
      url: https://tvrain.tv/
      type: scrape
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import yaml

from src.collect import fetch_feed
from src.db import get_conn
from src.extract import extract, fetch_html
from src.logging_setup import setup_logging
from src.scrape import scrape_homepage
from src.telegram_scrape import scrape_telegram_channel
from src.youtube_scrape import scrape_youtube_channel

log = setup_logging("pipeline")

SOURCES_PATH = Path("config/sources.yaml")
INTRA_SOURCE_DELAY = 0.1
MAX_WORKERS = 20
PAYS = "Russie"

# `type:` dans sources.yaml -> valeur de la colonne source_kind. Les types de
# collecte qui produisent de la presse classique (rss, scrape) partagent le
# meme kind : c'est la nature du contenu qui est filtree dans le dashboard,
# pas la technique de collecte.
SOURCE_KINDS = {"telegram": "telegram", "youtube": "youtube",
                "rutube": "tv", "hls": "tv", "vk": "vk"}


def load_sources(path: Path = SOURCES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def process_source(
    src: dict, conn, db_lock: Lock, seen_ids: set, seen_lock: Lock
) -> tuple[str, int, int, int]:
    name = src["name"]
    feed_url = src["url"]
    source_type = src.get("type", "rss")
    media_type = src.get("media_type")
    legal_status = src.get("legal_status")
    source_kind = SOURCE_KINDS.get(source_type, "press")
    new = 0
    extract_fails = 0

    try:
        if source_type == "scrape":
            entries = scrape_homepage(name, feed_url)
        elif source_type == "telegram":
            entries = scrape_telegram_channel(name, feed_url)
        elif source_type == "youtube":
            # seen_ids : permet de sauter les videos deja transcrites sans
            # aller chercher leurs metadonnees (cf. youtube_scrape).
            entries = scrape_youtube_channel(name, feed_url, seen_ids, seen_lock)
        elif source_type == "vk":
            # Import tardif : Playwright n'a pas a etre charge quand aucune
            # source VK n'est configuree.
            from src.vk_scrape import scrape_vk_community
            entries = scrape_vk_community(name, feed_url)
        elif source_type == "hls":
            # Import tardif : charge Playwright et le moteur de transcription.
            from src.hls_scrape import scrape_hls_program
            entries = scrape_hls_program(
                name, feed_url, src.get("episode_pattern"), seen_ids, seen_lock,
                src.get("program_pattern"))
        elif source_type == "rutube":
            # Import tardif : charge torch/ctranslate2, inutile de payer ce
            # cout de demarrage quand aucune source TV n'est configuree.
            from src.rutube_scrape import scrape_rutube_program
            entries = scrape_rutube_program(
                name, feed_url, src.get("program_pattern"), seen_ids, seen_lock,
                src.get("search_query"), src.get("min_duration"))
        else:
            entries = fetch_feed(name, feed_url)
    except Exception as e:
        log.error("[%s] source injoignable (%s): %s", name, source_type, e)
        return name, 0, 0, 1

    for art in entries:
        with seen_lock:
            if art.id in seen_ids:
                continue
            seen_ids.add(art.id)

        # Le scraper Telegram renvoie deja le texte complet en un seul
        # fetch (pas de page "article" separee a aller chercher) : on saute
        # le fetch_html + extract() sinon systematique.
        if art.content is not None:
            content, language, html = art.content, art.language, None
        else:
            html = fetch_html(art.url)
            if html is None:
                extract_fails += 1
                time.sleep(INTRA_SOURCE_DELAY)
                continue

            content, language = extract(html)

            # En mode scrape, si Trafilatura n'a rien extrait,
            # c'était sans doute une page de nav. On passe.
            if source_type == "scrape" and (content is None or len(content) < 200):
                extract_fails += 1
                time.sleep(INTRA_SOURCE_DELAY)
                continue

        with db_lock:
            try:
                conn.execute(
                    """INSERT INTO articles
                       (id, source_name, feed_url, url, title, author, summary,
                        content, language, published_at, raw_html,
                        pays, type_media, statut_legal_ru, source_kind,
                        view_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        art.id, art.source_name, art.feed_url, art.url, art.title,
                        art.author, art.summary, content, language,
                        art.published_at, html,
                        PAYS, media_type, legal_status, source_kind,
                        art.view_count,
                    ],
                )
                new += 1
            except Exception as e:
                log.error("[%s] insert: %s", name, e)

        time.sleep(INTRA_SOURCE_DELAY)

    return name, new, extract_fails, 0


def run():
    conn = get_conn()
    db_lock = Lock()
    seen_lock = Lock()

    seen_ids: set[str] = {
        row[0] for row in conn.execute("SELECT id FROM articles").fetchall()
    }
    log.info("DB existante: %d articles en mémoire pour dédup", len(seen_ids))

    sources = load_sources()
    rss_count = sum(1 for s in sources if s.get("type", "rss") == "rss")
    scrape_count = sum(1 for s in sources if s.get("type") == "scrape")
    telegram_count = sum(1 for s in sources if s.get("type") == "telegram")
    youtube_count = sum(1 for s in sources if s.get("type") == "youtube")
    log.info(
        "Sources: %d (RSS: %d, scrape: %d, telegram: %d, youtube: %d). Workers parallèles: %d",
        len(sources), rss_count, scrape_count, telegram_count, youtube_count, MAX_WORKERS,
    )

    started = time.time()
    total_new = 0
    total_extract_fails = 0
    total_feed_errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_source, src, conn, db_lock, seen_ids, seen_lock): src
            for src in sources
        }
        for future in as_completed(futures):
            name, new, fails, feed_err = future.result()
            total_new += new
            total_extract_fails += fails
            total_feed_errors += feed_err
            tag = "KO source" if feed_err else f"+{new} (rejets: {fails})"
            log.info("[%s] %s", name, tag)

    elapsed = time.time() - started
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    log.info(
        "Terminé en %.1fs. Nouveaux: %d. Sources KO: %d. Rejets: %d. Total: %d",
        elapsed, total_new, total_feed_errors, total_extract_fails, total,
    )
    conn.close()


if __name__ == "__main__":
    run()
