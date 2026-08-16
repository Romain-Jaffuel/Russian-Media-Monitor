"""Emissions de television, par interception du flux video.

Les extracteurs type yt-dlp codent en dur l'API de chaque plateforme : depuis
que VGTRK a migre son lecteur en JavaScript, l'extracteur smotrim est casse et
la chaine la plus regardee du pays devient inaccessible. Ici on ne
reimplemente rien -- on ouvre la page dans un navigateur et on ecoute son
trafic. Quel que soit le lecteur, il finit par demander un manifeste HLS, et
ffmpeg sait lire ce qu'il designe.

Valide sur smotrim.ru, 1tv.ru et ntv.ru. La suite est celle de
rutube_scrape.py : ffmpeg extrait la bande son, Whisper transcrit.
"""
import logging
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import httpx

from src.collect import RawArticle, url_hash
# Meme moteur de transcription et meme decoupage que les autres sources
# video : un segment de TV doit rester comparable a un segment de YouTube.
from src.rutube_scrape import _transcribe
from src.youtube_scrape import MAX_TITLE_CHARS, _segment

log = logging.getLogger("hls")

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

MAX_EPISODES = 1          # un episode par emission et par run : ces formats
                          # durent 1 h a 3 h, et trois emissions suivies font
                          # deja ~6 h d'audio, soit une demi-heure de GPU
MIN_DURATION = 600        # 10 min : ecarte les extraits
MAX_AGE_DAYS = 21
PAGE_TIMEOUT_MS = 45000
PLAYER_WAIT_MS = 9000     # temps laisse au lecteur pour demander son flux
FFMPEG_TIMEOUT = 1800

# Un seul navigateur a la fois, comme pour VK : le pipeline collecte en
# parallele et rien ne gagne a ouvrir plusieurs Chromium.
_browser_lock = Lock()

_MEDIA_RE = re.compile(r"\.m3u8", re.I)


def _episode_links(listing_url: str, pattern: str) -> list[str]:
    """URLs d'episodes trouvees sur la page de l'emission."""
    try:
        r = httpx.get(listing_url, headers={"User-Agent": USER_AGENT},
                      timeout=30.0, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        log.warning("listing %s : %s", listing_url, str(e)[:90])
        return []

    found = re.findall(pattern, r.text)
    # base = schema + domaine de la page de listing
    m = re.match(r"(https?://[^/]+)", str(r.url))
    base = m.group(1) if m else ""
    out, seen = [], set()
    for f in found:
        href = f if isinstance(f, str) else f[0]
        full = href if href.startswith("http") else base + href
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def _page_title(page_url: str) -> str:
    """Titre lu par une simple requete HTTP, avant d'ouvrir le navigateur.

    Les pages d'emission listent aussi des recommandations : sans ce controle
    prealable, on ouvrait un Chromium et on transcrivait une emission voisine
    (constate : une page « Vesti Nedeli » a ramene un « Vecher s Solovyovym »).
    Filtrer sur le titre ici coute une requete, contre une minute de
    navigateur et plusieurs minutes de GPU.
    """
    try:
        r = httpx.get(page_url, headers={"User-Agent": USER_AGENT},
                      timeout=20.0, follow_redirects=True)
        m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]{0,300})"', r.text)
        if m:
            return m.group(1)
        m = re.search(r"<title>([^<]{0,300})</title>", r.text)
        return m.group(1) if m else ""
    except Exception:
        return ""


# « Эфир 02.08.2026 » : ces sites ne posent pas de meta de date, mais leur
# titre porte la date de diffusion.
_DATE_IN_TITLE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")


def _date_from_title(title: str) -> datetime | None:
    m = _DATE_IN_TITLE.search(title or "")
    if not m:
        return None
    try:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)),
                        tzinfo=timezone.utc)
    except ValueError:
        return None


def _sniff_stream(page_url: str) -> tuple[str | None, dict]:
    """Ouvre la page et renvoie (URL du manifeste, metadonnees lues au passage).

    On ecoute toutes les requetes : le manifeste maitre (playlist.m3u8)
    apparait des que le lecteur s'initialise. Certains lecteurs attendent un
    clic, d'ou la tentative sur les boutons de lecture usuels.
    """
    from playwright.sync_api import sync_playwright

    urls: list[str] = []
    meta: dict = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            ctx = browser.new_context(
                viewport={"width": 1280, "height": 900}, locale="ru-RU",
                user_agent=USER_AGENT)
            page = ctx.new_page()
            page.on("request",
                    lambda r: urls.append(r.url) if _MEDIA_RE.search(r.url) else None)
            page.goto(page_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
            page.wait_for_timeout(5000)

            for sel in (".vjs-big-play-button", '[class*="play-button"]',
                        '[class*="Play"]', "video"):
                try:
                    loc = page.locator(sel).first
                    if loc.is_visible(timeout=1200):
                        loc.click(timeout=2500)
                        break
                except Exception:
                    pass
            page.wait_for_timeout(PLAYER_WAIT_MS)

            meta["title"] = (page.title() or "").strip()
            for prop in ("og:title", "og:video:duration", "article:published_time"):
                try:
                    v = page.locator(f'meta[property="{prop}"]').first.get_attribute(
                        "content", timeout=800)
                    if v:
                        meta[prop] = v
                except Exception:
                    pass
        except Exception as e:
            log.warning("lecture de %s : %s", page_url, str(e)[:100])
        finally:
            browser.close()

    # Le manifeste maitre liste les qualites ; les « chunklist » n'en sont
    # qu'une declinaison. On prefere le maitre, ffmpeg choisira.
    master = next((u for u in urls if "playlist.m3u8" in u.lower()), None)
    if not master:
        master = next((u for u in urls if "chunklist" not in u.lower()), None)
    if not master and urls:
        master = urls[0]
    return master, meta


def _extract_audio(manifest: str, workdir: Path) -> Path | None:
    """Bande son seule : `-vn` evite de telecharger la video pour rien.

    Deux garde-fous appris a l'usage : un segment HLS peut cesser de repondre
    et ffmpeg attend alors indefiniment (constate : 13 min sans un octet
    ecrit, a 95 % du telechargement). `-rw_timeout` le fait abandonner une
    lecture qui trainte, et si le delai global est malgre tout atteint on
    exploite le fichier partiel plutot que de tout jeter -- le MP3 se lit par
    flux, une transcription amputee de sa fin vaut mieux que rien.
    """
    out = workdir / "audio.mp3"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-user_agent", USER_AGENT,
           # microsecondes : 60 s sans octet lu -> on abandonne la lecture
           "-rw_timeout", "60000000",
           "-i", manifest,
           "-vn", "-acodec", "libmp3lame", "-ab", "64k", str(out)]
    partiel = False
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        rc, err = r.returncode, r.stderr or ""
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg : delai global depasse sur %s", manifest[:70])
        rc, err, partiel = 1, "timeout", True

    if not out.exists() or out.stat().st_size < 200_000:
        log.warning("ffmpeg (%s) : %s", rc, err[:130])
        return None
    if rc != 0:
        # ffmpeg interrompu, mais un fichier consequent existe : on continue
        # avec ce qu'on a, en le signalant.
        log.warning("ffmpeg interrompu (%s%s) -- on exploite les %d Mo ecrits",
                    "delai global" if partiel else f"code {rc}",
                    "", out.stat().st_size // (1024 * 1024))
    return out


def _duration_seconds(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float((r.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def _clean_title(meta: dict, fallback: str) -> str:
    t = meta.get("og:title") or meta.get("title") or fallback
    # les titres de page trainent souvent le nom du site derriere un separateur
    t = re.split(r"\s+[|—–-]\s+(?:Смотрим|СМОТРИМ|Первый канал|НТВ)\s*$", t)[0]
    return t.strip()[:MAX_TITLE_CHARS]


def _published_at(meta: dict) -> datetime | None:
    raw = meta.get("article:published_time")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _segment_id(page_url: str, index: int) -> str:
    return url_hash(f"{page_url}#seg{index}")


def scrape_hls_program(
    source_name: str, listing_url: str, episode_pattern: str | None = None,
    seen_ids: set | None = None, seen_lock=None,
    program_pattern: str | None = None,
) -> list[RawArticle]:
    """listing_url : page de l'emission. episode_pattern : regex des liens
    d'episode sur cette page (ex. `/video/\\d+` pour smotrim). program_pattern :
    regex sur le titre, pour ecarter les emissions recommandees a cote."""
    if not episode_pattern:
        log.warning("[%s] episode_pattern absent", source_name)
        return []

    liens = _episode_links(listing_url, episode_pattern)
    if not liens:
        log.warning("[%s] aucun lien d'episode sur %s", source_name, listing_url)
        return []

    prog_re = re.compile(program_pattern, re.I) if program_pattern else None
    now = datetime.now(timezone.utc)
    out: list[RawArticle] = []
    traites = 0
    for page_url in liens:
        if traites >= MAX_EPISODES:
            break
        if seen_ids is not None:
            probe = _segment_id(page_url, 0)
            with (seen_lock or _browser_lock):
                if probe in seen_ids:
                    continue

        # Tri en amont, sur une simple requete : titre attendu et date encore
        # dans la fenetre. Tout ce qui est ecarte ici economise une session de
        # navigateur et plusieurs minutes de GPU.
        titre_page = _page_title(page_url)
        if prog_re and titre_page and not prog_re.search(titre_page):
            continue
        d_titre = _date_from_title(titre_page)
        if d_titre and (now - d_titre) > timedelta(days=MAX_AGE_DAYS):
            continue

        with _browser_lock:
            manifest, meta = _sniff_stream(page_url)
        if not manifest:
            log.info("[%s] pas de flux detecte sur %s", source_name, page_url[:70])
            continue

        workdir = Path(tempfile.mkdtemp(prefix="hls_"))
        t0 = time.time()
        try:
            audio = _extract_audio(manifest, workdir)
            if audio is None:
                continue
            secs = _duration_seconds(audio)
            if secs < MIN_DURATION:
                log.info("[%s] %s : %.0f s, trop court", source_name,
                         page_url[:60], secs)
                continue
            published = _published_at(meta) or _date_from_title(
                meta.get("og:title") or meta.get("title") or titre_page)
            if published and (now - published) > timedelta(days=MAX_AGE_DAYS):
                continue
            cues = _transcribe(audio)
        except Exception as e:
            log.warning("[%s] %s : %s", source_name, page_url[:60], str(e)[:110])
            continue
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        segments = _segment(cues)
        if not segments:
            continue
        titre = _clean_title(meta, source_name)
        total = len(segments)
        for i, seg in enumerate(segments):
            out.append(RawArticle(
                id=_segment_id(page_url, i),
                source_name=source_name,
                feed_url=listing_url,
                url=f"{page_url}#t={int(seg['start'])}",
                title=f"{titre} [{i + 1}/{total}]",
                author=None,
                summary=None,
                published_at=published,
                content=seg["text"],
                language="ru",
            ))
        traites += 1
        log.info("[%s] %d segments en %.0f s (%.1fx temps reel) -- %s",
                 source_name, total, time.time() - t0,
                 secs / max(time.time() - t0, 1), titre[:44])
    return out
