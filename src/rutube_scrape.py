"""Collecte d'emissions de television russe : RuTube + transcription Whisper.

Pourquoi ce chemin, teste alternative par alternative (voir
note de couverture) : les chaines d'Etat ont ete retirees de
YouTube, smotrim.ru repond 403 hors de Russie, et les re-uploads YouTube par
des comptes tiers sont trop lacunaires pour une veille reguliere. RuTube est
la seule source ou les chaines publient officiellement et quotidiennement --
mais elle n'expose AUCUN sous-titre, ni manuel ni automatique, et aucun flux
audio separe. La transcription se fait donc en local, au GPU.

La television est le premier support d'information du pays (plus de la moitie
des Russes, 41 % de confiance, contre ~20 % et 11 % pour Telegram) : c'etait
le principal angle mort du corpus.

Cout : on telecharge la variante 144p (~250 kbps, soit ~120 Mo pour une heure)
puisque seule la bande son sert, puis faster-whisper transcrit a environ 8-12x
le temps reel sur une carte d'entree de gamme.
"""
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import httpx
from yt_dlp import YoutubeDL

from src.collect import RawArticle, url_hash
# Meme decoupage que pour YouTube : segments calibres sous la troncature des
# scripts d'analyse, coupes sur une fin de phrase, horodates.
from src.youtube_scrape import MAX_TITLE_CHARS, _segment

log = logging.getLogger("rutube")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RussianMediaMonitor/0.1"
API_SEARCH = "https://rutube.ru/api/search/video/"

MAX_EPISODES = 2       # deux numeros par emission et par run : les talk-shows
                       # quotidiens sont decoupes en « Часть 1/2/3 », un seul
                       # episode ne ramenait qu'un fragment de la journee
MIN_DURATION = 900     # 15 min : ecarte extraits, bandes-annonces et breves
MAX_DURATION = None    # plus de plafond. Il etait fixe a 2 h 30 et excluait
                       # les emissions phares de Soloviev (« Формула смысла »,
                       # « Полный контакт »), qui durent 4 h et rassemblent
                       # deux a trois fois l'audience des formats courts.
MAX_AGE_DAYS = 21
API_PAGES = 5          # la recherche RuTube melange les contenus voisins de
                       # la chaine : il faut plusieurs pages pour reunir assez
                       # d'episodes de l'emission visee

WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "int8_float16"   # ~1 Go de VRAM : tient sur 6 Go

_model = None
_model_lock = Lock()
# La transcription sature le GPU : une seule a la fois, quel que soit le
# nombre d'emissions collectees en parallele par le pipeline.
_gpu_lock = Lock()


def _register_cuda_dlls():
    """Sous Windows, cuBLAS et cuDNN sont installes par pip dans
    site-packages/nvidia/*/bin, un emplacement que ctranslate2 ne consulte pas :
    le modele se charge sans erreur puis l'inference echoue sur
    « cublas64_12.dll is not found ». On declare ces repertoires au chargeur de
    DLL avant tout appel."""
    import os
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return
    dirs = []
    for root in nvidia.__path__:
        for sub in Path(root).iterdir():
            binpath = sub / "bin"
            if binpath.is_dir():
                dirs.append(str(binpath))
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(binpath))
                    except OSError:
                        pass
    # add_dll_directory ne suffit pas : ctranslate2 charge la bibliotheque par
    # son nom court, ce qui suit le PATH du processus et ignore cette liste.
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")


def _build_model(device, compute_type):
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            _register_cuda_dlls()
            try:
                _model = _build_model(WHISPER_DEVICE, WHISPER_COMPUTE)
                log.info("Whisper %s charge sur %s", WHISPER_MODEL, WHISPER_DEVICE)
            except Exception as e:
                log.warning("GPU indisponible (%s) -- repli CPU, nettement plus lent",
                            str(e)[:90])
                _model = _build_model("cpu", "int8")
        return _model


def _reset_model_to_cpu():
    """Le chargement du modele peut reussir alors que l'inference echoue plus
    tard (bibliotheques CUDA incompletes) : le repli doit donc pouvoir se
    declencher au moment de la transcription, pas seulement au chargement."""
    global _model
    with _model_lock:
        _model = _build_model("cpu", "int8")
        log.warning("Bascule sur CPU pour la suite des transcriptions.")
        return _model


def _search_program(channel_id: str, query: str) -> list[dict]:
    """Episodes d'une emission, cherches par nom puis filtres sur la chaine.

    On ne parcourt pas le fil de la chaine : Pervyi Kanal publie plusieurs
    dizaines de breves par jour, si bien que trois pages de son fil ne
    couvrent qu'une seule journee -- l'hebdomadaire du dimanche s'y trouve
    enfoui hors de portee. La recherche va le chercher directement, et le
    filtre sur l'identifiant d'auteur ecarte les re-uploads par des tiers.
    """
    out = []
    with httpx.Client(timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as client:
        for page in range(1, API_PAGES + 1):
            try:
                r = client.get(API_SEARCH, params={"query": query, "page": page})
                r.raise_for_status()
                results = r.json().get("results") or []
            except Exception as e:
                log.warning("recherche '%s' page %d : %s", query, page, e)
                break
            if not results:
                break
            for v in results:
                if str((v.get("author") or {}).get("id")) == str(channel_id):
                    out.append(v)
            time.sleep(0.3)
    return out


def _download_audio(video_url: str, workdir: Path) -> Path | None:
    """Recupere la bande son. On force la plus basse definition disponible :
    la piste audio y est identique et le telechargement divise par dix."""
    opts = {
        "quiet": True, "no_warnings": True,
        # sans ca, la barre de progression de yt-dlp ecrit une ligne par
        # fragment : un seul episode gonflait le journal de 700 Ko
        "noprogress": True,
        "format": "worstvideo+bestaudio/worst",
        "outtmpl": str(workdir / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3", "preferredquality": "64"}],
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        log.warning("telechargement %s : %s", video_url, str(e)[:110])
        return None
    for f in workdir.iterdir():
        if f.suffix in (".mp3", ".m4a", ".opus", ".webm", ".wav"):
            return f
    return None


def _transcribe(audio: Path) -> list[tuple[float, str]]:
    """(instant, texte) pour chaque segment reconnu."""
    def _run(model):
        segments, _info = model.transcribe(
            str(audio), language="ru", vad_filter=True,
            # Le VAD coupe les silences (jingles, transitions) : moins d'audio
            # a traiter et moins d'hallucinations sur les passages muets.
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # La generation est paresseuse : c'est en consommant l'iterateur que
        # l'inference tourne, donc que d'eventuelles erreurs CUDA remontent.
        return [(float(s.start), s.text.strip()) for s in segments if s.text.strip()]

    with _gpu_lock:
        try:
            return _run(_get_model())
        except Exception as e:
            log.warning("transcription GPU en echec (%s)", str(e)[:110])
            return _run(_reset_model_to_cpu())


def _published_at(item: dict) -> datetime | None:
    ts = item.get("publication_ts") or item.get("created_ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    # RuTube renvoie un horodatage sans fuseau : sans ce rattachement, la
    # comparaison avec l'heure courante leve « can't subtract offset-naive
    # and offset-aware datetimes » et la source entiere tombe en erreur.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _segment_id(video_url: str, index: int) -> str:
    return url_hash(f"{video_url}#seg{index}")


def scrape_rutube_program(
    source_name: str, channel_url: str, program_pattern: str | None = None,
    seen_ids: set | None = None, seen_lock=None, search_query: str | None = None,
    min_duration: int | None = None,
) -> list[RawArticle]:
    """channel_url attendu : https://rutube.ru/channel/<id>/videos/

    `program_pattern` filtre les titres : une chaine comme Pervyi Kanal poste
    des dizaines de breves par jour, on ne veut que l'emission suivie.

    `min_duration` ajuste le plancher de duree, dans les deux sens : le seuil
    general de 15 min ecarte les extraits, mais il ecarterait aussi des
    journaux entiers (« Сегодня в Москве » fait 13 a 15 min), et il est trop
    bas pour les emissions de 4 h dont la chaine publie aussi des extraits de
    20 min sous le meme titre.
    """
    m = re.search(r"/channel/(\d+)", channel_url)
    if not m:
        log.warning("[%s] URL de chaine RuTube illisible : %s", source_name, channel_url)
        return []
    channel_id = m.group(1)

    if not program_pattern:
        log.warning("[%s] program_pattern absent : indispensable pour isoler "
                    "l'emission dans le catalogue de la chaine", source_name)
        return []
    pattern = re.compile(program_pattern, re.IGNORECASE)
    now = datetime.now(timezone.utc)

    # Le motif sert deux usages differents : filtrer les titres (regex) et
    # interroger la recherche RuTube (texte libre). Envoyer « A|B|C » ou un
    # motif a jokers tel quel comme requete ne ramenerait rien -- d'ou
    # `search_query` quand les deux ne peuvent pas coincider, et a defaut la
    # premiere variante du motif.
    query = (search_query or program_pattern.split("|")[0]).strip()
    items = _search_program(channel_id, query)
    items.sort(key=lambda v: str(v.get("publication_ts") or ""), reverse=True)

    candidates = []
    for item in items:
        title = item.get("title") or ""
        if not pattern.search(title):
            continue
        duration = item.get("duration") or 0
        if duration < (min_duration or MIN_DURATION):
            continue
        if MAX_DURATION and duration > MAX_DURATION:
            continue
        published = _published_at(item)
        if published and (now - published) > timedelta(days=MAX_AGE_DAYS):
            continue
        url = item.get("video_url") or f"https://rutube.ru/video/{item.get('id')}/"
        if seen_ids is not None:
            probe = _segment_id(url, 0)
            with (seen_lock or _model_lock):
                if probe in seen_ids:
                    continue
        candidates.append({"url": url, "title": title, "published": published,
                           "duration": item.get("duration") or 0,
                           "views": item.get("hits")})
        if len(candidates) >= MAX_EPISODES:
            break

    if not candidates:
        log.info("[%s] aucun nouvel episode", source_name)
        return []

    out: list[RawArticle] = []
    for ep in candidates:
        log.info("[%s] transcription : %s (%d min)", source_name,
                 ep["title"][:60], ep["duration"] // 60)
        t0 = time.time()
        workdir = Path(tempfile.mkdtemp(prefix="rutube_"))
        try:
            audio = _download_audio(ep["url"], workdir)
            if audio is None:
                continue
            cues = _transcribe(audio)
        except Exception as e:
            log.warning("[%s] echec sur %s : %s", source_name, ep["url"], str(e)[:110])
            continue
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        segments = _segment(cues)
        if not segments:
            log.warning("[%s] transcription vide pour %s", source_name, ep["url"])
            continue

        title = ep["title"][:MAX_TITLE_CHARS].strip()
        total = len(segments)
        for i, seg in enumerate(segments):
            out.append(RawArticle(
                id=_segment_id(ep["url"], i),
                source_name=source_name,
                feed_url=channel_url,
                # RuTube accepte ?t=<secondes> : chaque segment reste
                # verifiable a l'instant exact de l'emission.
                url=f"{ep['url']}?t={int(seg['start'])}",
                title=f"{title} [{i + 1}/{total}]",
                author=None,
                summary=None,
                published_at=ep["published"],
                content=seg["text"],
                language="ru",
                view_count=ep.get("views"),
            ))
        log.info("[%s] %d segments en %.0f s (%.1fx temps reel)", source_name,
                 total, time.time() - t0,
                 ep["duration"] / max(time.time() - t0, 1))
    return out
