"""Collecte de chaines YouTube : transcriptions decoupees en segments analysables.

Les chaines suivies publient des videos de 20 minutes a 4 heures. Inserer une
transcription entiere comme un seul "article" casserait les analyses en aval :

  - BERTopic embarque chaque document avec un modele de phrases qui tronque a
    ~512 tokens : seules les premieres minutes compteraient pour le theme ;
  - une posture unique par cible sur 3 h de parole ne veut rien dire -- une
    video peut defendre l'Armenie en debut d'emission et l'attaquer ensuite ;
  - l'onglet Themes compte les articles : une video de 4 h pesant autant
    qu'une depeche de 400 signes fausserait toutes les proportions.

D'ou le decoupage en segments d'environ 2000 signes -- l'ordre de grandeur
d'un article du corpus, donc aucun seuil ni ponderation a recalibrer en aval,
et sous la troncature des scripts d'analyse (cf. TARGET_SEGMENT_CHARS).
Les coupes tombent sur une fin de phrase (les sous-titres russes de ces
chaines sont ponctues) et chaque segment garde son horodatage : son URL
pointe sur l'instant exact de la video (watch?v=ID&t=1234s), donc n'importe
quel resultat du dashboard reste verifiable a la source en un clic.

La video parente reste identifiable sans colonne supplementaire : l'ID est
dans l'URL de chaque segment (regexp `v=([A-Za-z0-9_-]+)` cote SQL).
"""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx
from yt_dlp import YoutubeDL

from src.collect import RawArticle, url_hash

# getLogger et non setup_logging() : ce module est importe par le pipeline,
# qui a deja configure le logging -- rappeler setup_logging (basicConfig
# force=True) ecraserait sa configuration.
log = logging.getLogger("youtube")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 RussianMediaMonitor/0.1"
)

MAX_VIDEOS = 15          # par run et par chaine : large marge sur le rythme
                         # de publication reel (Kats, le plus prolifique, poste
                         # ~1,3 video/jour)
MIN_DURATION = 180       # sous 3 min : short ou bande-annonce, sans valeur
FILTER_POLITICAL = True  # cf. is_political() -- mettre a False pour tout
                         # collecter (le corpus se remplit alors de cinema,
                         # de voyage et de technologie)
MAX_AGE_DAYS = 60        # borne le backlog du premier run. Sans elle, les 15
                         # dernieres videos de vDud (2 a 4 h piece, ~3 par
                         # mois) remonteraient a 5 mois et produiraient a elles
                         # seules des centaines de segments a analyser, pour
                         # une periode que le reste du corpus ne couvre pas.
                         # Le test est fait apres extract_info (le listing plat
                         # ne donne pas les dates) mais avant le
                         # telechargement des sous-titres.
# Calibre sous la troncature la plus basse des analyses (2500 signes), titre
# compris. Au-dela, tous les segments video seraient tronques en silence.
TARGET_SEGMENT_CHARS = 2000
# Plafond dur, applique avant l'ajout de chaque sous-titre (cf. _segment) :
# aucun segment ne le depasse. 2300 + 150 de titre + 1 = 2451, sous la
# troncature la plus basse des analyses (2500).
MAX_SEGMENT_CHARS = 2300
MAX_TITLE_CHARS = 150     # ces chaines titrent en russe ET en anglais dans le
                          # meme champ ; sans borne, le titre mangerait la
                          # marge laissee sous la limite de 2500
MIN_TAIL_CHARS = 300      # sous ce seuil le dernier segment est fusionne avec
                          # le precedent : isole, il serait de toute facon
                          # ecarte par les analyses (seuil de 300 signes)
# YouTube limite par IP ("Sign in to confirm you're not a bot") a partir de
# quelques dizaines de resolutions rapprochees -- constate en rafale pendant
# la mise au point. Les 4 chaines etant collectees en parallele par le
# pipeline, ce delai s'applique a chacune : ~4 requetes toutes les 3 s.
INTER_VIDEO_DELAY = 3.0
# Au-dela, on arrete la chaine pour ce run : si YouTube a commence a refuser,
# insister ne fait qu'allonger la penalite, et les videos non collectees
# seront reprises au run suivant (rien n'est perdu, l'ID de segment est
# deterministe).
MAX_CONSECUTIVE_FAILURES = 3

# Un seul flux de requetes YouTube a la fois, quel que soit le nombre de
# chaines que le pipeline collecte en parallele (MAX_WORKERS = 20).
# C'est la simultaneite, plus que la cadence, qui declenche la detection :
# apres expiration d'un blocage, une requete isolee passe sans probleme alors
# que les quatre chaines lancees ensemble le redeclenchent en quelques
# secondes. Les autres sources, elles, continuent en parallele : ce verrou ne
# serialise que YouTube.
_YT_LOCK = Lock()

YDL_BASE = {"quiet": True, "no_warnings": True, "skip_download": True}

# Fin de phrase : on ne coupe qu'ici une fois la taille cible atteinte.
_SENTENCE_END = re.compile(r"[.!?…]['\"»)]?\s*$")

# Filtre de pertinence politique. Ces chaines alternent politique et culture,
# et une video de 4 h decoupee en 76 segments forme son propre theme : le
# cluster « Cinema et emotions » venait a 67 % d'une seule video de vDud. Le
# filtre porte sur la VIDEO, teste sur son titre, donc avant tout appel reseau
# ou analyse payante. Racines et non mots entiers, le russe decline.
POLITICAL_STEMS = (
    # Guerre, armee, mobilisation
    "войн", "воен", "арми", "фронт", "мобилизац", "призыв", "военкомат",
    "оборон", "удар", "обстрел", "дрон", "бпла", "ракет", "наступлен",
    "сво ", "всу", "солдат", "офицер", "генерал", "оккупац", "теракт",
    "спецслужб", "фсб", "гру", "кгб", "вагнер", "чвк",
    # Pouvoir et vie politique
    "путин", "кремл", "госдум", "дум", "выбор", "депутат", "министр",
    "правительств", "власт", "президент", "губернатор", "чиновник",
    "оппозиц", "навальн", "яблок", "партия", "митинг", "протест",
    "лавров", "медведев", "шойгу", "мишустин", "патрушев", "кадыров",
    # Repression, justice, droit
    "репресс", "арест", "суд", "приговор", "тюрьм", "колони", "уголовн",
    "иноагент", "экстремис", "цензур", "полиц", "росгварди", "закон",
    "заключен", "обыск", "штраф", "пытк",
    # Economie politique, sanctions
    "санкц", "бюджет", "инфляц", "рубл", "экономик", "нефт", "газпром",
    "энерг", "налог", "девальвац", "дефицит", "коррупц", "олигарх",
    # Geopolitique et pays
    "украин", "россия", "росси", "москв", "киев", "нато", "запад",
    "сша", "америк", "трамп", "европ", "евросоюз", "ес ", "китай",
    "белорус", "лукашенк", "казахстан", "армени", "грузи", "молдав",
    "иран", "израил", "сири", "кндр", "корея", "турци", "польш",
    "прибалтик", "эстони", "латви", "литв", "чечн", "крым", "донбасс",
    "зеленск", "переговор", "перемири", "дипломат", "посол", "саммит",
    # Societe politisee
    "мигрант", "эмиграц", "релокант", "пропаганд", "цензур", "церков",
    "патриарх", "демограф", "мобилизован", "беженц",
)


def _matches_political_stems(title: str, description: str = "") -> bool:
    haystack = f"{title} {description}".lower().replace("ё", "е")
    return any(stem in haystack for stem in POLITICAL_STEMS)


# Cache des decisions, en clair et versionnable a la main : c'est aussi la
# piece a consulter (ou a corriger) quand une video parait mal classee.
TOPIC_CACHE_PATH = Path("data/youtube_video_topics.json")
_cache_lock = Lock()

_CLASSIFY_PROMPT = """Tu tries des videos YouTube russophones pour un outil de
veille sur la politique russe.

Reponds UNIQUEMENT en JSON : {"politique": true|false, "raison": "<8 mots max>"}

politique = true si la video traite de : guerre en Ukraine, armee, mobilisation,
recrutement militaire, pouvoir russe, Kremlin, elections, opposition, repression,
justice politique, services de securite, sanctions, economie politique, budget,
corruption, geopolitique, relations internationales, societe russe sous l'angle
politique (emigration, demographie, mobilisation), catastrophes ou accidents
traites sous l'angle de la responsabilite des autorites.

politique = false si la video traite de : cinema, musique, sport, voyage,
tourisme, cuisine, celebrites etrangeres, technologie sans angle politique,
histoire ancienne sans lien avec l'actualite, divertissement.

En cas de doute, reponds true : mieux vaut garder une video hors sujet que
perdre une video politique."""


def _load_topic_cache() -> dict:
    if not TOPIC_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(TOPIC_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_topic_cache(cache: dict) -> None:
    try:
        TOPIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOPIC_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
    except Exception as e:
        log.warning("cache de classification non ecrit : %s", e)


def is_political(video_id: str, title: str, description: str = "") -> bool:
    """La video releve-t-elle du champ suivi par l'outil ?

    Un titre court demande une connaissance du monde qu'une liste de mots-cles
    n'a pas : « Вербуют с новой силой » (recrutement pour le front) ne contient
    aucune racine politique evidente, tandis que « Как устроено кино: деньги,
    музыка... » contient « деньги ». Teste sur les 50 videos deja collectees,
    le filtrage lexical se trompait sur 6 rejets sur 11 -- d'ou l'appel au
    modele, une fois par video et mis en cache (quelques dizaines d'appels par
    mois sur un prompt minuscule : cout negligeable).

    En cas d'indisponibilite de l'API : on garde la video. Ecarter est
    l'action destructrice, elle ne doit jamais resulter d'une panne.
    """
    with _cache_lock:
        cache = _load_topic_cache()
        hit = cache.get(video_id)
    if hit is not None:
        return bool(hit.get("politique", True))

    verdict, reason = None, ""
    try:
        from src.llm_mistral import complete_json
        payload = f"Titre : {title}"
        if description:
            payload += f"\nDescription : {description[:400]}"
        data = complete_json(_CLASSIFY_PROMPT, payload, max_tokens=80)
        if isinstance(data, dict) and "politique" in data:
            verdict = bool(data["politique"])
            reason = str(data.get("raison", ""))[:80]
    except Exception as e:
        log.warning("classification indisponible (%s) -- repli lexical", str(e)[:80])

    if verdict is None:
        verdict = _matches_political_stems(title, description)
        reason = "repli lexical"

    with _cache_lock:
        cache = _load_topic_cache()
        cache[video_id] = {"politique": verdict, "titre": title[:120],
                           "raison": reason}
        _save_topic_cache(cache)
    return verdict


def _list_recent(channel_url: str) -> list[dict]:
    """Listing "plat" : un seul appel reseau pour toute la chaine, sans
    resoudre chaque video (qui coute un aller-retour supplementaire)."""
    opts = {**YDL_BASE, "extract_flat": "in_playlist", "playlistend": MAX_VIDEOS}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
    except Exception:
        return []
    if not info:
        return []
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({
            "id": e["id"],
            "title": (e.get("title") or "").strip(),
            "duration": e.get("duration"),
            "view_count": e.get("view_count"),
        })
    return out


def _native_ru_track(info: dict) -> str | None:
    """URL de la piste de sous-titres russes d'origine, ou None.

    YouTube expose sous `automatic_captions` la transcription ASR d'origine
    ET sa traduction automatique dans ~100 langues. Seule la piste d'origine
    a une URL sans parametre `tlang` : sans ce test, une video anglophone
    entrerait dans le corpus via sa traduction russe automatique, avec le
    vocabulaire lisse d'une machine plutot que la parole reelle.
    """
    for key in ("subtitles", "automatic_captions"):   # sous-titres manuels d'abord
        for track in (info.get(key) or {}).get("ru") or []:
            url = track.get("url") or ""
            if track.get("ext") == "json3" and "tlang=" not in url:
                return url
    return None


def _fetch_cues(track_url: str) -> list[tuple[float, str]]:
    """(instant en secondes, texte) pour chaque sous-titre."""
    r = httpx.get(track_url, timeout=30.0, follow_redirects=True,
                  headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    data = r.json()

    cues: list[tuple[float, str]] = []
    previous = None
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        # Les pistes ASR repetent parfois la ligne precedente (effet de
        # defilement) : on ecarte les repetitions consecutives.
        if not text or text == previous:
            continue
        previous = text
        cues.append((float(event.get("tStartMs", 0)) / 1000.0, text))
    return cues


def _segment(cues: list[tuple[float, str]]) -> list[dict]:
    """Regroupe les sous-titres en blocs d'environ TARGET_SEGMENT_CHARS,
    coupes sur une fin de phrase."""
    segments: list[dict] = []
    buffer: list[str] = []
    start: float | None = None
    length = 0

    def flush():
        nonlocal buffer, start, length
        if buffer:
            segments.append({"start": start or 0.0, "text": " ".join(buffer)})
        buffer, start, length = [], None, 0

    for cue_start, text in cues:
        # Fermer AVANT d'ajouter quand l'ajout ferait deborder. En testant
        # apres coup, la taille finale depassait le plafond de la longueur du
        # sous-titre ajoute -- jusqu'a ~280 signes sur des cas reels, assez
        # pour repasser au-dessus de la troncature des analyses. Ce test-ci
        # rend le plafond exact, et non approche.
        if buffer and length + len(text) + 1 > MAX_SEGMENT_CHARS:
            flush()
        if start is None:
            start = cue_start
        buffer.append(text)
        length += len(text) + 1
        if length >= TARGET_SEGMENT_CHARS and _SENTENCE_END.search(text):
            flush()
    flush()

    # Une queue trop courte serait ecartee par les analyses (seuil 300
    # signes) et laisserait une ligne morte en base : on la recolle.
    if len(segments) > 1 and len(segments[-1]["text"]) < MIN_TAIL_CHARS:
        tail = segments.pop()
        segments[-1]["text"] += " " + tail["text"]

    return segments


def _published_at(info: dict) -> datetime | None:
    ts = info.get("timestamp")
    if ts:
        return datetime.fromtimestamp(ts, timezone.utc)
    day = info.get("upload_date")
    if day:
        try:
            return datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _segment_id(video_url: str, index: int) -> str:
    return url_hash(f"{video_url}#seg{index}")


def scrape_youtube_channel(
    source_name: str, channel_url: str, seen_ids: set | None = None, seen_lock=None
) -> list[RawArticle]:
    """Collecte une chaine, en serialisant les acces a YouTube (cf. _YT_LOCK)."""
    with _YT_LOCK:
        return _scrape_channel(source_name, channel_url, seen_ids, seen_lock)


def _scrape_channel(
    source_name: str, channel_url: str, seen_ids: set | None, seen_lock
) -> list[RawArticle]:
    """channel_url attendu : https://www.youtube.com/@<chaine>/videos.

    `seen_ids` (les ID d'articles deja en base) sert a sauter les videos deja
    transcrites AVANT d'aller chercher leurs metadonnees : l'ID du premier
    segment est deterministe, donc sa presence en base suffit a conclure, sans
    aucun appel reseau. Sans ce test, chaque run re-telechargerait la
    transcription complete des 15 dernieres videos de chaque chaine.
    """
    out: list[RawArticle] = []
    consecutive_failures = 0
    videos = _list_recent(channel_url)
    if not videos:
        log.warning("[%s] aucune video listee (chaine renommee, ou yt-dlp "
                    "a besoin d'une mise a jour)", source_name)
        return out

    skipped_apolitical = []
    for video in videos:
        duration = video.get("duration")
        if duration is not None and duration < MIN_DURATION:
            continue

        # Avant tout appel reseau supplementaire : une video hors sujet coute
        # sinon une resolution, un telechargement de sous-titres, et surtout
        # des dizaines de segments analyses par Mistral pour rien.
        if FILTER_POLITICAL and not is_political(video["id"], video["title"]):
            skipped_apolitical.append(video["title"][:70])
            continue

        video_url = f"https://www.youtube.com/watch?v={video['id']}"
        if seen_ids is not None:
            probe = _segment_id(video_url, 0)
            if seen_lock is not None:
                with seen_lock:
                    already = probe in seen_ids
            else:
                already = probe in seen_ids
            if already:
                continue

        try:
            with YoutubeDL(YDL_BASE) as ydl:
                info = ydl.extract_info(video_url, download=False)
        except Exception as e:
            consecutive_failures += 1
            log.warning("[%s] %s illisible (%d/%d) : %s",
                        source_name, video["id"], consecutive_failures,
                        MAX_CONSECUTIVE_FAILURES, e)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("[%s] arret de la chaine : %d echecs d'affilee. "
                          "Cause la plus probable : limitation anti-bot de "
                          "YouTube (reprise au prochain run).",
                          source_name, consecutive_failures)
                break
            time.sleep(INTER_VIDEO_DELAY)
            continue
        if not info:
            continue
        consecutive_failures = 0

        published = _published_at(info)
        if published is not None and MAX_AGE_DAYS:
            age = (datetime.now(timezone.utc) - published).days
            if age > MAX_AGE_DAYS:
                continue

        track_url = _native_ru_track(info)
        if not track_url:
            # video non russophone, ou sans transcription disponible
            log.info("[%s] %s : pas de piste russe d'origine", source_name, video["id"])
            time.sleep(INTER_VIDEO_DELAY)
            continue

        try:
            cues = _fetch_cues(track_url)
        except Exception as e:
            log.warning("[%s] %s : sous-titres injoignables : %s",
                        source_name, video["id"], e)
            time.sleep(INTER_VIDEO_DELAY)
            continue

        segments = _segment(cues)
        if not segments:
            continue

        title = (info.get("title") or video["title"])[:MAX_TITLE_CHARS].strip()
        author = info.get("channel") or info.get("uploader")
        total = len(segments)

        for i, seg in enumerate(segments):
            offset = int(seg["start"])
            out.append(RawArticle(
                id=_segment_id(video_url, i),
                source_name=source_name,
                feed_url=channel_url,
                # &t= : l'URL ouvre la video a l'instant du segment, ce qui
                # rend chaque ligne du dashboard verifiable a la source.
                url=f"{video_url}&t={offset}s",
                title=f"{title} [{i + 1}/{total}]",
                author=author,
                summary=None,
                published_at=published,
                content=seg["text"],
                language="ru",   # piste ASR d'origine russe, cf. _native_ru_track
                view_count=info.get("view_count") or video.get("view_count"),
            ))

        time.sleep(INTER_VIDEO_DELAY)

    # Journalise les rejets : c'est la seule facon de reperer un faux negatif
    # (video politique au titre allusif) et de completer POLITICAL_STEMS.
    if skipped_apolitical:
        log.info("[%s] %d video(s) ecartee(s), hors sujet politique : %s",
                 source_name, len(skipped_apolitical),
                 " | ".join(skipped_apolitical))

    return out
