"""Collecte de communautes VKontakte via un navigateur sans interface.

VK est le premier reseau social du pays et le premier poste de temps en ligne
(2 h 44/jour) : c'est la que les medias touchent le public qui ne visite pas
leurs sites.

Trois voies ont ete testees :
l'API sans jeton renvoie « token required », le HTML brut ne contient aucun
texte (page rendue entierement en JavaScript, y compris sur m.vk.com), et
seul un navigateur pilote restitue les publications -- sans mur de connexion.
C'est cette derniere voie qui est implementee ici.

Limite assumee : VK n'expose pas les compteurs de vues aux visiteurs non
connectes. Classer les publications « les plus vues » demanderait l'API
officielle, donc un compte VK. Ce module recupere le flux, pas la hierarchie
d'audience.
"""
import logging
import re
import time
from datetime import datetime, timedelta
from threading import Lock

from langdetect import LangDetectException, detect

from src.collect import RawArticle, url_hash

log = logging.getLogger("vk")

MIN_POST_LEN = 80      # sous ce seuil : renvoi vers une photo ou une video,
                       # sans contenu analysable
SCROLLS = 6            # VK charge le mur par paliers au defilement
SCROLL_PAUSE_MS = 1800
PAGE_TIMEOUT_MS = 45000
# VK cesse de servir le mur apres quelques visites anonymes rapprochees :
# constate sur une collecte de dix communautes, ou seules les deux premieres
# ont rendu des posts. On espace les visites et surtout on reste dans la meme
# session de navigateur (cf. _get_context).
INTER_COMMUNITY_DELAY = 8.0

# Un seul navigateur a la fois : le pipeline collecte les sources en
# parallele, et lancer un Chromium par communaute saturerait la machine.
_browser_lock = Lock()
_pw = _browser = _context = None
# Une fois la verification anti-robot opposee, elle vaut pour l'adresse IP :
# toutes les communautes suivantes echoueraient pareil. On arrete les frais
# pour ce run plutot que de payer un lancement de navigateur et 45 s de
# timeout par communaute restante.
_challenged = False


def _get_context():
    """Contexte de navigation unique, reutilise d'une communaute a l'autre.

    Relancer un Chromium par communaute presentait a VK une dizaine de
    visiteurs anonymes tout neufs depuis la meme adresse en quelques minutes,
    ce qui coupait le rendu du mur des la troisieme. Une session unique qui
    enchaine les pages ressemble a une navigation ordinaire, et conserve au
    passage les cookies que VK depose a la premiere visite.
    """
    global _pw, _browser, _context
    if _context is not None:
        return _context
    from playwright.sync_api import sync_playwright
    _pw = sync_playwright().start()
    _browser = _pw.chromium.launch()
    _context = _browser.new_context(
        viewport={"width": 1280, "height": 2200}, locale="ru-RU",
        user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"))
    import atexit
    atexit.register(close_browser)
    return _context


def close_browser():
    global _pw, _browser, _context
    for obj, meth in ((_context, "close"), (_browser, "close"), (_pw, "stop")):
        try:
            if obj is not None:
                getattr(obj, meth)()
        except Exception:
            pass
    _pw = _browser = _context = None

class VKChallenge(RuntimeError):
    """VK a substitue sa page de verification anti-robot au mur demande."""


_MONTHS_RU = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}


def parse_vk_date(text: str, now: datetime | None = None) -> datetime | None:
    """Convertit l'horodatage relatif de VK en date.

    VK n'expose aucune date absolue dans le DOM (ni attribut `datetime`, ni
    `title`) : il n'y a que le libelle affiche, en russe et relatif --
    « 5 мин назад », « сегодня в 14:30 », « вчера в 9:15 », « 14 авг в 10:00 »,
    « 3 мая 2025 ». Sans cette conversion, tous les posts arriveraient sans
    date et disparaitraient des series temporelles du dashboard.
    """
    if not text:
        return None
    t = text.strip().lower().replace(" ", " ")
    now = now or datetime.now()

    if t.startswith("только что"):
        return now

    m = re.match(r"(\d+)\s*(сек|мин|час|дн|ден|дня)", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"сек": timedelta(seconds=n), "мин": timedelta(minutes=n),
                 "час": timedelta(hours=n), "дн": timedelta(days=n),
                 "ден": timedelta(days=n), "дня": timedelta(days=n)}[unit]
        return now - delta

    m = re.match(r"(сегодня|вчера)(?:\s+в\s+(\d{1,2}):(\d{2}))?", t)
    if m:
        day = now.date() - (timedelta(days=1) if m.group(1) == "вчера" else timedelta())
        hh = int(m.group(2)) if m.group(2) else 0
        mm = int(m.group(3)) if m.group(3) else 0
        return datetime(day.year, day.month, day.day, hh, mm)

    # « 14 авг в 10:00 » (annee courante) ou « 3 мая 2025 »
    m = re.match(r"(\d{1,2})\s+([а-я]{3})[а-я]*\.?\s*(?:в\s+(\d{1,2}):(\d{2})|(\d{4}))?", t)
    if m:
        day = int(m.group(1))
        month = _MONTHS_RU.get(m.group(2))
        if not month:
            return None
        if m.group(5):                       # annee explicite
            return datetime(int(m.group(5)), month, day)
        year = now.year
        hh = int(m.group(3)) if m.group(3) else 0
        mm = int(m.group(4)) if m.group(4) else 0
        candidate = datetime(year, month, day, hh, mm)
        # VK omet l'annee pour les 12 derniers mois : une date qui tomberait
        # dans le futur appartient donc a l'annee precedente.
        if candidate > now + timedelta(days=1):
            candidate = candidate.replace(year=year - 1)
        return candidate
    return None


# Extraction faite en une passe dans la page : un aller-retour par post
# multiplierait les echanges avec le navigateur pour rien.
_EXTRACT_JS = """() => {
  const out = [];
  document.querySelectorAll('[id^="post-"]').forEach(p => {
    const body = p.querySelector('[class*="wall_post_text"]');
    if (!body) return;
    let href = null, dateTxt = null;
    // Le lien de date se trouve dans l'en-tete du post. Prendre le premier
    // lien /wall- venu attrapait parfois un renvoi vers un autre message
    // (« en reponse a »), dont le libelle n'est pas une date : le post
    // arrivait alors sans horodatage.
    const header = p.querySelector('[class*="PostHeaderSubtitle"], [class*="post_date"], [class*="rel_date"]');
    if (header) {
      const a = header.matches('a') ? header : header.querySelector('a');
      const el = a || header;
      dateTxt = el.innerText.trim();
      href = el.getAttribute && el.getAttribute('href');
    }
    for (const a of p.querySelectorAll('a')) {
      const h = a.getAttribute('href') || '';
      if (/^\\/wall-?\\d+_\\d+$/.test(h)) {
        if (!href || !/^\\/wall-?\\d+_\\d+$/.test(href)) href = h;
        if (!dateTxt) dateTxt = a.innerText.trim();
        break;
      }
    }
    out.push({ id: p.id, text: body.innerText.trim(), href: href, date: dateTxt });
  });
  return out;
}"""


def _collect_posts(page, community: str) -> list[dict]:
    page.goto(f"https://vk.com/{community}", wait_until="domcontentloaded",
              timeout=PAGE_TIMEOUT_MS)

    # VK oppose une page anti-robot (« Проверяем, что вы не робот ») au-dela
    # d'un certain rythme de visites anonymes depuis une meme adresse. Elle
    # s'applique a toutes les communautes d'un coup, y compris a celles qui
    # repondaient l'instant d'avant. La distinguer d'une communaute
    # reellement inaccessible evite de partir sur un faux diagnostic.
    if "challenge" in page.url or "не робот" in (page.title() or ""):
        raise VKChallenge(
            f"VK oppose sa verification anti-robot (vu sur {community}). "
            f"Elle vise l'adresse IP, pas la communaute, et se leve d'elle-meme "
            f"apres une pause. Espacer les collectes, ou passer a l'API "
            f"officielle qui ne subit pas ce filtrage.")

    try:
        page.wait_for_selector('[class*="wall_post_text"]', timeout=20000)
    except Exception:
        log.warning("[vk/%s] aucun post rendu : communaute privee, renommee, "
                    "ou page modifiee", community)
        return []

    seen, posts = set(), []
    for _ in range(SCROLLS):
        for item in page.evaluate(_EXTRACT_JS):
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            posts.append(item)
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(SCROLL_PAUSE_MS)
    return posts


def scrape_vk_community(source_name: str, community_url: str) -> list[RawArticle]:
    """community_url attendu : https://vk.com/<communaute>"""
    m = re.search(r"vk\.com/([A-Za-z0-9_.]+)", community_url)
    if not m:
        log.warning("[%s] URL de communaute VK illisible : %s", source_name, community_url)
        return []
    community = m.group(1)

    global _challenged
    with _browser_lock:
        if _challenged:
            log.info("[%s] ignoree : VK a oppose sa verification anti-robot "
                     "plus tot dans ce run", source_name)
            return []
        page = None
        try:
            page = _get_context().new_page()
            raw = _collect_posts(page, community)
        except VKChallenge as e:
            # Inutile d'enchainer sur les communautes suivantes : la
            # verification vise l'adresse IP, elles echoueraient toutes.
            _challenged = True
            log.error("[%s] %s", source_name, e)
            return []
        except Exception as e:
            log.error("[%s] navigateur : %s", source_name, str(e)[:130])
            return []
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
            time.sleep(INTER_COMMUNITY_DELAY)

    out: list[RawArticle] = []
    unparsed: list[str] = []
    now = datetime.now()
    for item in raw:
        text = (item.get("text") or "").strip()
        if len(text) < MIN_POST_LEN:
            continue
        href = item.get("href")
        url = f"https://vk.com{href}" if href else f"https://vk.com/{item['id'][5:]}"
        try:
            lang = detect(text)
        except LangDetectException:
            lang = None
        published = parse_vk_date(item.get("date"), now)
        if published is None and item.get("date"):
            # Journalise le libelle non reconnu : c'est la seule facon de
            # completer parse_vk_date sans avoir a re-observer la page.
            unparsed.append(str(item["date"])[:30])
        title = text.split("\n", 1)[0].strip()[:120] or text[:120]
        out.append(RawArticle(
            id=url_hash(url),
            source_name=source_name,
            feed_url=community_url,
            url=url,
            title=title,
            author=None,
            summary=None,
            published_at=published,
            content=text,
            language=lang,
        ))
    log.info("[%s] %d posts retenus sur %d lus", source_name, len(out), len(raw))
    if unparsed:
        log.warning("[%s] %d date(s) non reconnue(s) par parse_vk_date : %s",
                    source_name, len(unparsed), " | ".join(sorted(set(unparsed))[:5]))
    return out
