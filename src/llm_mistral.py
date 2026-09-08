"""Helper partagé pour les appels Mistral (remplace l'API Claude).

Tous les scripts d'analyse importent complete_json() depuis ce module.
Pour rebasculer sur Claude un jour, il suffit de réécrire ce seul fichier.

Prérequis :
  pip install mistralai
  $env:MISTRAL_API_KEY = "..."   (clé créée sur https://console.mistral.ai)

Plusieurs clés : renseigner MISTRAL_API_KEY, puis MISTRAL_API_KEY2, KEY3...
Elles sont consommées dans cet ordre. Quand l'API refuse une clé pour cause
de crédit épuisé, le module passe seul à la suivante et ne revient jamais en
arrière -- une clé déclarée morte le reste pour la durée du processus.

Modèles (mai 2026, prix par M tokens entrée/sortie) :
  - mistral-small-latest : 0,10 $ / 0,60 $  -> classification, extraction
  - mistral-large-latest : 0,50 $ / 1,50 $  -> nuance, raisonnement fin
"""
import json
import logging
import os
import re
import time
from threading import Lock

from dotenv import load_dotenv

load_dotenv()  # lit .env si present (MISTRAL_API_KEY), sans ecraser l'env existant

log = logging.getLogger("llm")

# Modeles reellement accessibles a ce compte, mesure le 08/09/2026 en
# interrogeant l'API directement :
#
#   mistral-small / medium   429, x-ratelimit-limit-req-minute = 0
#   mistral-large            403 tier_not_allowed
#   ministral-8b-2512        200, 188 requetes/minute
#   ministral-3b-2512        200, 750 requetes/minute
#
# La console affiche pourtant 1 req/s pour mistral-small : c'est la famille
# ministral qui porte le debit reel du palier. Les alias « -latest » ne
# valent donc rien ici, on epingle les versions datees.
MODEL_SMALL = "ministral-8b-2512"
#   ministral-14b-2512       200, 30 requetes/minute
#
# D'ou la repartition : le 8b encaisse le volume (sentiment, un appel par
# article), le 14b sert aux libelles de themes -- une vingtaine par jour,
# mais ce sont eux que l'utilisateur lit en permanence.
MODEL_LARGE = "ministral-14b-2512"

_client = None
_client_lock = Lock()

# Ordre de consommation. Quatre suffisent : au-dela, c'est un compte payant
# qu'il faut, pas une nieme cle gratuite.
_VARS_CLES = ("MISTRAL_API_KEY", "MISTRAL_API_KEY2",
              "MISTRAL_API_KEY3", "MISTRAL_API_KEY4")
_cle_courante = 0        # index dans _cles()

# Modeles refuses par le palier d'abonnement du compte. Mistral repond 403
# « tier_not_allowed » : inutile de le redemander a chaque theme, on note le
# refus et l'on retombe sur MODEL_SMALL pour la duree du processus. Le jour
# ou le compte monte de palier, la premiere tentative repasse.
_modeles_refuses = set()
_nom_actif = None        # nom de variable servant le client en cache


def _cles():
    """Les clés réellement renseignées, dans l'ordre de consommation."""
    return [(v, os.environ[v].strip()) for v in _VARS_CLES
            if os.environ.get(v, "").strip()]


def _statut(ex):
    """Code HTTP de l'erreur, quel que soit l'attribut où le SDK le range."""
    for attr in ("status_code", "http_status", "code"):
        v = getattr(ex, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"\b([45]\d\d)\b", str(ex))
    return int(m.group(1)) if m else None


def _modele_interdit(ex):
    """Le palier du compte refuse-t-il ce modele ?"""
    return _statut(ex) == 403 and "tier_not_allowed" in str(ex).lower()


def _cle_epuisee(ex):
    """L'erreur condamne-t-elle la clé, ou est-elle seulement passagère ?

    Distinction essentielle : un 429 ordinaire est une limite par minute, il
    suffit d'attendre et la clé reste bonne. Un 401/402/403, ou un message
    parlant de quota ou de crédit, veut dire que cette clé ne servira plus --
    insister ferait perdre le run entier en attentes inutiles.
    """
    if _statut(ex) in (401, 402, 403):
        return True
    txt = str(ex).lower()
    return any(m in txt for m in ("quota", "insufficient", "credit",
                                  "billing", "payment required",
                                  "exceeded your", "no longer active"))


def _basculer():
    """Passe à la clé suivante. Renvoie False s'il n'y en a plus."""
    global _cle_courante, _client
    with _client_lock:
        cles = _cles()
        if _cle_courante + 1 >= len(cles):
            log.error("Clé %s épuisée et aucune autre disponible.",
                      cles[_cle_courante][0] if cles else "?")
            return False
        _cle_courante += 1
        _client = None       # forcera la reconstruction sur la nouvelle clé
        log.warning("Clé épuisée : bascule sur %s.", cles[_cle_courante][0])
        return True


def _load_mistral_class():
    """Importe la classe Mistral quel que soit le chemin selon la version du SDK."""
    errors = []
    for path in ("mistralai", "mistralai.client"):
        try:
            module = __import__(path, fromlist=["Mistral"])
            return getattr(module, "Mistral")
        except (ImportError, AttributeError) as e:
            errors.append(f"{path}: {e}")
    raise SystemExit(
        "Impossible d'importer le SDK Mistral. Détails :\n  "
        + "\n  ".join(errors)
        + "\nEssayez : pip install --upgrade mistralai"
    )


def get_client():
    global _client, _nom_actif
    if _client is None:
        # Verrou pour eviter une double init si plusieurs threads appellent
        # complete_json() en meme temps au tout debut d'un run parallelise.
        with _client_lock:
            if _client is None:
                Mistral = _load_mistral_class()
                cles = _cles()
                if not cles:
                    raise SystemExit(
                        "Aucune clé Mistral définie.\n"
                        "  .env      : MISTRAL_API_KEY=...\n"
                        "  (facultatif) MISTRAL_API_KEY2=... pour le relais\n"
                        '  PowerShell : $env:MISTRAL_API_KEY = "..."'
                    )
                nom, valeur = cles[min(_cle_courante, len(cles) - 1)]
                _client = Mistral(api_key=valeur)
                if nom != _nom_actif:
                    # La valeur n'est jamais journalisee, seulement son nom.
                    log.info("Clé Mistral active : %s (%d disponible%s).",
                             nom, len(cles), "s" if len(cles) > 1 else "")
                    _nom_actif = nom
    return _client


def parse_json(text):
    """Extrait le JSON meme si le modele a ajoute du texte autour."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None


def complete_json(system, user, model=MODEL_SMALL, max_tokens=800, retries=2):
    """Appelle Mistral en mode JSON. Renvoie un dict ou None.

    Deux boucles imbriquées : les tentatives sur la clé courante, puis le
    passage à la clé suivante si celle-ci est déclarée morte. On ne bascule
    JAMAIS sur une erreur passagère, sinon un incident réseau consommerait
    une clé encore bonne.
    """
    if model in _modeles_refuses:
        model = MODEL_SMALL
    while True:
        epuisee = False
        dernier_429 = False
        for attempt in range(retries + 1):
            try:
                resp = get_client().chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                return parse_json(resp.choices[0].message.content)
            except Exception as ex:
                if _modele_interdit(ex) and model != MODEL_SMALL:
                    log.warning("%s refuse par le palier du compte : repli sur %s.",
                                model, MODEL_SMALL)
                    _modeles_refuses.add(model)
                    model = MODEL_SMALL
                    continue
                epuisee = _cle_epuisee(ex)
                dernier_429 = _statut(ex) == 429
                log.warning("Mistral erreur (tentative %d/%d)%s : %s",
                            attempt + 1, retries + 1,
                            " [clé épuisée]" if epuisee else "", ex)
                if epuisee:
                    break        # insister ne servirait a rien
                if attempt < retries:
                    # Le rate-limit (429) est generalement une fenetre par
                    # minute : un backoff court ne laisse pas le temps au quota
                    # de se reinitialiser, d'ou des echecs meme apres plusieurs
                    # tentatives.
                    time.sleep(20 if _statut(ex) == 429 else 3)
        # Un 429 passager se resout dans les tentatives ; un 429 qui survit a
        # toutes, espacees de 20 s, vient du compte et non du debit. Constate
        # sur ce projet : l'en-tete x-ratelimit-limit-req-minute valait 0, le
        # palier n'autorisant aucune requete par minute -- les cles etaient
        # valides (GET /models repondait 200) et creditees. Passer a la cle
        # suivante est la seule chose utile a tenter ici.
        if dernier_429 and not epuisee:
            log.warning("429 sur les %d tentatives : limite de debit du compte "
                        "ou quota, on tente la cle suivante. Verifier "
                        "x-ratelimit-limit-req-minute sur console.mistral.ai.",
                        retries + 1)
            epuisee = True
        if not (epuisee and _basculer()):
            return None
