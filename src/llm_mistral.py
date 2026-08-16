"""Helper partagé pour les appels Mistral (remplace l'API Claude).

Tous les scripts d'analyse importent complete_json() depuis ce module.
Pour rebasculer sur Claude un jour, il suffit de réécrire ce seul fichier.

Prérequis :
  pip install mistralai
  $env:MISTRAL_API_KEY = "..."   (clé créée sur https://console.mistral.ai)

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

MODEL_SMALL = "mistral-small-latest"
MODEL_LARGE = "mistral-large-latest"

_client = None
_client_lock = Lock()


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
    global _client
    if _client is None:
        # Verrou pour eviter une double init si plusieurs threads appellent
        # complete_json() en meme temps au tout debut d'un run parallelise.
        with _client_lock:
            if _client is None:
                Mistral = _load_mistral_class()
                if "MISTRAL_API_KEY" not in os.environ:
                    raise SystemExit(
                        "Variable MISTRAL_API_KEY non definie.\n"
                        '  PowerShell : $env:MISTRAL_API_KEY = "..."'
                    )
                _client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
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
    """Appelle Mistral en mode JSON. Renvoie un dict ou None."""
    client = get_client()
    for attempt in range(retries + 1):
        try:
            resp = client.chat.complete(
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
            log.warning("Mistral erreur (tentative %d/%d): %s",
                        attempt + 1, retries + 1, ex)
            if attempt < retries:
                # Le rate-limit (429) est generalement une fenetre par minute :
                # un backoff court ne laisse pas le temps au quota de se
                # reinitialiser, d'ou des echecs meme apres plusieurs tentatives.
                wait = 20 if "429" in str(ex) else 3
                time.sleep(wait)
    return None
