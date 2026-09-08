"""Dit quels modèles Mistral ce compte peut réellement appeler.

À lancer quand les libellés de thèmes reviennent en russe, ou après tout
changement d'abonnement chez Mistral. Le tableau de bord ne dit rien de tout
ça : il affiche simplement le repli, qui est la liste des mots-clés russes.

Le script interroge l'API directement, sans le SDK, et affiche pour chaque
modèle le code HTTP et le débit que Mistral annonce dans ses en-têtes. C'est
la seule source fiable : la page « Limites » de la console peut afficher un
débit pour un modèle que l'API refuse malgré tout.

Aucune clé n'est affichée, seulement ses quatre derniers caractères.

Usage :
  python scripts/maintenance/check_mistral.py
"""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from src.llm_mistral import MODEL_LARGE, MODEL_SMALL, _VARS_CLES

URL = "https://api.mistral.ai/v1"

# Les deux modèles configurés, puis les autres candidats du même compte : on
# veut savoir non seulement si la configuration marche, mais ce qui marcherait
# si elle cassait.
CANDIDATS = [MODEL_SMALL, MODEL_LARGE, "mistral-small-latest",
             "mistral-medium-latest", "mistral-large-latest",
             "ministral-3b-2512", "ministral-8b-2512", "ministral-14b-2512"]


def _essayer(entetes, modele):
    try:
        r = httpx.post(f"{URL}/chat/completions", headers=entetes, timeout=60,
                       json={"model": modele, "max_tokens": 5,
                             "messages": [{"role": "user", "content": "ok"}]})
    except Exception as e:
        return "EXC", "-", str(e)[:60]
    debit = r.headers.get("x-ratelimit-limit-req-minute", "-")
    if r.status_code == 200:
        return r.status_code, debit, ""
    try:
        motif = r.json().get("message", "")[:62]
    except Exception:
        motif = r.text[:62]
    return r.status_code, debit, motif


def run():
    load_dotenv()
    vus = set()
    modeles = [m for m in CANDIDATS if not (m in vus or vus.add(m))]
    utilisables = []

    for nom in _VARS_CLES:
        cle = os.environ.get(nom, "").strip()
        if not cle:
            continue
        entetes = {"Authorization": "Bearer " + cle}
        print(f"\n=== {nom}  (se termine par …{cle[-4:]}) ===")
        try:
            r = httpx.get(f"{URL}/models", headers=entetes, timeout=30)
            if r.status_code == 200:
                print(f"  clé valide, {len(r.json().get('data', []))} modèles visibles")
            else:
                print(f"  clé REFUSÉE : {r.status_code} {r.text[:80]}")
                continue
        except Exception as e:
            print("  injoignable :", str(e)[:70])
            continue

        print(f"  {'modèle':24} {'code':>5} {'req/min':>8}  motif du refus")
        for m in modeles:
            code, debit, motif = _essayer(entetes, m)
            marque = "OK " if code == 200 else "   "
            role = ""
            if m == MODEL_SMALL:
                role = "  <- configuré (volume)"
            elif m == MODEL_LARGE:
                role = "  <- configuré (libellés)"
            print(f"  {marque}{m:24} {code:>5} {debit:>8}  {motif}{role}")
            if code == 200:
                utilisables.append((nom, m))
            time.sleep(4)     # bien en dessous du débit le plus bas

    print()
    if utilisables:
        print("Modèles utilisables :")
        for nom, m in utilisables:
            print(f"  {nom} -> {m}")
        config = {m for _, m in utilisables}
        manquants = [m for m in (MODEL_SMALL, MODEL_LARGE) if m not in config]
        if manquants:
            print("\nATTENTION : les modèles configurés dans src/llm_mistral.py "
                  "ne répondent pas :")
            for m in manquants:
                print(f"  {m}")
            print("Remplacer MODEL_SMALL / MODEL_LARGE par un des modèles "
                  "utilisables ci-dessus.")
    else:
        print("Aucun modèle ne répond. Les libellés de thèmes resteront en "
              "russe (repli sur les mots-clés).")
        print("À vérifier sur console.mistral.ai : Admin > Limites, et "
              "l'activation du pay-as-you-go.")
    return 0 if utilisables else 1


if __name__ == "__main__":
    sys.exit(run())
