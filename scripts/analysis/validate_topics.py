"""Est-ce que nos themes tiennent ? Protocole ProxAnn.

D'apres Hoyle et al., « ProxAnn: Use-Oriented Evaluations of Topic Models and
Document Clustering » (2025). Les metriques automatiques de coherence
s'accordent mal avec le jugement humain ; le protocole propose imite ce que
fait vraiment un lecteur :

    1. on lui montre quelques documents d'un cluster,
    2. il en deduit une categorie,
    3. on teste si cette categorie s'applique a d'AUTRES documents.

Un modele joue ici le role de l'annotateur. Deux mesures en sortent :

    coherence      part des documents du meme theme que la categorie accepte
    distinction    part des documents d'autres themes qu'elle rejette

Un theme fourre-tout a une coherence haute et une distinction basse : sa
categorie est si large qu'elle accepte n'importe quoi. Un theme coupe en deux
a l'inverse. Les deux ensemble disent ce qu'aucune des deux ne dit seule.

Cout : deux appels par theme.

Usage :
  python scripts/analysis/validate_topics.py
  python scripts/analysis/validate_topics.py --topics 20 --reset
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from src.db import get_conn
from src.llm_mistral import complete_json, MODEL_SMALL
from src.logging_setup import setup_logging

log = setup_logging("valid_topics")

N_EXEMPLES = 5      # documents montres pour deduire la categorie
N_TEST = 6          # documents du meme theme, a accepter
N_CONTROLE = 6      # documents d'autres themes, a rejeter
EXTRAIT_CHARS = 600
MAX_WORKERS = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_quality (
    topic_key   INTEGER,
    categorie   VARCHAR,   -- la categorie deduite des exemples
    coherence   DOUBLE,    -- 0 a 1
    distinction DOUBLE,    -- 0 a 1
    n_test      INTEGER,
    n_controle  INTEGER,
    run_date    DATE
);
"""

PROMPT_DEDUIRE = """Vous lisez des extraits d'articles de presse russophone qui ont ete regroupes automatiquement.

Deduisez la categorie commune a ces extraits : une phrase courte qui dit ce qu'ils ont en commun, assez precise pour qu'on puisse tester si un autre article y appartient.

Si les extraits n'ont rien en commun, dites-le : categorie = "aucune categorie commune".

Repondez en JSON : {"categorie": "..."}"""

PROMPT_JUGER = """Vous verifiez si des articles appartiennent a une categorie donnee.

Pour chaque extrait numerote, repondez oui ou non : appartient-il a la categorie ?

Repondez en JSON : {"reponses": [{"n": 1, "appartient": true}, ...]}"""


def ensure_schema(conn, reset=False):
    if reset:
        conn.execute("DROP TABLE IF EXISTS topic_quality")
    conn.execute(SCHEMA)


def _extraits(rows):
    return "\n\n".join(f"[{i + 1}] {(t or '')[:EXTRAIT_CHARS]}"
                       for i, (t,) in enumerate(rows))


def _evaluer(topic_key, label, exemples, tests, controles):
    """Renvoie (categorie, coherence, distinction) pour un theme."""
    deduit = complete_json(PROMPT_DEDUIRE, _extraits(exemples),
                           model=MODEL_SMALL, max_tokens=120)
    categorie = (deduit or {}).get("categorie", "").strip()
    if not categorie:
        return None
    if categorie.lower().startswith("aucune"):
        # Le modele ne trouve aucun denominateur commun : le theme est
        # incoherent, inutile de depenser un second appel.
        return categorie, 0.0, 1.0

    melange = tests + controles
    juge = complete_json(
        PROMPT_JUGER,
        f"Categorie : {categorie}\n\nExtraits :\n{_extraits(melange)}",
        model=MODEL_SMALL, max_tokens=400)
    reponses = (juge or {}).get("reponses") or []
    verdicts = {}
    for r in reponses:
        try:
            verdicts[int(r["n"])] = bool(r["appartient"])
        except (KeyError, TypeError, ValueError):
            continue
    if not verdicts:
        return None

    # Les tests occupent les premieres positions, les controles la suite.
    acceptes = sum(1 for i in range(1, len(tests) + 1) if verdicts.get(i))
    rejetes = sum(1 for i in range(len(tests) + 1, len(melange) + 1)
                  if verdicts.get(i) is False)
    return (categorie,
            acceptes / max(len(tests), 1),
            rejetes / max(len(controles), 1))


def run(n_topics=None, reset=False):
    conn = get_conn()
    ensure_schema(conn, reset=reset)

    themes = conn.execute("""
        SELECT t.topic_key, t.label, COUNT(*) AS n
        FROM article_topics at_ JOIN topics t ON t.topic_key = at_.topic_key
        WHERE at_.topic_key <> -1
        GROUP BY 1, 2 HAVING COUNT(*) >= ?
        ORDER BY n DESC""", [N_EXEMPLES + N_TEST]).fetchall()
    if n_topics:
        themes = themes[:n_topics]
    log.info("%d themes a evaluer (%d appels Mistral)", len(themes), 2 * len(themes))

    # Les documents sont lus d'avance : DuckDB n'aime pas les requetes
    # concurrentes sur la meme connexion.
    travail = []
    for topic_key, label, _ in themes:
        docs = conn.execute("""
            SELECT a.content FROM article_topics at_
            JOIN articles a ON a.id = at_.article_id
            WHERE at_.topic_key = ? AND a.content IS NOT NULL
            ORDER BY at_.probability DESC
            LIMIT ?""", [topic_key, N_EXEMPLES + N_TEST]).fetchall()
        controles = conn.execute("""
            SELECT a.content FROM article_topics at_
            JOIN articles a ON a.id = at_.article_id
            WHERE at_.topic_key <> ? AND at_.topic_key <> -1
              AND a.content IS NOT NULL
            ORDER BY random() LIMIT ?""", [topic_key, N_CONTROLE]).fetchall()
        if len(docs) < N_EXEMPLES + 2 or len(controles) < 2:
            continue
        travail.append((topic_key, label, docs[:N_EXEMPLES],
                        docs[N_EXEMPLES:], controles))

    conn.execute("DELETE FROM topic_quality WHERE run_date = ?", [date.today()])
    resultats = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_evaluer, tk, lb, ex_, te, co): (tk, lb)
                   for tk, lb, ex_, te, co in travail}
        for f in as_completed(futures):
            tk, lb = futures[f]
            try:
                out = f.result()
            except Exception as e:
                log.warning("theme %s : %s", tk, str(e)[:90])
                continue
            if not out:
                continue
            categorie, coherence, distinction = out
            resultats.append((tk, categorie, coherence, distinction,
                              N_TEST, N_CONTROLE, date.today()))
            log.info("  #%-3s coherence %.2f  distinction %.2f  | %s",
                     tk, coherence, distinction, (lb or "")[:38])

    if resultats:
        conn.executemany(
            """INSERT INTO topic_quality (topic_key, categorie, coherence,
               distinction, n_test, n_controle, run_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""", resultats)

    moy_c = sum(r[2] for r in resultats) / max(len(resultats), 1)
    moy_d = sum(r[3] for r in resultats) / max(len(resultats), 1)
    faibles = sum(1 for r in resultats if r[2] < 0.5 or r[3] < 0.5)
    log.info("Termine. %d themes evalues. Coherence moyenne %.2f, "
             "distinction moyenne %.2f, %d themes faibles.",
             len(resultats), moy_c, moy_d, faibles)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=None,
                    help="n'evaluer que les N plus gros themes")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    run(n_topics=args.topics, reset=args.reset)
