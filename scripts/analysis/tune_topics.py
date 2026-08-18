"""Cherche les parametres de clustering au lieu de les deviner.

Jusqu'ici min_topic_size, n_neighbors et le plafond de themes etaient regles a
l'oeil, en regardant si les libelles « avaient l'air » mieux. Ce script teste
une grille et mesure chaque configuration sur des defauts qu'on a constates :

    dominant      part du plus gros theme -- un fourre-tout le fait monter
    non_classe    part laissee de cote par HDBSCAN
    purete_media  part moyenne de la nature de contenu majoritaire dans un
                  theme. Elevee = les themes separent les SUPPORTS (video
                  contre presse) au lieu des sujets, defaut mesure a 96 % sur
                  certains clusters
    mono_parent   part moyenne de la video ou emission dominante dans un theme
                  -- une seule video ne doit pas faire un theme

Aucun appel de modele de langue : ces mesures sont structurelles et gratuites.
L'embedding, seule etape couteuse, est calcule UNE fois et reutilise pour
toutes les configurations -- c'est ce qui rend la grille abordable (9 minutes
puis une minute par configuration).

La validation ProxAnn, elle, reste a lancer sur la configuration retenue :
elle seule dit si les themes ont un sens pour un lecteur.

Usage :
  python scripts/analysis/tune_topics.py
  python scripts/analysis/tune_topics.py --days 30 --sizes 10,15,25 --neighbors 5,10,15
"""
import argparse
import time
from collections import Counter
from datetime import date, timedelta

import numpy as np

from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("tune_topics")


def _charger(conn, window_days):
    debut = date.today() - timedelta(days=window_days)
    from scripts.analysis.analyze_topics import (MIN_CONTENT_LEN,
                                                 MIN_CONTENT_LEN_TELEGRAM)
    return conn.execute(
        """SELECT content, source_kind,
                  CASE WHEN source_kind = 'youtube'
                       THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1)
                       WHEN source_kind = 'tv'
                       THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1)
                       ELSE id END AS parent
           FROM articles
           WHERE content IS NOT NULL
             AND LENGTH(content) >= (CASE WHEN source_kind = 'telegram'
                                          THEN ? ELSE ? END)
             AND language = 'ru' AND published_at >= ?""",
        [MIN_CONTENT_LEN_TELEGRAM, MIN_CONTENT_LEN, debut]).fetchall()


def _mesurer(assignations, kinds, parents):
    """Les quatre defauts, sur une affectation donnee."""
    n = len(assignations)
    classes = [(t, k, p) for t, k, p in zip(assignations, kinds, parents) if t != -1]
    if not classes:
        return None
    tailles = Counter(t for t, _, _ in classes)

    par_theme_kind, par_theme_parent = {}, {}
    for t, k, p in classes:
        par_theme_kind.setdefault(t, Counter())[k] += 1
        par_theme_parent.setdefault(t, Counter())[p] += 1

    # Moyennes ponderees par la taille du theme : un micro-theme pur ne doit
    # pas peser autant qu'un gros theme melange.
    total_classes = len(classes)
    purete = sum(c.most_common(1)[0][1] for c in par_theme_kind.values()) / total_classes
    mono = sum(c.most_common(1)[0][1] for c in par_theme_parent.values()) / total_classes

    return {
        "themes": len(tailles),
        "dominant": 100 * max(tailles.values()) / n,
        "non_classe": 100 * sum(1 for t in assignations if t == -1) / n,
        "purete_media": 100 * purete,
        "mono_parent": 100 * mono,
    }


def run(window_days=30, sizes=(10, 15, 25), neighbors=(5, 10, 15), max_topics=90):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    from scripts.analysis.analyze_topics import (EMBED_MAX_TOKENS, EMBED_MODEL,
                                                 _lemmatizing_tokenizer,
                                                 _neutraliser_oral)

    conn = get_conn(read_only=True)
    rows = _charger(conn, window_days)
    conn.close()
    docs = [r[0] for r in rows]
    kinds = [r[1] or "press" for r in rows]
    parents = [r[2] or str(i) for i, r in enumerate(rows)]
    log.info("%d documents", len(docs))

    log.info("Embedding (une seule fois pour toute la grille)...")
    t0 = time.time()
    embed = SentenceTransformer(EMBED_MODEL)
    embed.max_seq_length = EMBED_MAX_TOKENS
    embeddings = embed.encode([_neutraliser_oral(d) for d in docs],
                              show_progress_bar=True)
    log.info("  %.0f s", time.time() - t0)

    resultats = []
    for n_neigh in neighbors:
        for size in sizes:
            t0 = time.time()
            modele = BERTopic(
                embedding_model=embed,
                umap_model=UMAP(n_neighbors=n_neigh, n_components=5,
                                min_dist=0.0, metric="cosine", random_state=42),
                vectorizer_model=CountVectorizer(
                    tokenizer=_lemmatizing_tokenizer, ngram_range=(1, 2),
                    min_df=2, max_df=0.5),
                min_topic_size=size, nr_topics=max_topics,
                language="multilingual", calculate_probabilities=False,
                verbose=False)
            ids, _ = modele.fit_transform(docs, embeddings=embeddings)
            m = _mesurer(list(ids), kinds, parents)
            if not m:
                continue
            m.update({"min_topic_size": size, "n_neighbors": n_neigh,
                      "duree_s": time.time() - t0})
            resultats.append(m)
            log.info("  size=%-3d voisins=%-3d -> %2d themes | dominant %5.1f%% | "
                     "non classe %5.1f%% | purete media %4.1f%% | mono-parent %4.1f%%",
                     size, n_neigh, m["themes"], m["dominant"], m["non_classe"],
                     m["purete_media"], m["mono_parent"])

    if not resultats:
        log.error("Aucune configuration exploitable.")
        return

    # Un score unique pour trancher : on veut peu de dominance, peu de themes
    # definis par leur support, peu de themes portes par une seule video. Le
    # taux de non-classes n'entre pas dans le score -- le rattachement le
    # corrige ensuite, et le penaliser pousserait vers des clusters mous.
    for m in resultats:
        m["score"] = m["dominant"] + m["purete_media"] + m["mono_parent"]
    resultats.sort(key=lambda m: m["score"])

    log.info("")
    log.info("Classement (score bas = mieux) :")
    for m in resultats:
        log.info("  %6.1f  size=%-3d voisins=%-3d  %d themes",
                 m["score"], m["min_topic_size"], m["n_neighbors"], m["themes"])
    best = resultats[0]
    log.info("")
    log.info("Retenu : min_topic_size=%d, n_neighbors=%d",
             best["min_topic_size"], best["n_neighbors"])
    log.info("A confirmer par : python scripts/analysis/analyze_topics.py --reset "
             "--min-size %d  puis validate_topics.py", best["min_topic_size"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sizes", default="10,15,25")
    ap.add_argument("--neighbors", default="5,10,15")
    ap.add_argument("--max-topics", type=int, default=90)
    args = ap.parse_args()
    run(window_days=args.days,
        sizes=tuple(int(x) for x in args.sizes.split(",")),
        neighbors=tuple(int(x) for x in args.neighbors.split(",")),
        max_topics=args.max_topics)
