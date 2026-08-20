"""Thèmes du jour : clustering par support sur 24 h, puis recoupement.

Pourquoi par support. Mesuré sur ce corpus : deux transcriptions quelconques
se ressemblent à 0,77 quand deux articles de presse ne se ressemblent qu'à
0,45. Mélangés dans un même espace, les documents se regroupent d'abord par
REGISTRE de parole et seulement ensuite par sujet -- la pureté de support des
thèmes plafonnait à 59 % avec le clustering unique. Chaque support est donc
regroupé chez lui, où la comparaison est à armes égales, et les clusters sont
rapprochés APRÈS coup, sur leurs centroïdes.

Pourquoi 24 h. Une fenêtre glissante de 30 jours répond à « de quoi parle-t-on
en ce moment » mais ne permet pas de dire ce qui a changé d'un jour sur
l'autre : les thèmes y sont des moyennes de mois. Une fenêtre d'un jour rend
chaque journée comparable à la précédente, l'identité des thèmes étant tenue
par le registre de centroïdes plutôt que par la fenêtre.

Ce que la volumétrie impose. Sur les journées en régime, la presse sort 200 à
1100 documents, Telegram 100 à 800, la télévision 30 à 475 : c'est assez pour
regrouper. YouTube tourne à ~25 segments issus d'une ou deux vidéos et VK à
moins de 10 : les regrouper seuls reviendrait à découper une seule vidéo en
morceaux. En dessous de MIN_DOCS_SUPPORT un support n'est donc pas clusterisé
-- ses documents sont rattachés aux sujets du jour s'ils en sont assez
proches, sinon laissés non classés.

Usage :
  python scripts/analysis/analyze_topics_daily.py                  # aujourd'hui
  python scripts/analysis/analyze_topics_daily.py --date 2026-08-18
  python scripts/analysis/analyze_topics_daily.py --backfill 10    # 10 jours
"""
import argparse
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

from scripts.analysis.analyze_topics import (EMBED_MAX_TOKENS, EMBED_MODEL,
                                             EMBED_PREFIXE, MIN_CONTENT_LEN,
                                             MIN_CONTENT_LEN_TELEGRAM,
                                             NOISE_KEY, _cosine_sim_matrix,
                                             _generate_readable_label,
                                             _lemmatizing_tokenizer,
                                             _match_to_registry,
                                             _neutraliser_oral, ensure_schema)
from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("topics_daily")

# Ordre d'affichage seulement ; les supports absents sont ignorés.
SUPPORTS = ("press", "telegram", "tv", "youtube", "vk")

# En dessous, un support n'est pas regroupé pour lui-même : HDBSCAN y
# trouverait la structure d'une poignée de documents, pas des sujets.
MIN_DOCS_SUPPORT = 40

# Et surtout : le volume de segments ne dit rien du nombre de sujets. Une
# journée YouTube, c'est ~25 segments issus d'une ou deux vidéos ; les
# regrouper produirait « vidéo A » contre « vidéo B », pas des thèmes. On
# exige donc aussi un minimum d'unités parentes distinctes (une émission, une
# vidéo, un article).
MIN_PARENTS_SUPPORT = 5

# HDBSCAN choisit par défaut les clusters les plus « stables » de son arbre
# (excess of mass), ce qui laisse systématiquement un cluster fourre-tout : sur
# 557 articles de presse, un seul en absorbait 400. La sélection par feuilles
# prend le bas de l'arbre -- beaucoup de clusters fins plutôt qu'un gros et des
# miettes. C'est le seul réglage qui attaque directement le fourre-tout ; la
# taille minimale ne fait que déplacer le problème.
CLUSTER_SELECTION = "leaf"

# Similarité minimale entre deux centroïdes de supports différents pour dire
# qu'ils parlent du même sujet. e5 tasse ses similarités entre 0,75 et 0,93 :
# un seuil bas soude tout. Le script imprime la distribution des paires à
# chaque run, c'est elle qui doit guider l'ajustement.
CROSS_THRESHOLD = 0.90

# Similarité minimale pour rapprocher un sujet du jour d'un thème déjà connu.
REGISTRY_THRESHOLD = 0.62

SCHEMA_SUPPORTS = """
CREATE TABLE IF NOT EXISTS topic_supports (
    topic_key   INTEGER,
    run_date    DATE,
    source_kind VARCHAR,
    n_docs      INTEGER,
    top_words   VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_topic_supports ON topic_supports(run_date, topic_key);
"""


def _ensure_schema_jour(conn, reset=False):
    ensure_schema(conn, reset=reset)
    if reset:
        conn.execute("DROP TABLE IF EXISTS topic_supports")
    for stmt in SCHEMA_SUPPORTS.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt + ";")


def _charger(conn, jour):
    """Documents publiés dans la journée, par support."""
    rows = conn.execute(
        """
        SELECT id, content, title, source_kind,
               CASE WHEN source_kind = 'youtube'
                    THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1)
                    WHEN source_kind = 'tv'
                    THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1)
                    ELSE id END AS parent
        FROM articles
        WHERE content IS NOT NULL AND language = 'ru'
          AND LENGTH(content) >= (CASE WHEN source_kind = 'telegram' THEN ? ELSE ? END)
          AND CAST(published_at AS DATE) = ?
        """,
        [MIN_CONTENT_LEN_TELEGRAM, MIN_CONTENT_LEN, jour],
    ).fetchall()
    par_support = defaultdict(list)
    for i, r in enumerate(rows):
        par_support[r[3] or "press"].append(i)
    return rows, par_support


def _taille_min(n):
    """Un cluster doit peser assez pour être un sujet, sans exiger un volume
    que la journée n'a pas. 500 documents -> 15, 100 -> 4."""
    return max(4, min(15, n // 25))


def _clusteriser_support(support, indices, embeddings, docs):
    """Regroupe un seul support. Renvoie la liste de ses clusters."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(indices)
    sous_emb = embeddings[indices]
    sous_docs = [docs[i] for i in indices]
    taille = _taille_min(n)
    # UMAP exige n_neighbors < n : sur une petite journée le défaut planterait.
    voisins = max(2, min(10, n - 1))

    modele = BERTopic(
        umap_model=UMAP(n_neighbors=voisins, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=taille, metric="euclidean",
                              cluster_selection_method=CLUSTER_SELECTION,
                              prediction_data=True),
        # min_df/max_df s'appliquent au niveau des clusters, pas des
        # documents : a 5 clusters par jour, « present dans >= 2 et <= 50 % »
        # ne laisse presque aucun mot -- et sklearn leve une erreur des que la
        # bande est vide. On laisse passer, la ponderation c-TF-IDF penalise
        # deja d'elle-meme les mots presents partout.
        vectorizer_model=CountVectorizer(tokenizer=_lemmatizing_tokenizer,
                                         ngram_range=(1, 2), min_df=1,
                                         max_df=1.0),
        min_topic_size=taille,
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
    )
    locaux, _ = modele.fit_transform(sous_docs, embeddings=sous_emb)

    clusters = []
    info = modele.get_topic_info()
    rangs = list(info["Topic"])
    for rang, bt_id in enumerate(rangs):
        bt_id = int(bt_id)
        if bt_id == NOISE_KEY:
            continue
        membres = [indices[k] for k, t in enumerate(locaux) if int(t) == bt_id]
        if not membres:
            continue
        mots = modele.get_topic(bt_id) or []
        clusters.append({
            "support": support,
            "membres": membres,
            "centroide": np.asarray(modele.topic_embeddings_[rang], dtype=float),
            "mots": ", ".join(w for w, _ in mots[:15]),
        })
    log.info("  %-9s %5d documents, min_topic_size=%-3d -> %2d clusters, "
             "%d non regroupés", support, n, taille, len(clusters),
             sum(1 for t in locaux if int(t) == NOISE_KEY))
    return clusters


def _recouper(clusters, seuil):
    """Relie les clusters de supports DIFFÉRENTS qui parlent du même sujet.

    Appariement mutuel, et non simple seuil : deux clusters ne sont reliés que
    si chacun est le plus proche de l'autre dans son support. Un seuil seul
    reliait A-B et B-C, et la fermeture transitive soudait alors A, B et C --
    mesuré : 89 % du corpus dans un seul sujet. L'exigence de réciprocité
    supprime ce chaînage : un cluster ne peut avoir qu'un seul partenaire par
    support, celui qui le désigne en retour.
    """
    n = len(clusters)
    parent = list(range(n))

    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    par_support = defaultdict(list)
    for i, c in enumerate(clusters):
        par_support[c["support"]].append(i)

    liens, scores = [], []
    supports = sorted(par_support)
    for a in range(len(supports)):
        for b in range(a + 1, len(supports)):
            ia, ib = par_support[supports[a]], par_support[supports[b]]
            if not ia or not ib:
                continue
            sims = _cosine_sim_matrix(
                np.array([clusters[i]["centroide"] for i in ia]),
                np.array([clusters[j]["centroide"] for j in ib]))
            scores.extend(sims.ravel().tolist())
            meilleur_b = sims.argmax(axis=1)   # pour chaque cluster de a
            meilleur_a = sims.argmax(axis=0)   # pour chaque cluster de b
            for x, y in enumerate(meilleur_b):
                if meilleur_a[y] == x and sims[x, y] >= seuil:
                    liens.append((ia[x], ib[y], float(sims[x, y])))

    if scores:
        log.info("  recoupement : %d paires inter-supports, déciles 50/75/90/99 "
                 "= %s | %d liens mutuels retenus (seuil %.2f)", len(scores),
                 " / ".join(f"{v:.3f}" for v in
                            np.percentile(scores, [50, 75, 90, 99])),
                 len(liens), seuil)
    for i, j, _ in liens:
        ri, rj = racine(i), racine(j)
        if ri != rj:
            parent[ri] = rj

    groupes = defaultdict(list)
    for i in range(n):
        groupes[racine(i)].append(i)
    return list(groupes.values())


def _rattacher_isoles(sujets, embeddings, deja_classes, tous_indices):
    """Les documents laissés de côté rejoignent le sujet le plus proche, si
    et seulement si leur proximité vaut celle du quart le plus faible des
    documents déjà classés. Un seuil absolu ne tiendrait pas : chaque modèle
    d'embedding a sa propre échelle de similarité."""
    isoles = [i for i in tous_indices if i not in deja_classes]
    if not isoles or not sujets:
        return {}, isoles

    cent = np.array([s["centroide"] for s in sujets])
    sims_classes = _cosine_sim_matrix(embeddings[sorted(deja_classes)], cent)
    propres = sims_classes.max(axis=1)
    seuil = float(np.percentile(propres, 25)) if len(propres) else 1.0

    sims = _cosine_sim_matrix(embeddings[isoles], cent)
    meilleurs = sims.argmax(axis=1)
    scores = sims.max(axis=1)
    affecte, restants = {}, []
    for k, i in enumerate(isoles):
        if scores[k] >= seuil:
            affecte[i] = (int(meilleurs[k]), float(scores[k]))
        else:
            restants.append(i)
    log.info("  rattachement : seuil %.2f (1er quartile des classés), "
             "%d/%d isolés rattachés", seuil, len(affecte), len(isoles))
    return affecte, restants


def traiter_jour(conn, jour, seuil_cross, seuil_registre, embed):
    rows, par_support = _charger(conn, jour)
    if not rows:
        log.warning("%s : aucun document éligible, journée ignorée.", jour)
        return
    docs = [r[1] for r in rows]
    titres = [r[2] or "" for r in rows]
    log.info("%s : %d documents (%s)", jour, len(rows),
             ", ".join(f"{k} {len(v)}" for k, v in
                       sorted(par_support.items(), key=lambda x: -len(x[1]))))

    # Les marqueurs d'oral sont retirés du texte encodé ; docs reste intact
    # pour le c-TF-IDF et les titres d'exemple.
    embeddings = embed.encode([EMBED_PREFIXE + _neutraliser_oral(d) for d in docs],
                              show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=float)

    clusters = []
    maigres = []
    for support in SUPPORTS:
        indices = par_support.get(support, [])
        if not indices:
            continue
        n_parents = len({rows[i][4] for i in indices})
        if len(indices) < MIN_DOCS_SUPPORT or n_parents < MIN_PARENTS_SUPPORT:
            maigres.append(support)
            log.info("  %-9s %5d documents / %d unités -- sous le seuil "
                     "(%d docs, %d unités), non regroupé pour lui-même",
                     support, len(indices), n_parents, MIN_DOCS_SUPPORT,
                     MIN_PARENTS_SUPPORT)
            continue
        clusters.extend(_clusteriser_support(support, indices, embeddings, docs))

    if not clusters:
        log.warning("%s : aucun cluster formé, journée ignorée.", jour)
        return

    groupes = _recouper(clusters, seuil_cross)
    sujets = []
    for g in groupes:
        membres = [i for k in g for i in clusters[k]["membres"]]
        poids = np.array([len(clusters[k]["membres"]) for k in g], dtype=float)
        cent = np.average([clusters[k]["centroide"] for k in g], axis=0,
                          weights=poids)
        sujets.append({
            "clusters": [clusters[k] for k in g],
            "membres": membres,
            "centroide": cent,
            "mots": " | ".join(clusters[k]["mots"] for k in g),
        })
    multi = sum(1 for s in sujets if len({c["support"] for c in s["clusters"]}) > 1)
    log.info("  %d sujets, dont %d présents sur plusieurs supports", len(sujets), multi)

    deja = {i for s in sujets for i in s["membres"]}
    affecte, non_classes = _rattacher_isoles(sujets, embeddings, deja,
                                             list(range(len(rows))))

    # --- Identité : rapprochement avec les thèmes déjà connus ---------------
    registre = conn.execute(
        "SELECT topic_key, centroid, label FROM topics WHERE topic_key != ? "
        "AND centroid IS NOT NULL AND len(centroid) > 0", [NOISE_KEY]).fetchall()
    centroides = [s["centroide"] for s in sujets]
    provisoires = [s["mots"][:60] for s in sujets]
    appariement, _ = _match_to_registry(centroides, provisoires,
                                        [(r[0], r[1], r[2]) for r in registre],
                                        seuil_registre)

    titres_par_sujet = {}
    for idx, s in enumerate(sujets):
        titres_par_sujet[idx] = [titres[i] for i in s["membres"][:10]]

    cles, n_repris, n_neufs = [], 0, 0
    for idx, s in enumerate(sujets):
        mots = ", ".join(dict.fromkeys(s["mots"].replace(" | ", ", ").split(", ")))[:400]
        if idx in appariement:
            cle, _sim = appariement[idx]
            conn.execute(
                "UPDATE topics SET top_words = ?, centroid = ?, last_seen = ?, "
                "active = TRUE WHERE topic_key = ?",
                [mots, list(map(float, s["centroide"])), jour, cle])
            n_repris += 1
        else:
            # Un appel Mistral seulement pour les sujets réellement nouveaux :
            # relabelliser chaque jour tous les thèmes coûterait le prix du
            # corpus entier pour un résultat identique.
            libelle = _generate_readable_label(
                mots, titres_par_sujet[idx],
                fallback=" / ".join(mots.split(", ")[:3]))
            cle = conn.execute("SELECT nextval('topic_key_seq')").fetchone()[0]
            conn.execute(
                "INSERT INTO topics (topic_key, label, top_words, centroid, "
                "first_seen, last_seen, active) VALUES (?, ?, ?, ?, ?, ?, TRUE)",
                [cle, libelle, mots, list(map(float, s["centroide"])), jour, jour])
            n_neufs += 1
            log.info("  nouveau thème : %s", libelle)
        cles.append(cle)

    # --- Écriture ----------------------------------------------------------
    ids = [r[0] for r in rows]
    conn.execute("CREATE OR REPLACE TEMP TABLE _ids_jour AS SELECT UNNEST(?) AS id", [ids])
    conn.execute("DELETE FROM article_topics WHERE article_id IN (SELECT id FROM _ids_jour)")
    conn.execute("DELETE FROM topic_supports WHERE run_date = ?", [jour])

    lignes = []
    for idx, s in enumerate(sujets):
        for i in s["membres"]:
            lignes.append((ids[i], cles[idx], 1.0, jour))
    for i, (idx, score) in affecte.items():
        lignes.append((ids[i], cles[idx], score, jour))
    for i in non_classes:
        lignes.append((ids[i], NOISE_KEY, 0.0, jour))
    conn.executemany(
        "INSERT INTO article_topics (article_id, topic_key, probability, run_date) "
        "VALUES (?, ?, ?, ?)", lignes)

    # La ventilation par support est ce qui rend le recoupement lisible :
    # elle dit si un thème vit dans la presse, à la télévision, ou dans les deux.
    ventilation = []
    for idx, s in enumerate(sujets):
        for c in s["clusters"]:
            ventilation.append((cles[idx], jour, c["support"],
                                len(c["membres"]), c["mots"][:400]))
    if ventilation:
        conn.executemany(
            "INSERT INTO topic_supports (topic_key, run_date, source_kind, "
            "n_docs, top_words) VALUES (?, ?, ?, ?, ?)", ventilation)

    _assurer_ligne_non_classe(conn, jour)
    log.info("  %s : %d sujets (%d repris, %d nouveaux), %d non classés (%.1f %%)"
             "%s", jour, len(sujets), n_repris, n_neufs, len(non_classes),
             100 * len(non_classes) / len(rows),
             f", supports non regroupés : {', '.join(maigres)}" if maigres else "")


def _assurer_ligne_non_classe(conn, jour):
    if conn.execute("SELECT 1 FROM topics WHERE topic_key = ?", [NOISE_KEY]).fetchone():
        conn.execute("UPDATE topics SET last_seen = ? WHERE topic_key = ?", [jour, NOISE_KEY])
    else:
        conn.execute(
            "INSERT INTO topics (topic_key, label, top_words, centroid, "
            "first_seen, last_seen, active) VALUES (?, 'Non classé', '', [], ?, ?, TRUE)",
            [NOISE_KEY, jour, jour])


def run(jour=None, backfill=0, seuil_cross=CROSS_THRESHOLD,
        seuil_registre=REGISTRY_THRESHOLD, reset=False):
    import duckdb
    from sentence_transformers import SentenceTransformer

    try:
        conn = get_conn()
    except duckdb.IOException:
        # Cas NORMAL : le tableau de bord est ouvert, ou une autre analyse
        # ecrit. DuckDB n'admet qu'un ecrivain -- le dire plutot que de
        # deverser une trace, la seule chose a faire etant d'attendre.
        log.error("La base est ouverte par un autre processus (tableau de bord "
                  "ou autre analyse). DuckDB n'admet qu'un ecrivain a la fois : "
                  "fermez-le et relancez.")
        return 1
    _ensure_schema_jour(conn, reset=reset)

    # La routine collecte puis analyse dans la meme passe : le jour a
    # regrouper est celui qui vient d'etre collecte, pas la veille.
    fin = jour or date.today()
    jours = [fin - timedelta(days=k) for k in range(backfill, -1, -1)]

    log.info("Chargement du modèle d'embedding (%s)...", EMBED_MODEL)
    embed = SentenceTransformer(EMBED_MODEL)
    embed.max_seq_length = EMBED_MAX_TOKENS

    for j in jours:
        traiter_jour(conn, j, seuil_cross, seuil_registre, embed)

    # Un thème sans document sur les 7 derniers jours passe en sommeil : il
    # garde sa clé et son historique, et se réactive si son sujet revient.
    conn.execute("""
        UPDATE topics SET active = (last_seen >= ?) WHERE topic_key != ?""",
        [fin - timedelta(days=7), NOISE_KEY])
    conn.execute(
        "UPDATE topics SET article_count = (SELECT COUNT(*) FROM article_topics "
        "WHERE article_topics.topic_key = topics.topic_key)")

    # Les evaluations ProxAnn portent sur un clustering donne : celles dont le
    # theme n'existe plus ne veulent plus rien dire, et laissees en place elles
    # remontaient jusqu'au graphique de qualite avec un volume vide.
    # La table n'existe que si validate_topics.py a deja tourne.
    if conn.execute("SELECT 1 FROM duckdb_tables() WHERE table_name = "
                    "'topic_quality'").fetchone():
        n_orph = conn.execute(
            "SELECT COUNT(*) FROM topic_quality WHERE topic_key NOT IN "
            "(SELECT topic_key FROM topics)").fetchone()[0]
        if n_orph:
            conn.execute("DELETE FROM topic_quality WHERE topic_key NOT IN "
                         "(SELECT topic_key FROM topics)")
            log.info("%d évaluations de thèmes disparus supprimées de "
                     "topic_quality.", n_orph)

    _resume(conn, jours)
    conn.close()
    return 0


def _resume(conn, jours):
    log.info("")
    log.info("=== Thèmes par jour ===")
    for j in jours:
        # Le decompte et la liste des supports se calculent separement : les
        # joindre multiplierait chaque article par le nombre de supports du
        # theme (un theme sur trois supports comptait triple).
        lignes = conn.execute("""
            WITH n AS (
              SELECT topic_key, COUNT(*) n FROM article_topics
              WHERE run_date = ? AND topic_key != ? GROUP BY 1),
            s AS (
              SELECT topic_key, STRING_AGG(DISTINCT source_kind, '+') supports
              FROM topic_supports WHERE run_date = ? GROUP BY 1)
            SELECT t.label, n.n, s.supports
            FROM n JOIN topics t ON t.topic_key = n.topic_key
            LEFT JOIN s ON s.topic_key = n.topic_key
            ORDER BY n.n DESC LIMIT 8""", [j, NOISE_KEY, j]).fetchall()
        if not lignes:
            continue
        log.info("%s", j)
        for label, n, supports in lignes:
            log.info("   %4d  %-46s %s", n, (label or "")[:46], supports or "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="jour à traiter (AAAA-MM-JJ), défaut : aujourd'hui")
    p.add_argument("--backfill", type=int, default=0,
                   help="traiter aussi les N jours précédents, du plus ancien "
                        "au plus récent")
    p.add_argument("--cross", type=float, default=CROSS_THRESHOLD,
                   help="similarité minimale pour recouper deux supports")
    p.add_argument("--threshold", type=float, default=REGISTRY_THRESHOLD,
                   help="similarité minimale pour retrouver un thème connu")
    p.add_argument("--reset", action="store_true",
                   help="vide topics/article_topics/topic_supports")
    a = p.parse_args()
    sys.exit(run(
        jour=datetime.strptime(a.date, "%Y-%m-%d").date() if a.date else None,
        backfill=a.backfill, seuil_cross=a.cross, seuil_registre=a.threshold,
        reset=a.reset))
