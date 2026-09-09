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

Segments et unités. Une transcription est découpée en segments d'environ
2 000 signes, parce que l'encodage s'arrête à 512 jetons et que les analyses
tronquent : le segment est donc l'objet du clustering. Mais une émission de
trois heures en produit jusqu'à quatre-vingt-treize, et si l'on COMPTE des
segments elle pèse autant que quatre-vingt-treize articles de presse. Tout ce
qui sert à décider ou à classer compte donc des unités parentes distinctes :
le seuil d'un support, la taille d'un thème, son rang. Un cluster qui ne
recouvre pas MIN_UNITES_CLUSTER unités est écarté -- ce n'est pas un sujet,
c'est une émission.

Sur les journées en régime, la presse sort 550 à 820 unités et Telegram 700 à
800, ce qui suffit ; la télévision tombe à une vingtaine et YouTube à deux ou
trois. Sous MIN_DOCS_SUPPORT le support est repris sur une fenêtre de plusieurs
jours, et s'il n'y arrive toujours pas ses documents sont rattachés aux sujets
du jour quand ils en sont assez proches, sinon laissés non classés.

Usage :
  python scripts/analysis/analyze_topics_daily.py                  # aujourd'hui
  python scripts/analysis/analyze_topics_daily.py --date 2026-08-18
  python scripts/analysis/analyze_topics_daily.py --backfill 10    # 10 jours
"""
import argparse
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

from src.topic_labels import _generate_readable_label
from src.topics import (EMBED_MAX_TOKENS, EMBED_MODEL, EMBED_PREFIXE,
                        MIN_CONTENT_LEN, MIN_CONTENT_LEN_TELEGRAM, NOISE_KEY,
                        SQL_PARENT, _cosine_sim_matrix, _lemmatizing_tokenizer,
                        _match_to_registry, _neutraliser_oral, ensure_schema)
from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("topics_daily")

# Ordre d'affichage seulement ; les supports absents sont ignorés.
SUPPORTS = ("press", "telegram", "tv", "youtube", "vk")

# En dessous, un support n'est pas regroupé pour lui-même : HDBSCAN y
# trouverait la structure d'une poignée de documents, pas des sujets. Le seuil
# compte des UNITÉS distinctes (une émission, une vidéo, un article) et non des
# segments : une seule émission découpée en quatre-vingt-dix morceaux
# franchissait autrement n'importe quel seuil sans apporter un seul sujet.
MIN_DOCS_SUPPORT = 12

# Un cluster doit recouvrir au moins ce nombre d'unités distinctes. C'est la
# réponse au fourre-tout de registre : quatre-vingts segments d'un même plateau
# forment un cluster dense et parfaitement homogène, que HDBSCAN retient
# volontiers, mais qui n'est pas un sujet -- c'est une émission.
#
# Mesuré : regrouper les segments d'une émission AVANT le clustering, plutôt
# que de filtrer après, donne un résultat pire. Le vecteur moyen d'une émission
# de trois heures est générique, toutes les émissions longues se ressemblent
# alors entre elles, et le thème dominant absorbait 50 à 67 % de la journée
# contre 8 à 19 % ici. Un plateau traite réellement plusieurs sujets : le
# résumer à un point est faux.
MIN_UNITES_CLUSTER = 3

# Fenetre de repli pour les supports qui n'atteignent jamais ce seuil en une
# journee. Mesure : YouTube publie 1 a 3 videos par jour, donc jamais assez
# pour former des sujets -- mais 15 videos distinctes sur 7 jours glissants.
# On regroupe alors sur la fenetre elargie et l'on n'en retient que les
# documents du jour : le support reste present dans les thematiques, et la
# comparaison d'une journee a l'autre reste possible.
FENETRE_SUPPORT_MAIGRE = 7


# HDBSCAN choisit par défaut les clusters les plus « stables » de son arbre
# (excess of mass), ce qui laisse systématiquement un cluster fourre-tout : sur
# 557 articles de presse, un seul en absorbait 400. La sélection par feuilles
# prend le bas de l'arbre -- beaucoup de clusters fins plutôt qu'un gros et des
# miettes. C'est le seul réglage qui attaque directement le fourre-tout ; la
# taille minimale ne fait que déplacer le problème.
CLUSTER_SELECTION = "leaf"

# Similarite minimale pour rapprocher un sujet du jour d'un theme deja connu.
# Mesure sur les 32 themes en base : entre themes DIFFERENTS, la similarite des
# centroides va de 0,810 a 0,981, mediane 0,889 -- e5 tasse tout dans le haut
# de l'echelle. L'ancien seuil de 0,62, herite de MiniLM, etait franchi par
# 100 % des paires : chaque sujet du jour se collait sur une cle arbitraire,
# d'ou des journees entieres a « 0 nouveaux » et des libelles sans rapport avec
# leur contenu. A 0,95 il ne reste que 5 % des paires non apparentees.
REGISTRY_THRESHOLD = 0.95

# Un theme repris garde son libelle tant que son vocabulaire reste proche. En
# dessous de ce recouvrement entre anciens et nouveaux mots-cles, le sujet a
# assez bouge pour meriter un nouveau libelle : sans cela « Fronts et combats
# locaux » finissait par designer le Jour du drapeau russe.
RECOUVREMENT_MIN = 0.4

# Deux clusterings sont produits pour chaque journee et cohabitent en base,
# distingues par la colonne « portee » :
#
#   portee = 'global'                  un seul regroupement, tous supports
#                                      melanges dans le meme espace
#   portee = 'press' | 'tv' | ...      un regroupement propre a ce support
#
# Ils repondent a deux questions differentes. Le global dit de quoi parle le
# corpus ; le clustering par support dit de quoi parle CHAQUE media, sans que
# le vocabulaire d'un autre vienne peser sur ses frontieres. Les comparer est
# l'interet du dispositif, d'ou leur conservation en parallele.
PORTEE_GLOBALE = "global"

SCHEMA_SUPPORTS = """
CREATE TABLE IF NOT EXISTS topic_supports (
    topic_key   INTEGER,
    run_date    DATE,
    source_kind VARCHAR,
    n_docs      INTEGER,
    top_words   VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_topic_supports ON topic_supports(run_date, topic_key);
ALTER TABLE topics ADD COLUMN IF NOT EXISTS portee VARCHAR;
ALTER TABLE article_topics ADD COLUMN IF NOT EXISTS portee VARCHAR;
CREATE INDEX IF NOT EXISTS idx_article_topics_portee
    ON article_topics(run_date, portee);
"""


def _ensure_schema_jour(conn, reset=False):
    ensure_schema(conn, reset=reset)
    # article_id etait PRIMARY KEY, ce qui interdisait qu'un article ait un
    # theme dans chacune des deux portees. La cle devient (article_id, portee).
    _cols = {r[0] for r in conn.execute(
        "SELECT column_name FROM duckdb_columns() "
        "WHERE table_name = 'article_topics'").fetchall()}
    if "portee" not in _cols:
        conn.execute("""
            CREATE TABLE _at_neuf (
                article_id  VARCHAR,
                topic_key   INTEGER,
                probability FLOAT,
                run_date    DATE,
                portee      VARCHAR,
                PRIMARY KEY (article_id, portee)
            )""")
        conn.execute("INSERT INTO _at_neuf SELECT article_id, topic_key, "
                     "probability, run_date, ? FROM article_topics",
                     [PORTEE_GLOBALE])
        conn.execute("DROP TABLE article_topics")
        conn.execute("ALTER TABLE _at_neuf RENAME TO article_topics")
        log.info("article_topics : cle portee a (article_id, portee).")
    if reset:
        conn.execute("DROP TABLE IF EXISTS topic_supports")
        # ensure_schema a supprime la sequence des cles : le prochain thema
        # cree reprendra a 1. Les evaluations ProxAnn deja stockees pointent
        # donc vers des cles qui existent toujours mais qui designent
        # desormais de tout autres clusters -- « Prix et carburants russes »
        # se retrouvait decrit comme le groupe AKHMAT. Une simple chasse aux
        # orphelines ne voit rien : les cles sont valides, c'est leur sens qui
        # a change. On repart donc de zero, quitte a relancer validate_topics.
        conn.execute("DROP TABLE IF EXISTS topic_quality")
    for stmt in SCHEMA_SUPPORTS.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt + ";")


def _charger(conn, jour):
    """Documents publiés dans la journée, par support."""
    rows = conn.execute(
        f"""
        SELECT id, content, title, source_kind,
               {SQL_PARENT} AS parent
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


def _charger_fenetre(conn, jour, support, jours):
    """Documents d'un support sur une fenetre de N jours finissant ce jour."""
    return conn.execute(
        f"""
        SELECT id, content, title, source_kind,
               {SQL_PARENT} AS parent
        FROM articles
        WHERE content IS NOT NULL AND language = 'ru' AND source_kind = ?
          AND LENGTH(content) >= (CASE WHEN source_kind = 'telegram' THEN ? ELSE ? END)
          AND CAST(published_at AS DATE) BETWEEN ? AND ?
        """,
        [support, MIN_CONTENT_LEN_TELEGRAM, MIN_CONTENT_LEN,
         jour - timedelta(days=jours - 1), jour]).fetchall()


def _unites(rows, indices):
    """Nombre d'unites parentes distinctes parmi ces documents."""
    return len({rows[i][4] for i in indices})


def _filtrer_une_seule_unite(clusters, rows):
    """Ecarte les clusters qui ne recouvrent qu'une poignee d'emissions.

    Un cluster fait de quatre-vingts segments d'une meme emission n'est pas un
    sujet : c'est une emission. Sans ce filtre, la television fabrique chaque
    jour des themes qui ne sont que le decoupage d'un plateau, et ils pesaient
    d'autant plus lourd qu'ils comptaient un document par segment.
    """
    gardes = [c for c in clusters if _unites(rows, c["membres"]) >= MIN_UNITES_CLUSTER]
    ecartes = len(clusters) - len(gardes)
    if ecartes:
        log.info("    %d cluster(s) ecarte(s) : moins de %d unités distinctes",
                 ecartes, MIN_UNITES_CLUSTER)
    return gardes


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
    # UMAP initialise par décomposition spectrale, qui demande à scipy plus de
    # vecteurs propres qu'il n'y a de points dès que le jeu est petit, et lève
    # « Cannot use scipy.linalg.eigh for sparse A with k >= N ». Depuis le
    # regroupement par émission, la télévision est passée de 475 segments à
    # 17 unités par jour et tombait exactement dans ce cas. Sous ce seuil on
    # réduit la projection et l'on initialise au hasard, ce qu'UMAP prévoit
    # pour les petits jeux -- la structure y est de toute façon lisible sans
    # amorçage spectral.
    petit = n <= 50
    composantes = max(2, min(5, n - 2)) if petit else 5

    modele = BERTopic(
        umap_model=UMAP(n_neighbors=voisins, n_components=composantes,
                        min_dist=0.0, metric="cosine", random_state=42,
                        init="random" if petit else "spectral"),
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


def _recouvrement(mots_a, mots_b, n=10):
    """Part des n premiers mots-cles communs aux deux versions d'un theme."""
    a = {m.strip() for m in (mots_a or "").split(",")[:n] if m.strip()}
    b = {m.strip() for m in (mots_b or "").split(",")[:n] if m.strip()}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rattacher_isoles(sujets, embeddings, deja_classes, tous_indices):
    """Les documents laissés de côté rejoignent le sujet le plus proche, si
    leur proximité vaut celle du quart le plus faible des documents déjà
    classés. Un seuil absolu ne tiendrait pas : chaque modèle d'embedding a sa
    propre échelle de similarité."""
    isoles = [i for i in tous_indices if i not in deja_classes]
    if not isoles or not sujets:
        return {}, isoles

    cent = np.array([s["centroide"] for s in sujets])
    propres = _cosine_sim_matrix(embeddings[sorted(deja_classes)], cent).max(axis=1)
    seuil = float(np.percentile(propres, 25)) if len(propres) else 1.0

    sims = _cosine_sim_matrix(embeddings[isoles], cent)
    meilleurs, scores = sims.argmax(axis=1), sims.max(axis=1)
    affecte, restants = {}, []
    for k, i in enumerate(isoles):
        if scores[k] >= seuil:
            affecte[i] = (int(meilleurs[k]), float(scores[k]))
        else:
            restants.append(i)
    return affecte, restants


def _sujets_depuis(clusters):
    """Un cluster = un sujet. Aucun recoupement : chaque portee vit seule."""
    sujets = []
    for c in clusters:
        sujets.append({"membres": c["membres"], "centroide": c["centroide"],
                       "mots": c["mots"], "support": c["support"]})
    return sujets


def _enregistrer(conn, jour, portee, sujets, rows, embeddings, seuil_registre,
                 indices_portee):
    """Apparie les sujets aux thèmes connus de CETTE portée, puis écrit.

    Le registre est cloisonné : un thème du clustering global ne peut pas
    servir d'identité à un thème du clustering Telegram, sinon les deux
    dispositifs se contamineraient et la comparaison n'aurait plus de sens.
    """
    ids = [r[0] for r in rows]
    titres = [r[2] or "" for r in rows]
    contenus = [r[1] or "" for r in rows]

    deja = {i for s in sujets for i in s["membres"]}
    affecte, non_classes = _rattacher_isoles(sujets, embeddings, deja,
                                             list(indices_portee))

    registre = conn.execute(
        "SELECT topic_key, centroid, label FROM topics "
        "WHERE topic_key != ? AND portee = ? AND centroid IS NOT NULL "
        "AND len(centroid) > 0", [NOISE_KEY, portee]).fetchall()
    appariement, _ = _match_to_registry(
        [s["centroide"] for s in sujets], [s["mots"][:60] for s in sujets],
        [(r[0], r[1], r[2]) for r in registre], seuil_registre)

    cles, n_repris, n_neufs = [], 0, 0
    for idx, s in enumerate(sujets):
        mots = ", ".join(dict.fromkeys(s["mots"].replace(" | ", ", ").split(", ")))[:400]
        # Toute la liste des membres : _generate_readable_label y prend un
        # echantillon reparti. Tronquer ici lui donnerait le haut du groupe,
        # souvent une meme emission decoupee en segments.
        exemples = [titres[i] for i in s["membres"]]
        morceaux = [contenus[i] for i in s["membres"]]
        if idx in appariement:
            cle, _sim = appariement[idx]
            anciens = conn.execute(
                "SELECT top_words FROM topics WHERE topic_key = ?",
                [cle]).fetchone()[0] or ""
            libelle = None
            if _recouvrement(anciens, mots) < RECOUVREMENT_MIN:
                libelle = _generate_readable_label(
                    mots, exemples, fallback=" / ".join(mots.split(", ")[:3]),
                    extraits=morceaux, n_articles=len(s["membres"]))
            conn.execute(
                "UPDATE topics SET top_words = ?, centroid = ?, last_seen = ?, "
                "active = TRUE, label = COALESCE(?, label) WHERE topic_key = ?",
                [mots, list(map(float, s["centroide"])), jour, libelle, cle])
            n_repris += 1
        else:
            libelle = _generate_readable_label(
                mots, exemples, fallback=" / ".join(mots.split(", ")[:3]),
                extraits=morceaux, n_articles=len(s["membres"]))
            cle = conn.execute("SELECT nextval('topic_key_seq')").fetchone()[0]
            conn.execute(
                "INSERT INTO topics (topic_key, label, top_words, centroid, "
                "first_seen, last_seen, active, portee) "
                "VALUES (?, ?, ?, ?, ?, ?, TRUE, ?)",
                [cle, libelle, mots, list(map(float, s["centroide"])), jour,
                 jour, portee])
            n_neufs += 1
        cles.append(cle)

    lignes = []

    def poser(i, cle, score):
        lignes.append((ids[i], cle, score, jour, portee))

    for idx, s in enumerate(sujets):
        for i in s["membres"]:
            poser(i, cles[idx], 1.0)
    for i, (idx, score) in affecte.items():
        poser(i, cles[idx], score)
    for i in non_classes:
        poser(i, NOISE_KEY, 0.0)
    if lignes:
        conn.executemany(
            "INSERT INTO article_topics (article_id, topic_key, probability, "
            "run_date, portee) VALUES (?, ?, ?, ?, ?)", lignes)
    return cles, n_repris, n_neufs, len(non_classes)


def traiter_jour(conn, jour, seuil_registre, embed):
    def embed_fenetre(textes):
        return embed.encode(textes, show_progress_bar=False,
                            normalize_embeddings=True)

    rows, par_support = _charger(conn, jour)
    if not rows:
        log.warning("%s : aucun document éligible, journée ignorée.", jour)
        return
    docs = [r[1] for r in rows]
    log.info("%s : %d documents pour %d unités (%s)", jour, len(rows),
             len({r[4] for r in rows}),
             ", ".join(f"{k} {_unites(rows, v)}" for k, v in
                       sorted(par_support.items(), key=lambda x: -len(x[1]))))

    # Les marqueurs d'oral sont retirés du texte encodé ; docs reste intact
    # pour le c-TF-IDF et les titres d'exemple. Un seul encodage sert aux deux
    # clusterings.
    embeddings = np.asarray(
        embed.encode([EMBED_PREFIXE + _neutraliser_oral(d) for d in docs],
                     show_progress_bar=False, normalize_embeddings=True),
        dtype=float)

    ids = [r[0] for r in rows]
    conn.execute("CREATE OR REPLACE TEMP TABLE _ids_jour AS SELECT UNNEST(?) AS id",
                 [ids])
    conn.execute("DELETE FROM article_topics WHERE article_id IN (SELECT id FROM _ids_jour)")
    conn.execute("DELETE FROM topic_supports WHERE run_date = ?", [jour])

    # --- 1. Un clustering par type de média -------------------------------
    for support in SUPPORTS:
        indices = par_support.get(support, [])
        if not indices:
            continue
        if _unites(rows, indices) >= MIN_DOCS_SUPPORT:
            clusters = _filtrer_une_seule_unite(
                _clusteriser_support(support, indices, embeddings, docs), rows)
            if not clusters:
                continue
            _, rep, neuf, nc = _enregistrer(conn, jour, support,
                                            _sujets_depuis(clusters), rows,
                                            embeddings, seuil_registre, indices)
            log.info("  [%-8s] %2d thèmes (%d repris, %d nouveaux), "
                     "%d non classés", support, len(clusters), rep, neuf, nc)
            continue

        # Trop maigre sur 24 h : on elargit la fenetre pour ce seul support.
        f_seg = _charger_fenetre(conn, jour, support, FENETRE_SUPPORT_MAIGRE)
        f_unites = len({r[4] for r in f_seg})
        if f_unites < MIN_DOCS_SUPPORT:
            log.info("  [%-8s] %5d documents / %d unités sur %d jours -- "
                     "toujours sous le seuil, non regroupé", support,
                     len(f_seg), f_unites, FENETRE_SUPPORT_MAIGRE)
            conn.executemany(
                "INSERT INTO article_topics (article_id, topic_key, "
                "probability, run_date, portee) VALUES (?, ?, ?, ?, ?)",
                [(ids[i], NOISE_KEY, 0.0, jour, support) for i in indices])
            continue

        f_rows = f_seg
        f_docs = [r[1] for r in f_rows]
        f_emb = np.asarray(
            embed_fenetre([EMBED_PREFIXE + _neutraliser_oral(d) for d in f_docs]),
            dtype=float)
        clusters = _filtrer_une_seule_unite(
            _clusteriser_support(support, list(range(len(f_rows))), f_emb,
                                 f_docs), f_rows)
        if not clusters:
            continue
        # Seuls les documents du jour sont enregistres : le regroupement se
        # calcule sur la fenetre, la restitution reste quotidienne.
        du_jour = {i for i, r in enumerate(f_rows)
                   if r[0] in {ids[k] for k in indices}}
        for c in clusters:
            c["membres"] = [i for i in c["membres"] if i in du_jour]
        clusters = [c for c in clusters if c["membres"]]
        if not clusters:
            continue
        _, rep, neuf, nc = _enregistrer(conn, jour, support,
                                        _sujets_depuis(clusters), f_rows,
                                        f_emb, seuil_registre, sorted(du_jour))
        log.info("  [%-8s] %2d thèmes sur fenêtre %dj (%d repris, %d nouveaux), "
                 "%d non classés", support, len(clusters),
                 FENETRE_SUPPORT_MAIGRE, rep, neuf, nc)

    # --- 2. Un clustering global, tous supports mêlés ---------------------
    tous = list(range(len(rows)))
    clusters_g = _filtrer_une_seule_unite(
        _clusteriser_support(PORTEE_GLOBALE, tous, embeddings, docs), rows)
    if clusters_g:
        sujets_g = _sujets_depuis(clusters_g)
        cles_g, rep, neuf, nc = _enregistrer(
            conn, jour, PORTEE_GLOBALE, sujets_g, rows, embeddings,
            seuil_registre, tous)
        log.info("  [%-8s] %2d thèmes (%d repris, %d nouveaux), %d non classés",
                 PORTEE_GLOBALE, len(sujets_g), rep, neuf, nc)

        # Composition par support de chaque theme global : c'est elle qui
        # permet de voir quel media alimente quoi.
        ventilation = []
        for idx, sj in enumerate(sujets_g):
            par_kind = {}
            for i in sj["membres"]:
                par_kind[rows[i][3] or "press"] = par_kind.get(rows[i][3] or "press", 0) + 1
            for kind, n in par_kind.items():
                ventilation.append((cles_g[idx], jour, kind, n, sj["mots"][:400]))
        if ventilation:
            conn.executemany(
                "INSERT INTO topic_supports (topic_key, run_date, source_kind, "
                "n_docs, top_words) VALUES (?, ?, ?, ?, ?)", ventilation)

    _assurer_ligne_non_classe(conn, jour)


def _assurer_ligne_non_classe(conn, jour):
    if conn.execute("SELECT 1 FROM topics WHERE topic_key = ?", [NOISE_KEY]).fetchone():
        conn.execute("UPDATE topics SET last_seen = ? WHERE topic_key = ?", [jour, NOISE_KEY])
    else:
        conn.execute(
            "INSERT INTO topics (topic_key, label, top_words, centroid, "
            "first_seen, last_seen, active) VALUES (?, 'Non classé', '', [], ?, ?, TRUE)",
            [NOISE_KEY, jour, jour])


def _jours_manquants(conn, fin, profondeur):
    """Journées ayant des articles mais aucun regroupement.

    Une passe interrompue laissait un trou definitif : --backfill 1 ne remonte
    que d'un jour, et rien ne revenait ensuite sur la journee sautee. Le 25/08
    a ainsi disparu des thematiques alors que ses 1776 articles etaient bien
    collectes.
    """
    debut = fin - timedelta(days=profondeur - 1)
    return [r[0] for r in conn.execute("""
        SELECT DISTINCT CAST(a.published_at AS DATE) AS j
        FROM articles a
        WHERE CAST(a.published_at AS DATE) BETWEEN ? AND ?
          AND NOT EXISTS (SELECT 1 FROM article_topics t
                          WHERE t.run_date = CAST(a.published_at AS DATE))
        ORDER BY 1""", [debut, fin]).fetchall()]


def run(jour=None, backfill=0, seuil_registre=REGISTRY_THRESHOLD,
        reset=False, combler=7):
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
    if combler and not reset:
        oublies = [j for j in _jours_manquants(conn, fin, combler)
                   if j not in jours]
        if oublies:
            log.info("Journées sans regroupement à rattraper : %s",
                     ", ".join(str(j) for j in oublies))
            jours = sorted(set(jours) | set(oublies))

    log.info("Chargement du modèle d'embedding (%s)...", EMBED_MODEL)
    embed = SentenceTransformer(EMBED_MODEL)
    embed.max_seq_length = EMBED_MAX_TOKENS

    for j in jours:
        traiter_jour(conn, j, seuil_registre, embed)

    # Un thème sans document sur les 7 derniers jours passe en sommeil : il
    # garde sa clé et son historique, et se réactive si son sujet revient.
    conn.execute("""
        UPDATE topics SET active = (last_seen >= ?) WHERE topic_key != ?""",
        [fin - timedelta(days=7), NOISE_KEY])
    conn.execute(
        "UPDATE topics SET article_count = (SELECT COUNT(DISTINCT "
        f"{SQL_PARENT}) FROM article_topics at_ JOIN articles a "
        "ON a.id = at_.article_id WHERE at_.topic_key = topics.topic_key)")

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
    p.add_argument("--threshold", type=float, default=REGISTRY_THRESHOLD,
                   help="similarité minimale pour retrouver un thème connu")
    p.add_argument("--combler", type=int, default=7,
                   help="rattrape les journées des N derniers jours qui ont "
                        "des articles mais aucun regroupement ; 0 désactive")
    p.add_argument("--reset", action="store_true",
                   help="vide topics/article_topics/topic_supports")
    a = p.parse_args()
    sys.exit(run(
        jour=datetime.strptime(a.date, "%Y-%m-%d").date() if a.date else None,
        backfill=a.backfill, seuil_registre=a.threshold, combler=a.combler,
        reset=a.reset))
