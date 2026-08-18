"""Themes suivis = clusters BERTopic sur une fenetre glissante (pas une
taxonomie ecrite a la main). Detecte les themes directement depuis les
articles : un theme sans article recent redevient inactif (sans perdre son
historique), un theme qui reapparait est reactive sous la meme identite si
son cluster ressemble assez a l'ancien (similarite cosinus des centroides).

Granularite : c'est min_topic_size (taille minimale d'un cluster) qui la
fixe ; max_topics n'est qu'un garde-fou contre l'emballement (fusion des
clusters les plus proches au-dela). Regler la granularite par max_topics
produit des fourre-tout : la fusion soude alors des sujets distincts.

Usage :
  python scripts/analysis/analyze_topics.py                  # fenetre 30j, <= 90 themes
  python scripts/analysis/analyze_topics.py --days 14
  python scripts/analysis/analyze_topics.py --max-topics 25 --reset
"""
import argparse
import re
import time
from datetime import date, timedelta

import numpy as np

from src.db import get_conn
from src.logging_setup import setup_logging

log = setup_logging("topics")

MIN_CONTENT_LEN = 300
# Les posts Telegram sont naturellement courts -- le seuil presse les
# excluait presque tous (le scraper ecarte deja tout ce qui fait moins de
# 50 car. a la collecte, cf. src/telegram_scrape.py).
MIN_CONTENT_LEN_TELEGRAM = 50
# Mesure faite sur 900 documents : avec l'ancien MiniLM, deux transcriptions
# quelconques se ressemblaient a 0,77 quand deux articles de presse ne se
# ressemblaient qu'a 0,45 -- la video formait un bloc dense et HDBSCAN y
# decoupait des themes par REGISTRE avant de le faire par sujet. e5-base divise
# cet ecart par cinq. Il ne devient abordable que sur GPU (2,5 min contre 33 en
# processeur pour 8 800 documents).
EMBED_MODEL = "intfloat/multilingual-e5-base"
# Les modeles e5 attendent un prefixe indiquant le role du texte.
EMBED_PREFIXE = "passage: "
# Longueur reellement encodee. Le defaut du modele (128) coupait la quasi
# totalite du corpus -- cf. le commentaire dans run().
EMBED_MAX_TOKENS = 512
NOISE_KEY = -1
# Similarite cosinus minimale pour rattacher un article laisse de cote par
# HDBSCAN au cluster le plus proche (cf. run()). Prudent par defaut : mieux
# vaut laisser un article non classe que le forcer dans un theme voisin, ce
# qui redilue exactement la precision qu'on cherche.
OUTLIER_THRESHOLD = 0.60
# Particules d'oral. Le probleme n'etait pas le volume de segments par video
# mais leur REGISTRE : l'analyse de divergence a montre que le vocabulaire
# distinctif de YouTube est « тип, короче, вообще, наверное, мол, угу » et
# celui de la television « действительно, собственно, значит, кстати » -- des
# marqueurs de parole, pas des sujets. L'embedding les capte et regroupe toutes
# les transcriptions ensemble quel que soit leur sujet. On les retire du texte
# envoye a l'embedding (le contenu stocke, lui, n'est jamais modifie).
_ORAL = (
    "вот", "ну", "ага", "угу", "короче", "типа", "тип", "мол", "как бы",
    "в общем", "собственно", "значит", "наверное", "кстати", "реально",
    "вообще", "как-то", "что-то", "какой-то", "какая-то", "какое-то",
    "прямо", "слушайте", "понимаете", "знаете", "так сказать", "это самое",
    "да ладно", "действительно", "конечно", "просто",
)
_ORAL_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ORAL, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


def _neutraliser_oral(texte):
    """Retire les particules de parole d'une transcription."""
    return _ORAL_RE.sub(" ", texte)

# --- Lemmatisation + stopwords pour les mots-cles de theme (c-TF-IDF) -----
#
# L'embedding (SentenceTransformer) tourne sur le texte brut : un transformer
# comprend deja les declinaisons russes semantiquement. Mais le vectorizer
# qui produit les MOTS affiches par theme (c-TF-IDF, sac-de-mots) n'a aucune
# notion de grammaire : sans lemmatisation, "Яблоко"/"Яблока"/"Яблоку" sont
# trois tokens distincts et polluent le meme theme en faux doublons (constate
# en pratique : "яблока / партии / яблоко", "гилман / роберт / роберт
# гилман"). pymorphy3 ramene chaque token a sa forme canonique avant comptage.
_PRESS_STOPWORDS = {
    # Vocabulaire d'agence/de compte-rendu, pas capte par la liste generaliste
    # de nltk -- sans ca, ces mots dominent le c-TF-IDF de tous les themes.
    "сообщает", "сообщил", "сообщила", "сообщили", "заявил", "заявила",
    "отметил", "отметила", "подчеркнул", "подчеркнула", "добавил",
    "добавила", "передает", "передают", "рассказал", "рассказала",
    "говорится", "уточнил", "уточнила", "пишет", "цитирует",
    "риа", "тасс", "рбк", "интерфакс", "ria", "tass",
    "год", "года", "году", "лет", "млн", "млрд", "тыс",
    # La liste russe de nltk ne fait que 151 entrees et laisse passer des mots
    # tres frequents -- "это", "наш", "который", "очень" n'y sont pas. Ils
    # remontaient en tete des mots-cles de themes et des divergences.
    "это", "этот", "тот", "весь", "наш", "ваш", "свой", "который", "такой",
    "какой", "самый", "очень", "просто", "тоже", "также", "просто", "давать",
    "сказать", "говорить", "мочь", "стать", "делать", "хотеть", "знать",
    "думать", "видеть", "идти", "получать", "считать", "понимать", "являться",
    "человек", "время", "дело", "вопрос", "случай", "работа", "слово",
    # Artefacts de transcription : Whisper et les sous-titres YouTube posent
    # ces marqueurs a la place des passages non verbaux.
    "музыка", "аплодисменты", "смех", "аплодировать", "неразборчиво",
    # Formules de plateforme, signatures de chaine et abreviations de date
    # ramassees par l'extraction.
    "подписаться", "подписываться", "подписывайтесь", "telegram", "канал",
    "видео", "смотреть", "читать", "источник", "фото", "авг", "сен", "окт",
}
_TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ][a-zA-Zа-яёА-ЯЁ\-']{2,}")

_morph = None
_stopwords = None
_lemma_cache: dict = {}


def _load_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("russian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        words = set(stopwords.words("russian"))
    return words | _PRESS_STOPWORDS


def _lemmatize(word):
    lemma = _lemma_cache.get(word)
    if lemma is None:
        lemma = _morph.parse(word)[0].normal_form.replace("ё", "е")
        _lemma_cache[word] = lemma
    return lemma


def _lemmatizing_tokenizer(text):
    """Tokenizer pour CountVectorizer : lemmatise chaque mot et filtre les
    stopwords sur la forme brute ET sur le lemme (les pronoms/particules
    irreguliers du russe ne se ramenent pas tous a une forme unique)."""
    global _morph, _stopwords
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    if _stopwords is None:
        _stopwords = _load_stopwords()

    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        # e/e trema : nltk ecrit "все" et "еще", les textes ecrivent souvent
        # "всё" et "ещё". Sans cette normalisation les mots vides passaient au
        # travers du filtre et arrivaient en tete des mots-cles.
        raw = raw.replace("ё", "е")
        if raw in _stopwords:
            continue
        lemma = _lemmatize(raw)
        if lemma in _stopwords:
            continue
        tokens.append(lemma)
    return tokens


# --- Label lisible par cluster (un appel Mistral par cluster, pas par --
# article : ~50-60 appels/run plutot que ~1000, cout negligeable). Les mots
# c-TF-IDF ("всу / нпз / бпла") restent dans top_words pour le detail ; le
# label affiche en priorite devient une phrase courte lisible.
_LABEL_SYSTEM_PROMPT = """Vous nommez des clusters d'articles de presse russophone (Russie) par un label court en francais, comme un titre de rubrique.

Regles :
- 2 a 5 mots, Format Titre
- decrit le sujet concret du cluster (pas une categorie generique comme "Actualites")
- basé sur les mots-cles ET les exemples de titres fournis

Repondez en JSON : {"label": "..."}"""


def _generate_readable_label(keywords_str, example_titles, fallback):
    from src.llm_mistral import complete_json, MODEL_SMALL

    titles_block = "\n".join(f"- {t}" for t in example_titles if t) or "(aucun titre disponible)"
    data = complete_json(
        _LABEL_SYSTEM_PROMPT,
        f"Mots-cles : {keywords_str}\n\nExemples de titres d'articles du cluster :\n{titles_block}",
        model=MODEL_SMALL, max_tokens=40,
    )
    if not data:
        return fallback
    label = (data.get("label") or "").strip().strip(".")
    return label[:150] if label else fallback


SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS topic_key_seq START 1;
CREATE TABLE IF NOT EXISTS topics (
    topic_key     INTEGER PRIMARY KEY,
    label         VARCHAR,
    top_words     VARCHAR,
    centroid      DOUBLE[],
    article_count INTEGER DEFAULT 0,
    first_seen    DATE,
    last_seen     DATE,
    active        BOOLEAN DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS article_topics (
    article_id  VARCHAR PRIMARY KEY,
    topic_key   INTEGER,
    probability FLOAT,
    run_date    DATE
);
CREATE INDEX IF NOT EXISTS idx_article_topics_topic ON article_topics(topic_key);
"""


def ensure_schema(conn, reset=False):
    if reset:
        # Utile quand on change la granularite du clustering (min_topic_size /
        # nr_topics) : les anciens themes ne sont plus comparables aux
        # nouveaux, les garder laisserait des dizaines de clusters orphelins
        # en sommeil. Ne touche QUE les tables derivees -- les articles
        # eux-memes ne sont jamais supprimes, tout est reconstructible.
        conn.execute("DROP TABLE IF EXISTS article_topics")
        conn.execute("DROP TABLE IF EXISTS topics")
        conn.execute("DROP SEQUENCE IF EXISTS topic_key_seq")
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt + ";")


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def _match_to_registry(new_centroids, new_labels, registry, threshold):
    """Greedy matching by descending cosine similarity.

    registry: list of (topic_key, centroid, label). Returns
    (assignment: {new_index: (topic_key, is_new)}, unmatched_registry_keys).
    """
    if not registry:
        return {}, set()
    if len(new_labels) == 0:
        return {}, {r[0] for r in registry}

    old_keys = [r[0] for r in registry]
    old_centroids = np.array([r[1] for r in registry])
    sims = _cosine_sim_matrix(np.array(new_centroids), old_centroids)

    pairs = [
        (sims[i, j], i, j)
        for i in range(sims.shape[0])
        for j in range(sims.shape[1])
    ]
    pairs.sort(key=lambda x: -x[0])

    assignment = {}
    used_old = set()
    for sim, i, j in pairs:
        if sim < threshold:
            break
        if i in assignment or old_keys[j] in used_old:
            continue
        assignment[i] = (old_keys[j], float(sim))
        used_old.add(old_keys[j])

    unmatched_registry = {k for k in old_keys if k not in used_old}
    return assignment, unmatched_registry


def run(window_days: int = 30, min_topic_size: int = 15, threshold: float = 0.62,
        max_topics: int = 90, reset: bool = False,
        outlier_threshold: float = OUTLIER_THRESHOLD):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    conn = get_conn()
    ensure_schema(conn, reset=reset)

    window_start = date.today() - timedelta(days=window_days)
    # L'unite parente se relit dans l'URL : tous les segments d'une meme video
    # ou d'une meme emission la partagent.
    rows = conn.execute(
        """
        SELECT id, content, title,
               CASE WHEN source_kind = 'youtube'
                    THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1)
                    WHEN source_kind = 'tv'
                    THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1)
                    ELSE id END AS parent
        FROM articles
        WHERE content IS NOT NULL
          AND LENGTH(content) >= (CASE WHEN source_kind = 'telegram' THEN ? ELSE ? END)
          AND language = 'ru' AND published_at >= ?
        """,
        [MIN_CONTENT_LEN_TELEGRAM, MIN_CONTENT_LEN, window_start],
    ).fetchall()
    log.info("Fenetre %dj (depuis %s) : %d articles a regrouper",
              window_days, window_start, len(rows))
    if len(rows) < 50:
        log.error("Pas assez d'articles dans la fenetre (< 50).")
        conn.close()
        return

    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    titles = [r[2] or "" for r in rows]

    n_video = sum(1 for r in rows if r[3] != r[0])
    log.info("Dont %d segments de video ou d'emission, dont on neutralise "
             "les marqueurs d'oral avant l'embedding", n_video)

    log.info("Chargement embedding model...")
    embed = SentenceTransformer(EMBED_MODEL)
    # Le defaut du modele est 128 tokens, ce qui tronquait 92 % des articles :
    # le clustering ne voyait que le chapeau, et les chapeaux d'agence se
    # ressemblent tous. D'ou des clusters fourre-tout. Encoder 512 tokens change
    # vraiment les vecteurs (0,55 de similarite avec les tronques) et coute
    # ~11 min au lieu de 4 sur ce corpus.
    embed.max_seq_length = EMBED_MAX_TOKENS
    # Le texte encode est nettoye de ses particules d'oral ; `docs` reste
    # intact pour le c-TF-IDF et les exemples affiches.
    embeddings = embed.encode(
        [EMBED_PREFIXE + _neutraliser_oral(d) for d in docs],
        show_progress_bar=True, normalize_embeddings=True)

    vectorizer = CountVectorizer(
        tokenizer=_lemmatizing_tokenizer,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
    )

    # random_state fixe, sinon deux runs sur la meme fenetre deplacent les
    # centroides et cassent la continuite d'identite des themes.
    # n_neighbors bas = structure plus locale, donc themes plus fins ; a 15 les
    # frappes, les negociations et les livraisons d'armes se confondaient.
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                       metric="cosine", random_state=42)

    model = BERTopic(
        embedding_model=embed,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        # Garde-fou contre l'emballement (Telegram produisait ~200
        # micro-themes), pas un reglage de granularite : a 40, la fusion
        # soudait des sujets distincts en un fourre-tout.
        nr_topics=max_topics,
        verbose=True,
        language="multilingual",
        calculate_probabilities=False,
    )

    log.info("Clustering BERTopic...")
    started = time.time()
    bertopic_ids, probs = model.fit_transform(docs, embeddings=embeddings)
    log.info("Clustering termine en %.1fs", time.time() - started)

    # HDBSCAN ecarte tout ce qui n'est pas au coeur d'une zone dense : 45 % du
    # corpus ici. On rattache ceux qui sont assez proches d'un cluster, les
    # autres restent non classes.
    idx_isoles = [i for i, t in enumerate(bertopic_ids) if t == NOISE_KEY]
    if idx_isoles:
        # La distribution dit ou poser le seuil : un essai a 0,30 avait
        # rattache 4013 isoles sur 4017 et redilue les themes.
        rangs = list(model.get_topic_info()["Topic"])
        vrais = [j for j, t in enumerate(rangs) if t != NOISE_KEY]
        proches = _cosine_sim_matrix(
            np.asarray(embeddings)[idx_isoles],
            np.asarray(model.topic_embeddings_)[vrais]).max(axis=1)
        log.info("Isoles (%d) : proximite au cluster le plus proche, "
                 "deciles 10/25/50/75/90 = %s", len(idx_isoles),
                 " / ".join(f"{v:.2f}" for v in
                            np.percentile(proches, [10, 25, 50, 75, 90])))

        # Le seuil se lit sur les documents DEJA classes plutot que d'etre fixe
        # en dur : chaque modele d'embedding a sa propre echelle de similarite
        # (MiniLM s'etale de 0,3 a 1,0, e5 se tasse entre 0,75 et 0,90), et un
        # seuil absolu rattachait tout avec l'un et rien avec l'autre. Regle
        # retenue : un isole rejoint un theme s'il en est au moins aussi proche
        # que le quart le plus faible de ses membres actuels.
        idx_classes = [i for i, t in enumerate(bertopic_ids) if t != NOISE_KEY]
        if idx_classes:
            rang_par_topic = {t: j for j, t in enumerate(
                [rangs[j] for j in vrais])}
            sims_classes = _cosine_sim_matrix(
                np.asarray(embeddings)[idx_classes],
                np.asarray(model.topic_embeddings_)[vrais])
            propres = [sims_classes[n, rang_par_topic[bertopic_ids[i]]]
                       for n, i in enumerate(idx_classes)
                       if bertopic_ids[i] in rang_par_topic]
            if propres:
                outlier_threshold = float(np.percentile(propres, 25))
                log.info("Seuil de rattachement deduit des documents classes : "
                         "%.2f (1er quartile de leur proximite a leur propre "
                         "theme)", outlier_threshold)

    avant = len(idx_isoles)
    bertopic_ids = model.reduce_outliers(
        docs, bertopic_ids, strategy="embeddings", embeddings=embeddings,
        threshold=outlier_threshold)
    apres = sum(1 for t in bertopic_ids if t == NOISE_KEY)
    log.info("Non classes : %d -> %d (%.1f %% du corpus, %d rattaches, "
             "seuil %.2f)", avant, apres, 100 * apres / max(len(docs), 1),
             avant - apres, outlier_threshold)

    # Quelques titres d'exemple par cluster, pour donner du contexte au label
    # Mistral au-dela des seuls mots-cles.
    titles_by_bt: dict = {}
    for i, bt_id in enumerate(bertopic_ids):
        bt_id = int(bt_id)
        if bt_id == NOISE_KEY:
            continue
        bucket = titles_by_bt.setdefault(bt_id, [])
        if len(bucket) < 10:
            bucket.append(titles[i])

    topic_info = model.get_topic_info()
    new_labels, new_centroids, new_top_words, bt_id_by_index = [], [], [], []
    for idx, row in topic_info.iterrows():
        bt_id = int(row["Topic"])
        if bt_id == NOISE_KEY:
            continue
        words = model.get_topic(bt_id)
        top_words_str = ", ".join(w for w, _ in words[:15]) if words else ""
        fallback_label = " / ".join(w for w, _ in words[:3]) if words else f"Theme {bt_id}"
        new_labels.append(fallback_label)
        new_top_words.append(top_words_str)
        new_centroids.append(model.topic_embeddings_[idx])
        bt_id_by_index.append(bt_id)

    log.info("Generation des labels lisibles (%d clusters, 1 appel Mistral/cluster)...",
              len(bt_id_by_index))
    for i, bt_id in enumerate(bt_id_by_index):
        new_labels[i] = _generate_readable_label(
            new_top_words[i], titles_by_bt.get(bt_id, []), fallback=new_labels[i])
        if (i + 1) % 10 == 0:
            log.info("  %d/%d labels generes", i + 1, len(bt_id_by_index))

    today = date.today()

    # Registre COMPLET (actifs + endormis) pour le rapprochement par
    # similarite : un theme endormi doit pouvoir etre reactive, donc il ne
    # faut pas se limiter aux themes actifs ici.
    registry = conn.execute(
        "SELECT topic_key, centroid, label, active FROM topics WHERE topic_key != ?",
        [NOISE_KEY],
    ).fetchall()
    match_candidates = [(r[0], r[1], r[2]) for r in registry]
    active_before = {r[0] for r in registry if r[3]}

    assignment, _ = _match_to_registry(new_centroids, new_labels, match_candidates, threshold)

    # bt_id -> topic_key final (cree ou reutilise), pour le mapping article->theme.
    bt_to_key = {}
    n_reactivated = n_new = n_kept = 0
    matched_keys = set()

    for i, bt_id in enumerate(bt_id_by_index):
        label, top_words, centroid = new_labels[i], new_top_words[i], new_centroids[i]
        if i in assignment:
            topic_key, sim = assignment[i]
            matched_keys.add(topic_key)
            was_active = topic_key in active_before
            conn.execute(
                """UPDATE topics SET label = ?, top_words = ?, centroid = ?,
                   last_seen = ?, active = TRUE WHERE topic_key = ?""",
                [label, top_words, list(map(float, centroid)), today, topic_key],
            )
            bt_to_key[bt_id] = topic_key
            if was_active:
                n_kept += 1
            else:
                n_reactivated += 1
                log.info("  Theme reactive (sim %.2f) : %s", sim, label)
        else:
            topic_key = conn.execute("SELECT nextval('topic_key_seq')").fetchone()[0]
            conn.execute(
                """INSERT INTO topics
                   (topic_key, label, top_words, centroid, first_seen, last_seen, active)
                   VALUES (?, ?, ?, ?, ?, ?, TRUE)""",
                [topic_key, label, top_words, list(map(float, centroid)), today, today],
            )
            bt_to_key[bt_id] = topic_key
            n_new += 1
            log.info("  Nouveau theme : %s", label)

    # Themes actifs non retrouves dans cette fenetre -> mis en sommeil (pas
    # supprimes). Les themes deja endormis et toujours non matches restent
    # simplement endormis, pas besoin d'y toucher.
    newly_dormant = active_before - matched_keys
    for topic_key in newly_dormant:
        conn.execute("UPDATE topics SET active = FALSE WHERE topic_key = ?", [topic_key])

    # Cluster de bruit BERTopic (non classe) : ligne evergreen, jamais soumise
    # au rapprochement par similarite (ce n'est pas un vrai theme).
    noise_present = any(int(r["Topic"]) == NOISE_KEY for _, r in topic_info.iterrows())
    if noise_present:
        exists = conn.execute(
            "SELECT 1 FROM topics WHERE topic_key = ?", [NOISE_KEY]
        ).fetchone()
        if exists:
            conn.execute(
                "UPDATE topics SET last_seen = ?, active = TRUE WHERE topic_key = ?",
                [today, NOISE_KEY],
            )
        else:
            conn.execute(
                """INSERT INTO topics
                   (topic_key, label, top_words, centroid, first_seen, last_seen, active)
                   VALUES (?, 'Non classe', '', [], ?, ?, TRUE)""",
                [NOISE_KEY, today, today],
            )
        bt_to_key[NOISE_KEY] = NOISE_KEY

    # Reassigne les articles de la fenetre (supprime puis reinsere ; les
    # articles hors fenetre gardent leur affectation d'un run precedent).
    conn.execute("CREATE OR REPLACE TEMP TABLE _window_ids AS SELECT UNNEST(?) AS id", [ids])
    conn.execute("DELETE FROM article_topics WHERE article_id IN (SELECT id FROM _window_ids)")
    for art_id, bt_id, prob in zip(ids, bertopic_ids, probs):
        conn.execute(
            "INSERT INTO article_topics (article_id, topic_key, probability, run_date) "
            "VALUES (?, ?, ?, ?)",
            [art_id, bt_to_key.get(int(bt_id), NOISE_KEY),
             float(prob) if prob is not None else 0.0, today],
        )

    conn.execute(
        "UPDATE topics SET article_count = "
        "(SELECT COUNT(*) FROM article_topics WHERE article_topics.topic_key = topics.topic_key)"
    )

    log.info(
        "Termine : %d themes conserves, %d reactives, %d nouveaux, %d mis en sommeil",
        n_kept, n_reactivated, n_new, len(newly_dormant),
    )

    print("\n=== THEMES ACTIFS (tries par volume) ===")
    for tid, label, n, words, first, last in conn.execute(
        "SELECT topic_key, label, article_count, top_words, first_seen, last_seen "
        "FROM topics WHERE active AND topic_key != ? ORDER BY article_count DESC",
        [NOISE_KEY],
    ).fetchall():
        print(f"  #{tid:>3}  [{n:>4} articles]  {label}  (depuis {first})")
        print(f"        {words}")

    dormant = conn.execute(
        "SELECT label, last_seen FROM topics WHERE NOT active AND topic_key != ? "
        "ORDER BY last_seen DESC LIMIT 10",
        [NOISE_KEY],
    ).fetchall()
    if dormant:
        print("\n=== THEMES EN SOMMEIL RECEMMENT (top 10) ===")
        for label, last in dormant:
            print(f"  {label}  (vu pour la derniere fois le {last})")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30,
                         help="Largeur de la fenetre glissante en jours")
    parser.add_argument("--min-size", type=int, default=15,
                         help="Taille minimale d'un cluster (plus haut = moins "
                              "de micro-themes)")
    parser.add_argument("--rattachement", type=float, default=OUTLIER_THRESHOLD,
                        help="proximite minimale pour rattacher un article "
                             "isole au cluster le plus proche ; 1.0 desactive "
                             "le rattachement et laisse les isoles non classes")
    parser.add_argument("--max-topics", type=int, default=90,
                         help="Plafond du nombre de themes : les clusters les "
                              "plus proches sont fusionnes au-dela")
    parser.add_argument("--threshold", type=float, default=0.62,
                         help="Similarite cosinus minimale pour rapprocher un "
                              "cluster d'un theme existant")
    parser.add_argument("--reset", action="store_true",
                         help="Repart de zero (vide topics/article_topics) : a "
                              "utiliser quand on change --min-size/--max-topics, "
                              "les anciens themes n'etant plus comparables")
    args = parser.parse_args()
    run(window_days=args.days, min_topic_size=args.min_size,
        threshold=args.threshold, max_topics=args.max_topics, reset=args.reset,
        outlier_threshold=args.rattachement)
