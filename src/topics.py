"""Primitives de clustering thematique, partagees par les passes et les outils.

Ce module ne fait rien tout seul : il porte les constantes de selection des
documents, le modele d'embedding, la neutralisation des particules d'oral, le
tokenizer lemmatise du c-TF-IDF, le schema des tables et le rapprochement d'un
cluster avec le registre des themes deja connus.

La passe quotidienne est dans scripts/analysis/analyze_topics_daily.py.
"""
import re

import numpy as np

# Un segment de transcription porte l'identifiant de son emission dans son URL.
# Cette expression la reconstitue, ce qui permet de compter et de regrouper par
# emission plutot que par segment : une emission de trois heures produit
# jusqu'a 93 segments, qui pesaient chacun autant qu'un article de presse.
# Definie ici et nulle part ailleurs -- elle existait en trois exemplaires
# (dashboard et deux fois dans la passe quotidienne), avec le risque qu'une
# correction n'en atteigne qu'un.
SQL_PARENT = ("CASE WHEN source_kind = 'youtube' "
              "THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1) "
              "WHEN source_kind = 'tv' "
              "THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1) "
              "ELSE id END")

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

