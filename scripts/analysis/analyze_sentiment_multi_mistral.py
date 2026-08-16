"""Sentiment multi-cibles via Mistral (16 acteurs).

Cibles : ukraine, etats_unis, union_europeenne, otan, allemagne, france,
pays_baltes, chine, inde, brics_global_south, georgie, opposition_russe,
kazakhstan, armenie, moldavie, iran

Pour chaque cible : stance + score + lean continu + reasoning.

Usage :
  python scripts/analysis/analyze_sentiment_multi_mistral.py --reset
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.db import get_conn
from src.logging_setup import setup_logging
from src.llm_mistral import complete_json, MODEL_SMALL

log = setup_logging("sent_multi")

MODEL = MODEL_SMALL
MIN_CONTENT_LEN = 300
# Les posts Telegram sont naturellement courts -- le seuil presse les
# excluait presque tous (le scraper ecarte deja tout ce qui fait moins de
# 50 car. a la collecte, cf. src/telegram_scrape.py).
MIN_CONTENT_LEN_TELEGRAM = 50
MAX_CONTENT_CHARS = 3500
# Le goulot est le temps reseau (attente Mistral), pas le CPU -- meme
# parallelisation que analyze_entities_mistral.py.
MAX_WORKERS = 8

TARGETS = [
    "ukraine", "etats_unis", "union_europeenne", "otan",
    "allemagne", "france", "pays_baltes", "chine", "inde",
    "brics_global_south", "georgie", "opposition_russe",
    "kazakhstan", "armenie", "moldavie", "iran",
]

# Mots-cles russes (cyrillique) servant a decider si un article merite un
# appel LLM. Racines volontairement larges (variations grammaticales russes)
# plutot que des formes exactes.
#
# Les racines courtes qui sont aussi des sous-chaines de mots courants sont
# ancrees par \b : sans ca "дели" matche "недели" (semaine) et "иран" matche
# "тиран" -- des faux positifs qui declenchent un appel LLM payant pour rien.
ALL_RE = re.compile(
    r"(украин|зеленск|киев|всу\b|донбасс|донецк|луганск|запорож|херсон|крым|"
    r"сша|американ|вашингтон|трамп|байден|пентагон|"
    r"европейский союз|\bес\b|брюссел|еврокомисси|"
    r"нато|североатлантич|"
    r"германи|немецк|берлин|шольц|мерц|"
    r"франци|французск|макрон|париж|"
    r"прибалтик|эстони|латви|литв|"
    r"кита[ий]|китайск|пекин|си цзиньпин|"
    r"инди[яйи]|индийск|\bдели\b|\bмоди\b|"
    r"западн[а-я]* стран|западн[а-я]* мир|западные партнеры|"
    r"брикс|глобальный юг|"
    r"грузи[яю]|грузинск|тбилиси|"
    r"навальн|оппозицион|иноагент|"
    r"казахстан|астана|токаев|"
    r"армени[яю]|ереван|пашинян|"
    r"молдав|кишинев|приднестров|санду|"
    r"\bиран|тегеран|хаменеи|"
    r"шахед|герань-2)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Vous etes un analyste geopolitique specialise dans la presse russophone (medias d'Etat, para-etatiques, independants et en exil). Pour chaque article, vous evaluez la posture editoriale vis-a-vis de 16 acteurs geopolitiques :

ukraine, etats_unis, union_europeenne, otan, allemagne, france,
pays_baltes (Estonie/Lettonie/Lituanie), chine, inde, brics_global_south,
georgie, opposition_russe, kazakhstan, armenie, moldavie, iran

Pour CHAQUE cible, renvoyez quatre champs :

1. stance : la POSTURE EDITORIALE de l'article envers la cible -- PAS la nature
   (positive/negative/violente) des evenements rapportes. C'est le piege le
   plus frequent : decrire une frappe, une perte militaire ou un evenement
   tragique concernant la cible n'est PAS en soi une posture "anti" envers
   cette cible. Un compte-rendu factuel d'une action ukrainienne ("les VSU
   ont frappe une raffinerie russe") n'est pas hostile a l'Ukraine -- c'est
   un fait de guerre rapporte sans jugement. A l'inverse, un article
   compatissant sur des victimes civiles cote cible (ex: TV Rain decrivant
   des morts ukrainiens sous une frappe russe) traduit plutot une posture
   favorable/sympathique envers la cible, pas hostile.
   Ne codez "anti" que si l'article porte un JUGEMENT explicite defavorable
   SUR la cible elle-meme (illegitimite, mauvaise foi, accusation, moquerie,
   negation de sa cause) -- jamais seulement parce qu'il mentionne une
   frappe, une perte ou un echec la concernant.
   - "pro" : soutien, valorisation, partenariat positif, mention favorable,
     ou sympathie/compassion implicite envers la cible
   - "anti" : jugement defavorable explicite SUR la cible (illegitimite,
     mauvaise foi, accusation, moquerie) -- pas le simple recit d'un
     evenement negatif qui la concerne
   - "neutre" : compte-rendu factuel sans charge emotionnelle ni jugement
     explicite, y compris pour des evenements negatifs (frappes, pertes,
     combats) rapportes de maniere descriptive
   - "non_concerne" : pas mentionnee ou totalement peripherique

2. score : confiance (0.0 a 1.0)

3. lean : inclination continue de -1.0 (tres anti) a +1.0 (tres pro), 0.0 = neutre.
   Meme un article "neutre" peut avoir un lean non nul. Pour "non_concerne", lean = 0.0.

3 bis. Attention a ne pas sur-interpreter : la plupart des articles de guerre
   sont factuels/neutres envers les deux camps qu'ils decrivent. "anti" et
   "pro" doivent rester minoritaires, reserves aux cas de cadrage editorial
   net -- pas au simple fait qu'un evenement rapporte soit violent ou triste.

4. reasoning : 1 phrase courte -- UNIQUEMENT pour "pro" et "anti" (justifie le
   jugement). Chaine vide "" pour "neutre" et "non_concerne" (pas de
   justification a donner, ca economise des tokens sur la grande majorite
   des cas -- la plupart des cibles sont non_concerne dans un article donne).

Precisions :
- pays_baltes : posture vis-a-vis du bloc Estonie/Lettonie/Lituanie, agrege.
- brics_global_south : posture vis-a-vis du bloc BRICS / pays du Sud global comme alternative geopolitique a l'Occident.
- opposition_russe : posture vis-a-vis de l'opposition russe en exil et des figures/medias independants russes (politique interieure, pas un pays).
- kazakhstan : posture vis-a-vis du Kazakhstan (partenaire CSTO/UEE, mais qui prend ses distances depuis 2022 -- route de contournement des sanctions, tensions recurrentes sur le statut du russe et l'alignement geopolitique).
- armenie : posture vis-a-vis de l'Armenie (allie CSTO historique, relation degradee depuis la guerre du Haut-Karabakh 2020 et le rapprochement recent avec l'Occident).
- moldavie : posture vis-a-vis de la Moldavie ET de la Transnistrie (contingent russe stationne, candidate a l'UE, cible declaree d'ingerence electorale russe). Agreger les deux.
- iran : posture vis-a-vis de l'Iran (partenariat militaire croissant, fourniture de drones Shahed/Geran, cooperation sur le contournement des sanctions).

Repondez UNIQUEMENT en JSON strict, avec exactement ces 16 cles :
{
  "ukraine": {"stance":"...","score":0.0,"lean":0.0,"reasoning":"..."},
  "etats_unis": {...}, "union_europeenne": {...}, "otan": {...},
  "allemagne": {...}, "france": {...},
  "pays_baltes": {...}, "chine": {...}, "inde": {...},
  "brics_global_south": {...}, "georgie": {...}, "opposition_russe": {...},
  "kazakhstan": {...}, "armenie": {...}, "moldavie": {...}, "iran": {...}
}"""


def ensure_schema(conn, reset=False):
    if reset:
        conn.execute("DROP TABLE IF EXISTS article_target_sentiment")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS article_target_sentiment (
            article_id VARCHAR, target VARCHAR,
            stance VARCHAR, score FLOAT, lean FLOAT, reasoning VARCHAR,
            method VARCHAR DEFAULT 'mistral',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (article_id, target)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ats_target ON article_target_sentiment(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ats_stance ON article_target_sentiment(stance)")


def needs_analysis(text):
    return bool(ALL_RE.search(text))


def write_neutre(conn, art_id):
    for t in TARGETS:
        conn.execute(
            "INSERT OR REPLACE INTO article_target_sentiment "
            "(article_id, target, stance, score, lean, method) VALUES (?, ?, ?, ?, ?, ?)",
            [art_id, t, "non_concerne", 0.0, 0.0, "mistral"],
        )


def write_target(conn, art_id, target, data):
    stance = data.get("stance", "non_concerne")
    try:
        score = float(data.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        lean = float(data.get("lean", 0.0) or 0.0)
    except (TypeError, ValueError):
        lean = 0.0
    lean = max(-1.0, min(1.0, lean))
    reasoning = (data.get("reasoning") or "")[:500]
    conn.execute(
        "INSERT OR REPLACE INTO article_target_sentiment "
        "(article_id, target, stance, score, lean, reasoning, method) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [art_id, target, stance, score, lean, reasoning, "mistral"],
    )


def run(reset=False, limit=None):
    conn = get_conn()
    ensure_schema(conn, reset=reset)

    sql = f"""
        SELECT id, COALESCE(title, '') || E'\\n' || content AS text
        FROM articles a
        WHERE content IS NOT NULL
          AND LENGTH(content) >= (CASE WHEN a.source_kind = 'telegram'
                                        THEN {MIN_CONTENT_LEN_TELEGRAM} ELSE {MIN_CONTENT_LEN} END)
          AND language = 'ru'
          AND NOT EXISTS (
              SELECT 1 FROM article_target_sentiment s WHERE s.article_id = a.id
          )
    """
    if limit:
        sql += f" LIMIT {limit * 10}"
    rows = conn.execute(sql).fetchall()

    to_call, to_skip = [], []
    for art_id, text in rows:
        (to_call if needs_analysis(text) else to_skip).append((art_id, text))
    if limit:
        to_call = to_call[:limit]
    log.info("A envoyer : %d, non_concerne sans appel : %d", len(to_call), len(to_skip))

    for art_id, _ in to_skip:
        write_neutre(conn, art_id)
    if not to_call:
        log.info("Rien a appeler.")
        conn.close()
        return

    started = time.time()
    ok = err = done = 0
    db_lock = Lock()

    def call_mistral(art_id, text):
        data = complete_json(SYSTEM_PROMPT, f"Article :\n\n{text[:MAX_CONTENT_CHARS]}",
                             model=MODEL, max_tokens=2500)
        return art_id, data

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(call_mistral, art_id, text) for art_id, text in to_call]
        for future in as_completed(futures):
            art_id, data = future.result()
            with db_lock:
                if not data:
                    err += 1
                else:
                    for target in TARGETS:
                        tdata = data.get(target) or {}
                        if not isinstance(tdata, dict):
                            tdata = {"stance": "non_concerne", "score": 0.0, "lean": 0.0}
                        write_target(conn, art_id, target, tdata)
                    ok += 1
                done += 1
                if done % 10 == 0:
                    rate = (time.time() - started) / done
                    log.info("  %d/%d (reste ~%.0fmin)", done, len(to_call),
                             (len(to_call) - done) * rate / 60)

    log.info("Termine : %d ok, %d echecs, %.1fs", ok, err, time.time() - started)
    conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    run(reset=a.reset, limit=a.limit)
