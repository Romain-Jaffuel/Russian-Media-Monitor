"""Pour chaque analyse, compare combien d'articles SONT eligibles
(content non-null, >=300 char -- 50 pour Telegram, lang=ru) vs combien ont des resultats.

Montre 3 articles eligibles non traites par analyse pour comprendre
si c'est un bug WHERE ou un cas de figure specifique.
"""
import duckdb

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

c = duckdb.connect("data/russia.duckdb", read_only=True)
_tables = {r[0] for r in c.execute("SHOW TABLES").fetchall()}

# 1. Total eligible
eligible = c.execute("""
    SELECT COUNT(*) FROM articles
    WHERE content IS NOT NULL AND LENGTH(content) >= CASE WHEN source_kind = 'telegram' THEN 50 ELSE 300 END AND language = 'ru'
""").fetchone()[0]
print(f"Articles ELIGIBLES (content >= 300/50 telegram, lang = ru) : {eligible}")
print()

# 2. Coverage par table
checks = [
    ("entities", "entities (NER)"),
    ("article_target_sentiment", "sentiment multi-cibles"),
    ("article_topics", "themes"),
    ("article_meta", "type + posture + evenements"),
]

print(f"{'ANALYSE':<32} {'ELIGIBLES':>10} {'TRAITES':>10} {'MANQUE':>10}")
print("=" * 70)
for tbl, label in checks:
    try:
        processed = c.execute(
            f"SELECT COUNT(DISTINCT article_id) FROM {tbl}"
        ).fetchone()[0]
        missing = eligible - processed
        print(f"  {label:<30} {eligible:>10} {processed:>10} {missing:>10}")
    except Exception as e:
        print(f"  {label:<30} ERREUR : {e}")

# 3. Echantillon d'articles eligibles SANS analyse entities
print()
print("=" * 70)
print("ECHANTILLON : 5 articles eligibles SANS entites (donc non traites)")
print("=" * 70)
if "entities" not in _tables:
    print("  Table 'entities' absente : analyze_entities_mistral.py n'a jamais tourne.")
else:
    rows = c.execute("""
        SELECT a.id, a.source_name, a.published_at, a.title
        FROM articles a
        WHERE a.content IS NOT NULL AND LENGTH(a.content) >= CASE WHEN a.source_kind = 'telegram' THEN 50 ELSE 300 END AND a.language = 'ru'
          AND NOT EXISTS (SELECT 1 FROM entities e WHERE e.article_id = a.id)
        ORDER BY a.published_at DESC NULLS LAST
        LIMIT 5
    """).fetchall()
    if not rows:
        print("  Aucun ! Tous les articles eligibles ont des entites.")
    else:
        for aid, src, dt, ttl in rows:
            print(f"  [{aid}] {str(dt)[:16]}  {src}  {(ttl or '')[:60]}")

# 4. Verifie si les analyses ont des lignes "incompletes"
# (cas du NOT EXISTS qui skip a tort)
print()
print("=" * 70)
print("VERIFICATION : articles avec analyses partielles")
print("=" * 70)
if "entities" not in _tables or "article_target_sentiment" not in _tables:
    print("  Table 'entities' ou 'article_target_sentiment' absente, verification sautee.")
else:
    n_partial = c.execute("""
        SELECT COUNT(DISTINCT a.id) FROM articles a
        WHERE a.content IS NOT NULL AND LENGTH(a.content) >= CASE WHEN a.source_kind = 'telegram' THEN 50 ELSE 300 END AND a.language = 'ru'
          AND EXISTS (SELECT 1 FROM entities e WHERE e.article_id = a.id)
          AND NOT EXISTS (SELECT 1 FROM article_target_sentiment s WHERE s.article_id = a.id)
    """).fetchone()[0]
    print(f"  Articles avec entites mais SANS sentiment multi-cibles : {n_partial}")

if "article_target_sentiment" not in _tables or "article_topics" not in _tables:
    print("  Table 'article_target_sentiment' ou 'article_topics' absente, verification sautee.")
else:
    n_partial2 = c.execute("""
        SELECT COUNT(DISTINCT a.id) FROM articles a
        WHERE a.content IS NOT NULL AND LENGTH(a.content) >= CASE WHEN a.source_kind = 'telegram' THEN 50 ELSE 300 END AND a.language = 'ru'
          AND EXISTS (SELECT 1 FROM article_target_sentiment s WHERE s.article_id = a.id)
          AND NOT EXISTS (SELECT 1 FROM article_topics t WHERE t.article_id = a.id)
    """).fetchone()[0]
    print(f"  Articles avec sentiment mais SANS themes : {n_partial2}")

c.close()

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)
print("Lancez UNE analyse manuellement et regardez le 1er message qu'elle imprime :")
print()
print("    python scripts/analysis/analyze_entities_mistral.py")
print()
print("Cherchez une ligne du type : 'X articles a traiter' ou 'Aucun nouveau article'.")
print("Si c'est 'Aucun nouveau', le WHERE de l'analyse a un bug.")
print("Si c'est 'X a traiter' avec X > 100, laissez tourner jusqu'au bout.")
