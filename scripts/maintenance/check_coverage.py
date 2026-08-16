"""Couverture par source - VERSION CORRIGEE (sans cross-product des joins).

Usage : python scripts/maintenance/check_coverage.py
"""
import duckdb

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

c = duckdb.connect("data/russia.duckdb", read_only=True)

# Tables optionnelles : CTE vide si l'analyse correspondante n'a jamais tourne,
# plutot que de planter sur une table absente.
_tables = {r[0] for r in c.execute("SHOW TABLES").fetchall()}
_empty_cte = "SELECT source_name, 0 AS n FROM articles WHERE FALSE GROUP BY source_name"
ent_sql = ("SELECT a.source_name, COUNT(DISTINCT e.article_id) AS n "
           "FROM articles a JOIN entities e ON e.article_id = a.id "
           "GROUP BY a.source_name") if "entities" in _tables else _empty_cte
snt_sql = ("SELECT a.source_name, COUNT(DISTINCT s.article_id) AS n "
           "FROM articles a JOIN article_target_sentiment s ON s.article_id = a.id "
           "GROUP BY a.source_name") if "article_target_sentiment" in _tables else _empty_cte
thm_sql = ("SELECT a.source_name, COUNT(DISTINCT t.article_id) AS n "
           "FROM articles a JOIN article_topics t ON t.article_id = a.id "
           "GROUP BY a.source_name") if "article_topics" in _tables else _empty_cte
# Pas de filtre sur topic_key != -1 ici : un article classe "bruit" par
# BERTopic a bien ete traite par analyze_topics.py, juste sans cluster net.
# L'exclure de ce diagnostic de SANTE du pipeline creait un faux signal
# PROBLEME sur les sources Telegram (contenu plus court/heterogene, donc
# plus souvent en bruit que la presse) alors que rien n'est casse.
met_sql = ("SELECT a.source_name, COUNT(DISTINCT m.article_id) AS n "
           "FROM articles a JOIN article_meta m ON m.article_id = a.id "
           "GROUP BY a.source_name") if "article_meta" in _tables else _empty_cte

# Seuil de contenu "exploitable" : les posts Telegram sont naturellement
# courts, un seuil de 300 car. (calibre presse) les flaguerait presque tous
# a tort en CONTENU MANQUANT.
_min_len = "CASE WHEN source_kind = 'telegram' THEN 50 ELSE 300 END"

# Base : counts depuis articles uniquement (pas de cross-join)
rows = c.execute(f"""
    WITH base AS (
        SELECT source_name,
               COUNT(*) AS total,
               SUM(CASE WHEN content IS NOT NULL AND LENGTH(content) >= {_min_len}
                        THEN 1 ELSE 0 END) AS with_content,
               SUM(CASE WHEN language = 'ru' THEN 1 ELSE 0 END) AS lang_ru
        FROM articles
        GROUP BY source_name
    ),
    ent AS ({ent_sql}),
    snt AS ({snt_sql}),
    thm AS ({thm_sql}),
    met AS ({met_sql})
    SELECT b.source_name, b.total, b.with_content, b.lang_ru,
           COALESCE(ent.n, 0) AS entities,
           COALESCE(snt.n, 0) AS sentiment,
           COALESCE(thm.n, 0) AS themes,
           COALESCE(met.n, 0) AS meta
    FROM base b
    LEFT JOIN ent ON ent.source_name = b.source_name
    LEFT JOIN snt ON snt.source_name = b.source_name
    LEFT JOIN thm ON thm.source_name = b.source_name
    LEFT JOIN met ON met.source_name = b.source_name
    ORDER BY b.total DESC
""").fetchall()

print("=" * 95)
print(f"{'SOURCE':<35} {'TOTAL':>6} {'CONT':>6} {'LANG':>6} {'ENT':>6} "
      f"{'SENT':>6} {'THM':>6} {'META':>6}")
print("=" * 95)
for src, tot, cont, lang, ent, sent, thm, met in rows:
    flag = ""
    if tot >= 5 and cont < tot * 0.5:
        flag = " <- CONTENU MANQUANT"
    elif tot >= 5 and lang < tot * 0.5:
        flag = " <- LANG NON DETECTEE"
    print(f"  {src:<33} {tot:>6} {cont:>6} {lang:>6} {ent:>6} "
          f"{sent:>6} {thm:>6} {met:>6}{flag}")

# Totaux globaux
tot_total = sum(r[1] for r in rows)
tot_cont = sum(r[2] for r in rows)
tot_ent = sum(r[4] for r in rows)
print()
print(f"  {'TOTAL CORPUS':<33} {tot_total:>6} {tot_cont:>6} {sum(r[3] for r in rows):>6} "
      f"{tot_ent:>6} {sum(r[5] for r in rows):>6} {sum(r[6] for r in rows):>6} {sum(r[7] for r in rows):>6}")

print()
print("Legende : TOTAL = articles stockes | CONT = content >= 300 car. "
      "(50 pour Telegram) | LANG = lang ru | ENT/SENT/THM/META = analyses appliquees")

c.close()
