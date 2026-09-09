"""Couverture des analyses : qui est éligible, qui a été traité, où sont les trous.

Remplace l'ancien check_coverage.py et diagnose_analyses.py, qui répondaient à
la même question et interrogeaient tous deux `entities` et `article_meta`,
tables disparues lors du passage de Gabon à Russia Monitor. Leurs colonnes
affichaient donc des zéros permanents, et `article_techniques`, bien réelle,
n'apparaissait nulle part.

Pour éviter que cela se reproduise, le script compare la liste des analyses
qu'il connaît à ce que contient réellement la base, et signale les deux sens de
l'écart.

Usage :
  python scripts/maintenance/check_coverage.py
"""
import sys
from pathlib import Path

import duckdb

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8
from src.topics import MIN_CONTENT_LEN, MIN_CONTENT_LEN_TELEGRAM

DB = Path("data/russia.duckdb")

# Analyses au grain de l'article, dans l'ordre du pipeline. La clé est la
# table, la valeur le nom lisible.
ANALYSES = {
    "article_target_sentiment": "sentiment multi-cibles",
    "article_topics": "thèmes",
    "article_techniques": "procédés de persuasion",
}

# Un article est analysable s'il a du texte dans la langue du corpus. Le seuil
# de la presse écarterait presque tous les posts Telegram, courts par nature.
ELIGIBLE = (f"content IS NOT NULL AND language = 'ru' AND LENGTH(content) >= "
            f"CASE WHEN source_kind = 'telegram' THEN {MIN_CONTENT_LEN_TELEGRAM} "
            f"ELSE {MIN_CONTENT_LEN} END")


def _tables_par_article(conn):
    """Tables qui portent une colonne article_id, donc une analyse par article."""
    return {r[0] for r in conn.execute(
        "SELECT DISTINCT table_name FROM duckdb_columns() "
        "WHERE column_name = 'article_id'").fetchall()}


def _derive(conn):
    """Signale les écarts entre les analyses connues du script et la base."""
    presentes = _tables_par_article(conn)
    inconnues = presentes - set(ANALYSES)
    absentes = set(ANALYSES) - presentes
    for table in sorted(absentes):
        print(f"  La table {table} ({ANALYSES[table]}) n'existe pas : cette "
              f"analyse n'a jamais tourné.")
    for table in sorted(inconnues):
        print(f"  La table {table} porte des analyses par article mais n'est "
              f"pas listée dans ce script : ajoutez-la à ANALYSES.")
    return [t for t in ANALYSES if t in presentes]


def _par_source(conn, tables):
    """Une ligne par source : volume, texte exploitable, part analysée."""
    colonnes = ", ".join(
        f"""(SELECT COUNT(DISTINCT x.article_id) FROM {t} x
             JOIN articles y ON y.id = x.article_id
             WHERE y.source_name = a.source_name) AS "{t}\""""
        for t in tables)
    lignes = conn.execute(f"""
        SELECT a.source_name, COUNT(*) AS total,
               COUNT(*) FILTER (WHERE {ELIGIBLE}) AS eligibles
               {", " + colonnes if colonnes else ""}
        FROM articles a GROUP BY a.source_name ORDER BY total DESC""").fetchall()

    entetes = "".join(f"{ANALYSES[t][:9]:>10}" for t in tables)
    print(f"\n{'SOURCE':<34}{'TOTAL':>8}{'ÉLIGIBLES':>11}{entetes}")
    print("-" * (53 + 10 * len(tables)))
    for ligne in lignes:
        nom, total, eligibles = ligne[0], ligne[1], ligne[2]
        cases = "".join(f"{n:>10}" for n in ligne[3:])
        # Une source dont la moitié des articles n'a pas de texte exploitable
        # a un problème de collecte, pas d'analyse.
        alerte = "  texte manquant" if total >= 5 and eligibles < total / 2 else ""
        print(f"{nom[:33]:<34}{total:>8}{eligibles:>11}{cases}{alerte}")
    return lignes


def _trous(conn, tables):
    """Pour chaque analyse, ce qui reste à traiter, et un échantillon."""
    eligibles = conn.execute(
        f"SELECT COUNT(*) FROM articles WHERE {ELIGIBLE}").fetchone()[0]
    print(f"\n{eligibles} articles éligibles dans le corpus.\n")
    print(f"{'ANALYSE':<28}{'TRAITÉS':>10}{'MANQUANTS':>11}{'COUVERTURE':>12}")
    print("-" * 61)
    manque_max = (None, 0)
    for table in tables:
        traites = conn.execute(
            f"SELECT COUNT(DISTINCT x.article_id) FROM {table} x "
            f"JOIN articles a ON a.id = x.article_id "
            f"WHERE {ELIGIBLE}").fetchone()[0]
        manquants = eligibles - traites
        part = 100 * traites / eligibles if eligibles else 0
        print(f"{ANALYSES[table]:<28}{traites:>10}{manquants:>11}{part:>11.1f} %")
        if manquants > manque_max[1]:
            manque_max = (table, manquants)

    table, manquants = manque_max
    if not table or manquants <= 0:
        print("\nToutes les analyses couvrent la totalité des articles éligibles.")
        return
    print(f"\nCinq articles éligibles que « {ANALYSES[table]} » n'a pas traités :")
    for aid, src, quand, titre in conn.execute(f"""
            SELECT a.id, a.source_name, a.published_at, a.title
            FROM articles a
            WHERE {ELIGIBLE}
              AND NOT EXISTS (SELECT 1 FROM {table} x WHERE x.article_id = a.id)
            ORDER BY a.published_at DESC NULLS LAST LIMIT 5""").fetchall():
        print(f"   {str(quand)[:16]:16}  {src[:22]:22}  {(titre or '')[:44]}")
    print("\nUn manque massif tient d'ordinaire à la clause WHERE de l'analyse ;\n"
          "un manque partiel, à des articles ajoutés depuis sa dernière passe.")


def run():
    if not DB.exists():
        print(f"Base introuvable : {DB}")
        return 1
    try:
        conn = duckdb.connect(str(DB), read_only=True)
    except duckdb.Error as ex:
        print(f"Ouverture impossible ({ex}). Fermez le tableau de bord et "
              f"réessayez.")
        return 1
    tables = _derive(conn)
    if tables:
        _par_source(conn, tables)
        _trous(conn, tables)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
