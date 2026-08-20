"""Routine complete russian-media-monitor : collecte + refetch + analyses + diagnostic.

Une seule commande pour tout mettre a jour :
    python update.py

Options :
    python update.py --skip-pipeline      # passe la collecte
    python update.py --skip-refetch       # passe le refetch des articles cassés
    python update.py --skip-analyses      # passe les analyses Mistral
    python update.py --dashboard          # lance streamlit a la fin
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/russia.duckdb")

# sys.executable et non "python" : sous le planificateur de taches, le venv
# n'est pas dans le PATH et tout echouait sur ModuleNotFoundError.
PY = f'"{sys.executable}"'

STEPS_PIPELINE = [
    ("Collecte des nouveaux articles", f"{PY} -m src.pipeline"),
    ("Recuperation dates manquantes (HTML)", f"{PY} scripts/maintenance/fix_missing_dates.py"),
]

STEPS_REFETCH = [
    ("Re-fetch articles sans contenu", f"{PY} scripts/maintenance/refetch_missing.py"),
]

# Entites et posture-par-article ont ete retirees en aout 2026 : un tiers du
# cout Mistral chacune, pour 67 % d'entites vues une seule fois d'un cote, et
# une posture deja renseignee a la main dans sources.yaml de l'autre.
STEPS_ANALYSES = [
    ("Extraction auteurs", f"{PY} scripts/analysis/extract_authors.py"),
    ("Sentiment multi-cibles", f"{PY} scripts/analysis/analyze_sentiment_multi_mistral.py"),
    # Fenetre de 24 h, un clustering par support puis recoupement entre
    # eux. --backfill 1 repasse la veille : au moment ou la routine tourne
    # elle est complete, alors que le run de la veille ne voyait qu'une
    # journee tronquee. analyze_topics.py (fenetre 30 j) reste dans le
    # depot, utilise par tune_topics.py et pour ses fonctions partagees.
    ("Themes (clustering 24 h par support)",
     f"{PY} scripts/analysis/analyze_topics_daily.py --backfill 1"),
    # Purement calculatoire, aucun appel d'API : sa place est dans la routine.
    ("Divergence lexicale", f"{PY} scripts/analysis/analyze_divergence.py"),
]

# Analyses facultatives, payantes, hors routine quotidienne. La validation des
# themes est un diagnostic qu'on relance apres un changement de parametres, pas
# tous les jours ; les procedes rhetoriques coutent l'ordre de grandeur du
# sentiment. Options explicites plutot qu'un cout qui grimpe sans qu'on le voie.
STEPS_VALIDATION = [
    ("Validation des themes (ProxAnn)", f"{PY} scripts/analysis/validate_topics.py"),
]

STEPS_TECHNIQUES = [
    ("Procedes de persuasion", f"{PY} scripts/analysis/analyze_techniques_mistral.py"),
]

STEPS_DIAG = [
    ("Diagnostic couverture", f"{PY} scripts/maintenance/check_coverage.py"),
]


def banner(text):
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)


def run_step(label, cmd, idx, total):
    print(f"\n[{idx}/{total}] {label}")
    print(f"  > {cmd}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else "ECHEC"
    print(f"  [{status}] en {dt:.1f}s")
    return label, status, dt


def check_db_available():
    """DuckDB n'autorise qu'un seul processus en ecriture : si le dashboard
    (ou un autre script) a encore la base ouverte, chacune des 9 etapes
    echouerait une par une avec la meme IOException -- autant le detecter
    tout de suite et arreter net avec un message clair."""
    if not DB_PATH.exists():
        return  # premiere collecte : la base n'existe pas encore, rien a verifier
    import duckdb
    try:
        duckdb.connect(str(DB_PATH)).close()
    except duckdb.IOException:
        print(f"\nERREUR : {DB_PATH} est deja ouverte par un autre processus "
              f"(le dashboard Streamlit, le plus souvent).")
        print("Fermez-le (Ctrl+C dans son terminal) puis relancez update.py.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-refetch", action="store_true")
    ap.add_argument("--skip-analyses", action="store_true")
    ap.add_argument("--with-validation", action="store_true",
                    help="evalue la qualite des themes (~0,16 $)")
    ap.add_argument("--with-techniques", action="store_true",
                    help="releve les procedes rhetoriques sur TV et YouTube")
    ap.add_argument("--dashboard", action="store_true")
    args = ap.parse_args()

    check_db_available()

    steps = []
    if not args.skip_pipeline:
        steps += STEPS_PIPELINE
    if not args.skip_refetch:
        steps += STEPS_REFETCH
    if not args.skip_analyses:
        steps += STEPS_ANALYSES
    if args.with_validation:
        steps += STEPS_VALIDATION
    if args.with_techniques:
        steps += STEPS_TECHNIQUES
    steps += STEPS_DIAG

    banner(f"ROUTINE RUSSIAN MEDIA MONITOR ({len(steps)} etapes)")
    print(f"Demarre a {datetime.now().strftime('%H:%M:%S')}")

    t_start = time.time()
    results = []
    for i, (label, cmd) in enumerate(steps, 1):
        results.append(run_step(label, cmd, i, len(steps)))

    total_min = (time.time() - t_start) / 60

    banner(f"TERMINE en {total_min:.1f} minutes")
    for label, status, dt in results:
        marker = "OK  " if status == "OK" else "FAIL"
        print(f"  [{marker}] {label:<40} {dt:>6.1f}s")

    fails = [r for r in results if r[1] != "OK"]
    if fails:
        print(f"\n{len(fails)} etapes en echec. Verifiez les logs ci-dessus.")
        sys.exit(1)

    if args.dashboard:
        print("\nLancement du dashboard...")
        subprocess.run("streamlit run dashboard\\app.py", shell=True)
    else:
        print("\nPour lancer le dashboard :")
        print("    streamlit run dashboard\\app.py")


if __name__ == "__main__":
    main()
