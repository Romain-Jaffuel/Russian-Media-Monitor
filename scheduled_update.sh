#!/usr/bin/env bash
# Wrapper de collecte automatique, execute par Git Bash.
# Lance update.py via le python du venv, logue tout dans logs/.
#
# Lancement manuel :
#   ./scheduled_update.sh
#   ./scheduled_update.sh --skip-analyses     # collecte seule, sans appels Mistral
#
# Planification (Planificateur de taches Windows) :
#   Programme/script : C:\Program Files\Git\bin\bash.exe
#   Arguments        : -lc "'/c/Users/<vous>/Classic/Russia-Monitor/scheduled_update.sh'"
#
# Note : ne pas lancer pendant que le dashboard Streamlit tourne -- DuckDB
# n'accepte qu'un seul processus en ecriture. update.py le detecte et
# s'arrete proprement, mais la collecte de ce creneau sera sautee.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="$LOG_DIR/update_$STAMP.log"

# Force UTF-8 : sans ca le cyrillique ressort en mojibake dans les logs.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

PYTHON_EXE="$PROJECT_ROOT/.venv/Scripts/python.exe"
if [ ! -f "$PYTHON_EXE" ]; then
    # Repli Linux/macOS, au cas ou le projet tourne ailleurs qu'en Windows.
    PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
fi
if [ ! -f "$PYTHON_EXE" ]; then
    echo "ERREUR : python du venv introuvable (lancez 'uv sync')" | tee -a "$LOG_FILE"
    exit 1
fi

{
    echo "==== Demarrage : $(date '+%Y-%m-%d %H:%M:%S') ===="
    echo "Projet : $PROJECT_ROOT"
} >> "$LOG_FILE"

# tee : l'avancement s'affiche dans le terminal ET part dans le log. Sans ca,
# un lancement interactif laisse l'ecran vide pendant des heures.
# "$@" transmet les arguments a update.py (--skip-analyses, --skip-pipeline...).
"$PYTHON_EXE" update.py "$@" 2>&1 | tee -a "$LOG_FILE"
# ${PIPESTATUS[0]} et non $? : dans un pipe, $? renverrait le code de tee
# (toujours 0), ce qui masquerait un echec de update.py.
EXIT_CODE=${PIPESTATUS[0]}

{
    echo "Code de sortie update.py : $EXIT_CODE"
    echo "==== Fin : $(date '+%Y-%m-%d %H:%M:%S') ===="
} >> "$LOG_FILE"

# Instantane publiable, uniquement si la passe s'est bien terminee : publier
# une base a moitie remplie serait pire que publier celle de la veille.
# Il n'est PAS pousse automatiquement -- le depot se fait a la main, quand on
# a regarde le resultat.
if [ "$EXIT_CODE" -eq 0 ]; then
    "$PYTHON_EXE" scripts/maintenance/export_snapshot.py >> "$LOG_FILE" 2>&1         && echo "Instantane : data/snapshot.duckdb.gz"         || echo "Instantane : ECHEC (voir $LOG_FILE)"
fi

# Nettoyage des logs de plus de 30 jours
find "$LOG_DIR" -name "update_*.log" -mtime +30 -delete 2>/dev/null

echo "Log : $LOG_FILE (code $EXIT_CODE)"
exit "$EXIT_CODE"
