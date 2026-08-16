"""Compresse data/russia.duckdb en gzip pour estimer sa taille compressee.

Contourne la limite 2 Go de Compress-Archive de PowerShell. gzip gere les
gros fichiers et compresse par flux (peu de RAM).

Le resultat data/russia.duckdb.gz sert a savoir si la base compressee rentre
sous le quota Git LFS gratuit (1 Go).

Usage : python scripts/maintenance/compress_db.py
"""
import gzip
import shutil
from pathlib import Path

from src import console_utf8  # noqa: F401 -- stdout/stderr en UTF-8

DB = Path("data/russia.duckdb")
OUT = DB.with_suffix(".duckdb.gz")


def human(n):
    for u in ["o", "Ko", "Mo", "Go"]:
        if abs(n) < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} To"


if not DB.exists():
    raise SystemExit(f"Introuvable : {DB}")

# Verifier l'espace : le .gz fait au pire ~ la taille de la base
libre = shutil.disk_usage(str(DB.parent)).free
print(f"Base          : {human(DB.stat().st_size)}")
print(f"Disque libre  : {human(libre)}")
if libre < DB.stat().st_size * 0.6:
    print("\nATTENTION : peu d'espace libre. Si la compression echoue, liberez")
    print("de la place (corbeille, TEMP) ou compressez vers une cle USB.")

print("\nCompression en cours (peut prendre 1-3 min)...")
with open(DB, "rb") as f_in, gzip.open(OUT, "wb", compresslevel=9) as f_out:
    shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

taille = OUT.stat().st_size
print(f"\nResultat : {OUT.name}")
print(f"  Compresse : {human(taille)}")
print(f"  Ratio     : {100 * taille / DB.stat().st_size:.0f}% de l'original")

print()
if taille < 1024 ** 3:
    print(">>> Sous 1 Go : rentre dans le quota Git LFS gratuit.")
    print("    On peut versionner cette archive compressee.")
else:
    print(">>> Au-dessus de 1 Go : depasse le quota LFS gratuit.")
    print("    Mieux vaut mettre la base sur un stockage externe (Drive) et")
    print("    ne versionner que le code, avec un lien dans le README.")
