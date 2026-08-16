"""Reconfigure stdout/stderr en UTF-8 (effet de bord a l'import).

La console Windows par defaut est en cp1252 : un print() contenant du
cyrillique plante en UnicodeEncodeError. setup_logging() fait deja ceci pour
les scripts qui journalisent ; les scripts plus legers qui se contentent de
print() (diagnostics, utilitaires ponctuels) importent ce module a la place.
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
