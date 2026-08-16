"""Shared logging setup: console + a timestamped file under logs/.

Lets every long-running script (collecte, analyses) leave a readable trace
in logs/ even when run by hand, without needing scheduled_update.ps1.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")

# httpx logs "HTTP Request: ..." at INFO, which is useful (status codes show
# rate-limits/blocks). httpcore's own internal chatter below that (e.g.
# "discarding data: None" on connection reuse) has no diagnostic value here.
_NOISY_LOGGERS = ["httpcore"]


class _DropNoise(logging.Filter):
    """Drops third-party log lines with no diagnostic value, regardless of
    which logger emits them (library internals can log through unexpected
    names)."""

    _PATTERNS = ("discarding data",)

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)


def setup_logging(name: str) -> logging.Logger:
    from src import console_utf8  # noqa: F401 -- effet de bord : stdout/stderr en UTF-8

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log_file = LOG_DIR / f"{name}_{stamp}.log"

    noise_filter = _DropNoise()
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    for h in handlers:
        h.addFilter(noise_filter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger(name)

    # Sans ca, une exception non catchee sort par le traceback par defaut
    # (stderr uniquement) et n'apparait jamais dans le fichier de log.
    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.error("Exception non geree", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _log_uncaught

    return log
