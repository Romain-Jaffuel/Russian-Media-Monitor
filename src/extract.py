"""Fetch full article HTML and extract clean text + detect language."""
import logging
import time

import httpx
import trafilatura
from langdetect import detect, LangDetectException

log = logging.getLogger("extract")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 RussianMediaMonitor/0.1"
)


def fetch_html(url: str, timeout: float = 30.0, retries: int = 1) -> str | None:
    if not url.startswith(("http://", "https://")):
        return None
    last_err = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "no-cache, no-store, max-age=0",
                    "Pragma": "no-cache",
                },
            ) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.text
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2.0)  # backoff avant retry
                continue
        except Exception as e:
            last_err = e
            break

    log.warning("fetch_html KO (%s) sur %s", type(last_err).__name__, url)
    return None


def extract(html: str | None) -> tuple[str | None, str | None]:
    """Return (clean_text, language_code) from raw HTML."""
    if not html:
        return None, None
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not text:
        return None, None
    try:
        lang = detect(text)
    except LangDetectException:
        lang = None
    return text, lang