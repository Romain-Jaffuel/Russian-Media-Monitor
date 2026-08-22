"""Regenere les captures du README depuis le tableau de bord.

Les images du README vieillissent a chaque changement de l'interface ou du
corpus. Les refaire a la main est long et donne des cadrages differents d'une
fois sur l'autre ; ce script les reproduit a l'identique.

Il demarre son propre Streamlit sur un port libre, attend qu'il reponde,
capture, puis l'arrete. Rien a lancer avant.

Prerequis : l'extra collecte (Playwright).
    uv sync --extra collecte
    uv run playwright install chromium

Usage :
  python scripts/maintenance/capture_dashboard.py
  python scripts/maintenance/capture_dashboard.py --garder   # laisse l'app up
"""
import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from src.logging_setup import setup_logging

log = setup_logging("captures")

RACINE = Path(__file__).resolve().parents[2]
ASSETS = RACINE / "assets"

# Fenetre volontairement haute : une capture plus grande que la fenetre oblige
# a passer par full_page, dont les coordonnees ne sont pas celles rendues par
# bounding_box(). Avec de la marge, un simple clip suffit.
LARGEUR, HAUTEUR = 1600, 1500
X0, LARG_UTILE = 348, 1232          # zone de contenu, barre laterale exclue

# (fichier, onglet, ancre, hauteur). ancre None = haut de page, la hauteur
# etant alors calculee jusqu'au bas du premier graphique.
PLAN = [
    ("vue-ensemble.png",        "Vue d'ensemble", None,                                   None),
    ("themes.png",              "Thèmes",         "Répartition des thèmes actifs",         700),
    ("themes-par-jour.png",     "Thèmes",         "Thèmes jour par jour",                  700),
    ("carte-influence.png",     "Sentiment",      "Carte d'influence",                     760),
    ("alignement-corpus.png",   "Alignement",     "Répartition du corpus par alignement",  720),
    ("procedes-persuasion.png", "Cadrage",        "Procédés de persuasion",                648),
]

# La barre d'outils Plotly et le bandeau Streamlit n'ont rien a faire sur une
# capture de documentation.
CSS_PROPRE = """
[data-testid="stToolbar"], header[data-testid="stHeader"] {display: none !important}
.modebar, .modebar-container {display: none !important}
"""

# Les onglets inactifs restent dans le DOM : sans ce filtre, le premier
# graphique trouve est celui d'un autre onglet, et l'on cadre a cote.
JS_VISIBLE = """(txt) => {
  const vu = e => e.offsetParent !== null && e.getBoundingClientRect().height > 20;
  const els = [...document.querySelectorAll('h1,h2,h3,h4,p,span,div')]
      .filter(e => vu(e) && e.textContent.trim().startsWith(txt)
                   && e.children.length === 0);
  if (!els.length) return null;
  const r = els[els.length - 1].getBoundingClientRect();
  return {y: r.y, h: r.height};
}"""


def _port_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _demarrer(port):
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(port), "--server.headless", "true"],
        cwd=RACINE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(60):
        if proc.poll() is not None:
            raise SystemExit("Streamlit s'est arrete au demarrage.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    raise SystemExit(f"Streamlit n'a pas repondu sur le port {port}.")


def _aller_a(page, onglet):
    for i in range(page.locator('[role="tab"]').count()):
        tab = page.locator('[role="tab"]').nth(i)
        if tab.inner_text().strip() == onglet:
            tab.click()
            page.wait_for_timeout(6500)
            return
    raise SystemExit(f"Onglet « {onglet} » introuvable. Les libelles ont-ils "
                     "change dans st.tabs() ?")


def capturer(page, fichier, onglet, ancre, hauteur):
    if ancre is None:
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(2000)
        cadre = page.locator('[data-testid="stPlotlyChart"]').first.bounding_box()
        y, h = 8, cadre["y"] + cadre["height"] + 14 - 8
    else:
        pos = page.evaluate(JS_VISIBLE, ancre)
        if pos is None:
            raise SystemExit(
                f"Ancre « {ancre} » absente de l'onglet {onglet}. Le titre "
                "a-t-il ete renomme ? Corriger PLAN plutot que de publier une "
                "capture cadree au hasard.")
        # scrollIntoView({block:'start'}) et non scroll_into_view_if_needed :
        # ce dernier s'arrete des que l'element est visible, souvent en bas de
        # fenetre, et la place manque alors sous lui.
        page.evaluate(
            """(txt) => {
                 const vu = e => e.offsetParent !== null;
                 const els = [...document.querySelectorAll('h1,h2,h3,h4,p,span,div')]
                     .filter(e => vu(e) && e.textContent.trim().startsWith(txt)
                                  && e.children.length === 0);
                 els[els.length - 1].scrollIntoView({block: 'start'});
               }""", ancre)
        page.evaluate("window.scrollBy(0, -30)")
        page.wait_for_timeout(2500)
        y = max(0, page.evaluate(JS_VISIBLE, ancre)["y"] - 22)
        h = hauteur
    h = min(h, HAUTEUR - y - 4)
    page.screenshot(path=str(ASSETS / fichier),
                    clip={"x": X0, "y": y, "width": LARG_UTILE, "height": h})
    log.info("  %-26s %-15s y=%4d h=%3d", fichier, onglet, y, h)


def run(garder=False):
    from playwright.sync_api import sync_playwright

    ASSETS.mkdir(exist_ok=True)
    port = _port_libre()
    log.info("Demarrage de Streamlit sur le port %d...", port)
    proc = _demarrer(port)
    try:
        with sync_playwright() as pw:
            nav = pw.chromium.launch()
            page = nav.new_context(
                viewport={"width": LARGEUR, "height": HAUTEUR},
                device_scale_factor=2).new_page()
            page.goto(f"http://localhost:{port}/", wait_until="networkidle",
                      timeout=180000)
            page.wait_for_timeout(17000)

            # Une base verrouillee donne une page d'avertissement sans onglets :
            # mieux vaut s'arreter que d'ecraser les images par des captures
            # vides.
            if page.locator('[role="tab"]').count() == 0:
                raise SystemExit(
                    "Le tableau de bord n'affiche aucun onglet. La base est "
                    "sans doute ouverte ailleurs (autre Streamlit, analyse en "
                    "cours). Fermez-la et relancez.")
            n_exc = page.locator('[data-testid="stException"]').count()
            if n_exc:
                raise SystemExit(
                    f"{n_exc} exception(s) affichee(s) : corriger le tableau de "
                    "bord avant de publier des captures.")

            page.add_style_tag(content=CSS_PROPRE)
            courant = None
            for fichier, onglet, ancre, hauteur in PLAN:
                if onglet != courant:
                    _aller_a(page, onglet)
                    courant = onglet
                capturer(page, fichier, onglet, ancre, hauteur)
            nav.close()
        log.info("%d captures ecrites dans assets/.", len(PLAN))
        return 0
    finally:
        if garder:
            log.info("Streamlit laisse en place : http://localhost:%d", port)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--garder", action="store_true",
                    help="ne pas arreter Streamlit a la fin")
    sys.exit(run(garder=ap.parse_args().garder))
