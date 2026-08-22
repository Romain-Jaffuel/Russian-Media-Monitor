"""Dashboard : refonte événements (carte par acteur), géographie (top 300, purple,
zoomable), acteurs (position des journalistes vs une cible géopolitique).
"""
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Lance directement par l'interprete (`python dashboard/app.py`), ce fichier
# s'execute sans serveur : Streamlit émet une salve de « missing
# ScriptRunContext », le script se termine et rien n'est servi. Le diagnostic
# n'est pas évident dans ce bruit, d'où ce garde-fou qui donne la commande.
if not st.runtime.exists():
    import sys
    sys.exit(
        "\nCe fichier est une application Streamlit : il lui faut son serveur.\n"
        "  uv run streamlit run dashboard/app.py\n"
    )

# ============================================================
# Recherche booléenne multi-mots (ET / OU / SAUF, guillemets, parentheses)
# ============================================================
import re as _re_kw

_chart_counter = {"n": 0}


def _next_chart_key(prefix="chart"):
    _chart_counter["n"] += 1
    return f"{prefix}_{_chart_counter['n']}"


def _kw_tokenize(q):
    tokens = []
    for m in _re_kw.finditer(r'"[^"]*"|\(|\)|[^\s()]+', q.strip()):
        tok = m.group(0)
        up = tok.upper()
        if tok == "(":
            tokens.append(("LP", None))
        elif tok == ")":
            tokens.append(("RP", None))
        elif up in ("AND", "ET", "&&", "&"):
            tokens.append(("AND", None))
        elif up in ("OR", "OU", "||", "|"):
            tokens.append(("OR", None))
        elif up in ("NOT", "SAUF", "NON"):
            tokens.append(("NOT", None))
        elif tok.startswith("-") and len(tok) > 1:
            tokens.append(("NOT", None))
            tokens.append(("TERM", tok[1:].strip('"')))
        else:
            tokens.append(("TERM", tok.strip('"')))
    return tokens


def _kw_insert_default(tokens, default_op):
    out = []
    for tok in tokens:
        if out:
            prev, cur = out[-1][0], tok[0]
            if prev in ("TERM", "RP") and cur in ("TERM", "LP", "NOT"):
                out.append(("AND" if cur == "NOT" else default_op, None))
        out.append(tok)
    return out


def _kw_parse(tokens):
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def p_or():
        nonlocal pos
        left = p_and()
        while peek()[0] == "OR":
            pos += 1
            left = ("OR", left, p_and())
        return left

    def p_and():
        nonlocal pos
        left = p_not()
        while peek()[0] == "AND":
            pos += 1
            left = ("AND", left, p_not())
        return left

    def p_not():
        nonlocal pos
        if peek()[0] == "NOT":
            pos += 1
            return ("NOT", p_not())
        return p_atom()

    def p_atom():
        nonlocal pos
        t = peek()
        if t[0] == "LP":
            pos += 1
            node = p_or()
            if peek()[0] == "RP":
                pos += 1
            return node
        if t[0] == "TERM":
            pos += 1
            return ("TERM", t[1])
        pos += 1
        return None

    return p_or()


def _kw_compile(node, params):
    if node is None:
        return "1=1"
    typ = node[0]
    if typ == "TERM":
        import unicodedata as _ud
        _t = "".join(ch for ch in _ud.normalize("NFD", node[1])
                     if _ud.category(ch) != "Mn")
        params.append(f"%{_t}%")
        params.append(f"%{_t}%")
        return ("(strip_accents(title) ILIKE ? OR strip_accents(content) ILIKE ?)")
    if typ == "NOT":
        return f"(NOT {_kw_compile(node[1], params)})"
    if typ in ("AND", "OR"):
        return f"({_kw_compile(node[1], params)} {typ} {_kw_compile(node[2], params)})"
    return "1=1"


def build_keyword_filter(query, default_op="AND"):
    """Compile une requête booléenne en (fragment SQL, params)."""
    if not query or not query.strip():
        return None, []
    toks = _kw_insert_default(_kw_tokenize(query), default_op)
    params = []
    return _kw_compile(_kw_parse(toks), params), params


DB_PATH = Path("data/russia.duckdb")
# Copie publiée : la seule chose dont dispose un hebergeur, la base
# complète restant en local. A déclarer ici, avant le premier usage --
# le chemin local ne passe jamais par cette branche, donc l'erreur ne se
# serait vue qu'une fois déployé.
SNAPSHOT_GZ = Path("data/snapshot.duckdb.gz")

# --- Identité publiée (pied de page) ---
URL_REPO = "https://github.com/Romain-Jaffuel/Russian-Media-Monitor"
URL_GITHUB = "https://github.com/Romain-Jaffuel"
URL_LINKEDIN = "https://www.linkedin.com/in/romain-jaffuel/"
URL_SITE = "https://romain-jaffuel.github.io/"
URL_GROLLEAU = "https://flor5378.github.io/"
URL_GABON = "https://github.com/Flor5378/Gabon-Monitor"
CREDIT_ORIGINE = ("Ossature initiale du pipeline reprise de "
                  f"[Gabon Monitor]({URL_GABON}), "
                  f"de [Florian Grolleau]({URL_GROLLEAU}).")
CREDIT_ORIGINE_HTML = (
    "Ossature initiale du pipeline reprise de "
    f'<a href="{URL_GABON}" target="_blank" rel="noopener">Gabon Monitor</a>, '
    f'de <a href="{URL_GROLLEAU}" target="_blank" rel="noopener">Florian Grolleau</a>.')

# Repris de primaryColor dans .streamlit/config.toml : le pied de page est le
# seul bloc HTML du fichier, Streamlit n'expose pas son theme en variables CSS.
ACCENT = "#00C2A8"
SOURCES_PATH = Path("config/sources.yaml")


def load_historical_stances(path=SOURCES_PATH):
    """source_name -> hand-written editorial history label (see config/sources.yaml)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {s["name"]: s.get("historical_stance", "") for s in data.get("sources", [])}


HISTORICAL_STANCE = load_historical_stances()


def load_source_config(path=SOURCES_PATH):
    """Les entrées de config/sources.yaml telles quelles, pour le panneau de
    couverture : il doit montrer les sources CONFIGURÉES, y compris celles qui
    n'ont encore rien rapporté -- c'est justement l'information utile."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("sources", [])


SOURCE_CONFIG = load_source_config()

# `type:` de la config -> source_kind en base. Copie volontaire de la table de
# src/pipeline.py : le dashboard lit la config sans importer le pipeline.
CFG_KIND = {"telegram": "telegram", "youtube": "youtube", "rutube": "tv",
            "hls": "tv", "vk": "vk"}

# Trois expressions SQL partagées par la vue d'ensemble et le panneau de
# couverture. L'unité parente se relit depuis l'URL du segment (`v=<id>` sur
# YouTube, `vidéo/<id>` sur RuTube et smotrim) et l'instant du segment
# (`?t=`, `&t=`, `#t=`) donne la minute couverte, ce qui évite de stocker une
# durée.
SQL_PARENT = ("CASE WHEN source_kind = 'youtube' "
              "THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1) "
              "WHEN source_kind = 'tv' "
              "THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1) "
              "ELSE id END")
SQL_OFFSET_S = "TRY_CAST(regexp_extract(url, '[?&#]t=(\\d+)', 1) AS INTEGER)"
SQL_MOTS = "LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1"


def fr_date(v, heure=False):
    """Date au format français. Renvoie une chaîne vide si la date manque.

    Les dates viennent de DuckDB en ISO (2026-08-15) : lisible pour une
    machine, mais ce tableau de bord est en français et 03/08 ne doit pas
    pouvoir se lire comme le 8 mars."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t):
        return ""
    return t.strftime("%d/%m/%Y %H:%M" if heure else "%d/%m/%Y")


# Format de date des colonnes horodatées d'un st.dataframe (syntaxe Moment.js,
# pas strftime -- c'est le composant JavaScript qui rend la cellule).
FMT_DATE_TABLE = "DD/MM/YYYY HH:mm"

# La collecte n'a atteint son regime qu'a cette date : avant, les sources
# entraient une a une et les journees sont quasi vides. Les garder ecrasait la
# courbe de volume contre l'axe. Le filtre ne s'applique qu'a ce graphique --
# les compteurs et les analyses portent bien sur toute la periode choisie.
DEBUT_SERIE = date(2026, 8, 11)

# Libelle court d'une nature de contenu (colonne source_kind).
UNITE_NATURE = {"press": "Presse", "tv": "Télévision", "youtube": "YouTube",
                "telegram": "Telegram", "vk": "VK"}

# Couleur par nature de contenu. Fixee ici plutot que laissee au cycle par
# defaut de Plotly : la frise et les graphiques empiles doivent donner la meme
# couleur au meme support d'un onglet a l'autre.
NATURE_COLORS = {"press": "#4C9AFF", "tv": "#B57BFF", "telegram": "#4CC38A",
                 "youtube": "#FF6B6B", "vk": "#FFA94D"}


def cols_article(**extra):
    """column_config commun aux tableaux d'articles : lien cliquable et date
    au format français. Sans DatetimeColumn, Streamlit rend l'horodatage
    DuckDB en ISO."""
    cfg = {"url": st.column_config.LinkColumn("Lien"),
           "published_at": st.column_config.DatetimeColumn(
               "Publié le", format=FMT_DATE_TABLE)}
    cfg.update(extra)
    return cfg


# Axe binaire, par-dessus les quatre familles. « independant » ne comptait
# qu'une source pour 102 documents : comme catégorie intermediaire il ne
# separait rien, et une pondération à parts égales lui donnait un quart du
# résultat. Regroupe en deux blocs, il rejoint le camp non aligné.
#
# Dénomination reprise de la littérature plutôt qu'inventée : sur les treize
# papiers de Bibliography/, « state média » (42 occurrences), « pro-government »
# (26), « pro-Kremlin » (23) et « state-controlled » (21) désignent le premier
# bloc. Pour le second, « independent média » (6) l'emporte largement sur
# « anti-Kremlin » (1) -- ces médias se définissent par leur indépendance, pas
# par leur opposition.
BLOC_SQL = ("CASE WHEN {a}.type_media IN ('etat', 'para_etat') "
            "THEN 'aligne_etat' ELSE 'independant_exil' END")
BLOC_LABEL = {"aligne_etat": "Aligné sur l'État",
              "independant_exil": "Indépendant / exil"}

# Schémas de pondération. Le corpus est un échantillon de commodité : on
# collecte ce qui est collectable, pas un tirage représentatif. Trois biais s'y
# superposent -- la composition (73 % de segments pro-Kremlin contre 26 %
# d'exil), la domination de volume (un rapport de 1 à 85 entre sources d'une
# même famille), et l'incommensurabilité des unités. Pondérer ne créé aucune
# information : cela redistribue le poids des documents pour répondre à une
# question précise, au prix d'une variance plus grande -- que l'on chiffre par
# la taille effective d'échantillon.
PONDERATIONS = {
    "Brut": ("Ce que contient le corpus. Chaque document compte pour un : "
             "les sources prolifiques dominent.", "CAST(1.0 AS DOUBLE)"),
    "Un média = une voix": (
        "Chaque source pèse autant, quel que soit son volume. Répond à "
        "« que disent les médias suivis ? » plutôt qu'a « qu'a-t-on lu ? ».",
        "1.0 / CAST(COUNT(*) OVER (PARTITION BY {a}.source_name) AS DOUBLE)"),
    "Familles a parts égales": (
        "État, para-etat, independant et exil pèsent autant. Répond à "
        "« qu'est-ce qui sépare les camps ? », en neutralisant leur poids "
        "respectif dans la collecte.",
        "1.0 / CAST(COUNT(*) OVER (PARTITION BY COALESCE({a}.type_media, 'inconnu')) AS DOUBLE)"),
    "Deux blocs à parts égales": (
        "Médias alignés sur l'État d'un côté, independants et en exil de "
        "l'autre, à 50/50. La comparaison la plus lisible entre les deux "
        "camps, et celle qui évite qu'une famille minuscule prenne le quart "
        "du résultat.",
        "1.0 / CAST(COUNT(*) OVER (PARTITION BY CASE WHEN {a}.type_media "
        "IN ('etat', 'para_etat') THEN 'aligne_etat' ELSE 'independant_exil' "
        "END) AS DOUBLE)"),
    "Familles égales, médias égaux": (
        "Les deux corrections à la fois : chaque source pèse autant au sein "
        "de sa famille, et chaque famille autant que les autres.",
        "1.0 / CAST(COUNT(*) OVER (PARTITION BY {a}.source_name) * "
        "COUNT(DISTINCT {a}.source_name) OVER "
        "(PARTITION BY COALESCE({a}.type_media, 'inconnu')) AS DOUBLE)"),
}


def poids_sql(schema, alias="a"):
    """Expression SQL du poids d'un document. A placer dans une CTE : les
    fonctions de fenêtre ne s'imbriquent pas dans une agrégation."""
    return PONDERATIONS[schema][1].format(a=alias)


def taille_effective(poids):
    """Taille effective d'échantillon (Kish) : (somme w)^2 / somme(w^2).

    Ce que la pondération coute. Si un petit groupe reçoit un poids énorme,
    la moyenne pondérée repose en pratique sur peu de documents, et le nombre
    brut de lignes ne le dit pas."""
    import numpy as _np
    w = _np.asarray(poids, dtype=float)
    w = w[w > 0]
    if not len(w):
        return 0
    return float(w.sum() ** 2 / (w ** 2).sum())


def col_entier(serie):
    """Colonne d'entiers pour st.dataframe, case vide quand la donnée manque.

    Streamlit affiche « None » pour toute valeur nulle, quel que soit le type
    -- Int64, objet ou flottant, et y compris avec un NumberColumn (vérifié
    sur cette version). Seule une chaîne vide rend une case vide. On n'y passe
    donc que les colonnes qui peuvent manquer -- la durée et les vues, absentes
    pour la presse et Telegram : le tri d'une colonne texte est alphabetique,
    ce qui n'a pas de sens pour un nombre, et les autres colonnes restent
    numeriques et triables."""
    return serie.map(lambda v: "" if pd.isna(v) else f"{int(v):,}".replace(",", " "))

CHART_FONT = 16
AXIS_FONT = 15
LEGEND_FONT = 14
TABLE_FONT = 14

TARGETS = [
    "ukraine", "etats_unis", "union_europeenne", "otan",
    "allemagne", "france", "pays_baltes", "chine", "inde",
    "brics_global_south", "georgie", "opposition_russe",
    "kazakhstan", "armenie", "moldavie", "iran",
]
TARGET_LABELS = {
    "ukraine": "Ukraine", "etats_unis": "États-Unis",
    "union_europeenne": "UE", "otan": "OTAN",
    "allemagne": "Allemagne", "france": "France",
    "pays_baltes": "Pays baltes", "chine": "Chine",
    "inde": "Inde",
    "brics_global_south": "BRICS / Global South",
    "georgie": "Georgie", "opposition_russe": "Opposition russe en exil",
    "kazakhstan": "Kazakhstan", "armenie": "Armenie",
    "moldavie": "Moldavie / Transnistrie", "iran": "Iran",
}
# Positions d'affichage, pas des capitales exactes : sur une carte du monde,
# huit acteurs europeens tiennent dans un mouchoir et leurs étiquettes se
# recouvrent. Les entités non geographiques (UE, OTAN, BRICS, opposition en
# exil) sont de toute façon symboliques ; les pays sont écartés juste assez
# pour rester reconnaissables.
TARGET_COORDS = {
    "ukraine": (50.45, 30.52), "etats_unis": (38.91, -77.04),
    "union_europeenne": (47.0, 8.0),      # au sud de Bruxelles, pour dégager
    "otan": (45.0, -30.0),                # en plein Atlantique : l'alliance
                                          # n'est pas un pays, et ca dégage l'Europe
    "allemagne": (52.52, 13.40), "france": (46.5, -1.5),
    "pays_baltes": (57.5, 24.1), "chine": (39.90, 116.41),
    "inde": (28.61, 77.21),
    "brics_global_south": (-15.79, -47.88),   # Brasilia
    "georgie": (43.5, 41.0), "armenie": (38.5, 46.0),
    "opposition_russe": (63.0, 10.0),         # hub d'exil balte, écarté au nord
    "kazakhstan": (51.17, 71.45),
    "moldavie": (43.0, 26.0), "iran": (32.0, 54.0),
}

# Côté ou poser l'étiquette de chaque bulle. Place a la main plutôt que
# calcule : ces seize acteurs ne bougent pas, et huit d'entre eux se serrent
# sur l'Europe -- les faire rayonner vers l'exterieur est la seule façon de
# les rendre tous lisibles. Le défaut vaut pour les acteurs isoles.
TARGET_TEXTPOS = {
    "otan": "middle left", "france": "bottom left",
    "union_europeenne": "bottom center", "allemagne": "top left",
    "opposition_russe": "top center", "pays_baltes": "middle right",
    "ukraine": "middle right", "moldavie": "bottom center",
    "georgie": "bottom right", "armenie": "bottom right",
    "kazakhstan": "top center", "iran": "bottom center",
}

# Trois libelles sont trop longs pour une carte : centres sur leur bulle, ils
# debordent sur les voisins. Version courte ici seulement -- ailleurs (menus,
# graphiques) le nom complet reste plus informatif.
TARGET_LABELS_CARTE = {
    "moldavie": "Moldavie", "opposition_russe": "Opposition en exil",
    "brics_global_south": "BRICS",
}

# Vocabulaire de cadrage kremlinien, d'après la littérature sur la propagande
# russe (Field et al. 2018, Pizzolo 2020). Détection lexicale simple : on
# mesure la PRÉSENCE du terme, pas l'adhesion -- un média independant peut
# citer ou critiquer ces mêmes termes.
FRAMING_TERMS = {
    "Monde russe": r"русск(ий|ого|ому|им|ом) мир",
    "Étranger proche": r"ближн(ее|его|ему|им|ем) зарубежь",
    "Régime de Kiev": r"киевск(ий|ого|ому|им|ом) режим",
    "Dénazification": r"денацифи",
    "Démilitarisation": r"демилитариз",
    "Russophobie": r"русофоб",
    "Neonazisme / Bandera": r"неонацист|необандер|бандеровц|бандеровск",
    "Occident collectif (lexical)": r"коллективн(ый|ого|ому|ым|ом) запад",
    "Genocide du Donbass": r"геноцид.{0,15}донбасс",
    "Junte illegitime": r"хунт|нелегитимн",
}

# Indicateurs de suivi : contrairement au vocabulaire de cadrage ci-dessus,
# ce ne sont pas des marqueurs de propagande mais des thermometres. Ils sont
# suivis en permanence même a bas bruit, là où le clustering BERTopic ne
# forme un thème que si le sujet devient assez dense pour émerger.
INDICATOR_TERMS = {
    # "повестка" seul veut aussi dire "ordre du jour" (повестка дня), très
    # frequent en actu politique. DuckDB utilise RE2, qui ne supporte pas le
    # lookahead négatif : on ne peut pas exclure "повестка дня" directement,
    # donc on ne garde "повестка" que dans ses collocations militaires.
    "Mobilisation": (r"мобилизац|уклонист|военкомат|призывник|"
                     r"повестк[а-я]* в военкомат|электронн[а-я]* повестк|"
                     r"вручил[а-я]* повестк|реестр повесток"),
    "Signaux de negociation": (r"переговор|перемири|мирн[а-я]* план|"
                               r"урегулировани|прекращени[ея] огня"),
    "Stress économique interne": (r"дефицит бюджет|бюджетн[а-я]* дефицит|"
                                  r"инфляц|девальвац"),
}

# Vue unifiée pour l'onglet Cadrage (les deux familles se mesurent
# de la même façon, seule leur lecture differe).
LEXICAL_CATEGORY = {
    **{k: "Cadrage (propagande)" for k in FRAMING_TERMS},
    **{k: "Indicateur de suivi" for k in INDICATOR_TERMS},
}
ALL_LEXICAL_TERMS = {**FRAMING_TERMS, **INDICATOR_TERMS}

STANCE_COLORS = {
    "pro": "#4C9AFF", "anti": "#FF6B6B", "neutre": "#A0A0A0", "non_concerne": "#3A3A3A",
}
SOURCE_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
TOP_PALETTE = ["#4C9AFF", "#FF6B6B", "#FFD93D", "#6BCB77", "#C77DFF"]


# Chrome des graphiques. Ces valeurs doivent rester coherentes avec le thème
# de .streamlit/config.toml : les figures Plotly ne heritent pas du thème
# Streamlit, il faut les habiller à la main.
GRID = "#252B38"      # grille : lisible mais en retrait des données
AXIS = "#38404F"      # ligne d'axe, un cran plus marquée que la grille
TEXT = "#E6E8EB"
MUTED = "#98A2B3"


def _unified_hover(fig):
    """Mode de survol groupe adapte à la figure, ou None si elle ne s'y prete
    pas.

    Le survol groupe suppose que les points compares partagent un axe. C'est
    vrai des barres empilées et des séries temporelles, faux d'une carte, d'un
    camembert ou d'un nuage de points. Et l'axe partage n'est pas toujours x :
    sur les barres horizontales (graphe des thèmes), c'est y -- s'y tromper
    regrouperait par valeur au lieu de regrouper par thème.
    """
    types = {(t.type or "scatter") for t in fig.data}
    if not types or not types <= {"bar", "scatter", "scattergl"}:
        return None
    shared_axes = set()
    for t in fig.data:
        if (t.type or "scatter") == "bar":
            shared_axes.add("y" if getattr(t, "orientation", None) == "h" else "x")
        else:
            if "lines" not in (getattr(t, "mode", None) or "lines"):
                return None
            shared_axes.add("x")
    if len(shared_axes) != 1:
        return None
    return f"{shared_axes.pop()} unified"


def _axe_temporel(fig):
    """L'axe des abscisses porte-t-il des dates ? Lu sur les données plutôt que
    sur le type d'axe déclaré : Plotly ne fixe `xaxis.type` qu'au rendu."""
    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None or len(x) == 0:
            continue
        return isinstance(x[0], (np.datetime64, pd.Timestamp, datetime, date))
    return False


def style(fig, height=None):
    fig.update_layout(
        font=dict(size=CHART_FONT, color=TEXT),
        legend=dict(font=dict(size=LEGEND_FONT), title_font=dict(size=LEGEND_FONT)),
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        # Fond transparent plutôt qu'une couleur en dur : la figure se pose
        # sur le fond de la page au lieu d'y découper un rectangle -- visible
        # des que Streamlit ajuste sa teinte (cartes, expanders, colonnes).
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#161A23", bordercolor=GRID,
                        font=dict(size=LEGEND_FONT, color=TEXT)),
    )
    # Un seul cadre au survol pour toutes les séries d'un même point : sur les
    # graphes empilés (volume par source, thèmes presse/telegram), comparer
    # série par série au survol était impraticable.
    hover = _unified_hover(fig)
    if hover:
        fig.update_layout(hovermode=hover)
    if height:
        fig.update_layout(height=height)
    axis_kw = dict(
        tickfont=dict(size=AXIS_FONT, color=MUTED),
        title_font=dict(size=AXIS_FONT, color=MUTED),
        gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
    )
    fig.update_xaxes(**axis_kw)
    fig.update_yaxes(**axis_kw)

    # Plotly date ses graduations en anglais (« Aug 9 ») et sa locale française
    # n'est pas embarquée par le composant Streamlit. Un format numérique
    # explicite évite d'en dépendre : jour/mois sur l'axe, date complète au
    # survol. On ne l'appliqué qu'aux axes réellement temporels -- sur un axe
    # numérique, un motif en %d serait interprete comme un format de nombre.
    if _axe_temporel(fig):
        fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
    return fig


st.set_page_config(
    page_title="Russian Media Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Couleurs, rayons, bordures et tailles de titre sont définis dans
# .streamlit/config.toml : Streamlit les appliqué lui-même à ses composants.
# Ne reste ici que la largeur de page, hors de portée du système de thème.
st.markdown("""
<style>
/* Au-dela de ~1650px les tableaux larges s'etirent et deviennent illisibles.
   Le retrait haut degage la barre d'outils flottante (Deploy, menu), qui sinon
   rogne le titre. */
.block-container{padding-top:4rem;padding-bottom:3rem;max-width:1650px;}

/* Dix onglets en file indifferenciee ne se lisent pas. Un filet avant le 2e,
   le 6e et le 8e detache les quatre familles : panorama | analyses de contenu |
   qui parle | lecture du corpus. st.tabs ne permet pas de grouper, d'ou le
   reperage par position -- a corriger si l'ordre des onglets change. */
.stTabs [role="tablist"] > [role="tab"]:nth-child(2),
.stTabs [role="tablist"] > [role="tab"]:nth-child(6),
.stTabs [role="tablist"] > [role="tab"]:nth-child(8){
  margin-left:14px;padding-left:14px;
  border-left:1px solid rgba(255,255,255,.14);}
</style>
""", unsafe_allow_html=True)

st.title("Russian Media Monitor")
st.caption("Veille des médias russophones — presse, Telegram, YouTube, "
           "télévision, VK")

if not DB_PATH.exists() and not SNAPSHOT_GZ.exists():
    st.warning(
        "Aucune base. En local, lancez `python update.py` ; pour un "
        "déploiement, publiez `data/snapshot.duckdb.gz` "
        "(`python scripts/maintenance/export_snapshot.py`).")
    st.stop()

@st.cache_resource(show_spinner="Décompression de l'instantané...")
def _decompresser_snapshot():
    """Prepare l'instantané publié pour la lecture, et renvoie son chemin.

    En local la base complète est la et rien de tout ceci ne sert. Sur un
    hebergeur, seul l'instantané compresse est versionne (13 Mo contre 3,5 Go) :
    DuckDB ne sachant pas lire un .gz, on le détend une fois par session dans un
    répertoire temporaire. `cache_resource` garantit que ca n'arrive qu'une fois
    même si plusieurs visiteurs arrivent en même temps.
    """
    import gzip
    import shutil
    import tempfile

    cible = Path(tempfile.gettempdir()) / "rmm-snapshot.duckdb"
    if not cible.exists() or cible.stat().st_mtime < SNAPSHOT_GZ.stat().st_mtime:
        with gzip.open(SNAPSHOT_GZ, "rb") as f, open(cible, "wb") as g:
            shutil.copyfileobj(f, g)
    return cible


def ouvrir_base(essais=3, attente=1.5):
    """Connexion en lecture, avec quelques tentatives.

    DuckDB n'admet qu'un écrivain OU plusieurs lecteurs : tant qu'une analyse
    écrit, l'ouverture échoué. C'est normal, mais sans ce garde-fou Streamlit
    affichait une trace Python, ce qui ressemble à une base cassée.
    """
    # Base complète si elle est la, instantané publié sinon.
    chemin = DB_PATH if DB_PATH.exists() else (
        _decompresser_snapshot() if SNAPSHOT_GZ.exists() else DB_PATH)
    derniere = None
    for reste in range(essais - 1, -1, -1):
        try:
            return duckdb.connect(str(chemin), read_only=True)
        except Exception as e:
            # On reessaie quelle que soit l'erreur : distinguer le verrou des
            # autres pannes supposerait de lire le texte du message, qui est
            # traduit dans la langue du système. Trois tentatives coutent
            # trois secondes dans le pire des cas.
            derniere = e
            if reste:
                time.sleep(attente)
    return derniere


_base = ouvrir_base()
if not isinstance(_base, duckdb.DuckDBPyConnection):
    st.warning(
        "**Analyse en cours d'ecriture.** La base est momentanement réservée "
        "par une analyse (collecte, thèmes, sentiment...). Le tableau de bord "
        "la relira des qu'elle aura rendu la main -- rien n'est perdu et rien "
        "n'est casse : DuckDB n'autorise qu'un écrivain à la fois."
    )
    if st.button("Reessayer"):
        st.rerun()
    with st.expander("Détail technique"):
        st.code(str(_base) if _base else "verrou toujours pris", language="text")
    st.stop()
conn = _base
tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}

# Entités et posture-par-article ont été retirées de la routine : leurs
# tables restent en base mais plus rien ne les lit.
has_target_sent = "article_target_sentiment" in tables
has_topics = "topics" in tables and "article_topics" in tables
# Analyses issues des travaux de recherche (cf. Bibliography/). Chacune est
# facultative : sans sa table, la section correspondante disparait.
has_divergence = "lexical_divergence" in tables
has_techniques = "article_techniques" in tables
has_topic_quality = "topic_quality" in tables
has_geo = "entity_geo" in tables


date_start = st.session_state.get("global_date_start", date(2026, 5, 1))
date_end = st.session_state.get("global_date_end", date.today())

with st.sidebar:
    st.header("Filtres")

    # Période et recherche en premier : les deux seuls filtres manipules a
    # chaque session. Le reste est replie -- 45 étiquettes de sources noyaient
    # tout, et les pastilles sous le titre rappellent ce qui est filtre.
    st.markdown("**Période**")
    c_date_start, c_date_end = st.columns(2)
    with c_date_start:
        date_start = st.date_input("Articles du", value=date_start, key="global_date_start")
    with c_date_end:
        date_end = st.date_input("au", value=date_end, key="global_date_end")

    st.markdown("**Recherche**")
    keyword = st.text_input(
        "Requête booléenne", label_visibility="collapsed",
        placeholder="ex : (sanctions OU иноагент) ET NON sport",
        help="ET / OU / SAUF (ou AND/OR/NOT, & | -). Guillemets pour une "
             "expression exacte, parentheses pour grouper.",
    )
    kw_default_op = st.radio(
        "Opérateur implicite entre deux mots", ["ET", "OU"],
        horizontal=True, index=0,
    )

    # ORDER BY sur tous les DISTINCT, et une clé explicite par widget.
    # Sans tri, DuckDB renvoie ces valeurs dans l'ordre de sa table de
    # hachage, qui change des que les données changent ; Streamlit voyait
    # alors une liste d'options différente d'un rerun a l'autre et remettait
    # la sélection à sa valeur par défaut -- d'où des filtres qui « se
    # remettent tout seuls ».
    with st.expander("Corpus", expanded=False):
        sources_all = conn.execute(
            "SELECT DISTINCT source_name FROM articles ORDER BY source_name"
        ).df()["source_name"].tolist()
        selected_sources = st.multiselect("Sources", sources_all, default=sources_all,
                                          key="f_sources")
        langs = conn.execute(
            "SELECT DISTINCT language FROM articles WHERE language IS NOT NULL "
            "ORDER BY language"
        ).df()["language"].tolist()
        # Par défaut le russe seul : l'outil suit les médias russophones, et
        # les 0,3 % restants sont les éditions traduites de Mediazona
        # (anglais, espagnol, portugais, polonais), hors sujet ici.
        default_langs = ["ru"] if "ru" in langs else langs
        selected_langs = st.multiselect(
            "Langues", langs, default=default_langs, key="f_langs",
            help="Le russe est seul coche par défaut. Les autres langues sont "
                 "des éditions traduites (Mediazona) ou des erreurs de "
                 "détection sur du cyrillique.",
        )
        types_media = conn.execute(
            "SELECT DISTINCT type_media FROM articles WHERE type_media IS NOT NULL "
            "ORDER BY type_media"
        ).df()["type_media"].tolist()
        selected_types_media = st.multiselect("Type de média", types_media,
                                              default=types_media, key="f_types")
        source_kinds = conn.execute(
            "SELECT DISTINCT source_kind FROM articles WHERE source_kind IS NOT NULL "
            "ORDER BY source_kind"
        ).df()["source_kind"].tolist()
        selected_source_kinds = st.multiselect(
            "Nature du contenu", source_kinds, default=source_kinds,
            help="Telegram : canaux officiels et milbloggers, texte brut du "
                 "post. YouTube : sous-titres de vidéo en segments d'environ "
                 "2000 signes -- ces chaînes sont toutes d'opposition en exil. "
                 "TV : transcription Whisper du journal de 21 h et des grands "
                 "talk-shows politiques, qui reequilibre le volet vidéo côté "
                 "pouvoir. VK : publications des communautes des grands "
                 "médias sur le premier réseau social du pays. "
                 "Voir l'onglet Alignement.",
            key="f_kinds",
        )
        statuts_legal = conn.execute(
            "SELECT DISTINCT statut_legal_ru FROM articles WHERE statut_legal_ru IS NOT NULL "
            "ORDER BY statut_legal_ru"
        ).df()["statut_legal_ru"].tolist()
        selected_statuts_legal = st.multiselect(
            "Statut légal (RU)", statuts_legal, default=statuts_legal, key="f_statuts")
        st.caption(
            "Statuts légaux : agent_etranger, organisation_indesirable, aucun. "
            "Plusieurs médias en exil sont interdits d'exploitation en Russie "
            "(cela n'empeche pas leur collecte depuis l'étranger)."
        )

    st.subheader("Affichage")
    _grains = {"Jour": "day", "Semaine": "week", "Mois": "month"}
    _grain_choisi = st.radio(
        "Pas de temps des courbes", list(_grains), index=0, horizontal=True,
        key="f_grain",
        help="S'appliqué a toutes les courbes d'évolution. Le jour montre les "
             "pics et les reprises d'un événement ; la semaine lisse le creux "
             "du week-end, très marque dans la presse d'agence.")
    GRAIN = _grains[_grain_choisi]
    GRAIN_LABEL = _grain_choisi

    PONDERATION = st.selectbox(
        "Pondération", list(PONDERATIONS), index=0, key="f_ponderation",
        help="Corrige la composition du corpus. Voir la note sous le "
             "graphique des thèmes pour ce que chaque choix signifie.")
    st.caption(PONDERATIONS[PONDERATION][0])

where = ["1 = 1"]
params: list = []
if date_start and date_end:
    where.append("(published_at >= ? AND published_at < ? + INTERVAL 1 DAY)")
    params.append(date_start)
    params.append(date_end)
elif date_start:
    where.append("published_at >= ?")
    params.append(date_start)
elif date_end:
    where.append("published_at < ? + INTERVAL 1 DAY")
    params.append(date_end)
if selected_sources:
    where.append(f"source_name IN ({','.join(['?'] * len(selected_sources))})")
    params.extend(selected_sources)
if selected_langs:
    where.append(f"language IN ({','.join(['?'] * len(selected_langs))})")
    params.extend(selected_langs)
if selected_types_media:
    where.append(f"type_media IN ({','.join(['?'] * len(selected_types_media))})")
    params.extend(selected_types_media)
if selected_source_kinds:
    where.append(f"source_kind IN ({','.join(['?'] * len(selected_source_kinds))})")
    params.extend(selected_source_kinds)
if selected_statuts_legal:
    where.append(f"statut_legal_ru IN ({','.join(['?'] * len(selected_statuts_legal))})")
    params.extend(selected_statuts_legal)
if keyword and keyword.strip():
    kw_sql, kw_params = build_keyword_filter(
        keyword, default_op=("AND" if kw_default_op == "ET" else "OR"))
    if kw_sql:
        where.append(kw_sql)
        params.extend(kw_params)
WHERE = " AND ".join(where)

# Même filtres que WHERE, sans la borne de date -- utilise par l'onglet
# Signaux qui a besoin de définir ses propres fenêtres temporelles (récente
# vs référence) par-dessus les filtres source/langue/mots-clés.
where_nodate = ["1 = 1"]
params_nodate: list = []
if selected_sources:
    where_nodate.append(f"source_name IN ({','.join(['?'] * len(selected_sources))})")
    params_nodate.extend(selected_sources)
if selected_langs:
    where_nodate.append(f"language IN ({','.join(['?'] * len(selected_langs))})")
    params_nodate.extend(selected_langs)
if selected_types_media:
    where_nodate.append(f"type_media IN ({','.join(['?'] * len(selected_types_media))})")
    params_nodate.extend(selected_types_media)
if selected_source_kinds:
    where_nodate.append(f"source_kind IN ({','.join(['?'] * len(selected_source_kinds))})")
    params_nodate.extend(selected_source_kinds)
if selected_statuts_legal:
    where_nodate.append(f"statut_legal_ru IN ({','.join(['?'] * len(selected_statuts_legal))})")
    params_nodate.extend(selected_statuts_legal)
if keyword and keyword.strip():
    kw_sql, kw_params = build_keyword_filter(
        keyword, default_op=("AND" if kw_default_op == "ET" else "OR"))
    if kw_sql:
        where_nodate.append(kw_sql)
        params_nodate.extend(kw_params)
WHERE_NODATE = " AND ".join(where_nodate)

# --- Aperçu de recherche (compteur + liste), affiche dans la sidebar ---
with st.sidebar:
    if keyword and keyword.strip():
        try:
            _n_res = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE {WHERE}", params
            ).fetchone()[0]
        except Exception as _e:
            _n_res = None
        if _n_res is None:
            st.warning("Recherche invalide, vérifiez la syntaxe.")
        elif _n_res == 0:
            st.info("Aucun article ne correspond.")
            st.caption("Astuce : essayez le mode OU, ou retirez un mot.")
        else:
            st.success(f"{_n_res} article(s) correspondent")
            with st.expander("Aperçu des articles trouves", expanded=False):
                _df_prev = conn.execute(
                    f"""SELECT published_at, source_name, title, url
                        FROM articles WHERE {WHERE}
                        ORDER BY published_at DESC NULLS LAST
                        LIMIT 50""",
                    params,
                ).df()
                for _, _r in _df_prev.iterrows():
                    _d = fr_date(_r["published_at"])
                    st.markdown(
                        f"**{_d}** - {_r['source_name']}  \n"
                        f"[{_r['title']}]({_r['url']})"
                    )
                if _n_res > 50:
                    st.caption(f"... et {_n_res - 50} autres. Affinez pour réduire.")



def with_a(c):
    return (c.replace("source_name", "a.source_name").replace("title", "a.title")
            .replace("content", "a.content").replace("language", "a.language")
            .replace("published_at", "a.published_at"))


def top5_plus_autres(df, gc, vc):
    top5 = df.groupby(gc)[vc].sum().nlargest(5).index.tolist()
    df = df.copy()
    df["display"] = df[gc].where(df[gc].isin(top5), "Autres")
    cmap = {n: TOP_PALETTE[i] for i, n in enumerate(top5)}
    cmap["Autres"] = "#6B7280"
    return df, cmap, top5 + ["Autres"]


# Périmètre courant, rappele sous le titre : les filtres vivent dans la barre
# laterale, qui peut être repliée -- sans ce bandeau, rien a l'écran ne dit
# sur quoi portent les chiffres qu'on est en train de lire.
total = conn.execute(f"SELECT COUNT(*) FROM articles WHERE {WHERE}", params).fetchone()[0]
n_src = conn.execute(
    f"SELECT COUNT(DISTINCT source_name) FROM articles WHERE {WHERE}", params
).fetchone()[0]
last_f = conn.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]

# Il n'y a pas d'identifiant de passe (fetched_at est pose ligne par ligne),
# on les reconstitue par les trous. 30 min : le plus court intervalle observe
# entre deux passes est de 37 min, et une transcription peut retarder de
# quelques minutes les insertions qui suivent.
nb_passes = conn.execute("""
    WITH t AS (SELECT DISTINCT fetched_at FROM articles),
         d AS (SELECT fetched_at - LAG(fetched_at) OVER (ORDER BY fetched_at) AS ecart
               FROM t)
    SELECT COUNT(*) FROM d WHERE ecart IS NULL OR ecart > INTERVAL 30 MINUTE
""").fetchone()[0]

# « X articles » serait faux : une émission de 2 h fait une soixantaine de
# lignes. On regroupe donc les segments sur leur unité parente, et le détail
# par nature part dans l'infobulle.
_df_unites = conn.execute(
    f"SELECT source_kind, COUNT(DISTINCT {SQL_PARENT}) AS n "
    f"FROM articles WHERE {WHERE} GROUP BY 1", params).df()
UNITE_MOT = {"press": "articles", "tv": "émissions de TV",
             "youtube": "videos", "telegram": "posts Telegram",
             "vk": "posts VK"}
_n_unites = int(_df_unites["n"].sum()) if not _df_unites.empty else 0
_detail_unites = " · ".join(
    f"{int(r['n'])} {UNITE_MOT.get(r['source_kind'], r['source_kind'])}"
    for _, r in _df_unites.sort_values("n", ascending=False).iterrows())
_total_fmt = f"{_n_unites:,}".replace(",", " ")  # espace fine insecable
# date_start/date_end peuvent être None : st.date_input rend un champ
# effaçable, et le filtre SQL plus haut prévoit déjà ce cas.
_periode = (f"{date_start:%d/%m/%Y}" if date_start else "debut") + " → " + \
           (f"{date_end:%d/%m/%Y}" if date_end else "aujourd'hui")
_chips = [f":gray-badge[**{_total_fmt}** contenus]",
          f":gray-badge[**{n_src}** sources]",
          f":gray-badge[{_periode}]"]
# On ne signale que les filtres réellement restrictifs : afficher "45 sources
# sur 45" à chaque écran serait du bruit. Les filtres actifs passent en
# couleur d'accent, le périmètre par défaut reste gris.
if len(selected_sources) < len(sources_all):
    _chips.append(f":primary-badge[sources : {len(selected_sources)}/{len(sources_all)}]")
if len(selected_source_kinds) < len(source_kinds):
    _chips.append(f":primary-badge[{' + '.join(selected_source_kinds)}]")
if len(selected_types_media) < len(types_media):
    _chips.append(f":primary-badge[média : {', '.join(selected_types_media)}]")
if keyword and keyword.strip():
    _chips.append(f":primary-badge[recherche : **{keyword.strip()}**]")
if last_f:
    _chips.append(f":gray-badge[collecte {fr_date(last_f, heure=True)}]")
st.markdown(" ".join(_chips), help=_detail_unites)

# Ordre de lecture : ce qui a change (Signaux), puis ce qui est dit (thèmes,
# sentiment, cadrage), puis qui le dit, puis la qualité des données.
# Les onglets vont par familles, separees visuellement dans la barre :
# le panorama, les quatre analyses de contenu, qui parle, puis ce qui aide a
# lire le reste (qualite de la collecte, paysage mediatique, references).
(tab_vue, tab_signaux, tab_themes, tab_sentiment, tab_cadrage,
 tab_alignement, tab_acteurs, tab_diagnostic, tab_couverture, tab_contexte,
 tab_references) = st.tabs([
    "Vue d'ensemble",
    "Signaux", "Thèmes", "Sentiment", "Cadrage",
    "Alignement", "Acteurs",
    "Diagnostic", "Couverture", "Contexte", "Références",
])

# ===== Tab Vue =================================================
with tab_vue:
    # « Articles » ne veut rien dire pour une émission de télévision : un
    # numéro de 2 h donne 60 lignes en base. Chaque nature de contenu est donc
    # comptée dans SON unité -- articles, émissions, vidéos, posts -- et les
    # segments ne servent qu'à comparer les natures entre elles.
    #
    # Unité parente, instant du segment, nombre de mots : cf. SQL_PARENT,
    # SQL_OFFSET_S et SQL_MOTS, définis en tête de fichier et partages avec le
    # panneau de couverture en bas de page.
    PARENT, OFFSET_S, MOTS = SQL_PARENT, SQL_OFFSET_S, SQL_MOTS

    df_nat = conn.execute(f"""
        WITH base AS (
            SELECT source_kind, source_name, id, {PARENT} AS parent,
                   {OFFSET_S} AS t_s, {MOTS} AS mots, view_count AS vues
            FROM articles WHERE {WHERE}
        ),
        par_parent AS (
            -- Les vues sont une propriete de la video, repetee sur chacun de
            -- ses segments : on prend le MAX par video avant de sommer, sinon
            -- une video de 60 segments compterait 60 fois son audience.
            SELECT source_kind, parent, MAX(t_s) AS fin_s, MAX(vues) AS vues
            FROM base GROUP BY 1, 2
        )
        SELECT b.source_kind AS nature,
               COUNT(DISTINCT b.parent) AS unites,
               COUNT(*) AS segments,
               SUM(b.mots) AS mots,
               COUNT(DISTINCT b.source_name) AS sources,
               (SELECT SUM(fin_s) FROM par_parent p
                 WHERE p.source_kind = b.source_kind) AS secondes,
               (SELECT SUM(vues) FROM par_parent p
                 WHERE p.source_kind = b.source_kind) AS vues
        FROM base b GROUP BY 1 ORDER BY mots DESC
    """, params).df()

    LIB = {"press": ("Articles de presse", "articles"),
           "tv": ("Émissions de télévision", "emissions"),
           "youtube": ("Vidéos YouTube", "videos"),
           "telegram": ("Publications Telegram", "posts"),
           "vk": ("Publications VK", "posts")}

    if not df_nat.empty:
        cols = st.columns(len(df_nat) + 1)
        for i, (_, r) in enumerate(df_nat.iterrows()):
            titre, unite = LIB.get(r["nature"], (r["nature"], "unites"))
            n = int(r["unites"])
            detail = f"{int(r['sources'])} sources"
            if r["nature"] in ("tv", "youtube"):
                detail = f"{int(r['segments'])} segments"
            cols[i].metric(titre, f"{n:,}".replace(",", " "), detail,
                           delta_color="off")
        # La ligne de détail n'est pas qu'informative : sans elle, cette carte
        # est plus courte que les cinq autres et la rangée se désaligné.
        cols[-1].metric("Dernière collecte",
                        last_f.strftime("%d/%m/%y") if last_f else "n/a",
                        f"{nb_passes} collectes", delta_color="off")

        with st.expander("Détail par nature de contenu", expanded=False):
            det = df_nat.copy()
            det["Nature"] = det["nature"].map(lambda k: LIB.get(k, (k, ""))[0])
            det["Type"] = det["nature"].map(lambda k: LIB.get(k, ("", "unites"))[1])
            det["Mots"] = det["mots"].astype("Int64")
            det["Segments"] = det["segments"].astype(int)
            det["Unités"] = det["unites"].astype(int)
            det["Sources"] = det["sources"].astype(int)
            # Durée et vues ne sont plus affichées ici : elles n'existent que
            # pour la vidéo et le panneau de couverture, en bas de page, les
            # donne par source. Elles restent dans df_nat.
            st.dataframe(
                det[["Nature", "Type", "Unités", "Segments", "Mots",
                     "Sources"]],
                width="stretch", hide_index=True,
                column_config={
                    "Mots": st.column_config.NumberColumn("Mots", format="%d"),
                })
            st.caption(
                "**Unités** : ce que l'on compte naturellement pour cette "
                "nature -- un article de presse, une émission de télévision, "
                "une vidéo, un post. **Segments** : les lignes réellement "
                "stockées et analysées ; une émission de 2 h en produit une "
                "soixantaine, un article un seul. Pour comparer les natures "
                "entre elles, c'est le nombre de **mots** qui fait foi."
            )

    df_vol = conn.execute(
        f"SELECT DATE_TRUNC('{GRAIN}', published_at) AS jour, source_name, COUNT(*) AS n "
        f"FROM articles WHERE {WHERE} AND published_at IS NOT NULL "
        f"GROUP BY 1, 2 ORDER BY 1", params).df()
    df_vol = df_vol[pd.to_datetime(df_vol["jour"]).dt.date >= DEBUT_SERIE]
    if not df_vol.empty:
        st.subheader(f"Volume par {GRAIN_LABEL.lower()}")
        df_d, cmap, order = top5_plus_autres(df_vol, "source_name", "n")
        df_agg = df_d.groupby(["jour", "display"], as_index=False)["n"].sum()
        fig = px.bar(df_agg, x="jour", y="n", color="display",
                     category_orders={"display": order}, color_discrete_map=cmap,
                     labels={"jour": "Date", "n": "Segments", "display": ""})
        fig.update_layout(barmode="stack", legend_title_text="")
        style(fig, 450)
        st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        st.caption(
            "En segments, l'unité réellement stockée et analysée : une "
            "émission de télévision de 2 h en produit une soixantaine, un "
            "article un seul. Une journée où la télévision pèse lourd n'est "
            "donc pas une journée où elle a plus parlé que la presse. "
            f"La courbe démarre au {DEBUT_SERIE:%d/%m/%Y}, date à laquelle la "
            "collecte a atteint son régime.")


# ===== Tab Thèmes ==============================================
with tab_themes:
    aw = with_a(WHERE)
    if not has_topics:
        st.info("Lancez : python analyze_topics.py")
    else:
        st.caption(
            "Thèmes décidés directement par les articles (clustering BERTopic "
            "sur une fenêtre glissante), pas une liste écrite à la main : un "
            "thème sans article récent devient inactif sans perdre son "
            "historique, et peut redevenir actif si le sujet revient."
        )
        # --- Frise des journées ---------------------------------------------
        # Le clustering travaille journée par journée : pouvoir en isoler une
        # rend deux journées comparables, ce que la période globale de la barre
        # latérale ne permet pas. Un menu déroulant aurait suffi à filtrer,
        # mais il ne dit rien du corpus -- la frise montre en plus le volume de
        # chaque journée et sa composition par support, et se clique.
        #
        # Le filtre porte sur la date de publication et non sur
        # article_topics.run_date : tout ce bloc partage la clause _aw, qui sert
        # aussi aux dénominateurs des pourcentages et à la pondération. Ne
        # filtrer que les thèmes aurait laissé les dénominateurs sur la période
        # entière, et les parts n'auraient plus sommé à 100 %.
        # DEBUT_SERIE, comme la courbe de volume : avant cette date la collecte
        # montait en charge et les journees sont trop maigres pour que leur
        # clustering veuille dire quelque chose.
        df_frise = conn.execute(
            """SELECT CAST(a.published_at AS DATE) AS jour,
                      COALESCE(a.source_kind, 'press') AS support,
                      COUNT(*) AS n
               FROM article_topics at_
               JOIN articles a ON a.id = at_.article_id
               WHERE at_.topic_key <> -1 AND at_.run_date >= ?
               GROUP BY 1, 2 ORDER BY 1""", [DEBUT_SERIE]).df()

        _aw, _pa, _sel_j = aw, params, None
        if not df_frise.empty:
            _jours_f = sorted(df_frise["jour"].unique())
            _par_jour = df_frise.groupby("jour")["n"].sum()

            # Une cle versionnee plutot qu'une remise a zero de l'etat du
            # widget : c'est le seul moyen fiable de vider la selection d'un
            # graphique Plotly, dont l'evenement survit au rerun.
            st.session_state.setdefault("frise_v", 0)
            _c_titre, _c_reset = st.columns([4, 1])
            _c_titre.markdown("**Journées regroupées**")
            if _c_reset.button("Toute la période", width="stretch",
                               key="frise_reset"):
                st.session_state.frise_v += 1
            _cle_frise = f"frise_jours_{st.session_state.frise_v}"

            # Selection lue AVANT de dessiner : la barre retenue doit ressortir
            # des le meme rendu, sinon il faut deux clics pour la voir.
            _ev = st.session_state.get(_cle_frise) or {}
            _pts = ((_ev.get("selection") or {}).get("points")) or []
            _idx = _pts[0].get("point_index") if _pts else None
            if _idx is not None and 0 <= _idx < len(_jours_f):
                _sel_j = _jours_f[_idx]

            fig_f = go.Figure()
            for _sup, _lib in UNITE_NATURE.items():
                _serie = (df_frise[df_frise["support"] == _sup]
                          .set_index("jour")["n"].reindex(_jours_f, fill_value=0))
                if _serie.sum() == 0:
                    continue
                fig_f.add_bar(
                    x=[f"{j:%d/%m}" for j in _jours_f], y=_serie.values,
                    name=_lib, marker_color=NATURE_COLORS.get(_sup, MUTED),
                    marker_opacity=[1.0 if (_sel_j is None or j == _sel_j) else 0.22
                                    for j in _jours_f],
                    hovertemplate="%{x} &mdash; " + _lib +
                                  " : %{y} documents<extra></extra>")
            fig_f.update_layout(barmode="stack", showlegend=True,
                                # Sans traceorder, la legende horizontale
                                # sort a l'envers de l'empilement.
                                legend=dict(orientation="h", y=1.25, x=0,
                                            title_text="", traceorder="normal"),
                                bargap=0.28)
            fig_f.update_yaxes(visible=False)
            # type="category" impose : sans lui Plotly lit « 14/08 » comme
            # une date, espace les etiquettes selon la largeur et ajoute
            # un 21/08 qui n'existe pas dans les donnees.
            fig_f.update_xaxes(type="category", tickangle=0, showgrid=False)
            style(fig_f, 190)
            st.plotly_chart(fig_f, width="stretch", key=_cle_frise,
                            on_select="rerun", selection_mode="points")

            if _sel_j is None:
                st.caption(
                    f"{len(_jours_f)} journées, "
                    f"{int(_par_jour.sum()):,}".replace(",", "\u202f")
                    + " documents classés. Cliquez une barre pour n'afficher "
                      "que cette journée.")
            else:
                _aw = f"{aw} AND CAST(a.published_at AS DATE) = ?"
                _pa = list(params) + [_sel_j]
                st.caption(
                    f"Filtré sur le **{fr_date(_sel_j)}** "
                    f"({int(_par_jour.get(_sel_j, 0))} documents classés). "
                    "« Toute la période » pour revenir à l'ensemble.")

        df_t = conn.execute(
            f"""SELECT t.topic_key, t.label, t.top_words, t.active,
                       t.first_seen, t.last_seen, COUNT(at_.article_id) AS n
            FROM topics t LEFT JOIN article_topics at_ ON at_.topic_key = t.topic_key
            LEFT JOIN articles a ON a.id = at_.article_id
            WHERE t.topic_key != -1 AND (at_.article_id IS NULL OR {_aw})
            GROUP BY t.topic_key, t.label, t.top_words, t.active, t.first_seen, t.last_seen
            ORDER BY n DESC""",
            _pa).df()
        df_t["pct"] = (df_t["n"] / df_t["n"].sum() * 100).round(1) if df_t["n"].sum() else 0.0
        nz = df_t[(df_t["n"] > 0) & (df_t["active"])].copy()
        archived = df_t[(df_t["n"] > 0) & (~df_t["active"])].sort_values("last_seen", ascending=False)

        st.subheader("Répartition des thèmes actifs")
        c_left, c_right = st.columns([2, 1])
        with c_left:
            if nz.empty:
                st.caption("Aucun thème actif sur la période/filtres choisis.")
            else:
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    unit_choice = st.radio(
                        "Compter en", ["Segments", "Volume de texte"],
                        horizontal=True, index=0, key="theme_unit",
                        help="Un segment = une ligne analysée : un article de "
                             "presse en vaut un, une émission de télévision une "
                             "soixantaine. Le volume de texte corrige ce biais "
                             "en ponderant par la longueur -- c'est la mesure a "
                             "prendre pour comparer des natures différentes.",
                    )
                with cm2:
                    scale_choice = st.radio(
                        "Afficher en", ["Nombre", "% du corpus"],
                        horizontal=True, index=1, key="theme_scale",
                        help="Le pourcentage rapporté au corpus entier tel que "
                             "filtre à gauche, pas seulement aux articles classes.",
                    )
                with cm3:
                    split_choice = st.selectbox(
                        "Répartir par",
                        ["Nature du contenu", "Bloc", "Type de média", "Média",
                         "Aucune"],
                        index=0, key="theme_split",
                        help="Décompose chaque barre. « Média » distingue les "
                             "sources une à une : lisible surtout après avoir "
                             "restreint la liste des sources à gauche.",
                    )

                SPLIT_COL = {"Nature du contenu": "COALESCE(a.source_kind, 'press')",
                             "Bloc": BLOC_SQL.format(a="a"),
                             "Type de média": "COALESCE(a.type_media, 'inconnu')",
                             "Média": "a.source_name",
                             "Aucune": "'Tous'"}[split_choice]

                # Dénominateur = corpus total selon les filtres actuels (pas
                # seulement les articles classifies) : la barre représente une
                # vraie part du corpus, pas une part entre thèmes seulement.
                denom_n, denom_chars = conn.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(LENGTH(a.content)), 0) "
                    f"FROM articles a WHERE {_aw}", _pa).fetchone()
                denom_n = float(denom_n or 1)
                denom_chars = float(denom_chars or 1)

                # Le poids se calcule sur le corpus FILTRE entier, pas sur les
                # seuls segments classes : sinon un thème absent d'une famille
                # ferait varier le poids des autres.
                W = poids_sql(PONDERATION)
                df_tk = conn.execute(
                    f"""WITH pesee AS (
                            SELECT a.id, {SPLIT_COL} AS decoupe,
                                   LENGTH(a.content) AS chars, {W} AS w
                            FROM articles a WHERE {_aw}
                        )
                        SELECT t.topic_key, t.label, p.decoupe,
                               CAST(SUM(p.w) AS DOUBLE) AS n,
                               CAST(COALESCE(SUM(p.w * p.chars), 0) AS DOUBLE) AS chars
                        FROM topics t
                        JOIN article_topics at_ ON at_.topic_key = t.topic_key
                        JOIN pesee p ON p.id = at_.article_id
                        WHERE t.active AND t.topic_key != -1
                        GROUP BY t.topic_key, t.label, p.decoupe""",
                    _pa).df()

                # Dénominateurs pondérés de la même façon, sinon les parts ne
                # sommeraient plus à 100 %.
                denom_n, denom_chars, n_eff = conn.execute(
                    f"""WITH pesee AS (
                            SELECT LENGTH(a.content) AS chars, {W} AS w
                            FROM articles a WHERE {_aw}
                        )
                        SELECT CAST(SUM(w) AS DOUBLE),
                               CAST(COALESCE(SUM(w * chars), 0) AS DOUBLE),
                               CAST(POWER(SUM(w), 2) / NULLIF(SUM(w * w), 0) AS DOUBLE)
                        FROM pesee""", _pa).fetchone()
                denom_n = float(denom_n or 1)
                denom_chars = float(denom_chars or 1)

                raw_col = "n" if unit_choice == "Segments" else "chars"
                denom = denom_n if unit_choice == "Segments" else denom_chars
                if scale_choice == "Nombre":
                    df_tk["val"] = df_tk[raw_col]
                    x_label = ("Nombre de segments" if unit_choice == "Segments"
                               else "Volume de texte (signes)")
                else:
                    df_tk["val"] = 100 * df_tk[raw_col] / denom
                    x_label = ("% des segments du corpus" if unit_choice == "Segments"
                               else "% du volume de texte du corpus")

                if PONDERATION != "Brut":
                    n_brut = conn.execute(
                        f"SELECT COUNT(*) FROM articles a WHERE {_aw}",
                        _pa).fetchone()[0] or 1
                    perte = 100 * (1 - (n_eff or 0) / n_brut)
                    st.caption(
                        f"**Pondération « {PONDERATION} ».** "
                        f"{PONDERATIONS[PONDERATION][0]} "
                        f"Taille effective d'échantillon : **{int(n_eff or 0)}** "
                        f"documents sur {n_brut} collectes, soit {perte:.0f} % "
                        f"de précision en moins. Pondérer ne créé pas "
                        f"d'information : cela redistribue le poids pour "
                        f"répondre à une autre question, au prix de la variance."
                    )
                    if (n_eff or 0) < 100:
                        st.warning(
                            "Taille effective inférieure à 100 : ces "
                            "proportions reposent en pratique sur trop peu de "
                            "documents pour être conclusives.")

                    # Une famille remontée à parts égales alors qu'elle repose
                    # sur une poignée de documents devient le point faible de
                    # tout le graphique : on la nomme au lieu de la noyer.
                    df_strates = conn.execute(
                        f"""WITH pesee AS (
                                SELECT CASE WHEN '{PONDERATION}' LIKE '%blocs%'
                                            THEN {BLOC_SQL.format(a="a")}
                                            ELSE COALESCE(a.type_media, 'inconnu')
                                       END AS strate,
                                       a.source_name, {W} AS w
                                FROM articles a WHERE {_aw}
                            )
                            SELECT strate, COUNT(*) AS docs,
                                   COUNT(DISTINCT source_name) AS sources,
                                   100 * SUM(w) / (SELECT SUM(w) FROM pesee) AS part
                            FROM pesee GROUP BY 1 ORDER BY part DESC""",
                        _pa).df()
                    fragiles = df_strates[(df_strates["part"] >= 10) &
                                          ((df_strates["sources"] < 3) |
                                           (df_strates["docs"] < 200))]
                    for _, r in fragiles.iterrows():
                        st.warning(
                            f"**{r['strate']}** pèse {r['part']:.0f} % du "
                            f"résultat avec seulement {int(r['docs'])} documents "
                            f"issus de {int(r['sources'])} source(s). Cette "
                            f"famille tire tout le graphique : ajoutez-y des "
                            f"sources avant de conclure, ou revenez au brut.")
                    with st.expander("Composition après pondération"):
                        st.dataframe(
                            df_strates.assign(
                                part=df_strates["part"].round(1)).rename(columns={
                                "strate": "Famille", "docs": "Documents",
                                "sources": "Sources", "part": "Poids (%)"}),
                            width="stretch", hide_index=True)

                theme_order = (
                    df_tk.groupby("label")["val"].sum()
                    .sort_values(ascending=False).index.tolist()
                )

                # Affichage par tranches de 20 : au-delà, les barres du bas
                # sont trop courtes pour se comparer à l'oeil et le graphe
                # devient un mur à faire défiler. Le reste reste accessible.
                THEME_PAGE = 20
                shown = st.session_state.get("theme_shown", THEME_PAGE)
                shown = min(shown, len(theme_order))
                theme_order = theme_order[:shown]
                df_tk = df_tk[df_tk["label"].isin(theme_order)]

                kind_colors = {"press": "#4C9AFF", "telegram": "#6BCB77",
                               "youtube": "#E4572E", "tv": "#C77DFF",
                               "vk": "#FFA94D",
                               "etat": "#FF6B6B", "para_etat": "#FFA94D",
                               "independant": "#4C9AFF", "exil": "#6BCB77",
                               "inconnu": "#6B7280", "Tous": "#4C9AFF"}
                fig = px.bar(
                    df_tk, x="val", y="label", color="decoupe", orientation="h",
                    category_orders={"label": theme_order},
                    color_discrete_map=kind_colors,
                    hover_data={"n": True, "chars": True},
                    labels={"val": x_label, "label": "", "decoupe": ""},
                )
                fig.update_layout(barmode="stack", legend_title_text="")
                # Plotly empilé les catégories du bas vers le haut : pour voir
                # le thème dominant EN HAUT, il faut lui passer l'ordre
                # croissant. Un `autorange="reversed"` par-dessus annulait ce
                # classement et remontait les thèmes les moins traités.
                fig.update_yaxes(categoryorder="array",
                                 categoryarray=list(reversed(theme_order)))
                style(fig, max(450, 34 * len(theme_order)))
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())

                total_themes = len(nz)
                cpg1, cpg2 = st.columns([1, 3])
                if shown < total_themes:
                    reste = total_themes - shown
                    if cpg1.button(f"Afficher {min(THEME_PAGE, reste)} de plus",
                                   key="theme_more"):
                        st.session_state["theme_shown"] = shown + THEME_PAGE
                        st.rerun()
                    cpg2.caption(f"{shown} thèmes sur {total_themes} affichés, "
                                 f"du plus traité au moins traité.")
                elif total_themes > THEME_PAGE:
                    if cpg1.button("Revenir aux 20 premiers", key="theme_less"):
                        st.session_state["theme_shown"] = THEME_PAGE
                        st.rerun()
                    cpg2.caption(f"Les {total_themes} thèmes actifs sont affichés.")
        with c_right:
            st.markdown("**Statistiques globales**")
            n_total = int(nz["n"].sum())
            st.metric("Segments classes (actifs)", f"{n_total:,}".replace(",", " "))
            st.metric("Thèmes actifs", f"{len(nz)}")
            st.metric("Thèmes en sommeil", f"{(~df_t['active']).sum()}")
            top = nz.iloc[0] if len(nz) else None
            if top is not None:
                st.metric("Thème dominant", top["label"], f"{top['pct']}%")
            top5 = nz.head(5)[["label", "n", "pct"]].rename(
                columns={"label": "Thème", "n": "N", "pct": "%"})
            st.dataframe(top5, hide_index=True, width="stretch")

        if not archived.empty:
            with st.expander(f"Thèmes en sommeil sur cette période ({len(archived)})"):
                st.caption(
                    "Plus de cluster correspondant dans la dernière fenêtre "
                    "d'analyse -- gardes en memoire, peuvent redevenir actifs."
                )
                df_arch = archived[["label", "n", "first_seen", "last_seen"]].rename(
                    columns={"label": "Thème", "n": "Segments",
                             "first_seen": "Vu depuis", "last_seen": "Vu jusqu'a"})
                st.dataframe(
                    df_arch, width="stretch", hide_index=True,
                    column_config={
                        c: st.column_config.DateColumn(c, format="DD/MM/YYYY")
                        for c in ("Vu depuis", "Vu jusqu'a")})

        # --- Thèmes jour par jour ------------------------------------------
        # Le clustering tourne sur une fenêtre de 24 h : chaque journée est
        # regroupée pour elle-même, donc deux journées sont comparables. Sans
        # cette section, cette dimension n'apparaissait nulle part -- le reste
        # de l'onglet agrège toute la période choisie en un seul classement.
        _jours = [r[0] for r in conn.execute(
            "SELECT DISTINCT run_date FROM article_topics "
            "WHERE run_date >= ? ORDER BY 1", [DEBUT_SERIE]).fetchall()]
        if _jours:
            st.markdown("---")
            st.subheader("Thèmes jour par jour")
            st.caption(
                "Chaque journée est regroupée séparément, sur ses seules 24 h. "
                "Cette section lit les journées telles qu'elles ont été "
                "regroupées : elle ne dépend ni de la pondération ni de la "
                "période choisies à gauche."
            )

            # --- Carte de chaleur : ce qui monte et ce qui tombe -----------
            df_hm = conn.execute("""
                WITH j AS (
                    SELECT run_date, topic_key, COUNT(*) AS n
                    FROM article_topics WHERE topic_key <> -1 AND run_date >= ?
                    GROUP BY 1, 2),
                     tot AS (SELECT run_date, SUM(n) AS t FROM j GROUP BY 1)
                SELECT j.run_date AS jour, t.label AS theme,
                       100.0 * j.n / tot.t AS part, j.n AS n
                FROM j
                JOIN tot ON tot.run_date = j.run_date
                JOIN topics t ON t.topic_key = j.topic_key
            """, [DEBUT_SERIE]).df()

            if not df_hm.empty:
                _rang = (df_hm.groupby("theme")["n"].sum()
                         .sort_values(ascending=False).head(15).index)
                _piv = (df_hm[df_hm["theme"].isin(_rang)]
                        .pivot_table(index="theme", columns="jour",
                                     values="part", fill_value=0)
                        .reindex(_rang))
                # Date courte : douze dates completes se chevauchaient
                # au-dessus des colonnes.
                _piv.columns = [f"{c:%d/%m}" for c in _piv.columns]
                fig = px.imshow(
                    _piv, aspect="auto", origin="upper",
                    color_continuous_scale=[[0.0, "#12161F"], [1.0, ACCENT]],
                    labels={"x": "", "y": "", "color": "% du jour"})
                fig.update_traces(
                    hovertemplate="%{y}<br>%{x} : %{z:.1f} %% des documents "
                                  "classés<extra></extra>")
                fig.update_xaxes(side="top", tickangle=0)
                style(fig, 34 * len(_piv) + 110)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
                st.caption(
                    "Les quinze thèmes les plus volumineux, en part des "
                    "documents classés de chaque journée. Une case vide veut "
                    "dire que le thème n'a pas été retrouvé ce jour-là."
                )

            # --- Détail d'une journée ou d'une semaine ---------------------
            _c1, _c2 = st.columns([1, 2])
            _grain = _c1.radio("Granularité", ["Jour", "Semaine"],
                               horizontal=True, key="th_jour_grain")
            if _grain == "Jour":
                _choix = _c2.selectbox("Journée", list(reversed(_jours)),
                                       format_func=fr_date, key="th_jour_sel")
                _deb, _fin = _choix, _choix
                _pas = timedelta(days=1)
            else:
                _lundis = sorted({j - timedelta(days=j.weekday()) for j in _jours},
                                 reverse=True)
                _choix = _c2.selectbox("Semaine du", _lundis, format_func=fr_date,
                                       key="th_sem_sel")
                _deb, _fin = _choix, _choix + timedelta(days=6)
                _pas = timedelta(days=7)

            def _themes_periode(debut, fin):
                return conn.execute("""
                    WITH n AS (
                        SELECT topic_key, COUNT(*) AS n FROM article_topics
                        WHERE run_date BETWEEN ? AND ? AND topic_key <> -1
                        GROUP BY 1),
                         s AS (
                        SELECT topic_key, STRING_AGG(DISTINCT source_kind, ' + ') AS supports
                        FROM topic_supports WHERE run_date BETWEEN ? AND ? GROUP BY 1)
                    SELECT t.label AS theme, n.n AS n, s.supports AS supports
                    FROM n JOIN topics t ON t.topic_key = n.topic_key
                    LEFT JOIN s ON s.topic_key = n.topic_key
                    ORDER BY n.n DESC
                """, [debut, fin, debut, fin]).df()

            df_p = _themes_periode(_deb, _fin)
            df_prec = _themes_periode(_deb - _pas, _fin - _pas)

            _n_cls, _n_tot = conn.execute(
                "SELECT SUM(CASE WHEN topic_key <> -1 THEN 1 ELSE 0 END), COUNT(*) "
                "FROM article_topics WHERE run_date BETWEEN ? AND ?",
                [_deb, _fin]).fetchone()
            _n_cls, _n_tot = int(_n_cls or 0), int(_n_tot or 0)

            if df_p.empty:
                st.caption("Aucun thème sur cette période.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Thèmes", f"{len(df_p)}")
                m2.metric("Documents classés", f"{_n_cls:,}".replace(",", "\u202f"))
                m3.metric("Non classés",
                          f"{100 * (_n_tot - _n_cls) / max(_n_tot, 1):.0f} %",
                          f"{_n_tot - _n_cls} documents", delta_color="off")

                # Part du jour plutot que volume brut : deux journees n'ont pas
                # le meme volume, comparer les effectifs induirait en erreur.
                df_p["Part"] = 100 * df_p["n"] / max(_n_cls, 1)
                _avant = df_prec.set_index("theme")["n"].to_dict()
                _tot_av = max(sum(_avant.values()), 1)
                df_p["Écart"] = [
                    ("nouveau" if t not in _avant else
                     f"{100 * r / max(_n_cls, 1) - 100 * _avant[t] / _tot_av:+.1f} pt")
                    for t, r in zip(df_p["theme"], df_p["n"])]

                st.dataframe(
                    df_p.rename(columns={"theme": "Thème", "n": "Documents",
                                         "supports": "Supports"})[
                        ["Thème", "Documents", "Part", "Supports", "Écart"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "Part": st.column_config.NumberColumn(
                            "Part", format="%.1f %%"),
                        "Documents": st.column_config.NumberColumn(format="%d"),
                    })
                st.caption(
                    "**Supports** : les natures de contenu où le thème a été "
                    "trouvé. Plusieurs supports signifient que leurs clusters "
                    "se sont révélés assez proches pour être recoupés. "
                    "**Écart** : évolution de la part par rapport à la période "
                    "précédente de même durée, en points de pourcentage."
                )

                _disparus = [t for t in _avant if t not in set(df_p["theme"])]
                if _disparus:
                    st.caption("**Plus vus sur cette période** : "
                               + ", ".join(sorted(_disparus)[:12])
                               + ("..." if len(_disparus) > 12 else ""))

        # --- Qualité des clusters (protocole ProxAnn) ---------------------
        if has_topic_quality:
            df_q = conn.execute(
                """SELECT topic_key, categorie, coherence, distinction
                   FROM topic_quality
                   WHERE run_date = (SELECT MAX(run_date) FROM topic_quality)"""
            ).df()
            if not df_q.empty:
                st.markdown("---")
                st.subheader("Qualité des thèmes")
                st.caption(
                    "Protocole ProxAnn (Hoyle et al., 2025) : un modèle déduit "
                    "une catégorie à partir de quelques articles du thème, puis "
                    "on vérifié qu'elle accepte d'autres articles du même thème "
                    "(**cohérence**) et qu'elle rejette ceux des autres thèmes "
                    "(**distinction**). Un fourre-tout se reconnaît à une "
                    "cohérence haute avec une distinction basse : sa définition "
                    "est si large qu'elle prend tout."
                )
                # Jointure interne, pas externe : topic_quality garde les
                # resultats de clusterings precedents, dont les cles n'existent
                # plus apres un --reset. En externe, leur volume ressortait a
                # NaN et Plotly refusait la taille des bulles.
                _q_avant = len(df_q)
                df_q = df_q.merge(
                    df_t[["topic_key", "label", "n"]], on="topic_key", how="inner")
                _q_perdus = _q_avant - len(df_q)
                k1, k2, k3 = st.columns(3)
                k1.metric("Cohérence moyenne", f"{df_q['coherence'].mean():.2f}")
                k2.metric("Distinction moyenne", f"{df_q['distinction'].mean():.2f}")
                faibles = int(((df_q["coherence"] < 0.5) |
                               (df_q["distinction"] < 0.5)).sum())
                k3.metric("Thèmes fragiles", f"{faibles} / {len(df_q)}",
                          "cohérence ou distinction < 0,5", delta_color="off")

                # Les deux mesures sont discretes -- six documents de test et
                # six de controle, donc des multiples de 1/6. Sans dispersion,
                # 83 thèmes se superposaient sur une quinzaine de positions et
                # le graphique montrait quinze points. Le décalage est calcule
                # à partir de la clé du thème : il est donc stable d'un
                # affichage a l'autre, contrairement à un tirage aleatoire.
                _pas = 1 / 6 * 0.34
                df_q["x_aff"] = df_q["coherence"] + (
                    (df_q["topic_key"] * 7 % 11) / 10 - 0.5) * _pas
                df_q["y_aff"] = df_q["distinction"] + (
                    (df_q["topic_key"] * 13 % 11) / 10 - 0.5) * _pas
                fig = px.scatter(
                    df_q, x="x_aff", y="y_aff", size="n",
                    hover_name="label",
                    hover_data={"categorie": True, "n": True,
                                "coherence": ":.2f", "distinction": ":.2f",
                                "x_aff": False, "y_aff": False},
                    labels={"x_aff": "Cohérence", "y_aff": "Distinction"})
                fig.update_traces(marker=dict(color="#4C9AFF", opacity=0.75,
                                              line=dict(width=1, color="#161A23")))
                # Les quadrants donnent la lecture : en haut à droite les thèmes
                # nets, en bas à droite les fourre-tout.
                fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED)
                fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED)
                fig.update_xaxes(range=[-0.05, 1.05])
                fig.update_yaxes(range=[-0.05, 1.05])
                style(fig, 430)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
                if _q_perdus:
                    st.caption(
                        f"{_q_perdus} évaluations portent sur des thèmes qui "
                        "n'existent plus (clustering refait depuis) et ne sont "
                        "pas affichées. Relancez `validate_topics.py` pour "
                        "évaluer les thèmes actuels.")
                st.caption(
                    "Chaque bulle est un thème, sa taille son nombre de "
                    "segments. Les points sont legerement disperses : les deux "
                    "mesures ne prennent que six valeurs chacune, et sans cela "
                    "les thèmes se superposeraient exactement. Les valeurs "
                    "exactes restent au survol.")

                with st.expander("Thèmes les plus fragiles"):
                    faible = df_q[(df_q["coherence"] < 0.5) |
                                  (df_q["distinction"] < 0.5)].copy()
                    faible = faible.sort_values(["coherence", "distinction"])
                    st.dataframe(
                        faible[["label", "n", "coherence", "distinction",
                                "categorie"]].rename(columns={
                            "label": "Thème", "n": "Segments",
                            "coherence": "Cohérence", "distinction": "Distinction",
                            "categorie": "Catégorie déduite"}),
                        width="stretch", hide_index=True)

        st.subheader("Explorer un thème")
        all_ids = df_t[df_t["n"] > 0]["topic_key"].tolist()
        if all_ids:
            label_lookup = dict(zip(df_t["topic_key"], df_t["label"]))
            n_lookup = dict(zip(df_t["topic_key"], df_t["n"]))
            active_lookup = dict(zip(df_t["topic_key"], df_t["active"]))
            tid = st.selectbox(
                "Thème", all_ids,
                format_func=lambda i: (
                    f"#{i:>3} - {label_lookup[i]} ({n_lookup[i]})"
                    f"{'' if active_lookup[i] else '  [en sommeil]'}"
                ),
                key="theme_selector",
            )
            theme_label = label_lookup[tid]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Sources**")
                df_src = conn.execute(
                    f"""SELECT a.source_name, COUNT(*) AS n FROM article_topics at_
                    JOIN articles a ON a.id = at_.article_id
                    WHERE at_.topic_key = ? AND {aw}
                    GROUP BY 1 ORDER BY 2 DESC LIMIT 10""",
                    [tid, *params]).df()
                if not df_src.empty:
                    fig = px.bar(df_src, x="n", y="source_name", orientation="h",
                                 color="n", color_continuous_scale="Tealgrn",
                                 labels={"n": "", "source_name": ""})
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_yaxes(autorange="reversed")
                    style(fig, 350)
                    st.plotly_chart(fig, width="stretch", key=_next_chart_key())
            with c2:
                st.markdown(f"**Évolution**")
                df_evo = conn.execute(
                    f"""SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS jour, COUNT(*) AS n
                    FROM article_topics at_ JOIN articles a ON a.id = at_.article_id
                    WHERE at_.topic_key = ? AND {aw} AND a.published_at IS NOT NULL
                    GROUP BY 1 ORDER BY 1""", [tid, *params]).df()
                if not df_evo.empty:
                    fig = px.line(df_evo, x="jour", y="n", markers=True,
                                  labels={"jour": GRAIN_LABEL, "n": "Segments"})
                    fig.update_traces(line_color="#4C9AFF", line_width=3, marker_size=10)
                    style(fig, 350)
                    st.plotly_chart(fig, width="stretch", key=_next_chart_key())
            with c3:
                # Ici se trouvait un camembert « Posture Ukraine ». La cible
                # était codée en dur parmi les seize suivies, sans raison de
                # privilegier celle-la, et la posture par cible a déjà son
                # onglet (Sentiment) avec un selecteur. A sa place, la question
                # qui manquait vraiment sur un thème : qui le porte -- la
                # presse, la télévision, Telegram ?
                st.markdown("**Ou ce thème est-il porte**")
                df_nat_t = conn.execute(
                    f"""SELECT a.source_kind, COUNT(*) AS n
                    FROM article_topics at_ JOIN articles a ON a.id = at_.article_id
                    WHERE at_.topic_key = ? AND {aw}
                    GROUP BY 1 ORDER BY 2 DESC""", [tid, *params]).df()
                if not df_nat_t.empty:
                    df_nat_t["nature"] = df_nat_t["source_kind"].map(
                        lambda k: UNITE_NATURE.get(k, k))
                    fig = px.pie(df_nat_t, values="n", names="nature", hole=0.5,
                                 color_discrete_sequence=TOP_PALETTE)
                    fig.update_traces(textinfo="percent+label", textfont_size=11,
                                      textposition="outside", automargin=True)
                    style(fig, 350)
                    st.plotly_chart(fig, width="stretch", key=_next_chart_key())
                    st.caption(
                        "En segments, pas en émissions : une émission de TV "
                        "pèse naturellement plus qu'un article. A lire comme "
                        "un partage de volume de texte.")

            top_words = conn.execute(
                "SELECT top_words FROM topics WHERE topic_key = ?", [tid]
            ).fetchone()
            if top_words and top_words[0]:
                st.caption(f"Mots-clés du cluster : {top_words[0]}")

            df_art = conn.execute(
                f"""SELECT a.published_at, a.source_name, ROUND(at_.probability, 2) AS proba,
                a.title, a.url FROM articles a
                JOIN article_topics at_ ON at_.article_id = a.id
                WHERE at_.topic_key = ? AND {aw}
                ORDER BY a.published_at DESC LIMIT 50""",
                [tid, *params]).df()
            st.caption("proba = probabilité d'appartenance au cluster (0 à 1)")
            st.dataframe(df_art, width="stretch", hide_index=True,
                         column_config=cols_article())


# ===== Tab Sentiment ===========================================
with tab_sentiment:
    aw = with_a(WHERE)
    if not has_target_sent:
        st.warning("Lancez : python analyze_sentiment_multi_mistral.py --reset")
    else:
        target = st.selectbox("Cible géopolitique", TARGETS,
                              format_func=lambda t: TARGET_LABELS[t], index=0,
                              key="sentiment_target")
        target_label = TARGET_LABELS[target]

        df_st = conn.execute(
            f"SELECT s.stance, COUNT(*) AS n FROM article_target_sentiment s "
            f"JOIN articles a ON a.id = s.article_id WHERE s.target = ? AND {aw} GROUP BY 1",
            [target, *params]).df()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Vue brute")
            if not df_st.empty:
                fig = px.pie(df_st, values="n", names="stance",
                             color="stance", color_discrete_map=STANCE_COLORS, hole=0.5)
                fig.update_traces(textinfo="percent+label", textfont_size=11, textposition="outside", automargin=True)
                fig.update_layout(showlegend=False)
                style(fig, 350)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        with c2:
            st.subheader(f"Focus {target_label}")
            df_f = df_st[df_st["stance"] != "non_concerne"]
            if not df_f.empty:
                fig = px.pie(df_f, values="n", names="stance",
                             color="stance", color_discrete_map=STANCE_COLORS, hole=0.5)
                fig.update_traces(textinfo="percent+label", textfont_size=11, textposition="outside", automargin=True)
                fig.update_layout(showlegend=False)
                style(fig, 350)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        with c3:
            st.subheader("Lean -1 anti / +1 pro")
            df_lean = conn.execute(
                f"SELECT s.lean, s.score FROM article_target_sentiment s "
                f"JOIN articles a ON a.id = s.article_id "
                f"WHERE s.target = ? AND s.stance != 'non_concerne' AND {aw}",
                [target, *params]).df()
            if not df_lean.empty:
                _w = df_lean["score"].fillna(0)
                avg = (df_lean["lean"] * _w).sum() / _w.sum() if _w.sum() > 0 else df_lean["lean"].mean()
                anti_count = int((df_lean["lean"] < -0.2).sum())
                neutre_count = int(((df_lean["lean"] >= -0.2) & (df_lean["lean"] <= 0.2)).sum())
                pro_count = int((df_lean["lean"] > 0.2).sum())
                total = len(df_lean)

                fig = go.Figure()
                n_bands = 50
                for i in range(n_bands):
                    x0b = -1 + i * (2 / n_bands)
                    x1b = x0b + (2 / n_bands)
                    xc = (x0b + x1b) / 2
                    band_color = "#FF6B6B" if xc < 0 else "#4C9AFF"
                    fig.add_vrect(x0=x0b, x1=x1b, fillcolor=band_color,
                                  opacity=min(abs(xc), 1) * 0.35, line_width=0)
                fig.add_trace(go.Violin(
                    x=df_lean["lean"], orientation="h",
                    box_visible=False, meanline_visible=False, points=False,
                    line_color="white", fillcolor="rgba(76, 154, 255, 0.5)",
                    name="", showlegend=False, hoveron="violins",
                    hovertemplate=(
                        f"<b>Distribution du lean ({target_label})</b><br>"
                        f"Total : {total} articles<br>"
                        f"Anti : {anti_count} ({100*anti_count/max(total,1):.0f}%)<br>"
                        f"Neutre : {neutre_count} ({100*neutre_count/max(total,1):.0f}%)<br>"
                        f"Pro : {pro_count} ({100*pro_count/max(total,1):.0f}%)<extra></extra>"
                    ),
                ))
                fig.add_vline(x=avg, line_dash="dash", line_color="yellow", line_width=3)
                fig.add_annotation(x=-0.6, y=0.92, xref="x", yref="paper",
                                   text=f"<b>Anti<br>{anti_count}</b>",
                                   showarrow=False, font=dict(size=14, color="#FF8888"))
                fig.add_annotation(x=0, y=0.92, xref="x", yref="paper",
                                   text=f"<b>Neutre<br>{neutre_count}</b>",
                                   showarrow=False, font=dict(size=14, color="#CCCCCC"))
                fig.add_annotation(x=0.6, y=0.92, xref="x", yref="paper",
                                   text=f"<b>Pro<br>{pro_count}</b>",
                                   showarrow=False, font=dict(size=14, color="#66B0FF"))
                fig.add_annotation(x=avg, y=0.55, xref="x", yref="paper",
                                   text=f"<b>Moy. {avg:+.2f}</b>",
                                   showarrow=False, font=dict(size=15, color="yellow"),
                                   bgcolor="rgba(0,0,0,0.6)", borderpad=4)
                fig.update_layout(xaxis_title="lean", xaxis_range=[-1.05, 1.05],
                                  yaxis=dict(showticklabels=False, showgrid=False,
                                             range=[-0.5, 0.5]),
                                  margin=dict(l=10, r=10, t=70, b=40))
                style(fig, 370)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())

        cp, cn, ca = st.columns(3)
        for col, st_label, title in [
            (cp, "pro", f"Top 15 pro-{target_label}"),
            (cn, "neutre", "Top 15 neutre"),
            (ca, "anti", f"Top 15 anti-{target_label}"),
        ]:
            with col:
                st.subheader(title)
                df = conn.execute(
                    f"SELECT a.source_name, ROUND(s.score, 2) AS score, "
                    f"ROUND(s.lean, 2) AS lean, a.title, s.reasoning, a.url "
                    f"FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id "
                    f"WHERE s.target = ? AND s.stance = ? AND {aw} "
                    f"ORDER BY s.score DESC LIMIT 15",
                    [target, st_label, *params]).df()
                st.dataframe(df, width="stretch", hide_index=True,
                             column_config=cols_article())

        st.subheader(f"Évolution de la posture vis-à-vis de {target_label}")
        # Le pas de temps est désormais un réglage global (barre laterale) :
        # deux commandes concurrentes sur la même notion pretaient a confusion.
        trunc = GRAIN
        df_evo = conn.execute(
            f"""SELECT DATE_TRUNC('{trunc}', a.published_at) AS periode,
                       s.stance, COUNT(*) AS n
            FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
            WHERE s.target = ? AND {aw} AND s.stance != 'non_concerne'
              AND a.published_at IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1""", [target, *params]).df()
        if not df_evo.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Volume par stance**")
                fig = px.bar(df_evo, x="periode", y="n", color="stance",
                             color_discrete_map=STANCE_COLORS,
                             labels={"periode": GRAIN_LABEL, "n": "Segments", "stance": ""})
                fig.update_layout(barmode="stack", legend_title_text="")
                style(fig, 400)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
            with c2:
                df_lean_evo = conn.execute(
                    f"""SELECT DATE_TRUNC('{trunc}', a.published_at) AS periode,
                              SUM(s.lean * s.score) / NULLIF(SUM(s.score), 0) AS lean_avg, COUNT(*) AS n
                    FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
                    WHERE s.target = ? AND {aw} AND s.stance != 'non_concerne'
                      AND a.published_at IS NOT NULL
                    GROUP BY 1 ORDER BY 1""", [target, *params]).df()
                if not df_lean_evo.empty:
                    st.markdown("**Lean moyen par période**")
                    fig = go.Figure()
                    fig.add_hrect(y0=0.2, y1=1, fillcolor="#4C9AFF", opacity=0.12, line_width=0)
                    fig.add_hrect(y0=-0.2, y1=0.2, fillcolor="#A0A0A0", opacity=0.12, line_width=0)
                    fig.add_hrect(y0=-1, y1=-0.2, fillcolor="#FF6B6B", opacity=0.12, line_width=0)
                    # Bulles avec aire proportionnelle au nombre d'articles
                    max_n = max(df_lean_evo["n"].max(), 1)
                    bubble_ref = 2.0 * max_n / (35 ** 2)
                    fig.add_trace(go.Scatter(
                        x=df_lean_evo["periode"], y=df_lean_evo["lean_avg"],
                        mode="lines+markers",
                        line=dict(color="#FFD93D", width=3),
                        marker=dict(
                            size=df_lean_evo["n"], sizemode="area",
                            sizeref=bubble_ref, sizemin=6,
                            color="#FFD93D",
                            line=dict(color="white", width=1.5),
                        ),
                        hovertemplate="%{x}<br>Lean: %{y:.2f}<br>Articles: %{text}<extra></extra>",
                        text=df_lean_evo["n"],
                    ))
                    fig.add_hline(y=0, line_dash="dot", line_color="white", line_width=1)
                    fig.update_layout(yaxis_range=[-1.05, 1.05], yaxis_title="Lean moyen",
                                      xaxis_title=GRAIN_LABEL)
                    style(fig, 400)
                    st.plotly_chart(fig, width="stretch", key=_next_chart_key())

        st.subheader(f"Posture des médias vis-à-vis de {target_label}")
        df_src_stance = conn.execute(
            f"""SELECT a.source_name, s.stance, COUNT(*) AS n
            FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
            WHERE s.target = ? AND {aw} AND s.stance != 'non_concerne'
            GROUP BY 1, 2""", [target, *params]).df()
        if not df_src_stance.empty:
            pv = df_src_stance.pivot_table(index="source_name", columns="stance",
                                           values="n", fill_value=0)
            pv["t"] = pv.sum(axis=1)
            pv = pv[pv["t"] >= 1].sort_values("t", ascending=False).drop(columns=["t"])
            pv_pct = pv.div(pv.sum(axis=1), axis=0) * 100
            order = [c for c in ["pro", "neutre", "anti"] if c in pv_pct.columns]
            fig = go.Figure()
            for col in order:
                fig.add_trace(go.Bar(
                    y=pv_pct.index, x=pv_pct[col], name=col, orientation="h",
                    marker_color=STANCE_COLORS.get(col, "#888"),
                    text=[f"{int(v)}%" if v > 5 else "" for v in pv_pct[col]],
                    textposition="inside", textfont=dict(color="white", size=14),
                ))
            fig.update_layout(barmode="stack", xaxis_title="Part (%)",
                              legend_title_text="", yaxis_autorange="reversed")
            style(fig, max(400, 28 * len(pv_pct)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

        st.subheader("Carte d'influence (16 acteurs)")
        df_map_data = conn.execute(
            f"""SELECT s.target,
                SUM(CASE WHEN s.stance = 'pro' THEN 1 ELSE 0 END) AS pro,
                SUM(CASE WHEN s.stance = 'anti' THEN 1 ELSE 0 END) AS anti,
                SUM(CASE WHEN s.stance NOT IN ('non_concerne') THEN 1 ELSE 0 END) AS mentions,
                SUM(s.lean * s.score) / NULLIF(SUM(s.score), 0) AS lean_avg
            FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
            WHERE {aw} GROUP BY 1""", params).df()

        if not df_map_data.empty:
            df_map_data = df_map_data[df_map_data["target"].isin(TARGETS)].copy()
            df_map_data["lat"] = df_map_data["target"].map(lambda t: TARGET_COORDS[t][0])
            df_map_data["lon"] = df_map_data["target"].map(lambda t: TARGET_COORDS[t][1])
            df_map_data["label"] = df_map_data["target"].map(TARGET_LABELS)

            # Aire proportionnelle au nombre de mentions (perceptuellement correct)
            max_m = max(df_map_data["mentions"].max(), 1)
            sizeref_val = 2.0 * max_m / (42 ** 2)  # diametre max en pixels ; au-delà
            # les bulles europeennes se recouvrent et masquent les étiquettes
        
            amplitude = max(df_map_data["lean_avg"].abs().max(), 0.1)

            fig = go.Figure()

            # Tous les acteurs, même traitement visuel (aucune cible privilégiée)
            fig.add_trace(go.Scattergeo(
                lat=df_map_data["lat"], lon=df_map_data["lon"],
                hovertext=df_map_data.apply(
                    lambda r: f"<b>{r['label']}</b><br>Pro: {int(r['pro'])}<br>"
                              f"Anti: {int(r['anti'])}<br>Mentions: {int(r['mentions'])}<br>"
                              f"Lean: {r['lean_avg']:+.2f}", axis=1),
                hoverinfo="text", mode="markers+text",
                marker=dict(
                    size=df_map_data["mentions"],
                    sizemode="area", sizeref=sizeref_val, sizemin=4,
                    # Échelle divergente sur l'orientation moyenne, pas sur le
                    # NOMBRE d'articles favorables : ce dernier suit le volume de
                    # mentions, si bien qu'une cible très commentée paraissait
                    # bienveillamment traitée même quand le solde était hostile
                    # (l'opposition en exil : 286 favorables, 439 défavorables).
                    # Mêmes couleurs que les camemberts de posture, gris neutre
                    # au centre.
                    color=df_map_data["lean_avg"],
                    # Rampe strictement lineaire : l'intensité doit rester
                    # proportionnelle a l'écart a zéro. Avec des paliers
                    # intermediaires, un +0,27 tombait juste sous le palier bleu
                    # vif et paraissait aussi marque qu'un -0,72 en rouge. Le
                    # bleu doit être pale ici, parce que la bienveillance
                    # maximale du corpus est trois fois plus faible que
                    # l'hostilité maximale.
                    colorscale=[[0.0, "#C0392B"], [0.5, "#8A8F98"],
                                [1.0, "#1B5FBF"]],
                    # Bornes calées sur l'amplitude observée et non sur [-1, 1] :
                    # les orientations réelles tiennent entre -0,72 et +0,27, si
                    # bien qu'une échelle théorique laissait tout le monde pale.
                    # Symetrique pour que le gris reste exactement sur zéro.
                    cmin=-amplitude, cmax=amplitude,
                    line=dict(width=1.5, color="white"), opacity=0.85,
                    showscale=True,
                    colorbar=dict(title="Orientation", thickness=15, len=0.55,
                                  tickvals=[-amplitude, 0, amplitude],
                                  ticktext=["hostile", "neutre", "favorable"],
                                  tickfont=dict(size=13), title_font=dict(size=13)),
                ),
                # Le nom seul : les chiffres sous chaque bulle faisaient deux
                # lignes par acteur et rendaient les noms illisibles. Ils
                # restent au survol, et le top 5 est encadré à gauche.
                text=df_map_data["target"].map(
                    lambda t: TARGET_LABELS_CARTE.get(t, TARGET_LABELS[t])),
                textposition=df_map_data["target"].map(
                    lambda t: TARGET_TEXTPOS.get(t, "bottom center")),
                textfont=dict(size=15, color="#F2F4F7"),
            ))

            # Encadré top 5 mentions en haut à gauche
            top5 = df_map_data.nlargest(5, "mentions")[
                ["label", "mentions", "target"]
            ].reset_index(drop=True)

            lines = ['<b style="color:#FFDC00">TOP 5 MENTIONS</b>',
                     '<span style="color:#888">──────────────────────</span>']
            for _, row in top5.iterrows():
                name = row["label"]
                pad = "&nbsp;" * max(2, 20 - len(name))
                lines.append(
                    f'<span style="color:#FFFFFF">{name}{pad}'
                    f'<b>{int(row["mentions"])}</b></span>'
                )
            top5_text = "<br>".join(lines)

            fig.add_annotation(
                xref="paper", yref="paper", x=0.015, y=0.98,
                text=top5_text, showarrow=False, align="left",
                bgcolor="rgba(14, 17, 23, 0.85)",
                bordercolor="#FFDC00", borderwidth=2, borderpad=12,
                font=dict(size=14, family="Consolas, Courier New, monospace"),
                xanchor="left", yanchor="top",
            )
            fig.update_geos(projection_type="natural earth",
                            showcountries=True, countrycolor="#444",
                            showcoastlines=True, coastlinecolor="#666",
                            bgcolor="#0E1117", landcolor="#1A1F2E", oceancolor="#0E1117",
                            lataxis=dict(range=[-45, 75]), lonaxis=dict(range=[-130, 160]))
            fig.update_layout(showlegend=False)
            style(fig, 750)
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())
            # On affiche la valeur SIGNÉE de l'acteur extreme, pas l'amplitude :
            # OTAN est a -0,56 et l'imprimer en +0,56 laissait croire à un
            # traitement favorable.
            _i_extreme = df_map_data["lean_avg"].abs().idxmax()
            _extreme = df_map_data.loc[_i_extreme, "label"]
            _val_extreme = df_map_data.loc[_i_extreme, "lean_avg"]
            st.caption(
                f"**Seule la couleur est mise à l'échelle**, relativement aux "
                f"acteurs affichés : « {_extreme} », le plus marque d'entre eux "
                f"({_val_extreme:+.2f}), sature la teinte, et les autres se "
                f"placent en proportion. L'échelle se recalculé donc quand les "
                f"filtres changent le corpus, et les cibles suivies sans "
                f"position sur la carte n'y entrent pas. La **taille** des "
                f"bulles, elle, reste le nombre de mentions."
            )

            st.markdown("**Synthèse chiffrée par acteur**")
            df_table = df_map_data[["label", "pro", "anti", "mentions", "lean_avg"]].copy()
            df_table["lean_avg"] = df_table["lean_avg"].round(2)
            df_table = df_table.sort_values("mentions", ascending=False).reset_index(drop=True)
            df_table.columns = ["Acteur", "Pro", "Anti", "Mentions", "Lean moyen"]
            st.dataframe(df_table, width="stretch", hide_index=True)


# ===== Tab Pouvoir/Type =========================================
with tab_alignement:
    # Les postures et types d'article derives du modèle ont été retirés de la
    # routine (cf. update.py) : ils coutaient un tiers du budget Mistral pour
    # une information que le label écrit à la main résumé déjà. Cet onglet
    # affiche désormais la classification curee, disponible pour toutes les
    # sources sans aucun appel API.
    aw = with_a(WHERE)
    st.subheader("Classification éditoriale des sources")
    st.caption(
        "Étiquettes saisies à la main dans config/sources.yaml, pas déduites "
        "des articles : elles valent pour toutes les sources, y compris celles "
        "dont aucun article n'a encore été analyse. Le type de média résumé "
        "l'alignement (État, para-État, independant, exil) ; le positionnement "
        "historique donne le détail éditorial."
    )

    df_class = conn.execute(
        f"""SELECT a.source_name AS Source,
                   COALESCE(a.type_media, 'non classe') AS "Type de média",
                   COALESCE(a.statut_legal_ru, 'aucun') AS "Statut légal (RU)",
                   ANY_VALUE(a.source_kind) AS "Nature",
                   COUNT(*) AS Articles
            FROM articles a WHERE {aw}
            GROUP BY 1, 2, 3 ORDER BY 2, 5 DESC""",
        params).df()
    df_class["Positionnement historique"] = (
        df_class["Source"].map(HISTORICAL_STANCE).fillna(""))

    st.dataframe(
        df_class, width="stretch", hide_index=True,
        column_config={
            "Positionnement historique": st.column_config.TextColumn(
                "Positionnement historique", width="large"),
        },
    )

    st.markdown("### Répartition du corpus par alignement")
    df_mix = conn.execute(
        f"""SELECT COALESCE(a.type_media, 'non classe') AS type_media,
                   COUNT(*) AS n, SUM(LENGTH(a.content)) AS signes
            FROM articles a WHERE {aw} GROUP BY 1 ORDER BY n DESC""",
        params).df()
    if not df_mix.empty:
        cmix1, cmix2 = st.columns([1, 2])
        with cmix1:
            fig = px.pie(df_mix, values="n", names="type_media", hole=0.5,
                         color="type_media",
                         color_discrete_map={"etat": "#FF6B6B",
                                             "para_etat": "#FFA94D",
                                             "independant": "#4C9AFF",
                                             "exil": "#6BCB77",
                                             "non classe": "#6B7280"})
            fig.update_traces(textinfo="percent+label", textfont_size=11,
                              textposition="outside", automargin=True)
            fig.update_layout(showlegend=False)
            style(fig, 380)
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        with cmix2:
            df_mix_src = conn.execute(
                f"""SELECT COALESCE(a.type_media, 'non classe') AS type_media,
                           ANY_VALUE(a.source_kind) AS nature,
                           a.source_name, COUNT(*) AS n
                    FROM articles a WHERE {aw}
                    GROUP BY 1, 3 ORDER BY n DESC LIMIT 25""",
                params).df()
            fig = px.bar(df_mix_src, x="n", y="source_name", color="type_media",
                         orientation="h",
                         color_discrete_map={"etat": "#FF6B6B",
                                             "para_etat": "#FFA94D",
                                             "independant": "#4C9AFF",
                                             "exil": "#6BCB77",
                                             "non classe": "#6B7280"},
                         labels={"n": "Segments", "source_name": "",
                                 "type_media": ""})
            fig.update_yaxes(autorange="reversed")
            style(fig, max(420, 26 * len(df_mix_src)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        st.caption(
            "A lire avec l'onglet Contexte : la composition du corpus ne "
            "reproduit pas celle de l'audience russe. Un alignement "
            "surrepresente ici l'est parce qu'il est facile à collecter, pas "
            "parce qu'il domine ce que les Russes consultent."
        )


# ===== Tab Acteurs (avec position vs cible) ====================
with tab_acteurs:
    df_auth = conn.execute(
        f"SELECT source_name, TRIM(author) AS auteur, COUNT(*) AS n FROM articles "
        f"WHERE {WHERE} AND author IS NOT NULL AND TRIM(author) != '' "
        f"GROUP BY 1, 2 HAVING n >= 1", params).df()

    if df_auth.empty:
        st.info("Lancez : python extract_authors.py")
    else:
        st.subheader("Classement des sources")
        sort_options = ["Nombre d'articles"]
        if has_target_sent:
            sort_options += [f"% pro-{TARGET_LABELS[t]}" for t in TARGETS]
            sort_options += [f"% anti-{TARGET_LABELS[t]}" for t in TARGETS]
            sort_options += [f"Lean moyen {TARGET_LABELS[t]}" for t in TARGETS]

        sort_by = st.selectbox("Trier par", sort_options)
        aw = with_a(WHERE)
        df_src = conn.execute(
            f"SELECT a.source_name, COUNT(*) AS n FROM articles a WHERE {WHERE} GROUP BY 1",
            params).df()

        color_col = "n"
        label_col = "Articles"

        if sort_by.startswith("% pro-") and has_target_sent:
            tn = sort_by.replace("% pro-", "")
            tk = next((k for k, v in TARGET_LABELS.items() if v == tn), None)
            ds = conn.execute(
                f"""SELECT a.source_name,
                    SUM(CASE WHEN s.stance = 'pro' THEN 1 ELSE 0 END) AS pro,
                    SUM(CASE WHEN s.stance NOT IN ('non_concerne') THEN 1 ELSE 0 END) AS concerne
                FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
                WHERE s.target = ? AND {aw} GROUP BY 1""", [tk, *params]).df()
            df_src = df_src.merge(ds, on="source_name", how="left").fillna(0)
            df_src["pct"] = (df_src["pro"] / df_src["concerne"].clip(lower=1) * 100).round(1)
            df_src = df_src[df_src["concerne"] >= 3].sort_values("pct", ascending=False)
            color_col, label_col = "pct", sort_by
        elif sort_by.startswith("% anti-") and has_target_sent:
            tn = sort_by.replace("% anti-", "")
            tk = next((k for k, v in TARGET_LABELS.items() if v == tn), None)
            ds = conn.execute(
                f"""SELECT a.source_name,
                    SUM(CASE WHEN s.stance = 'anti' THEN 1 ELSE 0 END) AS anti,
                    SUM(CASE WHEN s.stance NOT IN ('non_concerne') THEN 1 ELSE 0 END) AS concerne
                FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
                WHERE s.target = ? AND {aw} GROUP BY 1""", [tk, *params]).df()
            df_src = df_src.merge(ds, on="source_name", how="left").fillna(0)
            df_src["pct"] = (df_src["anti"] / df_src["concerne"].clip(lower=1) * 100).round(1)
            df_src = df_src[df_src["concerne"] >= 3].sort_values("pct", ascending=False)
            color_col, label_col = "pct", sort_by
        elif sort_by.startswith("Lean moyen ") and has_target_sent:
            tn = sort_by.replace("Lean moyen ", "")
            tk = next((k for k, v in TARGET_LABELS.items() if v == tn), None)
            ds = conn.execute(
                f"""SELECT a.source_name, SUM(s.lean * s.score) / NULLIF(SUM(s.score), 0) AS lean_avg, COUNT(*) AS concerne
                FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
                WHERE s.target = ? AND s.stance != 'non_concerne' AND {aw}
                GROUP BY 1 HAVING COUNT(*) >= 3""", [tk, *params]).df()
            df_src = df_src.merge(ds, on="source_name", how="left").fillna(0)
            df_src = df_src[df_src["concerne"] >= 3].sort_values("lean_avg", ascending=False)
            color_col, label_col = "lean_avg", sort_by
        else:
            df_src = df_src.sort_values("n", ascending=False)

        df_src = df_src.head(25)
        if color_col in df_src.columns:
            scale = "RdBu" if "lean" in str(color_col).lower() else "Tealgrn"
            fig = px.bar(df_src, x=color_col, y="source_name", orientation="h",
                         color=color_col, color_continuous_scale=scale,
                         hover_data={"n": True},
                         labels={color_col: label_col, "source_name": "", "n": "Segments"})
            fig.update_layout(coloraxis_showscale=True)
            fig.update_yaxes(autorange="reversed")
            style(fig, max(450, 28 * len(df_src)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

        # ===== NOUVEAU : Position des journalistes vs cible =====
        if has_target_sent:
            st.subheader("Position des journalistes vis-à-vis d'une cible")
            st.caption("Lean moyen et volume d'articles par journaliste. "
                       "Sont retenus les journalistes avec >= 3 articles concernés par la cible. "
                       "Attention au volume pour les auteurs de chaînes YouTube : une seule "
                       "vidéo compte pour une dizaine de segments, ce qui les place "
                       "mecaniquement en tête du classement. Le lean moyen, lui, reste "
                       "comparable (il est pondéré, pas cumule). Pour les écarter, "
                       "décochez YouTube dans \"Nature du contenu\".")

            target_author = st.selectbox(
                "Cible", TARGETS, format_func=lambda t: TARGET_LABELS[t],
                index=0, key="author_target",
            )
            target_author_label = TARGET_LABELS[target_author]

            df_author_stance = conn.execute(
                f"""SELECT TRIM(a.author) AS auteur,
                    a.source_name,
                    COUNT(*) AS articles_concernes,
                    ROUND(SUM(s.lean * s.score) / NULLIF(SUM(s.score), 0), 3) AS lean_moyen,
                    SUM(CASE WHEN s.stance = 'pro' THEN 1 ELSE 0 END) AS pro,
                    SUM(CASE WHEN s.stance = 'anti' THEN 1 ELSE 0 END) AS anti,
                    SUM(CASE WHEN s.stance = 'neutre' THEN 1 ELSE 0 END) AS neutre
                FROM articles a
                JOIN article_target_sentiment s ON s.article_id = a.id
                WHERE s.target = ? AND s.stance != 'non_concerne'
                  AND a.author IS NOT NULL AND TRIM(a.author) != ''
                  AND {WHERE}
                GROUP BY TRIM(a.author), a.source_name
                HAVING COUNT(*) >= 3
                ORDER BY articles_concernes DESC
                LIMIT 40""",
                [target_author, *params]).df()

            if df_author_stance.empty:
                st.info(f"Aucun journaliste avec assez d'articles concernés par {target_author_label}.")
            else:
                # Bar chart : auteurs tries par volume, couleur = lean
                fig = px.bar(
                    df_author_stance.head(30),
                    x="lean_moyen", y="auteur", orientation="h",
                    color="lean_moyen", color_continuous_scale="RdBu",
                    range_color=[-1, 1],
                    hover_data={"articles_concernes": True, "pro": True, "anti": True,
                                "neutre": True, "source_name": True, "lean_moyen": ":.2f"},
                    labels={"lean_moyen": f"Lean moyen vs {target_author_label}",
                            "auteur": ""},
                )
                fig.update_layout(coloraxis_showscale=True,
                                  coloraxis_colorbar=dict(title="Lean",
                                                          tickfont=dict(size=13)))
                fig.update_yaxes(autorange="reversed")
                style(fig, max(500, 28 * min(len(df_author_stance), 30)))
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())

                # Table complète pour exploitation
                st.markdown("**Table complète**")
                df_disp = df_author_stance.copy()
                df_disp.columns = ["Auteur", "Source", "Articles", "Lean moyen",
                                   "Pro", "Anti", "Neutre"]
                st.dataframe(df_disp, width="stretch", hide_index=True)

        # Treemap sources/auteurs
        st.subheader("Sources et leurs auteurs")
        st.caption("Hiérarchie source -> auteur. Cliquez pour zoomer.")

        TOP_SRC = 15
        TOP_AUTH = 10
        top_src_t = (df_auth.groupby("source_name", as_index=False)["n"].sum()
                     .nlargest(TOP_SRC, "n"))
        top_src_list = top_src_t["source_name"].tolist()
        parts = []
        total_corpus = int(df_auth["n"].sum())
        for s in top_src_list:
            sub = df_auth[df_auth["source_name"] == s].nlargest(TOP_AUTH, "n")
            parts.append(sub)
        df_tree = pd.concat(parts, ignore_index=True) if parts else df_auth.iloc[0:0]

        if not df_tree.empty:
            df_tree["pct"] = (df_tree["n"] / total_corpus * 100).round(2)
            fig = px.treemap(
                df_tree, path=[px.Constant("Corpus"), "source_name", "auteur"],
                values="n", color="source_name",
                color_discrete_sequence=SOURCE_PALETTE,
                custom_data=["pct"],
            )
            fig.update_traces(textinfo="label+value", textfont_size=16,
                              marker=dict(line=dict(width=2, color="#0E1117")),
                              hovertemplate=("<b>%{label}</b><br>Articles : %{value}<br>"
                                             "Part : %{customdata[0]:.2f}%<extra></extra>"))
            fig.update_layout(font=dict(size=CHART_FONT, color="#E0E0E0"),
                              height=750, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="#0E1117")
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

        st.subheader("Top auteurs globaux")
        df_global = (df_auth.groupby("auteur", as_index=False)["n"].sum()
                     .nlargest(40, "n"))
        st.dataframe(df_global, width="stretch", hide_index=True)
    
# ===== Tab Sources ============================================
with tab_diagnostic:
    st.subheader("Diagnostic de traitement par source")
    st.caption(
        "Pourcentage d'articles traités par chaque analyse, sur la période choisie. "
        "Vue 7j ou 30j pour voir la tendance récente (sinon la masse historique cache "
        "les bugs en cours). Vert > 95%%, orange 80-95%%, rouge < 80%%."
    )

    # Selecteur de période
    from datetime import datetime, timedelta
    cov_period = st.radio(
        "Période",
        ["7 derniers jours", "30 derniers jours", "Tout"],
        horizontal=True, index=0, key="cov_period",
    )
    if cov_period == "7 derniers jours":
        cov_cutoff = (datetime.now() - timedelta(days=7)).date()
    elif cov_period == "30 derniers jours":
        cov_cutoff = (datetime.now() - timedelta(days=30)).date()
    else:
        cov_cutoff = datetime(2000, 1, 1).date()

    # Tables optionnelles (pas encore alimentées si l'analyse correspondante
    # n'a jamais tourne) : on retombe sur un CTE vide plutôt que de planter
    # sur une table absente.
    _empty_cte = "SELECT source_name, 0 AS n FROM articles WHERE published_at >= ? AND FALSE GROUP BY source_name"
    snt_sql = ("SELECT a.source_name, COUNT(DISTINCT s.article_id) AS n "
               "FROM articles a JOIN article_target_sentiment s ON s.article_id = a.id "
               "WHERE a.published_at >= ? GROUP BY a.source_name") if has_target_sent else _empty_cte
    thm_sql = ("SELECT a.source_name, COUNT(DISTINCT t.article_id) AS n "
               "FROM articles a JOIN article_topics t ON t.article_id = a.id "
               "WHERE a.published_at >= ? GROUP BY a.source_name") if has_topics else _empty_cte

    df_cov_full = conn.execute(f"""
        WITH base AS (
            SELECT source_name,
                   COUNT(*) AS total,
                   SUM(CASE WHEN content IS NOT NULL AND LENGTH(content) >= 300
                            THEN 1 ELSE 0 END) AS with_content
            FROM articles WHERE published_at >= ?
            GROUP BY source_name
        ),
        snt AS ({snt_sql}),
        thm AS ({thm_sql}),
        auth AS (
            SELECT source_name,
                   SUM(CASE WHEN author IS NOT NULL AND TRIM(author) != ''
                            THEN 1 ELSE 0 END) AS n
            FROM articles WHERE published_at >= ?
            GROUP BY source_name
        )
        SELECT b.source_name AS Source,
               b.total AS Articles,
               CAST(ROUND(100.0 * b.with_content / NULLIF(b.total, 0)) AS INT) AS "Contenu",
               CAST(ROUND(100.0 * COALESCE(auth.n, 0) / NULLIF(b.total, 0)) AS INT) AS "Auteur",
               CAST(ROUND(100.0 * COALESCE(snt.n, 0) / NULLIF(b.with_content, 0)) AS INT) AS "Sentiment",
               CAST(ROUND(100.0 * COALESCE(thm.n, 0) / NULLIF(b.with_content, 0)) AS INT) AS "Thèmes"
        FROM base b
        LEFT JOIN snt ON snt.source_name = b.source_name
        LEFT JOIN thm ON thm.source_name = b.source_name
        LEFT JOIN auth ON auth.source_name = b.source_name
        WHERE b.total > 0
        ORDER BY b.total DESC
    """, [cov_cutoff] * 4).df()

    if df_cov_full.empty:
        st.warning(f"Aucun article sur la période {cov_period.lower()}.")
    else:
        df_cov_full["Analyses"] = df_cov_full[
            ["Sentiment", "Thèmes"]
        ].min(axis=1)

        def _statut(row):
            if row["Contenu"] < 80 or row["Analyses"] < 80:
                return "PROBLÈME"
            if row["Contenu"] < 95 or row["Analyses"] < 95:
                return "ATTENTION"
            return "OK"

        df_cov_full["Statut"] = df_cov_full.apply(_statut, axis=1)

        n_ok = int((df_cov_full["Statut"] == "OK").sum())
        n_warn = int((df_cov_full["Statut"] == "ATTENTION").sum())
        n_pb = int((df_cov_full["Statut"] == "PROBLÈME").sum())
        n_total_src = len(df_cov_full)
        km0, km1, km2, km3 = st.columns(4)
        km0.metric("Sources totales", n_total_src)
        km1.metric(f"Sources OK ({cov_period.lower()})", n_ok)
        km2.metric("Sources en attention", n_warn)
        km3.metric("Sources en problème", n_pb)

        filt = st.radio(
            "Filtre",
            ["Avec problèmes seulement", "Toutes les sources"],
            horizontal=True, index=0, key="cov_filter",
        )
        if filt == "Avec problèmes seulement":
            df_display = df_cov_full[df_cov_full["Statut"] != "OK"].copy()
        else:
            df_display = df_cov_full.copy()

        _order = {"PROBLÈME": 0, "ATTENTION": 1, "OK": 2}
        df_display["_o"] = df_display["Statut"].map(_order)
        df_display = df_display.sort_values(
            ["_o", "Articles"], ascending=[True, False]
        ).drop(columns=["_o"])

        if df_display.empty:
            st.success("Toutes les sources sont OK sur cette période.")
        else:
            df_compact = df_display[
                ["Statut", "Source", "Articles", "Contenu", "Auteur", "Analyses"]
            ]
            st.dataframe(
                df_compact, width="stretch", hide_index=True,
                column_config={
                    "Contenu": st.column_config.ProgressColumn(
                        "Contenu", min_value=0, max_value=100, format="%d%%"),
                    "Auteur": st.column_config.ProgressColumn(
                        "Auteur", min_value=0, max_value=100, format="%d%%"),
                    "Analyses": st.column_config.ProgressColumn(
                        "Analyses", min_value=0, max_value=100, format="%d%%",
                        help="Minimum entre Sentiment et Thèmes"),
                },
            )

            with st.expander("Voir le détail par analyse"):
                df_detail = df_display[
                    ["Source", "Articles", "Sentiment", "Thèmes"]
                ]
                st.dataframe(
                    df_detail, width="stretch", hide_index=True,
                    column_config={
                        c: st.column_config.ProgressColumn(
                            c, min_value=0, max_value=100, format="%d%%")
                        for c in ["Sentiment", "Thèmes"]
                    },
                )

    st.markdown("---")
    st.subheader(f"Évolution du taux de traitement, par {GRAIN_LABEL.lower()}")
    st.caption(
        "Selectionnez une source pour voir si elle a commencé à buguer à une date précise. "
        "Démarrage des courbes au 1er avril 2026."
    )
    sources_all_evo = ["(toutes sources)"] + sorted(
        conn.execute(
            "SELECT DISTINCT source_name FROM articles ORDER BY 1"
        ).df()["source_name"].tolist()
    )
    src_evo = st.selectbox("Source", sources_all_evo, key="evo_src")
    src_clause = ""
    src_params = []
    if src_evo != "(toutes sources)":
        src_clause = " AND a.source_name = ?"
        src_params = [src_evo]

    # Tables optionnelles : CTE vide (mais avec la même forme/paramètres) si
    # l'analyse correspondante n'a pas encore tourne, plutôt que de planter.
    _empty_cte_w = (
        f"SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS week, 0 AS n "
        f"FROM articles a WHERE a.published_at >= '2026-04-01' {src_clause} AND FALSE GROUP BY 1"
    )
    snt_w_sql = (
        f"SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS week, COUNT(DISTINCT s.article_id) AS n "
        f"FROM articles a JOIN article_target_sentiment s ON s.article_id = a.id "
        f"WHERE a.published_at >= '2026-04-01' {src_clause} GROUP BY 1"
    ) if has_target_sent else _empty_cte_w
    thm_w_sql = (
        f"SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS week, COUNT(DISTINCT t.article_id) AS n "
        f"FROM articles a JOIN article_topics t ON t.article_id = a.id "
        f"WHERE a.published_at >= '2026-04-01' {src_clause} GROUP BY 1"
    ) if has_topics else _empty_cte_w

    # Filtre date >= 2026-04-01 dans toutes les CTEs
    df_evo = conn.execute(f"""
        WITH weeks AS (
            SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS week,
                   COUNT(*) AS total,
                   SUM(CASE WHEN a.content IS NOT NULL AND LENGTH(a.content) >= 300
                            THEN 1 ELSE 0 END) AS with_content
            FROM articles a
            WHERE a.published_at >= '2026-04-01' {src_clause}
            GROUP BY 1
        ),
        snt_w AS ({snt_w_sql}),
        thm_w AS ({thm_w_sql})
        SELECT w.week, w.total,
               ROUND(100.0 * w.with_content / NULLIF(w.total, 0), 1) AS Contenu,
               ROUND(100.0 * COALESCE(snt_w.n, 0) / NULLIF(w.with_content, 0), 1) AS Sentiment,
               ROUND(100.0 * COALESCE(thm_w.n, 0) / NULLIF(w.with_content, 0), 1) AS "Thèmes"
        FROM weeks w
        LEFT JOIN snt_w ON snt_w.week = w.week
        LEFT JOIN thm_w ON thm_w.week = w.week
        ORDER BY w.week
    """.format(src_clause=src_clause), src_params * 3).df()

    if not df_evo.empty:
        fig_evo = go.Figure()
        line_colors = {"Contenu": "#FFD93D", "Sentiment": "#FF6B6B",
                       "Thèmes": "#6BCB77"}
        for col in ["Contenu", "Sentiment", "Thèmes"]:
            if col in df_evo.columns:
                fig_evo.add_trace(go.Scatter(
                    x=df_evo["week"], y=df_evo[col],
                    mode="lines+markers", name=col,
                    line=dict(color=line_colors[col], width=3),
                    marker=dict(size=9),
                ))
        fig_evo.add_hrect(y0=80, y1=105, fillcolor="green", opacity=0.06, line_width=0)
        fig_evo.add_hrect(y0=0, y1=80, fillcolor="red", opacity=0.06, line_width=0)
        fig_evo.add_hline(y=80, line_dash="dot", line_color="white", line_width=1)
        fig_evo.update_layout(yaxis_title="% articles traités",
                              xaxis_title=GRAIN_LABEL,
                              yaxis_range=[0, 105],
                              legend_title_text="")
        style(fig_evo, 480)
        st.plotly_chart(fig_evo, width="stretch", key=_next_chart_key())

    st.markdown("---")
    st.subheader("Explorer toutes les sources")

    # Vue d'ensemble
    df_overview = conn.execute("""
        SELECT source_name AS Source,
               ANY_VALUE(source_kind) AS Type,
               COUNT(*) AS Articles,
               -- Pour la video, "Articles" compte des segments de
               -- transcription : une emission de 2 h en produit une
               -- soixantaine. On remonte la video parente depuis l'URL du
               -- segment pour que la colonne reste comparable aux autres
               -- sources. Les deux plateformes n'ont pas la meme forme
               -- d'URL : watch?v=<id> pour YouTube, /video/<hash>/ pour
               -- RuTube.
               CASE WHEN ANY_VALUE(source_kind) = 'youtube'
                    THEN COUNT(DISTINCT regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1))
                    WHEN ANY_VALUE(source_kind) = 'tv'
                    THEN COUNT(DISTINCT regexp_extract(url, 'video/([a-f0-9]+)', 1))
                    END AS "Vidéos",
               CAST(MIN(published_at) AS DATE) AS Premier,
               CAST(MAX(published_at) AS DATE) AS Dernier,
               SUM(CASE WHEN published_at >= CURRENT_DATE - INTERVAL '7 days'
                        THEN 1 ELSE 0 END) AS Recent_7j,
               COUNT(DISTINCT author) AS Auteurs_distincts,
               COUNT(DISTINCT language) AS Langues
        FROM articles
        GROUP BY source_name
        ORDER BY Articles DESC
    """).df()
    df_overview["Positionnement historique"] = df_overview["Source"].map(HISTORICAL_STANCE).fillna("")

    st.markdown("**Vue d'ensemble**")
    st.caption("Cliquez sur une colonne pour trier. "
               "Recent_7j = articles publiés sur les 7 derniers jours. "
               "Vidéos = nombre de vidéos ou d'émissions sources, dont chaque "
               "transcription est découpée en plusieurs segments comptés dans "
               "Articles. "
               "Positionnement historique = label éditorial saisi à la main "
               "(config/sources.yaml), pas dérivé des données.")
    st.dataframe(
        df_overview, width="stretch", hide_index=True,
        column_config={
            "Positionnement historique": st.column_config.TextColumn(
                "Positionnement historique", width="large"),
        },
    )

    st.markdown("---")
    st.markdown("**Filtres pour explorer les articles**")

    c1, c2 = st.columns(2)
    sources_list = ["(toutes)"] + df_overview["Source"].tolist()
    with c1:
        sel_source = st.selectbox("Journal", sources_list, key="src_filter")
    with c2:
        sel_keyword = st.text_input("Mot-clé (titre ou contenu)", key="src_keyword")

    c3, c4, c5 = st.columns(3)
    with c3:
        date_start = st.date_input("Du", value=date(2026, 5, 1), key="src_date_start")
    with c4:
        date_end = st.date_input("Au", value=date.today(), key="src_date_end")
    with c5:
        sort_by = st.selectbox(
            "Tri",
            ["Date récente", "Date ancienne", "Source A->Z"],
            key="src_sort",
        )

    where_s = ["1=1"]
    params_s = []
    if sel_source != "(toutes)":
        where_s.append("source_name = ?")
        params_s.append(sel_source)
    if sel_keyword:
        where_s.append("(title ILIKE ? OR content ILIKE ?)")
        params_s.extend([f"%{sel_keyword}%", f"%{sel_keyword}%"])
    if date_start:
        where_s.append("published_at >= ?")
        params_s.append(date_start)
    if date_end:
        where_s.append("published_at < ? + INTERVAL 1 DAY")
        params_s.append(date_end)
    where_full = " AND ".join(where_s)

    order_by = {
        "Date récente": "published_at DESC NULLS LAST",
        "Date ancienne": "published_at ASC NULLS LAST",
        "Source A->Z": "source_name ASC, published_at DESC",
    }[sort_by]

    n_match = conn.execute(
        f"SELECT COUNT(*) FROM articles WHERE {where_full}", params_s
    ).fetchone()[0]

    cc1, cc2 = st.columns([3, 1])
    with cc1:
        st.markdown(f"**{n_match} articles correspondent aux filtres**")
    with cc2:
        limit = st.slider("A afficher", 50, 1000, 200, 50, key="src_limit")

    df_results = conn.execute(
        f"""SELECT published_at, source_name, author, language, title,
                   LEFT(content, 250) AS extrait, url
        FROM articles WHERE {where_full}
        ORDER BY {order_by} LIMIT ?""",
        params_s + [limit]
    ).df()

    st.dataframe(
        df_results, width="stretch", hide_index=True,
        column_config=cols_article(
            extrait=st.column_config.TextColumn("Extrait", width="large")),
    )

    # Stats par source pour les résultats filtres
    if n_match > 0 and sel_source == "(toutes)":
        st.markdown("---")
        st.markdown("**Répartition des résultats par source**")
        df_by_src = conn.execute(
            f"""SELECT source_name, COUNT(*) AS n
            FROM articles WHERE {where_full}
            GROUP BY 1 ORDER BY n DESC""", params_s
        ).df()
        st.dataframe(df_by_src, width="stretch", hide_index=True)


# ===== Tab Signaux (détection de changements) ===================
with tab_signaux:
    st.subheader("Signaux : ce qui change")
    st.caption(
        "Compare une période récente à la période de même durée qui la "
        "precede immédiatement (mêmes filtres source/langue/mots-clés que "
        "la barre laterale, hors filtre de date). Objectif : repérer un nom "
        "qui commence à circuler (ex: Yabloko), ou un traitement de Poutine "
        "/ de la guerre en Ukraine qui se durcit ou s'adoucit, sans avoir à "
        "comparer les onglets à la main."
    )

    from datetime import timedelta as _td

    window_days = st.select_slider(
        "Fenêtre de comparaison (jours)", options=[7, 14, 30, 45, 60, 90],
        value=30, key="sig_window",
        help="Sur une fenêtre courte, un jour de collecte manquant suffit a "
             "faire apparaître de faux signaux. 30 jours ou plus lisse ces "
             "a-coups.",
    )
    _today = date.today()
    recent_start = _today - _td(days=window_days - 1)
    recent_end_excl = _today + _td(days=1)
    ref_end_excl = recent_start
    ref_start = ref_end_excl - _td(days=window_days)
    recent_bounds = [recent_start, recent_end_excl]
    ref_bounds = [ref_start, ref_end_excl]
    st.caption(
        f"Récente : {recent_start} -> {_today} ({window_days}j)  |  "
        f"Référence : {ref_start} -> {ref_end_excl - _td(days=1)} ({window_days}j)"
    )

    # Volume de chaque fenêtre. Tout le reste de l'onglet en dépend : comparer
    # des effectifs bruts entre deux fenêtres de tailles différentes fait
    # passer une simple montée en charge de la collecte pour un signal.
    _vol_sql = (f"SELECT COUNT(*) FROM articles a WHERE a.published_at >= ? "
                f"AND a.published_at < ? AND {with_a(WHERE_NODATE)}")
    n_recent_total = conn.execute(_vol_sql, [*recent_bounds, *params_nodate]).fetchone()[0]
    n_ref_total = conn.execute(_vol_sql, [*ref_bounds, *params_nodate]).fetchone()[0]

    cvol1, cvol2, cvol3 = st.columns(3)
    cvol1.metric("Articles, période récente", f"{n_recent_total:,}".replace(",", " "))
    cvol2.metric("Articles, période de référence", f"{n_ref_total:,}".replace(",", " "))
    _ratio = (n_recent_total / n_ref_total) if n_ref_total else float("inf")
    cvol3.metric("Rapport de volume", "n/a" if n_ref_total == 0 else f"x{_ratio:.1f}")

    # Le corpus est jeune et la collecte est montée en charge : les fenêtres
    # anciennes peuvent être quasi vides. Le signaler explicitement, sinon on
    # lit des variations qui ne disent rien du discours mediatique.
    _MIN_WINDOW_ARTICLES = 100
    _comparable = True
    if n_ref_total < _MIN_WINDOW_ARTICLES or n_recent_total < _MIN_WINDOW_ARTICLES:
        _comparable = False
        st.warning(
            f"Fenêtres trop peu fournies pour conclure ({n_recent_total} vs "
            f"{n_ref_total} articles, seuil {_MIN_WINDOW_ARTICLES}). La collecte "
            f"a démarré récemment : élargissez la fenêtre ou attendez que "
            f"l'historique s'etoffe."
        )
    elif _ratio > 1.5 or _ratio < 0.67:
        st.warning(
            f"Les deux périodes n'ont pas le même volume (x{_ratio:.1f}). Les "
            f"comparaisons ci-dessous sont faites en **part du corpus** et non "
            f"en nombre d'articles, ce qui neutralise l'écart -- mais un rapport "
            f"aussi marque reste à garder en tête."
        )

    aw_nodate = with_a(WHERE_NODATE)

    def compare_periods(sql, extra_params=None):
        extra_params = extra_params or []
        df_r = conn.execute(sql, [*recent_bounds, *extra_params]).df()
        df_p = conn.execute(sql, [*ref_bounds, *extra_params]).df()
        return df_r, df_p

    def _share(df, key_col):
        s = df.set_index(key_col)["n"]
        total = s.sum()
        return (s / total * 100) if total else s

    if not (has_target_sent or has_topics):
        st.info("Aucune analyse disponible : lancez update.py.")

    # --- Dérive de la posture par cible (sentiment) --------------------
    if has_target_sent:
        st.markdown("---")
        st.markdown("### Dérive de la posture par cible")
        st.caption(
            "Lean moyen (-1 anti / +1 pro) par cible géopolitique, période "
            "récente vs référence -- utile pour voir si le traitement de "
            "l'Ukraine, de l'Occident, etc. se durcit ou s'adoucit. "
            "Une moyenne étant déjà independante du volume, seules les cibles "
            "ayant au moins 10 articles dans chacune des deux périodes sont "
            "retenues."
        )
        sent_sql = f"""
            SELECT s.target,
                   SUM(s.lean * s.score) / NULLIF(SUM(s.score), 0) AS lean,
                   COUNT(*) AS n
            FROM article_target_sentiment s JOIN articles a ON a.id = s.article_id
            WHERE s.stance != 'non_concerne' AND a.published_at >= ? AND a.published_at < ?
              AND {aw_nodate}
            GROUP BY 1
        """
        df_s_recent, df_s_prior = compare_periods(sent_sql, params_nodate)
        merged_s = df_s_recent.rename(columns={"lean": "lean_recent", "n": "n_recent"}).merge(
            df_s_prior.rename(columns={"lean": "lean_prior", "n": "n_prior"}),
            on="target", how="outer")
        cols4 = ["lean_recent", "lean_prior", "n_recent", "n_prior"]
        merged_s[cols4] = merged_s[cols4].fillna(0)
        # Une moyenne de lean sur 3 articles n'a aucune stabilité : elle bouge
        # de plusieurs dixiemes des qu'un article change de bord. Seuil relevé
        # à 10 pour que le delta affiche traduise une tendance et non le
        # hasard d'echantillonnage.
        _MIN_SENT = 10
        merged_s = merged_s[(merged_s["n_recent"] >= _MIN_SENT)
                            & (merged_s["n_prior"] >= _MIN_SENT)].copy()

        if merged_s.empty:
            st.info("Pas assez d'articles dans les deux périodes pour comparer les postures.")
        else:
            merged_s["delta"] = merged_s["lean_recent"] - merged_s["lean_prior"]
            merged_s["target_label"] = merged_s["target"].map(TARGET_LABELS)
            merged_s = merged_s.reindex(
                merged_s["delta"].abs().sort_values(ascending=False).index)
            top_movers = merged_s.head(8)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_movers["target_label"], x=top_movers["delta"], orientation="h",
                marker_color=["#FF6B6B" if d < 0 else "#4C9AFF" for d in top_movers["delta"]],
                text=[f"{d:+.2f}" for d in top_movers["delta"]], textposition="outside",
            ))
            fig.add_vline(x=0, line_color="white", line_width=1)
            fig.update_layout(xaxis_title="Delta du lean moyen (récent - référence)",
                              yaxis_autorange="reversed")
            style(fig, max(350, 45 * len(top_movers)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

            df_disp = merged_s[["target_label", "lean_prior", "lean_recent", "delta",
                                 "n_prior", "n_recent"]].round(2)
            df_disp.columns = ["Cible", "Lean (référence)", "Lean (récent)", "Delta",
                                "N (référence)", "N (récent)"]
            st.dataframe(df_disp, width="stretch", hide_index=True)

    # --- Dérive des thèmes traités --------------------------------------
    if has_topics:
        st.markdown("---")
        st.markdown("### Dérive des thèmes traités")
        st.caption(
            "Part de chaque thème (cluster BERTopic) dans le corpus, période "
            "récente vs référence. Un thème qui gagne ou perd du terrain "
            "signale un changement d'agenda éditorial ; un thème absent d'une "
            "des deux périodes peut correspondre à une apparition/disparition."
        )
        theme_sql = f"""
            SELECT t.label, COUNT(*) AS n
            FROM article_topics at_ JOIN articles a ON a.id = at_.article_id
            JOIN topics t ON t.topic_key = at_.topic_key
            WHERE t.topic_key != -1 AND a.published_at >= ? AND a.published_at < ? AND {aw_nodate}
            GROUP BY 1
        """
        df_t_recent, df_t_prior = compare_periods(theme_sql, params_nodate)
        pct_t_recent = _share(df_t_recent, "label")
        pct_t_prior = _share(df_t_prior, "label")
        theme_cmp = pd.DataFrame({"recent": pct_t_recent, "prior": pct_t_prior}).fillna(0)

        if theme_cmp.empty or theme_cmp[["recent", "prior"]].to_numpy().sum() == 0:
            st.info("Pas assez de données pour comparer les thèmes sur ces deux périodes.")
        else:
            theme_cmp["delta"] = theme_cmp["recent"] - theme_cmp["prior"]
            theme_cmp = theme_cmp.reindex(
                theme_cmp["delta"].abs().sort_values(ascending=False).index).head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=theme_cmp.index, x=theme_cmp["delta"], orientation="h",
                marker_color=["#FF6B6B" if d < 0 else "#6BCB77" for d in theme_cmp["delta"]],
                text=[f"{d:+.1f}pt" for d in theme_cmp["delta"]], textposition="outside",
            ))
            fig.add_vline(x=0, line_color="white", line_width=1)
            fig.update_layout(xaxis_title="Delta de part (points, récent - référence)",
                              yaxis_autorange="reversed")
            style(fig, max(350, 40 * len(theme_cmp)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

            df_disp = theme_cmp.round(1).reset_index().rename(
                columns={"label": "Thème", "recent": "% récent",
                         "prior": "% référence", "delta": "Delta (pt)"})
            st.dataframe(df_disp, width="stretch", hide_index=True)


# ===== Tab Cadrage lexical (vocabulaire de propagande, par source) ======
with tab_cadrage:
    st.subheader("Cadrage lexical")
    st.caption(
        "Deux familles mesurées de la même façon (fréquence d'un terme dans "
        "le texte), mais qui se lisent différemment. **Cadrage (propagande)** : "
        "vocabulaire documente par la littérature sur la propagande russe "
        "(monde russe, dénazification, russophobie...) -- mesure la PRÉSENCE "
        "du terme, pas l'adhesion : un média independant peut très bien citer "
        "ou critiquer ces mêmes termes. **Indicateur de suivi** : "
        "thermometres suivis en permanence même a bas bruit (mobilisation, "
        "signaux de negociation, stress économique), là où le clustering de "
        "l'onglet Thèmes ne fait émerger un sujet que s'il devient dense. "
        "Pour la posture éditoriale, voir l'onglet Sentiment."
    )

    aw = with_a(WHERE)
    n_total_ru = conn.execute(
        f"SELECT COUNT(*) FROM articles a WHERE {aw} AND a.language = 'ru' "
        f"AND a.content IS NOT NULL", params).fetchone()[0]

    if n_total_ru == 0:
        st.info("Aucun article russophone avec contenu sur la période/filtres choisis.")
    else:
        st.markdown("### Vue d'ensemble")
        overview_rows = []
        for term_label, pattern in ALL_LEXICAL_TERMS.items():
            n = conn.execute(
                f"SELECT COUNT(*) FROM articles a WHERE {aw} AND a.language = 'ru' "
                f"AND a.content IS NOT NULL AND regexp_matches(LOWER(a.content), ?)",
                [*params, pattern]).fetchone()[0]
            overview_rows.append({
                "Catégorie": LEXICAL_CATEGORY[term_label],
                "Terme": term_label, "Articles": n,
                "% du corpus": round(100 * n / n_total_ru, 2),
            })
        # Tri par volume et non par catégorie : les indicateurs de suivi sont
        # bien plus frequents que les termes de propagande, les grouper par
        # catégorie les enterrerait sous la ligne de flottaison du tableau.
        df_overview_terms = pd.DataFrame(overview_rows).sort_values(
            "Articles", ascending=False)
        st.dataframe(df_overview_terms, width="stretch", hide_index=True)
        st.caption(f"Sur {n_total_ru} articles russophones avec contenu (filtres actuels).")

        st.markdown("### Explorer un terme")
        term_label = st.selectbox(
            "Terme", list(ALL_LEXICAL_TERMS.keys()), key="framing_term",
            format_func=lambda t: f"[{LEXICAL_CATEGORY[t]}] {t}")
        pattern = ALL_LEXICAL_TERMS[term_label]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Évolution par {GRAIN_LABEL.lower()} (% des articles russophones)**")
            df_evo_term = conn.execute(
                f"""SELECT DATE_TRUNC('{GRAIN}', a.published_at) AS semaine,
                           COUNT(*) AS total,
                           SUM(CASE WHEN regexp_matches(LOWER(a.content), ?)
                                    THEN 1 ELSE 0 END) AS n
                    FROM articles a WHERE {aw} AND a.language = 'ru'
                      AND a.content IS NOT NULL AND a.published_at IS NOT NULL
                    GROUP BY 1 ORDER BY 1""",
                [pattern, *params]).df()
            if df_evo_term.empty or df_evo_term["total"].sum() == 0:
                st.caption("Pas assez de données.")
            else:
                df_evo_term["pct"] = 100 * df_evo_term["n"] / df_evo_term["total"]
                fig = px.line(df_evo_term, x="semaine", y="pct", markers=True,
                              labels={"semaine": GRAIN_LABEL, "pct": "% articles"})
                fig.update_traces(line_color="#FFD93D", line_width=3, marker_size=8)
                style(fig, 400)
                st.plotly_chart(fig, width="stretch", key=_next_chart_key())
        with c2:
            st.markdown("**Par source (analyse par journal)**")
            df_src_term = conn.execute(
                f"""SELECT a.source_name, ANY_VALUE(a.type_media) AS type_media,
                           COUNT(*) AS total,
                           SUM(CASE WHEN regexp_matches(LOWER(a.content), ?)
                                    THEN 1 ELSE 0 END) AS n
                    FROM articles a WHERE {aw} AND a.language = 'ru'
                      AND a.content IS NOT NULL
                    GROUP BY 1""",
                [pattern, *params]).df()
            df_src_term = df_src_term[df_src_term["total"] >= 5].copy()
            if df_src_term.empty:
                st.caption("Pas assez de données par source.")
            else:
                df_src_term["pct"] = 100 * df_src_term["n"] / df_src_term["total"]
                df_src_term = df_src_term[df_src_term["n"] > 0].sort_values(
                    "pct", ascending=False)
                if df_src_term.empty:
                    st.caption("Aucune source ne mentionne ce terme sur la période.")
                else:
                    type_colors = {"etat": "#FF6B6B", "para_etat": "#FFA94D",
                                   "independant": "#4C9AFF", "exil": "#6BCB77"}
                    fig = px.bar(
                        df_src_term, x="pct", y="source_name", orientation="h",
                        color="type_media", color_discrete_map=type_colors,
                        hover_data={"n": True, "total": True},
                        labels={"pct": "% des articles", "source_name": "",
                                "type_media": "Type"})
                    fig.update_yaxes(autorange="reversed")
                    style(fig, max(300, 32 * len(df_src_term)))
                    st.plotly_chart(fig, width="stretch", key=_next_chart_key())

    # --- Ce qui distingue chaque famille de médias -----------------------
    # Divergence de Kullback-Leibler, d'après Vestel & Degaetano-Ortlieb
    # (ICWSM 2025). Répond au défaut des agrégats melanges : ici on n'additionne
    # jamais les familles, on les oppose.
    if has_divergence:
        st.markdown("---")
        st.subheader("Ce qui distingue chaque famille de médias")
        st.caption(
            "Les mots qui rendent un groupe reconnaissable face à tous les "
            "autres, mesures par divergence de Kullback-Leibler. A la "
            "difference des graphiques qui melangent les sources, cette vue "
            "est contrastive par construction : elle ne calcule pas une "
            "moyenne du corpus, elle oppose des groupes. Un mot n'est retenu "
            "que s'il est employé par plusieurs sources du groupe."
        )
        c_axe, c_grp = st.columns([1, 2])
        with c_axe:
            axe = st.radio("Comparer par", ["type_media", "source_kind"],
                           format_func=lambda a: {"type_media": "Type de média",
                                                  "source_kind": "Nature du contenu"}[a],
                           key="div_axe", horizontal=True)
        groupes = conn.execute(
            "SELECT DISTINCT groupe FROM lexical_divergence WHERE axe = ? "
            "ORDER BY groupe", [axe]).df()["groupe"].tolist()
        with c_grp:
            groupe = st.selectbox("Groupe", groupes, key="div_groupe")

        df_div = conn.execute(
            """SELECT token, contribution, freq_groupe, freq_reste, n_groupe
               FROM lexical_divergence WHERE axe = ? AND groupe = ?
               ORDER BY rang LIMIT 25""", [axe, groupe]).df()
        if df_div.empty:
            st.info("Aucun mot distinctif pour ce groupe.")
        else:
            df_div["rapport"] = (df_div["freq_groupe"] /
                                 df_div["freq_reste"].clip(lower=0.01))
            fig = px.bar(df_div.iloc[::-1], x="contribution", y="token",
                         orientation="h", color="contribution",
                         color_continuous_scale="Tealgrn",
                         hover_data={"freq_groupe": ":.2f", "freq_reste": ":.2f",
                                     "rapport": ":.1f", "n_groupe": True,
                                     "contribution": False},
                         labels={"contribution": "Contribution a la divergence",
                                 "token": ""})
            fig.update_layout(coloraxis_showscale=False)
            # 30 px par barre : en dessous, Plotly masque un libelle sur
            # deux et la moitié du vocabulaire devient invisible.
            style(fig, max(420, 30 * len(df_div)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())
            st.caption(
                "Au survol : fréquence pour 10 000 mots dans le groupe et "
                "hors du groupe, et leur rapport. Les noms propres de médias "
                "et les formules de plateforme survivent parfois au filtrage "
                "-- ils sont distinctifs sans rien dire du discours."
            )

    # --- Procédés de persuasion ------------------------------------------
    # Taxonomie de Da San Martino et al. (EMNLP 2019), grille des plateaux de
    # Gulenko (2021). Les thèmes disent de quoi on parlé, le sentiment envers
    # qui on penche ; ceci dit comment le texte cherche a convaincre.
    if has_techniques:
        st.markdown("---")
        st.subheader("Procédés de persuasion")
        st.caption(
            "Repérage des procédés rhétoriques, fragment par fragment. "
            "Analyse limitée à la télévision et a YouTube, là où le cadrage "
            "est explicite. **Un procédé n'est pas un mensonge** : c'est une "
            "forme d'argumentation, et un média peut l'employer sur un fait "
            "exact."
        )
        df_tech = conn.execute(
            f"""SELECT t.technique, a.source_kind, a.type_media, COUNT(*) AS n
                FROM article_techniques t JOIN articles a ON a.id = t.article_id
                WHERE {aw} GROUP BY 1, 2, 3""", params).df()
        if df_tech.empty:
            st.info("Aucun procédé relevé sur la période et les filtres choisis.")
        else:
            base = conn.execute(
                f"""SELECT a.source_kind, COUNT(*) AS n FROM articles a
                    WHERE {aw} AND a.source_kind IN ('tv', 'youtube')
                    GROUP BY 1""", params).df()
            denom = dict(zip(base["source_kind"], base["n"]))
            df_tech["nature"] = df_tech["source_kind"].map(
                lambda k: UNITE_NATURE.get(k, k))
            # Un taux, pas un effectif : la TV et YouTube n'ont pas le même
            # nombre de segments, les comparer en brut serait trompeur.
            df_tech["pour_100"] = df_tech.apply(
                lambda r: 100 * r["n"] / max(denom.get(r["source_kind"], 1), 1),
                axis=1)
            agg = df_tech.groupby(["technique", "nature"], as_index=False)[
                "pour_100"].sum()
            ordre = (agg.groupby("technique")["pour_100"].sum()
                     .sort_values().index.tolist())
            fig = px.bar(agg, x="pour_100", y="technique", color="nature",
                         orientation="h", barmode="group",
                         category_orders={"technique": ordre},
                         color_discrete_sequence=TOP_PALETTE,
                         labels={"pour_100": "Segments concernés (%)",
                                 "technique": "", "nature": ""})
            style(fig, max(420, 26 * len(ordre)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

            with st.expander("Voir des exemples relevés"):
                tech_choisie = st.selectbox(
                    "Procédé", sorted(agg["technique"].unique()), key="tech_ex")
                df_ex = conn.execute(
                    f"""SELECT a.source_name, a.title, t.extrait, t.confiance, a.url
                        FROM article_techniques t
                        JOIN articles a ON a.id = t.article_id
                        WHERE t.technique = ? AND {aw}
                        ORDER BY t.confiance DESC LIMIT 15""",
                    [tech_choisie, *params]).df()
                st.dataframe(df_ex, width="stretch", hide_index=True,
                             column_config=cols_article())
                st.caption(
                    "L'extrait cite est le fragment sur lequel le modèle "
                    "s'appuie : il permet de vérifier chaque relevé.")


# ===== Tab Contexte (paysage mediatique russe) ==========================
# Onglet volontairement statique : il ne lit pas la base. Son rôle est de
# donner a quelqu'un qui découvre l'outil de quoi interpreter les autres
# onglets -- savoir que Telegram pèse peu dans la population réelle change la
# lecture d'un graphique ou Telegram représente un cinquieme du corpus.
with tab_contexte:
    st.subheader("Comment les Russes s'informent")
    st.caption(
        "Repères pour lire les autres onglets. Sources : Mediascope "
        "(mesure d'audience, T1 2026) et Levada (sondages, avril et juin 2026). "
        "Ces chiffres ne viennent pas du corpus collecte : ils servent à le "
        "situer."
    )

    # delta_color="off" : ces libellés secondaires sont des valeurs absolues,
    # pas des variations. Colorés, ils se liraient comme des hausses --
    # et une confiance en baisse affichée en vert serait contresens.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Télévision", "97 %", "3 h 18 / jour", delta_color="off",
              help="Part des Russes qui regardent la télévision au moins une "
                   "fois par semaine. 99 % chez les 55 ans et plus.")
    m2.metric("Internet", "86 %", "4 h 21 / jour", delta_color="off",
              help="105 millions de personnes, soit 86 % des 12 ans et plus.")
    m3.metric("Réseaux sociaux", "51 %", "2 h 44 / jour", delta_color="off",
              help="Part du temps passe en ligne. VKontakte, Telegram et "
                   "TikTok en tête.")
    m4.metric("Confiance : TV", "41 %", "-8 pts depuis mai 2025",
              delta_color="off",
              help="Premier rang malgre l'erosion. Réseaux sociaux 21 %, "
                   "sites d'info 14 %, chaînes Telegram 11 %.")

    st.markdown("### Ou les Russes prennent leur information")
    df_src_info = pd.DataFrame({
        "Support": ["Télévision", "Réseaux sociaux", "Sites d'information",
                    "Chaînes Telegram"],
        "Utilisent comme source": [55, 33, 25, 20],
        "Lui font confiance": [41, 21, 14, 11],
    })
    fig_ctx = go.Figure()
    fig_ctx.add_trace(go.Bar(
        y=df_src_info["Support"], x=df_src_info["Utilisent comme source"],
        name="S'en servent comme source", orientation="h", marker_color="#4C9AFF",
        hovertemplate="%{y}<br>%{x} % s'en servent<extra></extra>"))
    fig_ctx.add_trace(go.Bar(
        y=df_src_info["Support"], x=df_src_info["Lui font confiance"],
        name="Lui font confiance", orientation="h", marker_color="#00C2A8",
        hovertemplate="%{y}<br>%{x} % lui font confiance<extra></extra>"))
    fig_ctx.update_layout(barmode="group", xaxis_title="% des sondes",
                          legend_title_text="")
    fig_ctx.update_yaxes(categoryorder="array",
                         categoryarray=list(reversed(df_src_info["Support"])))
    style(fig_ctx, 340)
    st.plotly_chart(fig_ctx, width="stretch", key=_next_chart_key())
    st.caption(
        "L'écart entre les deux barres se lit comme un usage par défaut : on "
        "regarde la télévision sans forcement y croire. La confiance recule "
        "partout, le plus fortement pour Telegram (-9 points en un an, après "
        "les blocages)."
    )

    st.markdown("### Une fracture par age")
    df_age = pd.DataFrame({
        "Tranche": ["18-24 ans", "25-39 ans", "40-54 ans", "55 ans et plus"],
        "Télévision": [30, 45, 70, 90],
        "Internet et réseaux": [90, 85, 65, 35],
    })
    fig_age = px.line(df_age, x="Tranche", y=["Télévision", "Internet et réseaux"],
                      markers=True,
                      color_discrete_map={"Télévision": "#FF6B6B",
                                          "Internet et réseaux": "#4C9AFF"},
                      labels={"value": "% de la tranche", "variable": "",
                              "Tranche": ""})
    style(fig_age, 320)
    st.plotly_chart(fig_age, width="stretch", key=_next_chart_key())
    st.caption(
        "Ordres de grandeur, pas des mesures exactes : ils illustrent le "
        "croisement, documente par Mediascope, entre une audience télévisée "
        "âgée et une audience en ligne jeune. Les 18-34 ans regardent une "
        "heure de télévision de moins qu'en 2020. **Consequence pour l'outil** : "
        "une source ne pèse pas de la même façon selon le public vise -- la "
        "télévision touche l'electorat le plus age et le plus nombreux, "
        "YouTube et Telegram un public plus jeune et urbain."
    )

    st.markdown("### Les supports, un par un")
    MEDIA_NOTES = [
        ("Télévision", "97 % de couverture hebdomadaire",
         "Premier support d'information du pays et le plus cru. Trois chaînes "
         "dominent : Rossiya 1 (13,3 % de part d'audience), NTV, Pervyi Kanal "
         "(7,1 %). L'information y est encadrée par l'État, et les talk-shows "
         "politiques (« Vremya pokazhet », « Bolshaya igra », Soloviev) y "
         "formulent la ligne officielle de façon bien plus explicite que les "
         "journaux. **Dans l'outil** : transcrits automatiquement depuis "
         "RuTube, faute de sous-titres."),
        ("Sites de presse", "~25 % s'y informent, 14 % leur font confiance",
         "Va des agences d'État (TASS, RIA Novosti, RT) aux médias en exil "
         "(Meduza, Novaya Gazeta Europe, The Insider), en passant par une "
         "presse économique moins alignée (Vedomosti, RBK, The Bell). La "
         "plupart des titres independants ont été déclarés « agent étranger » "
         "ou « organisation indésirable » et sont bloqués en Russie -- ils "
         "publient depuis l'étranger pour un public qui les lit par VPN. "
         "**Dans l'outil** : le socle du corpus."),
        ("Telegram", "~20 % des sondes, 11 % de confiance",
         "Longtemps l'espace le plus libre du RuNet, en net recul depuis les "
         "restrictions. On y trouve les canaux officiels des médias, mais "
         "surtout les « voenkory » (correspondants de guerre pro-guerre comme "
         "Rybar ou Colonelcassad) et des tabloïdes a forte audience (Mash, "
         "Baza, SHOT) nourris de fuites policières. **Dans l'outil** : utile "
         "pour la vitesse et pour le discours militaire, mais son poids dans "
         "le corpus dépasse son poids réel dans la population."),
        ("VKontakte", "1er réseau social, 1er poste de temps en ligne",
         "L'équivalent russe de Facebook, propriété d'un groupe proche de "
         "l'État. Les grands médias y touchent un public qui ne visite jamais "
         "leur site, avec des formulations souvent plus directes. **Dans "
         "l'outil** : collecte limitée, VK opposant une verification "
         "anti-robot aux visiteurs automatises."),
        ("YouTube", "~22 M/jour, ralenti en Russie depuis 2024",
         "Principal espace de discours critique encore accessible. Les "
         "chaînes les plus vues (Maxime Kats, Varlamov, Chtefanov, vDud) sont "
         "toutes animées depuis l'exil et classées « agent étranger ». Les "
         "chaînes d'État, elles, en ont été retirées. **Dans l'outil** : "
         "attention, ce volet est structurellement d'opposition -- ce n'est "
         "pas un échantillon de la vidéo russophone."),
        ("Radio et Dzen", "audiences secondaires",
         "Vesti FM et Radio Rossii pour la radio ; Dzen, plateforme "
         "d'articles de Yandex, touche 31 millions de personnes par jour mais "
         "melange information et contenu de divertissement. **Dans l'outil** : "
         "non couverts à ce jour."),
    ]
    for titre, chiffre, texte in MEDIA_NOTES:
        with st.expander(f"{titre} — {chiffre}"):
            st.markdown(texte)

    st.info(
        "**A garder en tête en lisant les autres onglets.** La composition du "
        "corpus ne reproduit pas celle de la consommation réelle : la presse "
        "web et Telegram y sont surrepresentes parce qu'ils sont faciles a "
        "collecter, la télévision sous-représentée parce que chaque émission "
        "doit être transcrite. Un thème dominant dans le corpus n'est donc pas "
        "forcement un thème dominant dans ce que les Russes voient. Le filtre "
        "« Nature du contenu », à gauche, sert précisément à corriger cette "
        "lecture -- en isolant la télévision, par exemple."
    )


# ===== Onglet Couverture =======================================
# Ce que les autres onglets analysent, ce panneau le recense : sources
# suivies, volumes, et surtout celles qui ne rapportent rien. Il ignore
# volontairement les filtres -- une fiche de couverture qui retrecit quand
# on filtre ne renseigne plus.
with tab_couverture:
    st.caption("Ce qui est suivi, source par source, indépendamment des "
               "filtres appliqués à gauche. Les émissions de télévision "
               "sont regroupées par chaîne, avec leur part d'audience "
               "nationale.")
    _cov_src = conn.execute(f"""
        WITH base AS (
            SELECT source_kind, source_name, {SQL_PARENT} AS parent,
                   {SQL_OFFSET_S} AS t_s, {SQL_MOTS} AS mots,
                   view_count AS vues, published_at
            FROM articles
        ),
        par_parent AS (
            -- Les vues appartiennent a la video, pas au segment : on prend le MAX
            -- par unite avant de sommer, sinon une video de 60 segments compterait
            -- 60 fois son audience.
            SELECT source_name, parent, MAX(t_s) AS fin_s, MAX(vues) AS vues
            FROM base GROUP BY 1, 2
        ),
        duree AS (
            SELECT source_name, SUM(fin_s) AS secondes, SUM(vues) AS vues
            FROM par_parent GROUP BY 1
        )
        SELECT b.source_name, ANY_VALUE(b.source_kind) AS nature,
               COUNT(DISTINCT b.parent) AS unites, COUNT(*) AS segments,
               SUM(b.mots) AS mots, MAX(b.published_at) AS dernier,
               ANY_VALUE(d.secondes) AS secondes, ANY_VALUE(d.vues) AS vues
        FROM base b LEFT JOIN duree d USING (source_name)
        GROUP BY 1
    """).df().set_index("source_name")

    # Part d'audience TV : Mediascope, ensemble de la Russie, 2e trimestre 2026.
    # C'est la seule mesure d'audience réelle disponible -- les vues RuTube ou
    # YouTube mesurent le rattrapage en ligne, pas l'antenne.
    CHAINES = {
        "Rossiya 1": ("13,4 % de part d'audience nationale (1re chaîne)",
                      "Chaîne phare de VGTRK, groupe public. Ses talk-shows de "
                      "soirée sont le lieu où la ligne officielle est énoncée le "
                      "plus explicitement."),
        "NTV": ("9,5 % (2e chaîne)",
                "Groupe Gazprom-Média. Même ligne éditoriale que les chaînes "
                "publiques, registre plus sensationnaliste -- faits divers, "
                "sécurité, plateaux houleux."),
        "Pervyi Kanal": ("7,5 % (3e chaîne)",
                         "Héritière de la 1re chaîne sovietique, État actionnaire "
                         "majoritaire. Son journal de 21 h « Время » reste le "
                         "programme d'information de référence du pays."),
        "Soloviev LIVE": ("chaîne en ligne, hors mesure d'antenne",
                          "Studio personnel de Vladimir Soloviev (RuTube, "
                          "Telegram, VK). Formats très longs, 4 h, sans les "
                          "contraintes de l'antenne : le ton y est plus libre que "
                          "sur Rossiya 1."),
    }

    NATURES = [
        ("tv", "Télévision", "emissions"),
        ("press", "Presse", "articles"),
        ("telegram", "Telegram", "posts"),
        ("youtube", "YouTube", "videos"),
        ("vk", "VKontakte", "posts"),
    ]


    def _stat(nom):
        """Ligne de statistiques d'une source, ou des zéros si elle n'a rien
        rapporté -- une source configurée mais muette doit rester visible."""
        if nom not in _cov_src.index:
            return dict(unites=0, segments=0, mots=0, minutes=0, vues=None, dernier="")
        r = _cov_src.loc[nom]

        # `x or 0` ne suffit pas : les agrégats absents remontent en NaN, et
        # `NaN or 0` vaut NaN -- la presse, qui n'a ni durée ni vues, faisait alors
        # échouer la conversion en entier.
        def _n(v):
            return 0 if pd.isna(v) else int(v)

        return dict(unites=_n(r["unites"]), segments=_n(r["segments"]),
                    mots=_n(r["mots"]), minutes=_n(r["secondes"]) // 60,
                    vues=(None if pd.isna(r["vues"]) else int(r["vues"])),
                    dernier=fr_date(r["dernier"]))


    def _table(lignes, colonnes):
        df = pd.DataFrame(lignes)
        for c in ("Vues", "Minutes"):
            if c in df.columns:
                df[c] = col_entier(df[c])
        st.dataframe(df[colonnes], width="stretch", hide_index=True)


    for kind, libelle, unite in NATURES:
        cfg = [s for s in SOURCE_CONFIG if CFG_KIND.get(s.get("type", "rss"), "press") == kind]
        if not cfg:
            continue
        en_dev = any(s.get("status") == "en_developpement" for s in cfg)
        tot = sum(_stat(s["name"])["unites"] for s in cfg)
        titre = (f"{libelle} — {len(cfg)} sources suivies, "
                 f"{tot:,} {unite} collectée(s)".replace(",", " "))
        if en_dev:
            titre += "  ·  en développement"

        with st.expander(titre, expanded=False):
            if en_dev:
                st.warning(
                    "**Collecte en développement.** VK oppose une verification "
                    "anti-robot après une vingtaine de visites anonymes depuis une "
                    "même adresse : en pratique deux à trois communautes passent "
                    "par run, pas les cinq. Les chiffres ci-dessous sont donc un "
                    "plancher, et l'absence d'une communaute un jour donne ne dit "
                    "rien de son activité.")

            if kind == "tv":
                # Regroupement par chaîne : c'est la chaîne qui porte l'audience,
                # pas l'émission.
                par_chaine = {}
                for s in cfg:
                    par_chaine.setdefault(s.get("channel", "Autres"), []).append(s)
                for chaine, emissions in sorted(
                        par_chaine.items(),
                        key=lambda kv: -sum(_stat(s["name"])["mots"] for s in kv[1])):
                    part, note = CHAINES.get(chaine, ("", ""))
                    st.markdown(f"**{chaine}**" + (f" — {part}" if part else ""))
                    if note:
                        st.caption(note)
                    lignes = []
                    for s in sorted(emissions, key=lambda x: x["name"]):
                        st_ = _stat(s["name"])
                        lignes.append({
                            "Émission": s["name"].split(" (")[0],
                            "Acces": {"rutube": "RuTube", "hls": "flux intercepte"}
                                     .get(s.get("type"), s.get("type", "")),
                            "Épisodes": st_["unites"], "Segments": st_["segments"],
                            "Mots": st_["mots"], "Minutes": st_["minutes"],
                            "Vues": st_["vues"], "Dernier": st_["dernier"],
                        })
                    _table(lignes, ["Émission", "Acces", "Épisodes", "Segments",
                                    "Mots", "Minutes", "Vues", "Dernier"])
                    for s in sorted(emissions, key=lambda x: x["name"]):
                        if s.get("historical_stance"):
                            st.caption(f"**{s['name'].split(' (')[0]}** — "
                                       f"{s['historical_stance']}")
                    st.markdown("")
                st.caption(
                    "**Vues** : lectures en ligne sur RuTube, publiées par la "
                    "plateforme et renseignées depuis le 15 aout 2026 -- vides "
                    "pour les épisodes collectes avant. Elles ne mesurent PAS "
                    "l'audience d'antenne : « Время » reunit plusieurs millions de "
                    "telespectateurs pour quelques milliers de vues RuTube. Pour "
                    "l'audience réelle, c'est la part de chaîne ci-dessus qui fait "
                    "foi. **Minutes** : estimation basse, tirée de l'instant de "
                    "depart du dernier segment de chaque épisode.")
            else:
                lignes = []
                for s in sorted(cfg, key=lambda x: -_stat(x["name"])["mots"]):
                    st_ = _stat(s["name"])
                    ligne = {
                        "Source": s["name"], "Type": s.get("media_type", ""),
                        "Statut légal": s.get("legal_status", ""),
                        unite.capitalize(): st_["unites"],
                        "Mots": st_["mots"], "Dernier": st_["dernier"],
                    }
                    if kind == "youtube":
                        ligne["Segments"] = st_["segments"]
                        ligne["Minutes"] = st_["minutes"]
                        ligne["Vues"] = st_["vues"]
                    lignes.append(ligne)
                cols = ["Source", "Type", "Statut légal", unite.capitalize()]
                if kind == "youtube":
                    cols += ["Segments", "Minutes", "Vues"]
                cols += ["Mots", "Dernier"]
                _table(lignes, cols)
                if kind == "youtube":
                    st.caption(
                        "Volet structurellement d'opposition : les quatre chaînes "
                        "les plus vues du YouTube politique russophone sont animées "
                        "depuis l'exil et classées « agent étranger ». Les chaînes "
                        "d'État, elles, ont été retirées de la plateforme. Ce n'est "
                        "donc pas un échantillon de la vidéo russophone.")
                elif kind == "telegram":
                    st.caption(
                        "Trois familles y cohabitent : les canaux officiels des "
                        "médias (doublons rapides de leur site), les « voenkory » "
                        "correspondants de guerre, souvent plus critiques du "
                        "ministere de la Défense que la presse d'État, et les "
                        "tabloïdes a forte audience nourris de fuites policières.")
                elif kind == "press":
                    st.caption(
                        "Une source « scrape » est lue sur sa page d'accueil, faute "
                        "de flux RSS exploitable : sa couverture est plus "
                        "irrégulière qu'un flux, et un site refondu peut cesser de "
                        "rendre sans erreur visible. Un « Dernier » qui prend du "
                        "retard est le signal à surveiller.")

    # Sources configurées mais absentes de la base : le signal le plus utile du
    # panneau, une source qui ne rapporté rien ne se voit nulle part ailleurs.
    _muettes = [s["name"] for s in SOURCE_CONFIG if s["name"] not in _cov_src.index]
    if _muettes:
        st.caption(f"**{len(_muettes)} sources sans aucun contenu en base** : "
                   + ", ".join(sorted(_muettes))
                   + ". Émission en relâche, source récemment ajoutée, ou collecte "
                     "en échec -- à vérifier dans les journaux de la dernière passe.")



# --- Onglet Références ------------------------------------------
# Un outil qui emprunte ses méthodes doit dire à qui. Chaque entrée porte un
# lien : le lecteur doit pouvoir remonter à la source et juger lui-même.

REMERCIEMENTS = [
    ("Approche générale et traitement des médias",
     "NATO Strategic Communications Centre of Excellence",
     "https://stratcomcoe.org/publications/crisis-control-crusade-russias-propaganda-architecture/347",
     "« Crisis, Control, Crusade: Russia's Propaganda Architecture » (2026)",
     "Le rapport de référence sur l'architecture de communication russe : il "
     "analyse les mêmes trois piliers -- sites d'État, télévision, Telegram -- "
     "et établit que chaque plateforme joue un rôle fonctionnel distinct. C'est "
     "ce qui justifie de ne jamais agréger les supports dans une même moyenne, "
     "principe suivi dans tout le tableau de bord."),
    ("Regroupement thématique",
     "Russian Propaganda Analysis — Ukraina.ru",
     "https://github.com/Romain-Jaffuel/Russian-Propaganda-Analysis-Ukraina.ru",
     "Projet antérieur du même auteur",
     "La démarche de clustering thématique appliquée ici -- détecter les thèmes "
     "dans les données plutôt que de les fixer d'avance -- vient de ce travail "
     "mené sur le corpus d'Ukraina.ru."),
    ("Procédés de propagande",
     "Da San Martino et al., EMNLP 2019",
     "https://scholar.google.fr/citations?view_op=view_citation&hl=en&user=URABLy0AAAAJ&citation_for_view=URABLy0AAAAJ:2P1L_qKh6hAC",
     "« Fine-Grained Analysis of Propaganda in News Articles »",
     "La taxonomie de procédés rhétoriques employée dans l'onglet Cadrage "
     "lexical, annotée au niveau du fragment de phrase. L'onglet en reprend "
     "quinze, ceux qui apparaissent effectivement dans le corpus russophone."),
    ("Architecture initiale",
     f"Florian Grolleau — Gabon Monitor",
     "https://github.com/Flor5378/Gabon-Monitor",
     "Cellule influence, détachement interarmées au Gabon",
     "Ossature de départ du pipeline : collecte RSS, stockage DuckDB, tableau "
     "de bord Streamlit."),
]
with tab_references:
    st.caption("Ce que cet outil doit à d'autres travaux, et où les consulter.")
    for role, titre, url, ref, texte in REMERCIEMENTS:
        with st.expander(f"{role} — {titre}"):
            st.markdown(f"[{ref}]({url})")
            st.caption(texte)
    st.caption(
        "Les travaux de recherche qui ont directement donné lieu à du code "
        "sont détaillés dans les commentaires des scripts concernés : "
        "`analyze_divergence.py` (divergence de Kullback-Leibler), "
        "`validate_topics.py` (protocole ProxAnn) et "
        "`analyze_techniques_mistral.py` (taxonomie des procédés)."
    )


# --- Pied de page ------------------------------------------------------
# Un tableau de bord public doit dire qui le publie, sous quelle licence, et
# d'où vient le code : sans ça, un visiteur ne sait ni à qui s'adresser ni ce
# qu'il a le droit d'en faire.
#
# Seul bloc HTML du fichier, et pour une raison précise : Streamlit n'a pas de
# primitive de pied de page, et Material Symbols ne fournit pas de marque
# GitHub ni LinkedIn. Les trois icônes sont donc des SVG en ligne -- aucun
# appel réseau, rien à héberger. Mise en page reprise de romain-jaffuel.github.io :
# centrée, icônes en boutons carrés, crédit en retrait dessous.
_ICONES = {
    "Site": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.7" stroke-linecap="round">'
             '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4.2" ry="9"/>'
             '<path d="M3.4 9h17.2M3.4 15h17.2"/></svg>'),
    "GitHub": ('<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 '
               '00-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8'
               '-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3'
               '.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.2-3.2 0 0 '
               '1-.3 3.3 1.2a11.5 11.5 0 016 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.7.3 2.9.1 3.2'
               '.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6'
               'A12 12 0 0012 .3"/></svg>'),
    "LinkedIn": ('<svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.4 20.5h-3.6v-5.6'
                 'c0-1.3 0-3-1.8-3s-2.1 1.4-2.1 2.9v5.7H9.4V9h3.4v1.6h.04c.5-.9 1.6-1.9 '
                 '3.4-1.9 3.6 0 4.3 2.4 4.3 5.5v6.3zM5.3 7.4a2.1 2.1 0 110-4.2 2.1 2.1 0 '
                 '010 4.2zm1.8 13H3.6V9h3.5v11.5zM22.2 0H1.8C.8 0 0 .8 0 1.7v20.6C0 23.2.8 '
                 '24 1.8 24h20.4c1 0 1.8-.8 1.8-1.7V1.7c0-.9-.8-1.7-1.8-1.7"/></svg>'),
}
_LIENS_ICONES = {"Site": URL_SITE, "GitHub": URL_GITHUB, "LinkedIn": URL_LINKEDIN}

st.markdown(f"""
<style>
.rm-pied{{border-top:1px solid rgba(255,255,255,.14);margin-top:2.6rem;
  padding:26px 0 10px;text-align:center;}}
.rm-pied-nom{{font-size:.95rem;font-weight:600;letter-spacing:.01em;}}
.rm-pied-sous{{font-size:.8rem;opacity:.6;margin-top:3px;}}
.rm-pied-icones{{display:flex;justify-content:center;gap:10px;margin:16px 0 14px;}}
/* color:inherit neutralise la couleur de lien du theme : trois pastilles
   d'accent en pleine couleur tiraient l'oeil plus que les liens eux-memes. */
.rm-pied-icones a{{width:38px;height:38px;border:1px solid rgba(255,255,255,.18);
  border-radius:9px;display:flex;align-items:center;justify-content:center;
  color:inherit;opacity:.62;transition:all .15s;}}
.rm-pied-icones a:hover{{opacity:1;border-color:{ACCENT};color:{ACCENT};}}
.rm-pied-icones svg{{width:17px;height:17px;}}
.rm-pied-liens{{font-size:.82rem;}}
.rm-pied-liens a{{color:{ACCENT};text-decoration:none;}}
.rm-pied-liens a:hover{{text-decoration:underline;}}
.rm-pied-liens .sep{{opacity:.28;margin:0 12px;}}
/* Le crédit d'origine n'est pas au même niveau que le reste : il parle d'un
   autre projet. Filet court et italique pour le dire sans le noyer. */
.rm-pied-credit{{font-size:.76rem;font-style:italic;opacity:.55;margin:16px auto 0;
  padding-top:14px;max-width:640px;border-top:1px solid rgba(255,255,255,.12);}}
.rm-pied-credit a{{color:inherit;text-decoration:underline;
  text-decoration-color:rgba(255,255,255,.25);}}
</style>
<div class="rm-pied">
  <div class="rm-pied-nom">Russian Media Monitor</div>
  <div class="rm-pied-sous">Veille des médias russophones sur la Russie &mdash; Romain Jaffuel</div>
  <div class="rm-pied-icones">
    {"".join(f'<a href="{_LIENS_ICONES[n]}" target="_blank" rel="noopener" '
             f'title="{n}" aria-label="{n}">{s}</a>' for n, s in _ICONES.items())}
  </div>
  <div class="rm-pied-liens">
    <a href="{URL_REPO}" target="_blank" rel="noopener">Code source</a>
    <span class="sep">|</span>
    <a href="{URL_REPO}/blob/master/LICENSE" target="_blank" rel="noopener">Licence MIT</a>
  </div>
  <div class="rm-pied-credit">{CREDIT_ORIGINE_HTML}</div>
</div>
""", unsafe_allow_html=True)

conn.close()
