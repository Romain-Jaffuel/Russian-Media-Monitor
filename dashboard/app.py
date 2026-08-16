"""Dashboard : refonte evenements (carte par acteur), geographie (top 300, purple,
zoomable), acteurs (position des journalistes vs une cible geopolitique).
"""
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Lance directement par l'interprete (`python dashboard/app.py`), ce fichier
# s'execute sans serveur : Streamlit emet une salve de « missing
# ScriptRunContext », le script se termine et rien n'est servi. Le diagnostic
# n'est pas evident dans ce bruit, d'ou ce garde-fou qui donne la commande.
if not st.runtime.exists():
    import sys
    sys.exit(
        "\nCe fichier est une application Streamlit : il lui faut son serveur.\n"
        "  uv run streamlit run dashboard/app.py\n"
    )

# ============================================================
# Recherche booleenne multi-mots (ET / OU / SAUF, guillemets, parentheses)
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
    """Compile une requete booleenne en (fragment SQL, params)."""
    if not query or not query.strip():
        return None, []
    toks = _kw_insert_default(_kw_tokenize(query), default_op)
    params = []
    return _kw_compile(_kw_parse(toks), params), params


DB_PATH = Path("data/russia.duckdb")
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
    """Les entrees de config/sources.yaml telles quelles, pour le panneau de
    couverture : il doit montrer les sources CONFIGUREES, y compris celles qui
    n'ont encore rien rapporte -- c'est justement l'information utile."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("sources", [])


SOURCE_CONFIG = load_source_config()

# `type:` de la config -> source_kind en base. Copie volontaire de la table de
# src/pipeline.py : le dashboard lit la config sans importer le pipeline.
CFG_KIND = {"telegram": "telegram", "youtube": "youtube", "rutube": "tv",
            "hls": "tv", "vk": "vk"}

# Trois expressions SQL partagees par la vue d'ensemble et le panneau de
# couverture. L'unite parente se relit depuis l'URL du segment (`v=<id>` sur
# YouTube, `video/<id>` sur RuTube et smotrim) et l'instant du segment
# (`?t=`, `&t=`, `#t=`) donne la minute couverte, ce qui evite de stocker une
# duree.
SQL_PARENT = ("CASE WHEN source_kind = 'youtube' "
              "THEN regexp_extract(url, 'v=([A-Za-z0-9_-]+)', 1) "
              "WHEN source_kind = 'tv' "
              "THEN regexp_extract(url, 'video/([A-Za-z0-9]+)', 1) "
              "ELSE id END")
SQL_OFFSET_S = "TRY_CAST(regexp_extract(url, '[?&#]t=(\\d+)', 1) AS INTEGER)"
SQL_MOTS = "LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1"


def fr_date(v, heure=False):
    """Date au format francais. Renvoie une chaine vide si la date manque.

    Les dates viennent de DuckDB en ISO (2026-08-15) : lisible pour une
    machine, mais ce tableau de bord est en francais et 03/08 ne doit pas
    pouvoir se lire comme le 8 mars."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t):
        return ""
    return t.strftime("%d/%m/%Y %H:%M" if heure else "%d/%m/%Y")


# Format de date des colonnes horodatees d'un st.dataframe (syntaxe Moment.js,
# pas strftime -- c'est le composant JavaScript qui rend la cellule).
FMT_DATE_TABLE = "DD/MM/YYYY HH:mm"

# Libelle court d'une nature de contenu (colonne source_kind).
UNITE_NATURE = {"press": "Presse", "tv": "Television", "youtube": "YouTube",
                "telegram": "Telegram", "vk": "VK"}


def cols_article(**extra):
    """column_config commun aux tableaux d'articles : lien cliquable et date
    au format francais. Sans DatetimeColumn, Streamlit rend l'horodatage
    DuckDB en ISO."""
    cfg = {"url": st.column_config.LinkColumn("Lien"),
           "published_at": st.column_config.DatetimeColumn(
               "Publie le", format=FMT_DATE_TABLE)}
    cfg.update(extra)
    return cfg


def col_entier(serie):
    """Colonne d'entiers pour st.dataframe, case vide quand la donnee manque.

    Streamlit affiche « None » pour toute valeur nulle, quel que soit le type
    -- Int64, objet ou flottant, et y compris avec un NumberColumn (verifie
    sur cette version). Seule une chaine vide rend une case vide. On n'y passe
    donc que les colonnes qui peuvent manquer -- la duree et les vues, absentes
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
    "ukraine": "Ukraine", "etats_unis": "Etats-Unis",
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
# huit acteurs europeens tiennent dans un mouchoir et leurs etiquettes se
# recouvrent. Les entites non geographiques (UE, OTAN, BRICS, opposition en
# exil) sont de toute facon symboliques ; les pays sont ecartes juste assez
# pour rester reconnaissables.
TARGET_COORDS = {
    "ukraine": (50.45, 30.52), "etats_unis": (38.91, -77.04),
    "union_europeenne": (47.0, 8.0),      # au sud de Bruxelles, pour degager
    "otan": (53.5, -2.0),                 # decale vers la mer du Nord
    "allemagne": (52.52, 13.40), "france": (46.5, -1.5),
    "pays_baltes": (57.5, 24.1), "chine": (39.90, 116.41),
    "inde": (28.61, 77.21),
    "brics_global_south": (-15.79, -47.88),   # Brasilia
    "georgie": (43.5, 41.0), "armenie": (38.5, 46.0),
    "opposition_russe": (63.0, 10.0),         # hub d'exil balte, ecarte au nord
    "kazakhstan": (51.17, 71.45),
    "moldavie": (43.0, 26.0), "iran": (32.0, 54.0),
}

# Cote ou poser l'etiquette de chaque bulle. Place a la main plutot que
# calcule : ces seize acteurs ne bougent pas, et huit d'entre eux se serrent
# sur l'Europe -- les faire rayonner vers l'exterieur est la seule facon de
# les rendre tous lisibles. Le defaut vaut pour les acteurs isoles.
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

# Vocabulaire de cadrage kremlinien, d'apres la litterature sur la propagande
# russe (Field et al. 2018, Pizzolo 2020). Detection lexicale simple : on
# mesure la PRESENCE du terme, pas l'adhesion -- un media independant peut
# citer ou critiquer ces memes termes.
FRAMING_TERMS = {
    "Monde russe": r"русск(ий|ого|ому|им|ом) мир",
    "Etranger proche": r"ближн(ее|его|ему|им|ем) зарубежь",
    "Regime de Kiev": r"киевск(ий|ого|ому|им|ом) режим",
    "Denazification": r"денацифи",
    "Demilitarisation": r"демилитариз",
    "Russophobie": r"русофоб",
    "Neonazisme / Bandera": r"неонацист|необандер|бандеровц|бандеровск",
    "Occident collectif (lexical)": r"коллективн(ый|ого|ому|ым|ом) запад",
    "Genocide du Donbass": r"геноцид.{0,15}донбасс",
    "Junte illegitime": r"хунт|нелегитимн",
}

# Indicateurs de suivi : contrairement au vocabulaire de cadrage ci-dessus,
# ce ne sont pas des marqueurs de propagande mais des thermometres. Ils sont
# suivis en permanence meme a bas bruit, la ou le clustering BERTopic ne
# forme un theme que si le sujet devient assez dense pour emerger.
INDICATOR_TERMS = {
    # "повестка" seul veut aussi dire "ordre du jour" (повестка дня), tres
    # frequent en actu politique. DuckDB utilise RE2, qui ne supporte pas le
    # lookahead negatif : on ne peut pas exclure "повестка дня" directement,
    # donc on ne garde "повестка" que dans ses collocations militaires.
    "Mobilisation": (r"мобилизац|уклонист|военкомат|призывник|"
                     r"повестк[а-я]* в военкомат|электронн[а-я]* повестк|"
                     r"вручил[а-я]* повестк|реестр повесток"),
    "Signaux de negociation": (r"переговор|перемири|мирн[а-я]* план|"
                               r"урегулировани|прекращени[ея] огня"),
    "Stress economique interne": (r"дефицит бюджет|бюджетн[а-я]* дефицит|"
                                  r"инфляц|девальвац"),
}

# Vue unifiee pour l'onglet Cadrage lexical (les deux familles se mesurent
# de la meme facon, seule leur lecture differe).
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


# Chrome des graphiques. Ces valeurs doivent rester coherentes avec le theme
# de .streamlit/config.toml : les figures Plotly ne heritent pas du theme
# Streamlit, il faut les habiller a la main.
GRID = "#252B38"      # grille : lisible mais en retrait des donnees
AXIS = "#38404F"      # ligne d'axe, un cran plus marquee que la grille
TEXT = "#E6E8EB"
MUTED = "#98A2B3"


def _unified_hover(fig):
    """Mode de survol groupe adapte a la figure, ou None si elle ne s'y prete
    pas.

    Le survol groupe suppose que les points compares partagent un axe. C'est
    vrai des barres empilees et des series temporelles, faux d'une carte, d'un
    camembert ou d'un nuage de points. Et l'axe partage n'est pas toujours x :
    sur les barres horizontales (graphe des themes), c'est y -- s'y tromper
    regrouperait par valeur au lieu de regrouper par theme.
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
    """L'axe des abscisses porte-t-il des dates ? Lu sur les donnees plutot que
    sur le type d'axe declare : Plotly ne fixe `xaxis.type` qu'au rendu."""
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
        # Fond transparent plutot qu'une couleur en dur : la figure se pose
        # sur le fond de la page au lieu d'y decouper un rectangle -- visible
        # des que Streamlit ajuste sa teinte (cartes, expanders, colonnes).
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#161A23", bordercolor=GRID,
                        font=dict(size=LEGEND_FONT, color=TEXT)),
    )
    # Un seul cadre au survol pour toutes les series d'un meme point : sur les
    # graphes empiles (volume par source, themes presse/telegram), comparer
    # serie par serie au survol etait impraticable.
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

    # Plotly date ses graduations en anglais (« Aug 9 ») et sa locale francaise
    # n'est pas embarquee par le composant Streamlit. Un format numerique
    # explicite evite d'en dependre : jour/mois sur l'axe, date complete au
    # survol. On ne l'applique qu'aux axes reellement temporels -- sur un axe
    # numerique, un motif en %d serait interprete comme un format de nombre.
    if _axe_temporel(fig):
        fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
    return fig


st.set_page_config(
    page_title="Russia Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Les tokens sont declares une fois ici et reutilisees par toutes les regles :
# une seule valeur a changer pour retoucher l'identite. Les couleurs de
# donnees (pro/anti/postures) restent dans les palettes Python plus haut --
# elles portent du sens, contrairement a celles-ci qui n'habillent que le
# chrome de l'interface.
st.markdown("""
<style>
:root{
  --rm-accent:#00C2A8;
  --rm-panel:#161A23;
  --rm-border:#252B38;
  --rm-text:#E6E8EB;
  --rm-muted:#98A2B3;
}
/* Au-dela de ~1600px les tableaux larges s'etirent et deviennent illisibles.
   padding-top : la barre d'outils de Streamlit (Deploy, menu) flotte au-dessus
   du contenu ; en descendre moins, le titre passe dessous et se fait rogner. */
.block-container{padding-top:4rem;padding-bottom:3rem;max-width:1650px;}

/* --- En-tete d'identite --- */
.rm-head{display:flex;align-items:center;gap:14px;margin-bottom:.2rem;}
.rm-mark{width:40px;height:40px;border-radius:10px;flex:0 0 40px;
  background:linear-gradient(135deg,var(--rm-accent),#0B6E5F);
  display:flex;align-items:center;justify-content:center;
  font-size:18px;color:#04150F;font-weight:700;}
.rm-title{font-size:1.6rem;font-weight:700;line-height:1.1;margin:0;
  letter-spacing:-.01em;}
.rm-sub{color:var(--rm-muted);font-size:.88rem;margin:3px 0 0;}

/* --- Bandeau de perimetre : rappelle en permanence ce qui est filtre --- */
.rm-scope{display:flex;flex-wrap:wrap;gap:7px;margin:.9rem 0 .2rem;}
.rm-chip{background:var(--rm-panel);border:1px solid var(--rm-border);
  border-radius:999px;padding:3px 11px;font-size:.77rem;color:var(--rm-muted);
  white-space:nowrap;}
.rm-chip b{color:var(--rm-text);font-weight:600;}
.rm-chip.on{border-color:var(--rm-accent);color:var(--rm-text);}

/* --- Onglets : 9 rubriques, il faut pouvoir les balayer et voir ou on est ---
   Streamlit >= 1.60 rend les onglets via React Aria ([data-testid="stTab"]),
   les versions anterieures via BaseWeb ([data-baseweb="tab"]). On vise les
   deux : l'habillage survit ainsi a une montee de version comme a un retour
   en arriere, au lieu de disparaitre sans bruit.
   Le soulignement de l'onglet actif n'est pas redefini ici : Streamlit le
   dessine deja avec primaryColor, defini dans .streamlit/config.toml. */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"]{
  gap:0;border-bottom:1px solid var(--rm-border);}
.stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"]{
  height:44px;padding:0 15px;display:flex;align-items:center;
  color:var(--rm-muted);}
.stTabs [data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p{
  font-size:.9rem;color:inherit;font-weight:inherit;}
.stTabs [data-testid="stTab"]:hover, .stTabs [data-baseweb="tab"]:hover{
  color:var(--rm-text);}
.stTabs [aria-selected="true"]{color:var(--rm-text) !important;font-weight:600;}

/* --- Metriques en cartes plutot qu'en chiffres flottants --- */
[data-testid="stMetric"]{background:var(--rm-panel);border:1px solid var(--rm-border);
  border-radius:10px;padding:13px 16px;}
[data-testid="stMetricLabel"] p{color:var(--rm-muted) !important;font-size:.76rem !important;
  text-transform:uppercase;letter-spacing:.05em;}

/* --- Titres de section : le filet d'accent sert de reperage vertical --- */
h3{font-size:1.1rem !important;font-weight:650 !important;
  padding-left:11px;border-left:3px solid var(--rm-accent);
  margin-top:1.3rem !important;margin-bottom:.5rem !important;}

/* --- Barre laterale --- */
[data-testid="stSidebar"]{border-right:1px solid var(--rm-border);}
[data-testid="stSidebar"] h2{font-size:1.05rem !important;}

/* --- Panneau de couverture (bas de page) ---
   Volontairement hors de la barre d'onglets : ce n'est pas une rubrique
   d'analyse de plus mais la fiche d'identite du corpus, consultee de temps en
   temps. Le liseré d'accent et le fond plein la detachent du reste. */
.rm-cover{border:1px solid var(--rm-border);border-left:3px solid var(--rm-accent);
  background:var(--rm-panel);border-radius:10px;padding:14px 18px;margin-top:2.5rem;}
.rm-cover h4{margin:0;font-size:1.02rem;font-weight:650;color:var(--rm-text);}
.rm-cover p{margin:.35rem 0 0;color:var(--rm-muted);font-size:.82rem;line-height:1.5;}

/* Les legendes sous les graphiques portent les mises en garde de lecture :
   elles doivent rester lisibles, pas s'effacer. */
[data-testid="stCaptionContainer"] p{color:var(--rm-muted);font-size:.79rem;line-height:1.45;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rm-head">
  <div class="rm-mark">◈</div>
  <div>
    <p class="rm-title">Russia Monitor</p>
    <p class="rm-sub">Veille des medias russophones &mdash; presse, Telegram, YouTube, television, VK</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not DB_PATH.exists():
    st.warning("Aucune base.")
    st.stop()

def ouvrir_base(essais=3, attente=1.5):
    """Connexion en lecture, avec quelques tentatives.

    DuckDB n'admet qu'un ecrivain OU plusieurs lecteurs : tant qu'une analyse
    ecrit, l'ouverture echoue. C'est normal, mais sans ce garde-fou Streamlit
    affichait une trace Python, ce qui ressemble a une base cassee.
    """
    derniere = None
    for reste in range(essais - 1, -1, -1):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except Exception as e:
            # On reessaie quelle que soit l'erreur : distinguer le verrou des
            # autres pannes supposerait de lire le texte du message, qui est
            # traduit dans la langue du systeme. Trois tentatives coutent
            # trois secondes dans le pire des cas.
            derniere = e
            if reste:
                time.sleep(attente)
    return derniere


_base = ouvrir_base()
if not isinstance(_base, duckdb.DuckDBPyConnection):
    st.warning(
        "**Analyse en cours d'ecriture.** La base est momentanement reservee "
        "par une analyse (collecte, themes, sentiment...). Le tableau de bord "
        "la relira des qu'elle aura rendu la main -- rien n'est perdu et rien "
        "n'est casse : DuckDB n'autorise qu'un ecrivain a la fois."
    )
    if st.button("Reessayer"):
        st.rerun()
    with st.expander("Detail technique"):
        st.code(str(_base) if _base else "verrou toujours pris", language="text")
    st.stop()
conn = _base
tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}

# Entites et posture-par-article ont ete retirees de la routine : leurs
# tables restent en base mais plus rien ne les lit.
has_target_sent = "article_target_sentiment" in tables
has_topics = "topics" in tables and "article_topics" in tables
has_geo = "entity_geo" in tables


date_start = st.session_state.get("global_date_start", date(2026, 5, 1))
date_end = st.session_state.get("global_date_end", date.today())

with st.sidebar:
    st.header("Filtres")

    # Periode et recherche en premier : les deux seuls filtres manipules a
    # chaque session. Le reste est replie -- 45 etiquettes de sources noyaient
    # tout, et les pastilles sous le titre rappellent ce qui est filtre.
    st.markdown("**Periode**")
    c_date_start, c_date_end = st.columns(2)
    with c_date_start:
        date_start = st.date_input("Articles du", value=date_start, key="global_date_start")
    with c_date_end:
        date_end = st.date_input("au", value=date_end, key="global_date_end")

    st.markdown("**Recherche**")
    keyword = st.text_input(
        "Requete booleenne", label_visibility="collapsed",
        placeholder="ex : (sanctions OU иноагент) ET NON sport",
        help="ET / OU / SAUF (ou AND/OR/NOT, & | -). Guillemets pour une "
             "expression exacte, parentheses pour grouper.",
    )
    kw_default_op = st.radio(
        "Operateur implicite entre deux mots", ["ET", "OU"],
        horizontal=True, index=0,
    )

    # ORDER BY sur tous les DISTINCT, et une cle explicite par widget.
    # Sans tri, DuckDB renvoie ces valeurs dans l'ordre de sa table de
    # hachage, qui change des que les donnees changent ; Streamlit voyait
    # alors une liste d'options differente d'un rerun a l'autre et remettait
    # la selection a sa valeur par defaut -- d'ou des filtres qui « se
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
        # Par defaut le russe seul : l'outil suit les medias russophones, et
        # les 0,3 % restants sont les editions traduites de Mediazona
        # (anglais, espagnol, portugais, polonais), hors sujet ici.
        default_langs = ["ru"] if "ru" in langs else langs
        selected_langs = st.multiselect(
            "Langues", langs, default=default_langs, key="f_langs",
            help="Le russe est seul coche par defaut. Les autres langues sont "
                 "des editions traduites (Mediazona) ou des erreurs de "
                 "detection sur du cyrillique.",
        )
        types_media = conn.execute(
            "SELECT DISTINCT type_media FROM articles WHERE type_media IS NOT NULL "
            "ORDER BY type_media"
        ).df()["type_media"].tolist()
        selected_types_media = st.multiselect("Type de media", types_media,
                                              default=types_media, key="f_types")
        source_kinds = conn.execute(
            "SELECT DISTINCT source_kind FROM articles WHERE source_kind IS NOT NULL "
            "ORDER BY source_kind"
        ).df()["source_kind"].tolist()
        selected_source_kinds = st.multiselect(
            "Nature du contenu", source_kinds, default=source_kinds,
            help="Telegram : canaux officiels et milbloggers, texte brut du "
                 "post. YouTube : sous-titres de video en segments d'environ "
                 "2000 signes -- ces chaines sont toutes d'opposition en exil. "
                 "TV : transcription Whisper du journal de 21 h et des grands "
                 "talk-shows politiques, qui reequilibre le volet video cote "
                 "pouvoir. VK : publications des communautes des grands "
                 "medias sur le premier reseau social du pays. "
                 "Voir l'onglet Sources.",
            key="f_kinds",
        )
        statuts_legal = conn.execute(
            "SELECT DISTINCT statut_legal_ru FROM articles WHERE statut_legal_ru IS NOT NULL "
            "ORDER BY statut_legal_ru"
        ).df()["statut_legal_ru"].tolist()
        selected_statuts_legal = st.multiselect(
            "Statut legal (RU)", statuts_legal, default=statuts_legal, key="f_statuts")
        st.caption(
            "Statuts legaux : agent_etranger, organisation_indesirable, aucun. "
            "Plusieurs medias en exil sont interdits d'exploitation en Russie "
            "(cela n'empeche pas leur collecte depuis l'etranger)."
        )

    st.subheader("Affichage")
    _grains = {"Jour": "day", "Semaine": "week", "Mois": "month"}
    _grain_choisi = st.radio(
        "Pas de temps des courbes", list(_grains), index=0, horizontal=True,
        key="f_grain",
        help="S'applique a toutes les courbes d'evolution. Le jour montre les "
             "pics et les reprises d'un evenement ; la semaine lisse le creux "
             "du week-end, tres marque dans la presse d'agence.")
    GRAIN = _grains[_grain_choisi]
    GRAIN_LABEL = _grain_choisi

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

# Meme filtres que WHERE, sans la borne de date -- utilise par l'onglet
# Signaux qui a besoin de definir ses propres fenetres temporelles (recente
# vs reference) par-dessus les filtres source/langue/mots-cles.
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

# --- Apercu de recherche (compteur + liste), affiche dans la sidebar ---
with st.sidebar:
    if keyword and keyword.strip():
        try:
            _n_res = conn.execute(
                f"SELECT COUNT(*) FROM articles WHERE {WHERE}", params
            ).fetchone()[0]
        except Exception as _e:
            _n_res = None
        if _n_res is None:
            st.warning("Recherche invalide, verifiez la syntaxe.")
        elif _n_res == 0:
            st.info("Aucun article ne correspond.")
            st.caption("Astuce : essayez le mode OU, ou retirez un mot.")
        else:
            st.success(f"{_n_res} article(s) correspondent")
            with st.expander("Apercu des articles trouves", expanded=False):
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
                    st.caption(f"... et {_n_res - 50} autres. Affinez pour reduire.")



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


# Perimetre courant, rappele sous le titre : les filtres vivent dans la barre
# laterale, qui peut etre repliee -- sans ce bandeau, rien a l'ecran ne dit
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

# « X articles » serait faux : une emission de 2 h fait une soixantaine de
# lignes. On regroupe donc les segments sur leur unite parente, et le detail
# par nature part dans l'infobulle.
_df_unites = conn.execute(
    f"SELECT source_kind, COUNT(DISTINCT {SQL_PARENT}) AS n "
    f"FROM articles WHERE {WHERE} GROUP BY 1", params).df()
UNITE_MOT = {"press": "articles", "tv": "emissions de TV",
             "youtube": "videos", "telegram": "posts Telegram",
             "vk": "posts VK"}
_n_unites = int(_df_unites["n"].sum()) if not _df_unites.empty else 0
_detail_unites = " · ".join(
    f"{int(r['n'])} {UNITE_MOT.get(r['source_kind'], r['source_kind'])}"
    for _, r in _df_unites.sort_values("n", ascending=False).iterrows())
_total_fmt = f"{_n_unites:,}".replace(",", "&nbsp;")
# date_start/date_end peuvent etre None : st.date_input rend un champ
# effacable, et le filtre SQL plus haut prevoit deja ce cas.
_periode = (f"{date_start:%d/%m/%Y}" if date_start else "debut") + " &rarr; " + \
           (f"{date_end:%d/%m/%Y}" if date_end else "aujourd'hui")
_chips = [
    f"<span class='rm-chip' title='{_detail_unites}'><b>{_total_fmt}</b> "
    f"contenus</span>",
    f"<span class='rm-chip'><b>{n_src}</b> sources</span>",
    f"<span class='rm-chip'>{_periode}</span>",
]
# On ne signale que les filtres reellement restrictifs : afficher "45 sources
# sur 45" a chaque ecran serait du bruit.
if len(selected_sources) < len(sources_all):
    _chips.append(f"<span class='rm-chip on'>sources : {len(selected_sources)}"
                  f"/{len(sources_all)}</span>")
if len(selected_source_kinds) < len(source_kinds):
    _chips.append("<span class='rm-chip on'>"
                  + " + ".join(selected_source_kinds) + "</span>")
if len(selected_types_media) < len(types_media):
    _chips.append(f"<span class='rm-chip on'>media : "
                  f"{', '.join(selected_types_media)}</span>")
if keyword and keyword.strip():
    _safe_kw = keyword.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    _chips.append(f"<span class='rm-chip on'>recherche : <b>{_safe_kw}</b></span>")
if last_f:
    _chips.append(f"<span class='rm-chip'>collecte {fr_date(last_f, heure=True)}</span>")
st.markdown(f"<div class='rm-scope'>{''.join(_chips)}</div>", unsafe_allow_html=True)

# Ordre de lecture : ce qui a change (Signaux), puis ce qui est dit (themes,
# sentiment, cadrage), puis qui le dit, puis la qualite des donnees.
(tab_vue, tab_signaux, tab_themes, tab_sentiment, tab_cadrage,
 tab_pouvoir, tab_acteurs, tab_sources, tab_contexte) = st.tabs([
    "Vue d'ensemble", "Signaux", "Themes", "Sentiment geopolitique",
    "Cadrage lexical", "Sources et alignement", "Acteurs", "Diagnostic",
    "Contexte",
])

# ===== Tab Vue =================================================
with tab_vue:
    # « Articles » ne veut rien dire pour une emission de television : un
    # numero de 2 h donne 60 lignes en base. Chaque nature de contenu est donc
    # comptee dans SON unite -- articles, emissions, videos, posts -- et les
    # segments ne servent qu'a comparer les natures entre elles.
    #
    # Unite parente, instant du segment, nombre de mots : cf. SQL_PARENT,
    # SQL_OFFSET_S et SQL_MOTS, definis en tete de fichier et partages avec le
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
           "tv": ("Emissions de television", "emissions"),
           "youtube": ("Videos YouTube", "videos"),
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
        # La ligne de detail n'est pas qu'informative : sans elle, cette carte
        # est plus courte que les cinq autres et la rangee se desaligne.
        cols[-1].metric("Derniere collecte",
                        last_f.strftime("%d/%m/%y") if last_f else "n/a",
                        f"{nb_passes} collectes", delta_color="off")

        with st.expander("Detail par nature de contenu", expanded=False):
            det = df_nat.copy()
            det["Nature"] = det["nature"].map(lambda k: LIB.get(k, (k, ""))[0])
            det["Type"] = det["nature"].map(lambda k: LIB.get(k, ("", "unites"))[1])
            det["Mots"] = det["mots"].astype("Int64")
            det["Segments"] = det["segments"].astype(int)
            det["Unites"] = det["unites"].astype(int)
            det["Sources"] = det["sources"].astype(int)
            # Duree et vues ne sont plus affichees ici : elles n'existent que
            # pour la video et le panneau de couverture, en bas de page, les
            # donne par source. Elles restent dans df_nat.
            st.dataframe(
                det[["Nature", "Type", "Unites", "Segments", "Mots",
                     "Sources"]],
                width="stretch", hide_index=True,
                column_config={
                    "Mots": st.column_config.NumberColumn("Mots", format="%d"),
                })
            st.caption(
                "**Unites** : ce que l'on compte naturellement pour cette "
                "nature -- un article de presse, une emission de television, "
                "une video, un post. **Segments** : les lignes reellement "
                "stockees et analysees ; une emission de 2 h en produit une "
                "soixantaine, un article un seul. Pour comparer les natures "
                "entre elles, c'est le nombre de **mots** qui fait foi."
            )

    df_vol = conn.execute(
        f"SELECT DATE_TRUNC('{GRAIN}', published_at) AS jour, source_name, COUNT(*) AS n "
        f"FROM articles WHERE {WHERE} AND published_at IS NOT NULL "
        f"GROUP BY 1, 2 ORDER BY 1", params).df()
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
            "En segments, l'unite reellement stockee et analysee : une "
            "emission de television de 2 h en produit une soixantaine, un "
            "article un seul. Une journee ou la television pese lourd n'est "
            "donc pas une journee ou elle a plus parle que la presse.")


# ===== Tab Themes ==============================================
with tab_themes:
    aw = with_a(WHERE)
    if not has_topics:
        st.info("Lancez : python analyze_topics.py")
    else:
        st.caption(
            "Themes decides directement par les articles (clustering BERTopic "
            "sur une fenetre glissante), pas une liste ecrite a la main : un "
            "theme sans article recent devient inactif sans perdre son "
            "historique, et peut redevenir actif si le sujet revient."
        )
        df_t = conn.execute(
            f"""SELECT t.topic_key, t.label, t.top_words, t.active,
                       t.first_seen, t.last_seen, COUNT(at_.article_id) AS n
            FROM topics t LEFT JOIN article_topics at_ ON at_.topic_key = t.topic_key
            LEFT JOIN articles a ON a.id = at_.article_id
            WHERE t.topic_key != -1 AND (at_.article_id IS NULL OR {aw})
            GROUP BY t.topic_key, t.label, t.top_words, t.active, t.first_seen, t.last_seen
            ORDER BY n DESC""",
            params).df()
        df_t["pct"] = (df_t["n"] / df_t["n"].sum() * 100).round(1) if df_t["n"].sum() else 0.0
        nz = df_t[(df_t["n"] > 0) & (df_t["active"])].copy()
        archived = df_t[(df_t["n"] > 0) & (~df_t["active"])].sort_values("last_seen", ascending=False)

        st.subheader("Repartition des themes actifs")
        c_left, c_right = st.columns([2, 1])
        with c_left:
            if nz.empty:
                st.caption("Aucun theme actif sur la periode/filtres choisis.")
            else:
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    unit_choice = st.radio(
                        "Compter en", ["Segments", "Volume de texte"],
                        horizontal=True, index=0, key="theme_unit",
                        help="Un segment = une ligne analysee : un article de "
                             "presse en vaut un, une emission de television une "
                             "soixantaine. Le volume de texte corrige ce biais "
                             "en ponderant par la longueur -- c'est la mesure a "
                             "prendre pour comparer des natures differentes.",
                    )
                with cm2:
                    scale_choice = st.radio(
                        "Afficher en", ["Nombre", "% du corpus"],
                        horizontal=True, index=1, key="theme_scale",
                        help="Le pourcentage rapporte au corpus entier tel que "
                             "filtre a gauche, pas seulement aux articles classes.",
                    )
                with cm3:
                    split_choice = st.selectbox(
                        "Repartir par",
                        ["Nature du contenu", "Type de media", "Media", "Aucune"],
                        index=0, key="theme_split",
                        help="Decompose chaque barre. « Media » distingue les "
                             "sources une a une : lisible surtout apres avoir "
                             "restreint la liste des sources a gauche.",
                    )

                SPLIT_COL = {"Nature du contenu": "COALESCE(a.source_kind, 'press')",
                             "Type de media": "COALESCE(a.type_media, 'inconnu')",
                             "Media": "a.source_name",
                             "Aucune": "'Tous'"}[split_choice]

                # Denominateur = corpus total selon les filtres actuels (pas
                # seulement les articles classifies) : la barre represente une
                # vraie part du corpus, pas une part entre themes seulement.
                denom_n, denom_chars = conn.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(LENGTH(a.content)), 0) "
                    f"FROM articles a WHERE {aw}", params).fetchone()
                denom_n = denom_n or 1
                denom_chars = denom_chars or 1

                df_tk = conn.execute(
                    f"""SELECT t.topic_key, t.label,
                               {SPLIT_COL} AS decoupe,
                               COUNT(*) AS n, COALESCE(SUM(LENGTH(a.content)), 0) AS chars
                    FROM topics t
                    JOIN article_topics at_ ON at_.topic_key = t.topic_key
                    JOIN articles a ON a.id = at_.article_id
                    WHERE t.active AND t.topic_key != -1 AND {aw}
                    GROUP BY t.topic_key, t.label, decoupe""",
                    params).df()

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

                theme_order = (
                    df_tk.groupby("label")["val"].sum()
                    .sort_values(ascending=False).index.tolist()
                )

                # Affichage par tranches de 20 : au-dela, les barres du bas
                # sont trop courtes pour se comparer a l'oeil et le graphe
                # devient un mur a faire defiler. Le reste reste accessible.
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
                # Plotly empile les categories du bas vers le haut : pour voir
                # le theme dominant EN HAUT, il faut lui passer l'ordre
                # croissant. Un `autorange="reversed"` par-dessus annulait ce
                # classement et remontait les themes les moins traites.
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
                    cpg2.caption(f"{shown} themes sur {total_themes} affiches, "
                                 f"du plus traite au moins traite.")
                elif total_themes > THEME_PAGE:
                    if cpg1.button("Revenir aux 20 premiers", key="theme_less"):
                        st.session_state["theme_shown"] = THEME_PAGE
                        st.rerun()
                    cpg2.caption(f"Les {total_themes} themes actifs sont affiches.")
        with c_right:
            st.markdown("**Statistiques globales**")
            n_total = int(nz["n"].sum())
            st.metric("Segments classes (actifs)", f"{n_total:,}".replace(",", " "))
            st.metric("Themes actifs", f"{len(nz)}")
            st.metric("Themes en sommeil", f"{(~df_t['active']).sum()}")
            top = nz.iloc[0] if len(nz) else None
            if top is not None:
                st.metric("Theme dominant", top["label"], f"{top['pct']}%")
            top5 = nz.head(5)[["label", "n", "pct"]].rename(
                columns={"label": "Theme", "n": "N", "pct": "%"})
            st.dataframe(top5, hide_index=True, width="stretch")

        if not archived.empty:
            with st.expander(f"Themes en sommeil sur cette periode ({len(archived)})"):
                st.caption(
                    "Plus de cluster correspondant dans la derniere fenetre "
                    "d'analyse -- gardes en memoire, peuvent redevenir actifs."
                )
                df_arch = archived[["label", "n", "first_seen", "last_seen"]].rename(
                    columns={"label": "Theme", "n": "Segments",
                             "first_seen": "Vu depuis", "last_seen": "Vu jusqu'a"})
                st.dataframe(
                    df_arch, width="stretch", hide_index=True,
                    column_config={
                        c: st.column_config.DateColumn(c, format="DD/MM/YYYY")
                        for c in ("Vu depuis", "Vu jusqu'a")})

        st.subheader("Explorer un theme")
        all_ids = df_t[df_t["n"] > 0]["topic_key"].tolist()
        if all_ids:
            label_lookup = dict(zip(df_t["topic_key"], df_t["label"]))
            n_lookup = dict(zip(df_t["topic_key"], df_t["n"]))
            active_lookup = dict(zip(df_t["topic_key"], df_t["active"]))
            tid = st.selectbox(
                "Theme", all_ids,
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
                st.markdown(f"**Evolution**")
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
                # etait codee en dur parmi les seize suivies, sans raison de
                # privilegier celle-la, et la posture par cible a deja son
                # onglet (Sentiment) avec un selecteur. A sa place, la question
                # qui manquait vraiment sur un theme : qui le porte -- la
                # presse, la television, Telegram ?
                st.markdown("**Ou ce theme est-il porte**")
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
                        "En segments, pas en emissions : une emission de TV "
                        "pese naturellement plus qu'un article. A lire comme "
                        "un partage de volume de texte.")

            top_words = conn.execute(
                "SELECT top_words FROM topics WHERE topic_key = ?", [tid]
            ).fetchone()
            if top_words and top_words[0]:
                st.caption(f"Mots-cles du cluster : {top_words[0]}")

            df_art = conn.execute(
                f"""SELECT a.published_at, a.source_name, ROUND(at_.probability, 2) AS proba,
                a.title, a.url FROM articles a
                JOIN article_topics at_ ON at_.article_id = a.id
                WHERE at_.topic_key = ? AND {aw}
                ORDER BY a.published_at DESC LIMIT 50""",
                [tid, *params]).df()
            st.caption("proba = probabilite d'appartenance au cluster (0 a 1)")
            st.dataframe(df_art, width="stretch", hide_index=True,
                         column_config=cols_article())


# ===== Tab Sentiment ===========================================
with tab_sentiment:
    aw = with_a(WHERE)
    if not has_target_sent:
        st.warning("Lancez : python analyze_sentiment_multi_mistral.py --reset")
    else:
        target = st.selectbox("Cible geopolitique", TARGETS,
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

        st.subheader(f"Evolution de la posture vis-a-vis de {target_label}")
        # Le pas de temps est desormais un reglage global (barre laterale) :
        # deux commandes concurrentes sur la meme notion pretaient a confusion.
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
                    st.markdown("**Lean moyen par periode**")
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

        st.subheader(f"Posture des medias vis-a-vis de {target_label}")
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
            sizeref_val = 2.0 * max_m / (42 ** 2)  # diametre max en pixels ; au-dela
            # les bulles europeennes se recouvrent et masquent les etiquettes
            max_pro = max(df_map_data["pro"].max(), 1)

            fig = go.Figure()

            # Tous les acteurs, meme traitement visuel (aucune cible privilegiee)
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
                    color=df_map_data["pro"], colorscale="Reds",
                    cmin=0, cmax=max_pro,
                    line=dict(width=1.5, color="white"), opacity=0.85,
                    showscale=True,
                    colorbar=dict(title="Articles<br>pro", thickness=15, len=0.5,
                                  tickfont=dict(size=13), title_font=dict(size=13)),
                ),
                # Le nom seul : les chiffres sous chaque bulle faisaient deux
                # lignes par acteur et rendaient les noms illisibles. Ils
                # restent au survol, et le top 5 est encadre a gauche.
                text=df_map_data["target"].map(
                    lambda t: TARGET_LABELS_CARTE.get(t, TARGET_LABELS[t])),
                textposition=df_map_data["target"].map(
                    lambda t: TARGET_TEXTPOS.get(t, "bottom center")),
                textfont=dict(size=15, color="#F2F4F7"),
            ))

            # Encadre top 5 mentions en haut a gauche
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

            st.markdown("**Synthese chiffree par acteur**")
            df_table = df_map_data[["label", "pro", "anti", "mentions", "lean_avg"]].copy()
            df_table["lean_avg"] = df_table["lean_avg"].round(2)
            df_table = df_table.sort_values("mentions", ascending=False).reset_index(drop=True)
            df_table.columns = ["Acteur", "Pro", "Anti", "Mentions", "Lean moyen"]
            st.dataframe(df_table, width="stretch", hide_index=True)


# ===== Tab Pouvoir/Type =========================================
with tab_pouvoir:
    # Les postures et types d'article derives du modele ont ete retires de la
    # routine (cf. update.py) : ils coutaient un tiers du budget Mistral pour
    # une information que le label ecrit a la main resume deja. Cet onglet
    # affiche desormais la classification curee, disponible pour toutes les
    # sources sans aucun appel API.
    aw = with_a(WHERE)
    st.subheader("Classification editoriale des sources")
    st.caption(
        "Etiquettes saisies a la main dans config/sources.yaml, pas deduites "
        "des articles : elles valent pour toutes les sources, y compris celles "
        "dont aucun article n'a encore ete analyse. Le type de media resume "
        "l'alignement (Etat, para-Etat, independant, exil) ; le positionnement "
        "historique donne le detail editorial."
    )

    df_class = conn.execute(
        f"""SELECT a.source_name AS Source,
                   COALESCE(a.type_media, 'non classe') AS "Type de media",
                   COALESCE(a.statut_legal_ru, 'aucun') AS "Statut legal (RU)",
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

    st.markdown("### Repartition du corpus par alignement")
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
            "surrepresente ici l'est parce qu'il est facile a collecter, pas "
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
            st.subheader("Position des journalistes vis-a-vis d'une cible")
            st.caption("Lean moyen et volume d'articles par journaliste. "
                       "Sont retenus les journalistes avec >= 3 articles concernes par la cible. "
                       "Attention au volume pour les auteurs de chaines YouTube : une seule "
                       "video compte pour une dizaine de segments, ce qui les place "
                       "mecaniquement en tete du classement. Le lean moyen, lui, reste "
                       "comparable (il est pondere, pas cumule). Pour les ecarter, "
                       "decochez YouTube dans \"Nature du contenu\".")

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
                st.info(f"Aucun journaliste avec assez d'articles concernes par {target_author_label}.")
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

                # Table complete pour exploitation
                st.markdown("**Table complete**")
                df_disp = df_author_stance.copy()
                df_disp.columns = ["Auteur", "Source", "Articles", "Lean moyen",
                                   "Pro", "Anti", "Neutre"]
                st.dataframe(df_disp, width="stretch", hide_index=True)

        # Treemap sources/auteurs
        st.subheader("Sources et leurs auteurs")
        st.caption("Hierarchie source -> auteur. Cliquez pour zoomer.")

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
with tab_sources:
    st.subheader("Diagnostic de traitement par source")
    st.caption(
        "Pourcentage d'articles traites par chaque analyse, sur la periode choisie. "
        "Vue 7j ou 30j pour voir la tendance recente (sinon la masse historique cache "
        "les bugs en cours). Vert > 95%%, orange 80-95%%, rouge < 80%%."
    )

    # Selecteur de periode
    from datetime import datetime, timedelta
    cov_period = st.radio(
        "Periode",
        ["7 derniers jours", "30 derniers jours", "Tout"],
        horizontal=True, index=0, key="cov_period",
    )
    if cov_period == "7 derniers jours":
        cov_cutoff = (datetime.now() - timedelta(days=7)).date()
    elif cov_period == "30 derniers jours":
        cov_cutoff = (datetime.now() - timedelta(days=30)).date()
    else:
        cov_cutoff = datetime(2000, 1, 1).date()

    # Tables optionnelles (pas encore alimentees si l'analyse correspondante
    # n'a jamais tourne) : on retombe sur un CTE vide plutot que de planter
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
               CAST(ROUND(100.0 * COALESCE(thm.n, 0) / NULLIF(b.with_content, 0)) AS INT) AS "Themes"
        FROM base b
        LEFT JOIN snt ON snt.source_name = b.source_name
        LEFT JOIN thm ON thm.source_name = b.source_name
        LEFT JOIN auth ON auth.source_name = b.source_name
        WHERE b.total > 0
        ORDER BY b.total DESC
    """, [cov_cutoff] * 4).df()

    if df_cov_full.empty:
        st.warning(f"Aucun article sur la periode {cov_period.lower()}.")
    else:
        df_cov_full["Analyses"] = df_cov_full[
            ["Sentiment", "Themes"]
        ].min(axis=1)

        def _statut(row):
            if row["Contenu"] < 80 or row["Analyses"] < 80:
                return "PROBLEME"
            if row["Contenu"] < 95 or row["Analyses"] < 95:
                return "ATTENTION"
            return "OK"

        df_cov_full["Statut"] = df_cov_full.apply(_statut, axis=1)

        n_ok = int((df_cov_full["Statut"] == "OK").sum())
        n_warn = int((df_cov_full["Statut"] == "ATTENTION").sum())
        n_pb = int((df_cov_full["Statut"] == "PROBLEME").sum())
        n_total_src = len(df_cov_full)
        km0, km1, km2, km3 = st.columns(4)
        km0.metric("Sources totales", n_total_src)
        km1.metric(f"Sources OK ({cov_period.lower()})", n_ok)
        km2.metric("Sources en attention", n_warn)
        km3.metric("Sources en probleme", n_pb)

        filt = st.radio(
            "Filtre",
            ["Avec problemes seulement", "Toutes les sources"],
            horizontal=True, index=0, key="cov_filter",
        )
        if filt == "Avec problemes seulement":
            df_display = df_cov_full[df_cov_full["Statut"] != "OK"].copy()
        else:
            df_display = df_cov_full.copy()

        _order = {"PROBLEME": 0, "ATTENTION": 1, "OK": 2}
        df_display["_o"] = df_display["Statut"].map(_order)
        df_display = df_display.sort_values(
            ["_o", "Articles"], ascending=[True, False]
        ).drop(columns=["_o"])

        if df_display.empty:
            st.success("Toutes les sources sont OK sur cette periode.")
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
                        help="Minimum entre Sentiment et Themes"),
                },
            )

            with st.expander("Voir le detail par analyse"):
                df_detail = df_display[
                    ["Source", "Articles", "Sentiment", "Themes"]
                ]
                st.dataframe(
                    df_detail, width="stretch", hide_index=True,
                    column_config={
                        c: st.column_config.ProgressColumn(
                            c, min_value=0, max_value=100, format="%d%%")
                        for c in ["Sentiment", "Themes"]
                    },
                )

    st.markdown("---")
    st.subheader(f"Evolution du taux de traitement, par {GRAIN_LABEL.lower()}")
    st.caption(
        "Selectionnez une source pour voir si elle a commence a buguer a une date precise. "
        "Demarrage des courbes au 1er avril 2026."
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

    # Tables optionnelles : CTE vide (mais avec la meme forme/parametres) si
    # l'analyse correspondante n'a pas encore tourne, plutot que de planter.
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
               ROUND(100.0 * COALESCE(thm_w.n, 0) / NULLIF(w.with_content, 0), 1) AS Themes
        FROM weeks w
        LEFT JOIN snt_w ON snt_w.week = w.week
        LEFT JOIN thm_w ON thm_w.week = w.week
        ORDER BY w.week
    """.format(src_clause=src_clause), src_params * 3).df()

    if not df_evo.empty:
        fig_evo = go.Figure()
        line_colors = {"Contenu": "#FFD93D", "Sentiment": "#FF6B6B",
                       "Themes": "#6BCB77"}
        for col in ["Contenu", "Sentiment", "Themes"]:
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
        fig_evo.update_layout(yaxis_title="% articles traites",
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
                    END AS Videos,
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
               "Recent_7j = articles publies sur les 7 derniers jours. "
               "Videos = nombre de videos ou d'emissions sources, dont chaque "
               "transcription est decoupee en plusieurs segments comptes dans "
               "Articles. "
               "Positionnement historique = label editorial saisi a la main "
               "(config/sources.yaml), pas derive des donnees.")
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
        sel_keyword = st.text_input("Mot-cle (titre ou contenu)", key="src_keyword")

    c3, c4, c5 = st.columns(3)
    with c3:
        date_start = st.date_input("Du", value=date(2026, 5, 1), key="src_date_start")
    with c4:
        date_end = st.date_input("Au", value=date.today(), key="src_date_end")
    with c5:
        sort_by = st.selectbox(
            "Tri",
            ["Date recente", "Date ancienne", "Source A->Z"],
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
        "Date recente": "published_at DESC NULLS LAST",
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

    # Stats par source pour les resultats filtres
    if n_match > 0 and sel_source == "(toutes)":
        st.markdown("---")
        st.markdown("**Repartition des resultats par source**")
        df_by_src = conn.execute(
            f"""SELECT source_name, COUNT(*) AS n
            FROM articles WHERE {where_full}
            GROUP BY 1 ORDER BY n DESC""", params_s
        ).df()
        st.dataframe(df_by_src, width="stretch", hide_index=True)


# ===== Tab Signaux (detection de changements) ===================
with tab_signaux:
    st.subheader("Signaux : ce qui change")
    st.caption(
        "Compare une periode recente a la periode de meme duree qui la "
        "precede immediatement (memes filtres source/langue/mots-cles que "
        "la barre laterale, hors filtre de date). Objectif : reperer un nom "
        "qui commence a circuler (ex: Yabloko), ou un traitement de Poutine "
        "/ de la guerre en Ukraine qui se durcit ou s'adoucit, sans avoir a "
        "comparer les onglets a la main."
    )

    from datetime import timedelta as _td

    window_days = st.select_slider(
        "Fenetre de comparaison (jours)", options=[7, 14, 30, 45, 60, 90],
        value=30, key="sig_window",
        help="Sur une fenetre courte, un jour de collecte manquant suffit a "
             "faire apparaitre de faux signaux. 30 jours ou plus lisse ces "
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
        f"Recente : {recent_start} -> {_today} ({window_days}j)  |  "
        f"Reference : {ref_start} -> {ref_end_excl - _td(days=1)} ({window_days}j)"
    )

    # Volume de chaque fenetre. Tout le reste de l'onglet en depend : comparer
    # des effectifs bruts entre deux fenetres de tailles differentes fait
    # passer une simple montee en charge de la collecte pour un signal.
    _vol_sql = (f"SELECT COUNT(*) FROM articles a WHERE a.published_at >= ? "
                f"AND a.published_at < ? AND {with_a(WHERE_NODATE)}")
    n_recent_total = conn.execute(_vol_sql, [*recent_bounds, *params_nodate]).fetchone()[0]
    n_ref_total = conn.execute(_vol_sql, [*ref_bounds, *params_nodate]).fetchone()[0]

    cvol1, cvol2, cvol3 = st.columns(3)
    cvol1.metric("Articles, periode recente", f"{n_recent_total:,}".replace(",", " "))
    cvol2.metric("Articles, periode de reference", f"{n_ref_total:,}".replace(",", " "))
    _ratio = (n_recent_total / n_ref_total) if n_ref_total else float("inf")
    cvol3.metric("Rapport de volume", "n/a" if n_ref_total == 0 else f"x{_ratio:.1f}")

    # Le corpus est jeune et la collecte est montee en charge : les fenetres
    # anciennes peuvent etre quasi vides. Le signaler explicitement, sinon on
    # lit des variations qui ne disent rien du discours mediatique.
    _MIN_WINDOW_ARTICLES = 100
    _comparable = True
    if n_ref_total < _MIN_WINDOW_ARTICLES or n_recent_total < _MIN_WINDOW_ARTICLES:
        _comparable = False
        st.warning(
            f"Fenetres trop peu fournies pour conclure ({n_recent_total} vs "
            f"{n_ref_total} articles, seuil {_MIN_WINDOW_ARTICLES}). La collecte "
            f"a demarre recemment : elargissez la fenetre ou attendez que "
            f"l'historique s'etoffe."
        )
    elif _ratio > 1.5 or _ratio < 0.67:
        st.warning(
            f"Les deux periodes n'ont pas le meme volume (x{_ratio:.1f}). Les "
            f"comparaisons ci-dessous sont faites en **part du corpus** et non "
            f"en nombre d'articles, ce qui neutralise l'ecart -- mais un rapport "
            f"aussi marque reste a garder en tete."
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

    # --- Derive de la posture par cible (sentiment) --------------------
    if has_target_sent:
        st.markdown("---")
        st.markdown("### Derive de la posture par cible")
        st.caption(
            "Lean moyen (-1 anti / +1 pro) par cible geopolitique, periode "
            "recente vs reference -- utile pour voir si le traitement de "
            "l'Ukraine, de l'Occident, etc. se durcit ou s'adoucit. "
            "Une moyenne etant deja independante du volume, seules les cibles "
            "ayant au moins 10 articles dans chacune des deux periodes sont "
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
        # Une moyenne de lean sur 3 articles n'a aucune stabilite : elle bouge
        # de plusieurs dixiemes des qu'un article change de bord. Seuil releve
        # a 10 pour que le delta affiche traduise une tendance et non le
        # hasard d'echantillonnage.
        _MIN_SENT = 10
        merged_s = merged_s[(merged_s["n_recent"] >= _MIN_SENT)
                            & (merged_s["n_prior"] >= _MIN_SENT)].copy()

        if merged_s.empty:
            st.info("Pas assez d'articles dans les deux periodes pour comparer les postures.")
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
            fig.update_layout(xaxis_title="Delta du lean moyen (recent - reference)",
                              yaxis_autorange="reversed")
            style(fig, max(350, 45 * len(top_movers)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

            df_disp = merged_s[["target_label", "lean_prior", "lean_recent", "delta",
                                 "n_prior", "n_recent"]].round(2)
            df_disp.columns = ["Cible", "Lean (reference)", "Lean (recent)", "Delta",
                                "N (reference)", "N (recent)"]
            st.dataframe(df_disp, width="stretch", hide_index=True)

    # --- Derive des themes traites --------------------------------------
    if has_topics:
        st.markdown("---")
        st.markdown("### Derive des themes traites")
        st.caption(
            "Part de chaque theme (cluster BERTopic) dans le corpus, periode "
            "recente vs reference. Un theme qui gagne ou perd du terrain "
            "signale un changement d'agenda editorial ; un theme absent d'une "
            "des deux periodes peut correspondre a une apparition/disparition."
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
            st.info("Pas assez de donnees pour comparer les themes sur ces deux periodes.")
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
            fig.update_layout(xaxis_title="Delta de part (points, recent - reference)",
                              yaxis_autorange="reversed")
            style(fig, max(350, 40 * len(theme_cmp)))
            st.plotly_chart(fig, width="stretch", key=_next_chart_key())

            df_disp = theme_cmp.round(1).reset_index().rename(
                columns={"label": "Theme", "recent": "% recent",
                         "prior": "% reference", "delta": "Delta (pt)"})
            st.dataframe(df_disp, width="stretch", hide_index=True)


# ===== Tab Cadrage lexical (vocabulaire de propagande, par source) ======
with tab_cadrage:
    st.subheader("Cadrage lexical")
    st.caption(
        "Deux familles mesurees de la meme facon (frequence d'un terme dans "
        "le texte), mais qui se lisent differemment. **Cadrage (propagande)** : "
        "vocabulaire documente par la litterature sur la propagande russe "
        "(monde russe, denazification, russophobie...) -- mesure la PRESENCE "
        "du terme, pas l'adhesion : un media independant peut tres bien citer "
        "ou critiquer ces memes termes. **Indicateur de suivi** : "
        "thermometres suivis en permanence meme a bas bruit (mobilisation, "
        "signaux de negociation, stress economique), la ou le clustering de "
        "l'onglet Themes ne fait emerger un sujet que s'il devient dense. "
        "Pour la posture editoriale, voir l'onglet Sentiment."
    )

    aw = with_a(WHERE)
    n_total_ru = conn.execute(
        f"SELECT COUNT(*) FROM articles a WHERE {aw} AND a.language = 'ru' "
        f"AND a.content IS NOT NULL", params).fetchone()[0]

    if n_total_ru == 0:
        st.info("Aucun article russophone avec contenu sur la periode/filtres choisis.")
    else:
        st.markdown("### Vue d'ensemble")
        overview_rows = []
        for term_label, pattern in ALL_LEXICAL_TERMS.items():
            n = conn.execute(
                f"SELECT COUNT(*) FROM articles a WHERE {aw} AND a.language = 'ru' "
                f"AND a.content IS NOT NULL AND regexp_matches(LOWER(a.content), ?)",
                [*params, pattern]).fetchone()[0]
            overview_rows.append({
                "Categorie": LEXICAL_CATEGORY[term_label],
                "Terme": term_label, "Articles": n,
                "% du corpus": round(100 * n / n_total_ru, 2),
            })
        # Tri par volume et non par categorie : les indicateurs de suivi sont
        # bien plus frequents que les termes de propagande, les grouper par
        # categorie les enterrerait sous la ligne de flottaison du tableau.
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
            st.markdown(f"**Evolution par {GRAIN_LABEL.lower()} (% des articles russophones)**")
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
                st.caption("Pas assez de donnees.")
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
                st.caption("Pas assez de donnees par source.")
            else:
                df_src_term["pct"] = 100 * df_src_term["n"] / df_src_term["total"]
                df_src_term = df_src_term[df_src_term["n"] > 0].sort_values(
                    "pct", ascending=False)
                if df_src_term.empty:
                    st.caption("Aucune source ne mentionne ce terme sur la periode.")
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


# ===== Tab Contexte (paysage mediatique russe) ==========================
# Onglet volontairement statique : il ne lit pas la base. Son role est de
# donner a quelqu'un qui decouvre l'outil de quoi interpreter les autres
# onglets -- savoir que Telegram pese peu dans la population reelle change la
# lecture d'un graphique ou Telegram represente un cinquieme du corpus.
with tab_contexte:
    st.subheader("Comment les Russes s'informent")
    st.caption(
        "Reperes pour lire les autres onglets. Sources : Mediascope "
        "(mesure d'audience, T1 2026) et Levada (sondages, avril et juin 2026). "
        "Ces chiffres ne viennent pas du corpus collecte : ils servent a le "
        "situer."
    )

    # delta_color="off" : ces libellés secondaires sont des valeurs absolues,
    # pas des variations. Colorés, ils se liraient comme des hausses --
    # et une confiance en baisse affichee en vert serait contresens.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Television", "97 %", "3 h 18 / jour", delta_color="off",
              help="Part des Russes qui regardent la television au moins une "
                   "fois par semaine. 99 % chez les 55 ans et plus.")
    m2.metric("Internet", "86 %", "4 h 21 / jour", delta_color="off",
              help="105 millions de personnes, soit 86 % des 12 ans et plus.")
    m3.metric("Reseaux sociaux", "51 %", "2 h 44 / jour", delta_color="off",
              help="Part du temps passe en ligne. VKontakte, Telegram et "
                   "TikTok en tete.")
    m4.metric("Confiance : TV", "41 %", "-8 pts depuis mai 2025",
              delta_color="off",
              help="Premier rang malgre l'erosion. Reseaux sociaux 21 %, "
                   "sites d'info 14 %, chaines Telegram 11 %.")

    st.markdown("### Ou les Russes prennent leur information")
    df_src_info = pd.DataFrame({
        "Support": ["Television", "Reseaux sociaux", "Sites d'information",
                    "Chaines Telegram"],
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
        "L'ecart entre les deux barres se lit comme un usage par defaut : on "
        "regarde la television sans forcement y croire. La confiance recule "
        "partout, le plus fortement pour Telegram (-9 points en un an, apres "
        "les blocages)."
    )

    st.markdown("### Une fracture par age")
    df_age = pd.DataFrame({
        "Tranche": ["18-24 ans", "25-39 ans", "40-54 ans", "55 ans et plus"],
        "Television": [30, 45, 70, 90],
        "Internet et reseaux": [90, 85, 65, 35],
    })
    fig_age = px.line(df_age, x="Tranche", y=["Television", "Internet et reseaux"],
                      markers=True,
                      color_discrete_map={"Television": "#FF6B6B",
                                          "Internet et reseaux": "#4C9AFF"},
                      labels={"value": "% de la tranche", "variable": "",
                              "Tranche": ""})
    style(fig_age, 320)
    st.plotly_chart(fig_age, width="stretch", key=_next_chart_key())
    st.caption(
        "Ordres de grandeur, pas des mesures exactes : ils illustrent le "
        "croisement, documente par Mediascope, entre une audience televisee "
        "agee et une audience en ligne jeune. Les 18-34 ans regardent une "
        "heure de television de moins qu'en 2020. **Consequence pour l'outil** : "
        "une source ne pese pas de la meme facon selon le public vise -- la "
        "television touche l'electorat le plus age et le plus nombreux, "
        "YouTube et Telegram un public plus jeune et urbain."
    )

    st.markdown("### Les supports, un par un")
    MEDIA_NOTES = [
        ("Television", "97 % de couverture hebdomadaire",
         "Premier support d'information du pays et le plus cru. Trois chaines "
         "dominent : Rossiya 1 (13,3 % de part d'audience), NTV, Pervyi Kanal "
         "(7,1 %). L'information y est encadree par l'Etat, et les talk-shows "
         "politiques (« Vremya pokazhet », « Bolshaya igra », Soloviev) y "
         "formulent la ligne officielle de facon bien plus explicite que les "
         "journaux. **Dans l'outil** : transcrits automatiquement depuis "
         "RuTube, faute de sous-titres."),
        ("Sites de presse", "~25 % s'y informent, 14 % leur font confiance",
         "Va des agences d'Etat (TASS, RIA Novosti, RT) aux medias en exil "
         "(Meduza, Novaya Gazeta Europe, The Insider), en passant par une "
         "presse economique moins alignee (Vedomosti, RBK, The Bell). La "
         "plupart des titres independants ont ete declares « agent etranger » "
         "ou « organisation indesirable » et sont bloques en Russie -- ils "
         "publient depuis l'etranger pour un public qui les lit par VPN. "
         "**Dans l'outil** : le socle du corpus."),
        ("Telegram", "~20 % des sondes, 11 % de confiance",
         "Longtemps l'espace le plus libre du RuNet, en net recul depuis les "
         "restrictions. On y trouve les canaux officiels des medias, mais "
         "surtout les « voenkory » (correspondants de guerre pro-guerre comme "
         "Rybar ou Colonelcassad) et des tabloides a forte audience (Mash, "
         "Baza, SHOT) nourris de fuites policieres. **Dans l'outil** : utile "
         "pour la vitesse et pour le discours militaire, mais son poids dans "
         "le corpus depasse son poids reel dans la population."),
        ("VKontakte", "1er reseau social, 1er poste de temps en ligne",
         "L'equivalent russe de Facebook, propriete d'un groupe proche de "
         "l'Etat. Les grands medias y touchent un public qui ne visite jamais "
         "leur site, avec des formulations souvent plus directes. **Dans "
         "l'outil** : collecte limitee, VK opposant une verification "
         "anti-robot aux visiteurs automatises."),
        ("YouTube", "~22 M/jour, ralenti en Russie depuis 2024",
         "Principal espace de discours critique encore accessible. Les "
         "chaines les plus vues (Maxime Kats, Varlamov, Chtefanov, vDud) sont "
         "toutes animees depuis l'exil et classees « agent etranger ». Les "
         "chaines d'Etat, elles, en ont ete retirees. **Dans l'outil** : "
         "attention, ce volet est structurellement d'opposition -- ce n'est "
         "pas un echantillon de la video russophone."),
        ("Radio et Dzen", "audiences secondaires",
         "Vesti FM et Radio Rossii pour la radio ; Dzen, plateforme "
         "d'articles de Yandex, touche 31 millions de personnes par jour mais "
         "melange information et contenu de divertissement. **Dans l'outil** : "
         "non couverts a ce jour."),
    ]
    for titre, chiffre, texte in MEDIA_NOTES:
        with st.expander(f"{titre} — {chiffre}"):
            st.markdown(texte)

    st.info(
        "**A garder en tete en lisant les autres onglets.** La composition du "
        "corpus ne reproduit pas celle de la consommation reelle : la presse "
        "web et Telegram y sont surrepresentes parce qu'ils sont faciles a "
        "collecter, la television sous-representee parce que chaque emission "
        "doit etre transcrite. Un theme dominant dans le corpus n'est donc pas "
        "forcement un theme dominant dans ce que les Russes voient. Le filtre "
        "« Nature du contenu », a gauche, sert precisement a corriger cette "
        "lecture -- en isolant la television, par exemple."
    )


# ===== Panneau de couverture ===================================
# Hors de la barre d'onglets : les neuf onglets disent ce que racontent les
# medias russes, ce panneau dit ce qu'on suit. Il ignore les filtres -- une
# fiche de couverture qui retrecit quand on filtre ne renseigne plus.
st.markdown("""
<div class="rm-cover">
  <h4>Couverture du corpus</h4>
  <p>Ce qui est suivi, source par source, independamment des filtres appliques
  ci-dessus. Les emissions de television sont regroupees par chaine, avec leur
  part d'audience nationale.</p>
</div>
""", unsafe_allow_html=True)

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
# C'est la seule mesure d'audience reelle disponible -- les vues RuTube ou
# YouTube mesurent le rattrapage en ligne, pas l'antenne.
CHAINES = {
    "Rossiya 1": ("13,4 % de part d'audience nationale (1re chaine)",
                  "Chaine phare de VGTRK, groupe public. Ses talk-shows de "
                  "soiree sont le lieu ou la ligne officielle est enoncee le "
                  "plus explicitement."),
    "NTV": ("9,5 % (2e chaine)",
            "Groupe Gazprom-Media. Meme ligne editoriale que les chaines "
            "publiques, registre plus sensationnaliste -- faits divers, "
            "securite, plateaux houleux."),
    "Pervyi Kanal": ("7,5 % (3e chaine)",
                     "Heritiere de la 1re chaine sovietique, Etat actionnaire "
                     "majoritaire. Son journal de 21 h « Время » reste le "
                     "programme d'information de reference du pays."),
    "Soloviev LIVE": ("chaine en ligne, hors mesure d'antenne",
                      "Studio personnel de Vladimir Soloviev (RuTube, "
                      "Telegram, VK). Formats tres longs, 4 h, sans les "
                      "contraintes de l'antenne : le ton y est plus libre que "
                      "sur Rossiya 1."),
}

NATURES = [
    ("tv", "Television", "emissions"),
    ("press", "Presse", "articles"),
    ("telegram", "Telegram", "posts"),
    ("youtube", "YouTube", "videos"),
    ("vk", "VKontakte", "posts"),
]


def _stat(nom):
    """Ligne de statistiques d'une source, ou des zeros si elle n'a rien
    rapporte -- une source configuree mais muette doit rester visible."""
    if nom not in _cov_src.index:
        return dict(unites=0, segments=0, mots=0, minutes=0, vues=None, dernier="")
    r = _cov_src.loc[nom]

    # `x or 0` ne suffit pas : les agregats absents remontent en NaN, et
    # `NaN or 0` vaut NaN -- la presse, qui n'a ni duree ni vues, faisait alors
    # echouer la conversion en entier.
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
             f"{tot:,} {unite} collectee(s)".replace(",", " "))
    if en_dev:
        titre += "  ·  en developpement"

    with st.expander(titre, expanded=False):
        if en_dev:
            st.warning(
                "**Collecte en developpement.** VK oppose une verification "
                "anti-robot apres une vingtaine de visites anonymes depuis une "
                "meme adresse : en pratique deux a trois communautes passent "
                "par run, pas les cinq. Les chiffres ci-dessous sont donc un "
                "plancher, et l'absence d'une communaute un jour donne ne dit "
                "rien de son activite.")

        if kind == "tv":
            # Regroupement par chaine : c'est la chaine qui porte l'audience,
            # pas l'emission.
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
                        "Emission": s["name"].split(" (")[0],
                        "Acces": {"rutube": "RuTube", "hls": "flux intercepte"}
                                 .get(s.get("type"), s.get("type", "")),
                        "Episodes": st_["unites"], "Segments": st_["segments"],
                        "Mots": st_["mots"], "Minutes": st_["minutes"],
                        "Vues": st_["vues"], "Dernier": st_["dernier"],
                    })
                _table(lignes, ["Emission", "Acces", "Episodes", "Segments",
                                "Mots", "Minutes", "Vues", "Dernier"])
                for s in sorted(emissions, key=lambda x: x["name"]):
                    if s.get("historical_stance"):
                        st.caption(f"**{s['name'].split(' (')[0]}** — "
                                   f"{s['historical_stance']}")
                st.markdown("")
            st.caption(
                "**Vues** : lectures en ligne sur RuTube, publiees par la "
                "plateforme et renseignees depuis le 15 aout 2026 -- vides "
                "pour les episodes collectes avant. Elles ne mesurent PAS "
                "l'audience d'antenne : « Время » reunit plusieurs millions de "
                "telespectateurs pour quelques milliers de vues RuTube. Pour "
                "l'audience reelle, c'est la part de chaine ci-dessus qui fait "
                "foi. **Minutes** : estimation basse, tiree de l'instant de "
                "depart du dernier segment de chaque episode.")
        else:
            lignes = []
            for s in sorted(cfg, key=lambda x: -_stat(x["name"])["mots"]):
                st_ = _stat(s["name"])
                ligne = {
                    "Source": s["name"], "Type": s.get("media_type", ""),
                    "Statut legal": s.get("legal_status", ""),
                    unite.capitalize(): st_["unites"],
                    "Mots": st_["mots"], "Dernier": st_["dernier"],
                }
                if kind == "youtube":
                    ligne["Segments"] = st_["segments"]
                    ligne["Minutes"] = st_["minutes"]
                    ligne["Vues"] = st_["vues"]
                lignes.append(ligne)
            cols = ["Source", "Type", "Statut legal", unite.capitalize()]
            if kind == "youtube":
                cols += ["Segments", "Minutes", "Vues"]
            cols += ["Mots", "Dernier"]
            _table(lignes, cols)
            if kind == "youtube":
                st.caption(
                    "Volet structurellement d'opposition : les quatre chaines "
                    "les plus vues du YouTube politique russophone sont animees "
                    "depuis l'exil et classees « agent etranger ». Les chaines "
                    "d'Etat, elles, ont ete retirees de la plateforme. Ce n'est "
                    "donc pas un echantillon de la video russophone.")
            elif kind == "telegram":
                st.caption(
                    "Trois familles y cohabitent : les canaux officiels des "
                    "medias (doublons rapides de leur site), les « voenkory » "
                    "correspondants de guerre, souvent plus critiques du "
                    "ministere de la Defense que la presse d'Etat, et les "
                    "tabloides a forte audience nourris de fuites policieres.")
            elif kind == "press":
                st.caption(
                    "Une source « scrape » est lue sur sa page d'accueil, faute "
                    "de flux RSS exploitable : sa couverture est plus "
                    "irreguliere qu'un flux, et un site refondu peut cesser de "
                    "rendre sans erreur visible. Un « Dernier » qui prend du "
                    "retard est le signal a surveiller.")

# Sources configurees mais absentes de la base : le signal le plus utile du
# panneau, une source qui ne rapporte rien ne se voit nulle part ailleurs.
_muettes = [s["name"] for s in SOURCE_CONFIG if s["name"] not in _cov_src.index]
if _muettes:
    st.caption(f"**{len(_muettes)} sources sans aucun contenu en base** : "
               + ", ".join(sorted(_muettes))
               + ". Emission en relache, source recemment ajoutee, ou collecte "
                 "en echec -- a verifier dans les journaux de la derniere passe.")

conn.close()