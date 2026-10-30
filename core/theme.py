"""Thème « Institutionnel sobre » : menu sombre, champs vert clair, angles droits, tableau d'accueil, formulaires alignés.
Injecté une seule fois depuis app.py. Les modules n'ont rien à faire : le CSS restyle leurs formulaires et boutons.
Réglages faciles à changer : les constantes ci-dessous."""
import streamlit as st

CHAMP_BG = "#E3F1EA"      # fond des champs du formulaire (vert clair)
CHAMP_BORD = "#A9CDB9"    # contour des champs
ACCENT = "#4CAF87"        # barre verte du menu, focus des champs
SIDEBAR = "#1C2B25"       # fond du menu de gauche
TEXTE_MENU = "#B9C7BF"    # texte du menu
LIGNE = "108px"           # hauteur d'une ligne de formulaire (étiquette jusqu'à 3 lignes + champ)

S = 'section[data-testid="stSidebar"]'
G = S + ' div[role="radiogroup"]'
R = G + ' label'

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@600;700&display=swap');
.stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre){font-family:'IBM Plex Sans',system-ui,sans-serif}
.stApp{background:#FFFFFF}
[data-testid="stHeader"]{background:transparent}
h1,h2,h3,h4{color:#111827 !important;font-weight:600 !important;letter-spacing:-.01em}

@S@{background:@SIDEBAR@ !important;width:270px !important;min-width:270px !important}
@S@ h1{font-size:1.05rem !important;text-transform:uppercase;letter-spacing:.14em;color:#CFE8DB !important}
@S@ [data-testid="stWidgetLabel"] p{color:#8FBFAA !important;text-transform:uppercase;letter-spacing:.1em;font-size:11px !important}
@S@ [data-testid="stCaptionContainer"] *{color:#7C9488 !important;font-size:12px !important}
@R@ > *:not(:has(p)):not(input){display:none !important}
@R@{padding:9px 14px !important;border-radius:0;border-left:3px solid transparent;margin:0 0 2px;width:100%}
@R@ p{font-size:14px !important;color:@MENU@ !important}
@R@:hover{background:rgba(255,255,255,.05)}
@R@:has(input:checked){background:rgba(255,255,255,.08);border-left:3px solid @ACCENT@}
@R@:has(input:checked) p{color:#FFFFFF !important;font-weight:500}

/* « Accueil » : seulement le premier élément du menu, plus grand, police à empattement, trait de séparation */
@G@ > *:first-child:first-child{padding-bottom:10px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.16)}
@G@ > *:first-child:first-child p{font-family:'Source Serif 4',Georgia,serif !important;font-size:21px !important;font-weight:700 !important;letter-spacing:.01em;color:#FFFFFF !important}

[data-testid="stWidgetLabel"] p{text-transform:uppercase;letter-spacing:.02em;font-size:11.5px !important;color:#4B5563 !important;font-weight:500}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div{background:@CHAMP@ !important;border:1px solid @BORD@ !important;border-radius:2px !important}
[data-baseweb="input"] input,[data-baseweb="base-input"] input{background:transparent !important}
[data-testid="stNumberInput"] button{background:transparent !important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:@ACCENT@ !important}

.stButton>button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"],[data-testid="stFormSubmitButton"] button{background:@SIDEBAR@ !important;color:#FFFFFF !important;border:0 !important;border-radius:2px !important;padding:.5rem 1.3rem !important;font-weight:500}
.stButton>button:hover,[data-testid="stBaseButton-secondary"]:hover,[data-testid="stBaseButton-primary"]:hover{background:#2D4539 !important}
[data-testid="stBaseButton-secondary"] p,[data-testid="stBaseButton-primary"] p,.stButton>button p{color:#FFFFFF !important}

[data-testid="stVerticalBlockBorderWrapper"]{border-color:#C7CDD3;border-radius:2px}
[data-testid="stExpander"],[data-testid="stExpander"] details{border-radius:2px !important}

/* Accueil : contenu en texte simple (sans cadre), avec le statut coloré */
.card{margin-bottom:18px}
.card b{color:#111827;font-size:15px;font-weight:600}
.card p{color:#4B5563;font-size:13px;margin:6px 0}
.badge{display:inline-block;font-size:11px;padding:2px 9px;border-radius:2px;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.badge.ok{background:#D5F0DF;color:#14632F}
.badge.wip{background:#FDE9C4;color:#8A4B06}
.badge.err{background:#FBD5D5;color:#991B1B}

/* Accueil organisé : sections et tableau aligné */
.sec{font-size:12px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:@SIDEBAR@;border-bottom:2px solid @ACCENT@;padding-bottom:6px;margin:28px 0 6px}
.sec span{color:#6B7280;font-weight:500;margin-left:8px}
[data-testid="stMarkdownContainer"] table.mods{width:100%;border-collapse:collapse;table-layout:fixed;margin:0 0 6px;border:0}
[data-testid="stMarkdownContainer"] table.mods th{width:auto;text-align:left;font-size:11px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:#6B7280;padding:8px 10px;border:0;border-bottom:1px solid #D1D5DB;background:transparent}
[data-testid="stMarkdownContainer"] table.mods td{padding:13px 10px;border:0;border-bottom:1px solid #E5E7EB;vertical-align:middle;color:#374151;font-size:14px;background:transparent}
[data-testid="stMarkdownContainer"] table.mods td:first-child{font-weight:600;color:#111827}
[data-testid="stMarkdownContainer"] table.mods th:nth-child(1){width:32%}
[data-testid="stMarkdownContainer"] table.mods th:nth-child(2){width:30%}
[data-testid="stMarkdownContainer"] table.mods th:nth-child(3){width:22%}
[data-testid="stMarkdownContainer"] table.mods th:nth-child(4){width:16%}
[data-testid="stMarkdownContainer"] table.mods tr:hover td{background:#F5F8F6}

/* Formulaires : chaque ligne a la même hauteur dans toutes les colonnes, donc les champs restent alignés même si une étiquette est longue */
[data-testid="stWidgetLabel"]{min-height:56px;align-items:flex-end !important}
@S@ [data-testid="stWidgetLabel"]{min-height:0}
[data-testid="stHorizontalBlock"]{align-items:flex-start !important}
[data-baseweb="input"],[data-baseweb="select"]>div{min-height:40px}
[data-testid="stColumn"] [data-testid="stElementContainer"],[data-testid="column"] [data-testid="stElementContainer"]{min-height:@LIGNE@}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]),[data-testid="column"] [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]){padding-top:40px}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(h1,h2,h3,h4,h5,h6,p>strong:only-child),[data-testid="column"] [data-testid="stElementContainer"]:has(h1,h2,h3,h4,h5,h6,p>strong:only-child){min-height:86px}
"""

for _token, _value in {"@S@": S, "@G@": G, "@R@": R, "@SIDEBAR@": SIDEBAR, "@MENU@": TEXTE_MENU,
                       "@ACCENT@": ACCENT, "@LIGNE@": LIGNE, "@CHAMP@": CHAMP_BG, "@BORD@": CHAMP_BORD}.items():
    CSS = CSS.replace(_token, _value)


def apply_theme():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
