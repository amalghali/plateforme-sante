"""Plateforme d'aide à la décision en santé : accueil organisé en sections (tableau aligné) et navigation par module."""
import html

import streamlit as st
from core.registry import discover
from core.team import TEAM
from core.ui import disclaimer
from core.theme import apply_theme

st.set_page_config(page_title="Plateforme d'aide à la décision en santé", page_icon="🏥", layout="wide")
apply_theme()
modules = discover()

st.sidebar.title("Plateforme santé")
labels = {"home": "Accueil"}
for t in TEAM:
    m = modules.get(t["id"])
    labels[t["id"]] = (m.META["title"] if m is not None and not isinstance(m, Exception) else t["dataset"])
choice = st.sidebar.radio("Modules", list(labels), format_func=labels.get)
st.sidebar.caption("Projet de validation ML · 4CCE12 · CRISP-DM")

SECTIONS = (("ok", "Modules disponibles", "disponible"),
            ("wip", "En cours de développement", "en cours"),
            ("err", "Modules en erreur", "erreur"))


def kind_of(t):
    m = modules.get(t["id"])
    if isinstance(m, Exception):
        return "err"
    return "ok" if m is not None else "wip"


def table_html(rows, label):
    e = html.escape
    head = "<tr><th>Dataset</th><th>Type de problème</th><th>Responsable</th><th>Statut</th></tr>"
    body = "".join(
        f'<tr><td>{e(t["dataset"])}</td><td>{e(t["problem"])}</td><td>{e(t["owner"])}</td>'
        f'<td><span class="badge {k}">{label}</span></td></tr>'
        for t, k in rows)
    return f'<table class="mods"><thead>{head}</thead><tbody>{body}</tbody></table>'


if choice == "home":
    st.title("Plateforme web d'aide à la décision en santé")
    st.write("6 modèles de Machine Learning, un par domaine médical. Choisissez un module dans le menu de gauche.")
    disclaimer()
    entries = [(t, kind_of(t)) for t in TEAM]
    for kind, title, label in SECTIONS:
        rows = [(t, k) for t, k in entries if k == kind]
        if rows:
            st.markdown(f'<div class="sec">{title}<span>{len(rows)}</span></div>', unsafe_allow_html=True)
            st.markdown(table_html(rows, label), unsafe_allow_html=True)
else:
    m = modules.get(choice)
    if m is None:
        st.title(labels[choice]); st.info("Ce module sera ajouté par son responsable (voir README, section « Ajouter mon module »).")
    elif isinstance(m, Exception):
        st.error(f"Le module « {choice} » a une erreur : {m}")
    else:
        m.render()
