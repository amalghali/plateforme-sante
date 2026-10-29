import streamlit as st
from core.registry import discover
from core.team import TEAM
from core.ui import disclaimer

st.set_page_config(page_title="Plateforme d'aide à la décision en santé", page_icon="🏥", layout="wide")
modules = discover()

st.sidebar.title("🏥 Plateforme santé")
labels = {"home": "🏠 Accueil"}
for t in TEAM:
    m = modules.get(t["id"])
    labels[t["id"]] = (f"{m.META.get('icon', '🔹')} {m.META['title']}" if m is not None and not isinstance(m, Exception)
                       else f"⏳ {t['dataset']}")
choice = st.sidebar.radio("Modules", list(labels), format_func=labels.get)
st.sidebar.caption("Projet de validation ML · 4CCE12 · CRISP-DM")

if choice == "home":
    st.title("Plateforme web d'aide à la décision en santé")
    st.write("6 modèles de Machine Learning, un par domaine médical. Choisissez un module dans le menu de gauche.")
    disclaimer()
    rows = []
    for t in TEAM:
        m = modules.get(t["id"])
        status = "✅ disponible" if m is not None and not isinstance(m, Exception) else ("❌ erreur" if isinstance(m, Exception) else "⏳ en cours")
        rows.append({"Module": t["dataset"], "Problème": t["problem"], "Responsable": t["owner"], "Statut": status})
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    m = modules.get(choice)
    if m is None:
        st.title(labels[choice]); st.info("Ce module sera ajouté par son responsable (voir README, section « Ajouter mon module »).")
    elif isinstance(m, Exception):
        st.error(f"Le module « {choice} » a une erreur : {m}")
    else:
        m.render()
