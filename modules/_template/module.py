"""MODÈLE DE MODULE — copier ce dossier : cp -r modules/_template modules/<mon_id>
<mon_id> = l'id de core/team.py (diabetes_readmission, bone_marrow, dengai, heart_failure, stroke_anomaly).
Ne modifier QUE votre dossier. L'application détecte automatiquement module.py : META + render()."""
import streamlit as st
from core.ui import module_header, disclaimer

META = dict(title="Titre court du module", icon="🩺", dataset="Nom du dataset",
            problem="Type de problème", owner="Votre nom", version="0.1")


def render():
    module_header(META)
    disclaimer()
    # 1) charger votre artefact : joblib.load(Path(__file__).parent / "artifacts" / "mon_modele.joblib")
    # 2) formulaire de saisie (st.form) -> dict d'entrées
    # 3) appeler votre predict.py (logique SANS streamlit, testable)
    # 4) afficher : prédiction + niveau/interprétation + facteurs explicatifs + performance/limites du modèle
    st.info("Module en cours de construction.")
