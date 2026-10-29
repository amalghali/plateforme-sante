import streamlit as st

DISCLAIMER = ("⚠️ Outil d'aide à la décision destiné aux professionnels de santé. Il fournit une estimation statistique, "
              "**pas un diagnostic** : la décision finale reste celle du médecin. Aucune donnée saisie n'est enregistrée.")


def module_header(meta: dict):
    st.markdown(f"### {meta.get('icon', '')} {meta['title']}")
    st.caption(f"{meta['problem']} · dataset : {meta['dataset']} · responsable : {meta['owner']}")


def disclaimer():
    st.info(DISCLAIMER)
