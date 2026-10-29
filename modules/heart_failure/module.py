import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(title="Profils rétiniens", icon="👁️",
            dataset="Diabetic Retinopathy Debrecen — UCI",
            problem="Clustering exploratoire (K-means, 3 groupes)",
            owner="Ahmed Jribi", version="1.0")

def render():
    module_header(META)
    disclaimer()
    st.write("Affecter une observation aux groupes décrits dans le projet rétinien.")
    st.caption("Saisir les descripteurs extraits selon le protocole du dataset Debrecen. "
               "Cette interface n’analyse pas directement une photographie de la rétine.")
    try:
        art = P.load_artifact()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return
    with st.form("retine_ahmed"):
        quality = st.checkbox("Image de qualité suffisante (quality = 1)")
        cols = st.columns(3)
        inputs = {}
        for i, name in enumerate(art["features"]):
            default = art["defaults"][name]
            if name in ("ma1", "ma5"):
                inputs[name] = cols[i % 3].number_input(
                    name, min_value=0, value=int(default), step=1)
            else:
                inputs[name] = cols[i % 3].number_input(
                    name, min_value=0.0, value=float(default), format="%.6f")
        go = st.form_submit_button("Identifier le groupe exploratoire")
    if go:
        try:
            result = P.predict({**inputs, "quality": int(quality)}, art)
        except ValueError as error:
            st.error(str(error))
        else:
            st.metric("Groupe", str(result["cluster"]))
            st.write(result["description"])
            st.write(f"Effectif de référence : {result['effectif']} observations.")
            st.caption(result["message"])
    with st.expander("Évaluation et limites", expanded=True):
        st.write("Silhouette : 0,3013 · ARI moyen : 0,9477 · ARI minimum : 0,8709.")
        st.write("Effectifs : 290 / 844 / 13. Séparation partielle ; petit groupe de 13 "
                 "observations ; sensibilité au prétraitement ; aucune validation clinique "
                 "ou évaluation de généralisation. Une nouvelle observation est toujours "
                 "affectée au centre le plus proche, même si elle est très différente des données.")
