import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(title="Survie en réanimation", icon="🫀", dataset="Patient Survival WiDS 2020",
            problem="Classification binaire (survie / décès hospitalier)", owner="Amal Ghali", version="1.0")

COLORS = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴"}


def render():
    module_header(META)
    disclaimer()
    try:
        art = P.load_artifact()
    except FileNotFoundError as e:
        st.error(str(e)); return

    st.write("Saisir les mesures des **24 premières heures** de réanimation.")
    with st.form("survival"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge (ans)", 16, 100, 65)
        hr = c1.number_input("Fréquence cardiaque max (bpm)", 30, 250, 100)
        smin = c2.number_input("TA systolique min (mmHg)", 30, 250, 96)
        smax = c2.number_input("TA systolique max (mmHg)", 40, 300, 140)
        eyes = c3.selectbox("Glasgow – ouverture des yeux", [4, 3, 2, 1], help="4 = spontanée … 1 = aucune")
        motor = c3.selectbox("Glasgow – réponse motrice", [6, 5, 4, 3, 2, 1], help="6 = obéit … 1 = aucune")
        verbal = c3.selectbox("Glasgow – réponse verbale", [5, 4, 3, 2, 1], help="5 = orientée … 1 = aucune")
        vent = c1.checkbox("Ventilation mécanique")
        elec = c2.checkbox("Chirurgie programmée")
        src_opts = art["cat_choices"].get("icu_admit_source", [])
        body_opts = art["cat_choices"].get("apache_2_bodysystem", [])
        src = c2.selectbox("Provenance à l'admission", src_opts) if src_opts else None
        body = c3.selectbox("Système atteint (APACHE)", body_opts) if body_opts else None
        go = st.form_submit_button("Estimer le risque")

    if go:
        inp = dict(age=age, d1_heartrate_max=hr, d1_sysbp_min=smin, d1_sysbp_max=smax, gcs_eyes_apache=eyes,
                   gcs_motor_apache=motor, gcs_verbal_apache=verbal, ventilated_apache=vent, elective_surgery=elec,
                   icu_admit_source=src, apache_2_bodysystem=body)
        try:
            r = P.predict(inp, art)
        except ValueError as e:
            st.error(f"Saisie invalide : {e}"); return
        a, b = st.columns([1, 2])
        a.metric("Risque de décès estimé", f"{r['risk']*100:.1f} %", help="Probabilité calibrée sur les données de test")
        b.markdown(f"#### {COLORS[r['level']]} Niveau d'alerte : **{r['level']}**")
        b.write(r["message"])
        if r["factors"]:
            st.write("**Facteurs qui pèsent le plus dans cette estimation** (positif = augmente le risque)")
            st.bar_chart({n: v for n, v in r["factors"]}, horizontal=True)

    with st.expander("Performance et limites du modèle"):
        m = art.get("metrics", {})
        st.write(f"XGBoost · ROC-AUC {m.get('roc_auc', float('nan')):.3f} · PR-AUC {m.get('pr_auc', float('nan')):.3f} "
                 f"(test, {m.get('n_test', 0):,} séjours). Seuils d'alerte choisis sur le train.")
        st.markdown("- Entraîné sur 10 variables seulement (lisibilité) · pas de validation externe sur d'autres hôpitaux\n"
                    "- Recall faible chez les patients non ventilés ou opérés en programmé\n- Données de 2020 : à recalibrer avant tout usage réel")
