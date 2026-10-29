import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(title="Réadmission des patients diabétiques", icon="💉", dataset="Diabetes 130-US Hospitals",
            problem="Classification multiclasse (NO / >30 / <30)", owner="Malek Rahmouni", version="1.0")

COLORS = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🔴"}
CLASS_LABELS = {"NO": "Pas de réadmission", ">30": "Après 30 jours", "<30": "Dans les 30 jours"}


def age_label(cat: str) -> str:
    lo, hi = cat.strip("[)").split("-")
    return f"{lo} à {int(hi) - 1} ans"


def render():
    module_header(META)
    disclaimer()
    try:
        art = P.load_artifact()
    except FileNotFoundError as e:
        st.error(str(e)); return

    st.write("Saisir les informations disponibles **à la sortie du patient**. "
             "Le modèle estime s'il risque d'être **réadmis dans les 30 jours** pour organiser son suivi.")
    with st.form("diabetes_readmission"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Patient et admission**")
        age = c1.selectbox("Tranche d'âge", art["age_categories"], index=7, format_func=age_label)
        adm_type = c1.selectbox("Type d'admission", list(P.ADMISSION_TYPE), format_func=P.ADMISSION_TYPE.get)
        adm_src = c1.selectbox("Provenance du patient", list(P.ADMISSION_SOURCE), format_func=P.ADMISSION_SOURCE.get)
        discharge = c1.selectbox("Mode de sortie", list(P.DISCHARGE), format_func=P.DISCHARGE.get)

        c2.markdown("**Séjour**")
        days = c2.number_input("Durée du séjour (jours)", 1, 14, 3)
        labs = c2.number_input("Nombre d'analyses de laboratoire", 1, 132, 44)
        procs = c2.number_input("Nombre d'actes médicaux (hors analyses)", 0, 6, 1)
        meds = c2.number_input("Nombre de médicaments administrés", 1, 81, 15)
        diags = c2.number_input("Nombre de diagnostics", 1, 16, 9)

        c3.markdown("**Traitement du diabète et historique**")
        diab_med = c3.radio("Médicament du diabète prescrit", ["Oui", "Non"], horizontal=True) == "Oui"
        insulin = c3.selectbox("Insuline", list(P.INSULIN), format_func=P.INSULIN.get)
        changes = c3.number_input("Nb de médicaments du diabète dont la dose a changé", 0, 23, 0)
        outp = c3.number_input("Consultations externes l'année d'avant", 0, 42, 0)
        emerg = c3.number_input("Passages aux urgences l'année d'avant", 0, 76, 0)
        inpat = c3.number_input("Hospitalisations l'année d'avant", 0, 21, 0)
        go = st.form_submit_button("Estimer le risque de réadmission")

    if go:
        inp = dict(age=age, admission_type=adm_type, admission_source=adm_src, discharge=discharge,
                   time_in_hospital=days, num_lab_procedures=labs, num_procedures=procs, num_medications=meds,
                   number_diagnoses=diags, diabetes_med=diab_med, insulin=insulin, n_med_changes=changes,
                   number_outpatient=outp, number_emergency=emerg, number_inpatient=inpat)
        try:
            r = P.predict(inp, art)
        except ValueError as e:
            st.error(f"Saisie invalide : {e}"); return
        a, b = st.columns([1, 2])
        a.metric("Réadmission prédite", CLASS_LABELS[r["decision"]])
        b.markdown(f"#### {COLORS[r['level']]} Niveau d'alerte : **{r['level']}**")
        b.write(r["message"])
        if r["capacity"]:
            b.write(f"Ce séjour fait partie du **{r['capacity']}** des scores <30 les plus élevés : "
                    "à appeler en priorité si l'équipe a une capacité de suivi limitée.")
        st.write("**Score du modèle pour chaque classe** (sert à classer les patients, ce n'est pas une probabilité en %)")
        st.bar_chart(r["scores"], horizontal=True, color="#1F7A5A")
        st.caption("NO = pas de réadmission · >30 = réadmission après 30 jours · <30 = réadmission dans les 30 jours")
        st.caption(f"Règle de décision : alerte « réadmission dans les 30 jours » dès que son score atteint "
                   f"{art['threshold_lt30']:.2f}".replace(".", ",") + " ; sinon, la classe au plus haut score entre NO et >30.")

    with st.expander("Ce que le modèle regarde le plus"):
        st.markdown("1. **Nombre de visites l'année d'avant** (consultations + urgences + hospitalisations), de loin le plus important\n"
                    "2. **Mode de sortie** (retour à domicile, soins à domicile…)\n3. **Âge**\n\n"
                    "Importance mesurée par permutation sur le test (phase 4 du notebook).")

    with st.expander("Performance et limites du modèle"):
        m = art.get("metrics", {})
        fr = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")
        lo, hi = m.get("rappel_lt30_ic", (float("nan"), float("nan")))
        n_test = f"{m.get('n_test', 0):,}".replace(",", " ")
        st.write(f"Gradient Boosting + class_weight · seuil {fr(art['threshold_lt30'], 2)} · test de {n_test} séjours : "
                 f"trouve **{fr(m.get('rappel_lt30', 0) * 100)} %** des patients réadmis dans les 30 jours "
                 f"(intervalle de confiance {fr(lo * 100)} – {fr(hi * 100)} %), soit environ "
                 f"{fr(m.get('gain_hasard', 0))} fois plus qu'un choix au hasard du même nombre de patients. "
                 f"F1 macro {fr(m.get('f1_macro_seuil', 0), 3)}.")
        st.markdown("- Environ **1 alerte sur 6** est juste : outil d'aide au tri, **pas une décision automatique**\n"
                    "- Environ la moitié des patients réadmis dans les 30 jours ne sont pas repérés\n"
                    "- Scores non calibrés : un score de 0,34 ne veut pas dire 34 % de risque\n"
                    "- Formulaire réduit à 15 champs (18 variables) : un modèle avec toutes les variables ferait un peu mieux\n"
                    "- Données de 130 hôpitaux américains (1999-2008) : à revérifier sur des données récentes avant tout usage réel")
