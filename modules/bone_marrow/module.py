import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(title="Durée de séjour (diabète)", icon="🩺", dataset="Diabetes 130-US Hospitals",
            problem="Régression (durée de séjour en jours)", owner="Wael Ben Hadj Yahia", version="1.0")


def _select(col, label, choices, default=None, help=None):
    idx = choices.index(default) if default in choices else 0
    return col.selectbox(label, choices, index=idx, help=help)


def render():
    module_header(META)
    disclaimer()
    try:
        art = P.load_artifact()
    except FileNotFoundError as e:
        st.error(str(e)); return

    rg, ch = art["num_ranges"], art["cat_choices"]

    def num(col, key, label, help=None):
        r = rg[key]
        return col.number_input(label, int(r["min"]), int(r["max"]), int(r["median"]), help=help)

    st.write("Estimer la **durée de séjour hospitalier** (en jours) d'un patient diabétique à partir de son dossier d'admission.")
    ages = list(range(5, 96, 10))
    with st.form("diabetes_los"):
        c1, c2, c3 = st.columns(3)
        age = c1.selectbox("Tranche d'âge", ages, index=ages.index(65), format_func=lambda a: f"{a - 5}–{a + 5} ans")
        sex = c1.selectbox("Sexe", ["Femme", "Homme"])
        a1c = c1.checkbox("Test d'HbA1c réalisé", help="Dosage de l'hémoglobine glyquée effectué pendant le séjour")
        glucose = _select(c1, "Glycémie sérique (max)", ch["glucose_group"], "None", help="None = non mesurée")
        n_med = num(c2, "num_medications", "Nombre de médicaments administrés")
        n_lab = num(c2, "num_lab_procedures", "Nombre d'examens de laboratoire")
        n_diag = num(c2, "number_diagnoses", "Nombre de diagnostics saisis")
        n_inp = num(c2, "number_inpatient", "Hospitalisations de l'année précédente")
        n_chg = num(c3, "n_med_changes", "Médicaments dont la dose a changé", help="Parmi les traitements du diabète")
        n_act = num(c3, "n_med_active", "Médicaments du diabète actifs")
        src = _select(c3, "Provenance à l'admission", ch["admission_source_group"], "Emergency_Room")
        spec = _select(c3, "Spécialité du médecin", ch["medical_specialty_group"], "InternalMedicine")
        d1, d2, d3 = st.columns(3)
        diag1 = _select(d1, "Diagnostic principal (code CIM-9)", ch["diag_1_detail"], "250",
                        help="Code à 3 chiffres ; « Rare_... » = regroupement des codes peu fréquents")
        diag2 = _select(d2, "Diagnostic secondaire 2 (groupe)", ch["diag_2_group"], "Circulatory")
        diag3 = _select(d3, "Diagnostic secondaire 3 (groupe)", ch["diag_3_group"], "None")
        go = st.form_submit_button("Estimer la durée de séjour")

    if go:
        inp = dict(num_medications=n_med, num_lab_procedures=n_lab, diag_1_detail=diag1, number_diagnoses=n_diag,
                   n_med_changes=n_chg, medical_specialty_group=spec, diag_3_group=diag3, diag_2_group=diag2,
                   age_ordinal=age, number_inpatient=n_inp, n_med_active=n_act, admission_source_group=src,
                   a1c_tested=int(a1c), glucose_group=glucose, is_male=int(sex == "Homme"))
        try:
            r = P.predict(inp, art)
        except ValueError as e:
            st.error(f"Saisie invalide : {e}"); return
        a, b = st.columns([1, 2])
        a.metric("Durée de séjour estimée", f"{r['days']:.1f} jours")
        b.markdown(f"#### Intervalle à {r['level'] * 100:.0f} % : **{r['low']:.1f} à {r['high']:.1f} jours**")
        b.write("Cet intervalle est dérivé des erreurs observées sur le jeu de test : environ 8 séjours sur 10 "
                "se sont terminés dans une fourchette de cette largeur autour de la prédiction.")

    with st.expander("Performance et limites du modèle"):
        m, t = art["metrics"], art["target"]
        st.write(f"Réseau de neurones (MLP) · MAE {m['mae']:.2f} j · RMSE {m['rmse']:.2f} j · R² {m['r2']:.2f} "
                 f"(test, {m['n_test']:,} séjours). Cible bornée entre {t['lo']} et {t['hi']} jours.")
        st.markdown("- Le modèle n'explique qu'environ 41 % de la variance : il donne un ordre de grandeur, pas une date de sortie\n"
                    "- L'intervalle a la même largeur pour tous les patients (erreur globale de test)\n"
                    "- Aucune validation externe sur d'autres hôpitaux ; données historiques américaines à recalibrer avant tout usage réel")