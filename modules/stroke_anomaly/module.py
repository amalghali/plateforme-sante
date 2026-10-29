import pandas as pd
import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(title="AVC · profils atypiques", icon="🧠", dataset="Accident Vasculaire Cérébral (AVC)",
            problem="Détection d'anomalies (Elliptic Envelope, meilleur modèle de la comparaison)",
            owner="Nour Islem Ben Achour", version="1.0")

COLORS = {"Habituel": "🟢", "Légèrement atypique": "🟡", "Atypique": "🟠", "Très atypique": "🔴"}
YESNO = {"Non": 0, "Oui": 1}
# valeurs les plus fréquentes du jeu de données, proposées par défaut (profil banal = « Habituel »)
DEFAULTS = {"gender": "Female", "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
            "smoking_status": "never smoked"}


def _idx(ch, col):
    return ch[col].index(DEFAULTS[col]) if DEFAULTS[col] in ch[col] else 0


def render():
    module_header(META)
    disclaimer()
    try:
        art = P.load_artifact()
    except FileNotFoundError as e:
        st.error(str(e)); return

    ch = art["cat_choices"]
    st.write("Saisir le profil du patient. Le modèle indique **à quel point ce profil s'éloigne du profil habituel** "
             "des patients du jeu de données. Un profil atypique n'est **pas** un diagnostic d'AVC : il peut servir à prioriser un suivi.")
    with st.form("stroke_anomaly"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge (ans)", 0.0, 120.0, 55.0, step=1.0)
        glu = c1.number_input("Glycémie moyenne (mg/dL)", 30.0, 400.0, 105.0, step=1.0)
        bmi = c1.number_input("Indice de masse corporelle (bmi)", 10.0, 100.0, 28.0, step=0.1)
        hyp = c2.selectbox("Hypertension", list(YESNO), help="Antécédent d'hypertension artérielle")
        heart = c2.selectbox("Maladie cardiaque", list(YESNO))
        gender = c2.selectbox("Genre", ch["gender"], index=_idx(ch, "gender"))
        married = c3.selectbox("Déjà marié(e)", ch["ever_married"], index=_idx(ch, "ever_married"), help="Yes = oui, No = non")
        work = c3.selectbox("Type de travail", ch["work_type"], index=_idx(ch, "work_type"))
        res = c3.selectbox("Résidence", ch["Residence_type"], index=_idx(ch, "Residence_type"))
        smoke = c3.selectbox("Tabagisme", ch["smoking_status"], index=_idx(ch, "smoking_status"))
        go = st.form_submit_button("Analyser le profil")

    if go:
        inp = dict(age=age, avg_glucose_level=glu, bmi=bmi, hypertension=YESNO[hyp], heart_disease=YESNO[heart],
                   gender=gender, ever_married=married, work_type=work, Residence_type=res, smoking_status=smoke)
        try:
            r = P.predict(inp, art)
        except ValueError as e:
            st.error(f"Saisie invalide : {e}"); return
        a, b = st.columns([1, 2])
        a.metric("Plus atypique que", f"{r['percentile']:.0f} % des patients",
                 help=f"Score d'anomalie = {r['score']:.1f} (plus il est élevé, plus le profil est inhabituel)")
        b.markdown(f"#### {COLORS[r['level']]} Profil : **{r['level']}**")
        b.write(r["message"])
        if r["flagged"]:
            st.warning("Profil signalé comme anomalie (seuil ≈ 5 %). À interpréter avec un avis médical.")
        if r["factors"]:
            st.write("**Ce qui rend ce profil atypique** (contribution à l'éloignement ; plus la barre est grande, plus la variable pèse)")
            st.bar_chart({n: v for n, v in r["factors"]}, horizontal=True)

    with st.expander("Performance et limites du modèle"):
        m = art["metrics"]
        st.write(f"{m['model']} · AUC {m['auc']:.3f} · recall {m['recall']:.3f} · précision {m['precision']:.3f} "
                 f"(validation croisée à 5 plis, {m['n_patients']:,} patients dont {m['n_avc']} AVC).")
        st.write("**Effet du seuil** : part des patients signalés → part des AVC détectés (recall) et part des signalés ayant fait un AVC (précision).")
        t = pd.DataFrame(m["effet_seuil"]).rename(columns={"% de patients signalés": "% signalés", "Recall": "Recall", "Precision": "Précision"})
        st.dataframe(t, hide_index=True)
        st.markdown(
            "- Un AVC n'est pas une « vraie » anomalie : beaucoup de patients âgés ou hypertendus ressemblent à ceux qui font un AVC sans en faire un.\n"
            "- Le modèle est **modeste** : au seuil de 5 %, il détecte moins de 1 AVC sur 10 et la majorité des patients signalés n'ont pas fait d'AVC.\n"
            "- Le modèle est entraîné sans la cible ; la cible ne sert qu'à évaluer. Le taux d'anomalies attendu (≈ 4,9 %) vient du taux d'AVC du jeu de données.\n"
            "- Jeu de données public unique (5 109 patients), pas de validation externe : à utiliser pour **prioriser un suivi**, jamais pour poser un diagnostic.")
