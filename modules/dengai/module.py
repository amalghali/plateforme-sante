"""Module DengAI — Prédiction du nombre de cas hebdomadaires de dengue.

Interface Streamlit : formulaire de saisie des conditions climatiques → prédiction → interprétation.
Suit la template du projet (META + render).
"""
import streamlit as st
from core.ui import module_header, disclaimer
from . import predict as P

META = dict(
    title="Prédiction de la dengue",
    icon="🦟",
    dataset="DengAI",
    problem="Régression (cas hebdomadaires de dengue)",
    owner="Adem Ben Ismail",
    version="1.0",
)

COLORS = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴"}
CITY_NAMES = {"sj": "San Juan (Porto Rico)", "iq": "Iquitos (Pérou)"}


def render():
    module_header(META)
    disclaimer()
    try:
        art = P.load_artifact()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.write(
        "Saisir les **conditions climatiques** d'une semaine pour estimer le nombre de cas de dengue. "
        "Le modèle utilise les données météorologiques et de végétation pour anticiper la transmission."
    )

    with st.form("dengai"):
        c1, c2, c3 = st.columns(3)

        c1.markdown("**📍 Localisation et période**")
        city = c1.selectbox("Ville", list(CITY_NAMES), format_func=CITY_NAMES.get)
        week = c1.number_input("Semaine de l'année", 1, 53, 26)

        c1.markdown("**🌡️ Température**")
        avg_temp = c1.number_input("Température moyenne (°C)", 15.0, 35.0, 27.0, step=0.5)
        max_temp = c1.number_input("Température maximale (°C)", 20.0, 45.0, 33.0, step=0.5)
        min_temp = c1.number_input("Température minimale (°C)", 10.0, 30.0, 22.0, step=0.5)

        c2.markdown("**💧 Humidité et précipitations**")
        humidity = c2.number_input("Humidité spécifique (g/kg)", 5.0, 25.0, 16.0, step=0.5)
        dew = c2.number_input("Point de rosée (K)", 280.0, 305.0, 295.0, step=0.5)
        precip = c2.number_input("Précipitations (mm)", 0.0, 400.0, 30.0, step=5.0)
        precip_st = c2.number_input("Précipitations station (mm)", 0.0, 400.0, 25.0, step=5.0)
        rel_hum = c2.number_input("Humidité relative (%)", 50.0, 100.0, 78.0, step=1.0)

        c3.markdown("**🌿 Végétation (NDVI)**")
        ndvi_se = c3.number_input("NDVI Sud-Est", -0.5, 0.6, 0.15, step=0.05)
        ndvi_sw = c3.number_input("NDVI Sud-Ouest", -0.5, 0.6, 0.10, step=0.05)

        with st.expander("Variables supplémentaires (optionnel — valeurs par défaut si non renseigné)"):
            s1, s2, s3 = st.columns(3)
            ndvi_ne = s1.number_input("NDVI Nord-Est", -0.5, 0.6, 0.10, step=0.05)
            ndvi_nw = s1.number_input("NDVI Nord-Ouest", -0.5, 0.6, 0.10, step=0.05)
            reanalysis_air = s2.number_input("Température air réanalyse (K)", 290.0, 315.0, 299.0, step=0.5)
            reanalysis_avg = s2.number_input("Température moy. réanalyse (K)", 290.0, 315.0, 299.0, step=0.5)
            reanalysis_max = s2.number_input("Température max réanalyse (K)", 290.0, 320.0, 303.0, step=0.5)
            reanalysis_min = s3.number_input("Température min réanalyse (K)", 280.0, 305.0, 295.0, step=0.5)
            reanalysis_precip = s3.number_input("Précipitations réanalyse (kg/m²)", 0.0, 600.0, 30.0, step=5.0)
            reanalysis_sat = s3.number_input("Précipitations saturées (mm)", 0.0, 600.0, 30.0, step=5.0)
            reanalysis_tdtr = s1.number_input("Amplitude thermique diurne (K)", 0.0, 20.0, 3.0, step=0.5)
            diur_rng = s2.number_input("Amplitude thermique station (°C)", 2.0, 20.0, 8.0, step=0.5)

        go = st.form_submit_button("🔍 Estimer le nombre de cas")

    if go:
        inp = dict(
            city=city, weekofyear=week,
            station_avg_temp_c=avg_temp, station_max_temp_c=max_temp, station_min_temp_c=min_temp,
            reanalysis_specific_humidity_g_per_kg=humidity, reanalysis_dew_point_temp_k=dew,
            precipitation_amt_mm=precip, station_precip_mm=precip_st,
            reanalysis_relative_humidity_percent=rel_hum,
            ndvi_se=ndvi_se, ndvi_sw=ndvi_sw, ndvi_ne=ndvi_ne, ndvi_nw=ndvi_nw,
            reanalysis_air_temp_k=reanalysis_air, reanalysis_avg_temp_k=reanalysis_avg,
            reanalysis_max_air_temp_k=reanalysis_max, reanalysis_min_air_temp_k=reanalysis_min,
            reanalysis_precip_amt_kg_per_m2=reanalysis_precip,
            reanalysis_sat_precip_amt_mm=reanalysis_sat,
            reanalysis_tdtr_k=reanalysis_tdtr, station_diur_temp_rng_c=diur_rng,
        )
        try:
            r = P.predict(inp, art)
        except ValueError as e:
            st.error(f"Saisie invalide : {e}")
            return

        # -- Résultat --
        a, b = st.columns([1, 2])
        a.metric(
            "Cas de dengue estimés",
            f"{r['cases']}",
            help=f"Intervalle : [{r['lower']} – {r['upper']}] cas (±MAE du test)",
        )
        b.markdown(f"#### {COLORS[r['level']]} Niveau d'alerte : **{r['level']}**")
        b.write(r["message"])

        st.write(f"**Intervalle de confiance** : entre **{r['lower']}** et **{r['upper']}** cas "
                 f"(basé sur l'erreur moyenne du modèle en test)")
        st.caption(f"Ville : {CITY_NAMES[r['city']]} · Semaine {week}")

        # -- Facteurs importants --
        if r["factors"]:
            st.write("**Variables les plus influentes** (importance du modèle)")
            st.bar_chart({n: v for n, v in r["factors"]}, horizontal=True, color="#1F7A5A")

    # -- Performance et limites --
    with st.expander("Performance et limites du modèle"):
        m = art.get("metrics", {})
        for cid, cname in CITY_NAMES.items():
            cm = m.get(cid, {})
            st.write(
                f"**{cname}** : MAE = {cm.get('mae', '?')} · RMSE = {cm.get('rmse', '?')} · "
                f"R² = {cm.get('r2', '?')} (test, {cm.get('n_test', '?')} semaines)"
            )
        st.markdown(
            f"- Modèle : **{art.get('model_name', 'AdaBoost')}** "
            f"(n_estimators={art['model_params']['n_estimators']}, "
            f"learning_rate={art['model_params']['learning_rate']}, "
            f"max_depth={art['model_params']['max_depth']})\n"
            "- Entraînement **séparé par ville** (San Juan et Iquitos)\n"
            "- Validation croisée temporelle (TimeSeriesSplit, 5 folds)\n"
            "- L'intervalle de confiance est basé sur la MAE brute du test (erreur moyenne)\n"
            "- Ce modèle simplifié n'utilise pas les données des semaines précédentes (lags) ; "
            "les performances en production pourraient être améliorées avec un historique climatique\n"
            "- Données de 1990-2010 : à revalider avant tout usage opérationnel"
        )
