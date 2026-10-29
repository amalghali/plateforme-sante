"""Logique de prédiction du module DengAI (sans Streamlit → testable).

Entrées : ville (sj/iq), semaine de l'année (1-52), 20 variables climatiques.
Sorties : nombre de cas de dengue prédit, intervalle d'incertitude (±MAE brut),
          niveau d'alerte, facteurs les plus importants.
Le modèle (AdaBoost) attend les features climatiques + encodage cyclique, normalisées
par le StandardScaler du train. La cible est normalisée (stabilisation numérique) :
la prédiction est re-transformée en nombre de cas réels avant affichage.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).parent / "artifacts" / "dengai_deploy.joblib"

# Plages acceptées pour les variables climatiques (bornes larges, pour éviter les fautes de frappe)
RANGES = {
    "weekofyear": (1, 53),
    "ndvi_ne": (-0.5, 0.6), "ndvi_nw": (-0.5, 0.6), "ndvi_se": (-0.5, 0.6), "ndvi_sw": (-0.5, 0.6),
    "precipitation_amt_mm": (0, 600), "reanalysis_air_temp_k": (290, 315),
    "reanalysis_avg_temp_k": (290, 315), "reanalysis_dew_point_temp_k": (280, 305),
    "reanalysis_max_air_temp_k": (290, 320), "reanalysis_min_air_temp_k": (280, 305),
    "reanalysis_precip_amt_kg_per_m2": (0, 600),
    "reanalysis_relative_humidity_percent": (50, 100),
    "reanalysis_sat_precip_amt_mm": (0, 600),
    "reanalysis_specific_humidity_g_per_kg": (5, 25),
    "reanalysis_tdtr_k": (0, 20),
    "station_avg_temp_c": (15, 35), "station_diur_temp_rng_c": (2, 20),
    "station_max_temp_c": (20, 45), "station_min_temp_c": (10, 30),
    "station_precip_mm": (0, 600),
}

# Variables principales affichées dans le formulaire simplifié
FORM_FEATURES = [
    "station_avg_temp_c", "station_max_temp_c", "station_min_temp_c",
    "reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
    "precipitation_amt_mm", "station_precip_mm",
    "ndvi_se", "ndvi_sw",
    "reanalysis_relative_humidity_percent",
]

LABELS = {
    "station_avg_temp_c": "Température moyenne (°C)",
    "station_max_temp_c": "Température maximale (°C)",
    "station_min_temp_c": "Température minimale (°C)",
    "reanalysis_specific_humidity_g_per_kg": "Humidité spécifique (g/kg)",
    "reanalysis_dew_point_temp_k": "Point de rosée (K)",
    "precipitation_amt_mm": "Précipitations (mm)",
    "station_precip_mm": "Précipitations station (mm)",
    "ndvi_se": "NDVI Sud-Est",
    "ndvi_sw": "NDVI Sud-Ouest",
    "ndvi_ne": "NDVI Nord-Est",
    "ndvi_nw": "NDVI Nord-Ouest",
    "reanalysis_relative_humidity_percent": "Humidité relative (%)",
    "reanalysis_air_temp_k": "Température de l'air (K)",
    "reanalysis_avg_temp_k": "Température moyenne réanalyse (K)",
    "reanalysis_max_air_temp_k": "Température max réanalyse (K)",
    "reanalysis_min_air_temp_k": "Température min réanalyse (K)",
    "reanalysis_precip_amt_kg_per_m2": "Précipitations réanalyse (kg/m²)",
    "reanalysis_sat_precip_amt_mm": "Précipitations saturées (mm)",
    "reanalysis_tdtr_k": "Amplitude thermique diurne (K)",
    "station_diur_temp_rng_c": "Amplitude thermique station (°C)",
    "week_sin": "Saisonnalité (sin)", "week_cos": "Saisonnalité (cos)",
}

ALERT_THRESHOLDS = {"sj": (20, 50, 100), "iq": (8, 15, 30)}


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else ARTIFACT
    if not p.exists():
        raise FileNotFoundError(
            f"Artefact introuvable : {p}. Exécuter le script d'export du notebook "
            f"et copier dengai_deploy.joblib dans modules/dengai/artifacts/."
        )
    return joblib.load(p)


def validate(inp: dict) -> list[str]:
    """Retourne la liste des erreurs de saisie (vide = OK)."""
    errs = []
    city = inp.get("city")
    if city not in ("sj", "iq"):
        errs.append(f"city = {city!r} doit être 'sj' ou 'iq'")
    for k, (lo, hi) in RANGES.items():
        v = inp.get(k)
        if v is None:
            continue  # les champs non fournis seront remplis par la médiane
        if not lo <= v <= hi:
            errs.append(f"{k} = {v} hors de la plage [{lo}, {hi}]")
    if inp.get("weekofyear") is None:
        errs.append("weekofyear manquant")
    return errs


def fill_defaults(inp: dict, art: dict) -> dict:
    """Remplit les variables climatiques manquantes par la médiane du dataset."""
    filled = dict(inp)
    for col in art["climate_cols"]:
        if col not in filled or filled[col] is None:
            filled[col] = art["feature_ranges"][col]["median"]
    return filled


def to_features(inp: dict, art: dict) -> pd.DataFrame:
    """Construit le vecteur de features dans l'ordre attendu par le modèle."""
    filled = fill_defaults(inp, art)

    # Encodage cyclique de la semaine
    w = float(filled["weekofyear"])
    filled["week_sin"] = float(np.sin(2 * np.pi * w / 52))
    filled["week_cos"] = float(np.cos(2 * np.pi * w / 52))

    # Extraire les features dans l'ordre attendu
    row = {f: float(filled[f]) for f in art["features"]}
    df = pd.DataFrame([row], columns=art["features"])

    # Normalisation z-score
    city = inp["city"]
    means = art["scalers"][city]["means"]
    scales = art["scalers"][city]["scales"]
    for i, col in enumerate(art["features"]):
        df[col] = (df[col] - means[i]) / scales[i]

    return df


def level_from(cases: int, city: str) -> tuple[str, str]:
    """Détermine le niveau d'alerte en fonction du nombre de cas prédit."""
    low, med, high = ALERT_THRESHOLDS.get(city, (20, 50, 100))
    if cases >= high:
        return "Très élevé", (
            "Risque épidémique majeur : mobiliser les équipes de démoustication "
            "et les centres de soins. Surveiller les hospitalisations."
        )
    if cases >= med:
        return "Élevé", (
            "Transmission active : renforcer la surveillance épidémiologique "
            "et les mesures de prévention (moustiquaires, répulsifs)."
        )
    if cases >= low:
        return "Modéré", (
            "Transmission modérée : maintenir les programmes de lutte antivectorielle habituels."
        )
    return "Faible", "Activité faible. Surveillance de routine."


def top_factors(inp: dict, art: dict, city: str, top: int = 5) -> list[tuple[str, float]]:
    """Retourne les features les plus importantes pour la ville donnée."""
    key = f"importances_{city}"
    if key not in art:
        return []
    imp = art[key]
    sorted_imp = sorted(imp.items(), key=lambda t: -abs(t[1]))[:top]
    return [(LABELS.get(f, f), float(v)) for f, v in sorted_imp]


def predict(inp: dict, art: dict | None = None) -> dict:
    """Prédit le nombre de cas de dengue pour une semaine donnée.

    inp: dict avec au minimum city, weekofyear, et idéalement les variables climatiques.
    Retourne: dict avec cases (int), level, message, interval, factors, features.
    """
    art = art or load_artifact()
    errs = validate(inp)
    if errs:
        raise ValueError("; ".join(errs))

    city = inp["city"]
    X = to_features(inp, art)
    model = art["models"][city]

    # Prédiction (espace normalisé) → reconversion en nombre de cas
    pred_norm = float(model.predict(X)[0])
    cases_raw = float(np.expm1(pred_norm))
    cases = int(max(0, round(cases_raw)))

    # Intervalle d'incertitude basé sur la MAE brute du test
    mae_raw = art["metrics"][city]["mae_raw"]
    lower = max(0, int(round(cases_raw - mae_raw)))
    upper = int(round(cases_raw + mae_raw))

    level, msg = level_from(cases, city)
    factors = top_factors(inp, art, city)

    return dict(
        cases=cases, cases_raw=cases_raw,
        lower=lower, upper=upper,
        level=level, message=msg,
        mae=art["metrics"][city]["mae"],
        rmse=art["metrics"][city]["rmse"],
        r2=art["metrics"][city]["r2"],
        factors=factors,
        features=X,
        city=city,
    )
