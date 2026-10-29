"""Logique de prédiction du module Patient Survival (sans Streamlit → testable).

Entrées : 12 mesures saisies par le clinicien (24 premières heures de réanimation).
Sorties : probabilité de décès hospitalier (brute + calibrée), niveau d'alerte, facteurs explicatifs.
Le modèle (XGBoost) attend 10 variables préparées comme en phase 3 : on rejoue ici le même prétraitement
(plafonnement aux percentiles 1/99 du train, standardisation z-score, one-hot) à partir des paramètres exportés.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd

ARTIFACT = Path(__file__).parent / "artifacts" / "survival_deploy.joblib"

LABELS = {
    "shock_index": "Indice de choc (FC max / TA systolique min)", "gcs_total": "Score de Glasgow total",
    "d1_sysbp_min": "TA systolique minimale", "d1_sysbp_max": "TA systolique maximale",
    "sysbp_range": "Amplitude de la TA systolique", "age": "Âge", "ventilated_apache": "Ventilation mécanique",
    "elective_surgery": "Chirurgie programmée", "icu_admit_source_Operating Room / Recovery": "Admission depuis le bloc / réveil",
    "apache_2_bodysystem_Metabolic": "Atteinte métabolique",
}

# plages acceptées dans le formulaire (bornes cliniques larges, pour éviter les fautes de frappe)
RANGES = {"age": (16, 100), "d1_heartrate_max": (30, 250), "d1_sysbp_min": (30, 250), "d1_sysbp_max": (40, 300),
          "gcs_eyes_apache": (1, 4), "gcs_motor_apache": (1, 6), "gcs_verbal_apache": (1, 5)}


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else ARTIFACT
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}. Exécuter la cellule d'export du notebook (phase 6) "
                                f"et copier survival_deploy.joblib dans modules/patient_survival/artifacts/.")
    art = joblib.load(p)
    if art.get("model") is None:
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(str(p.parent / "survival_model.ubj"))
        art["model"] = m
    return art


def validate(inp: dict) -> list[str]:
    """Retourne la liste des erreurs de saisie (vide = OK)."""
    errs = []
    for k, (lo, hi) in RANGES.items():
        v = inp.get(k)
        if v is None:
            errs.append(f"{k} manquant")
        elif not lo <= v <= hi:
            errs.append(f"{k} = {v} hors de la plage [{lo}, {hi}]")
    if not errs and inp["d1_sysbp_max"] < inp["d1_sysbp_min"]:
        errs.append("La TA systolique maximale doit être ≥ à la minimale")
    return errs


def build_raw(inp: dict) -> dict:
    """Variables brutes + variables créées (feature engineering de la phase 3), avant plafonnement."""
    smin, smax = float(inp["d1_sysbp_min"]), float(inp["d1_sysbp_max"])
    return {
        "shock_index": float(inp["d1_heartrate_max"]) / smin,
        "gcs_total": float(inp["gcs_eyes_apache"] + inp["gcs_motor_apache"] + inp["gcs_verbal_apache"]),
        "d1_sysbp_min": smin, "d1_sysbp_max": smax, "sysbp_range": smax - smin,
        "age": float(inp["age"]),
        "ventilated_apache": int(bool(inp["ventilated_apache"])), "elective_surgery": int(bool(inp["elective_surgery"])),
    }


def to_features(inp: dict, art: dict) -> pd.DataFrame:
    raw, row = build_raw(inp), {}
    for f, sp in art["spec"].items():
        if sp["type"] == "num":
            v = min(max(raw[f], sp["lo"]), sp["hi"])                 # plafonnement percentiles 1/99 (train)
            row[f] = (v - sp["mean"]) / sp["scale"]                  # z-score (train)
        elif sp["type"] == "bin":
            row[f] = float(raw[f])
        else:                                                         # one-hot
            row[f] = float(inp.get(sp["column"]) == sp["value"])
    return pd.DataFrame([row], columns=art["features"])


def level_from(p_raw: float, thr: dict) -> tuple[str, str]:
    if p_raw >= thr["f1"]:
        return "Très élevé", "Le profil ressemble fortement à celui des patients décédés : surveillance rapprochée recommandée."
    if p_raw >= thr["recall80"]:
        return "Élevé", "Profil à risque : à discuter avec l'équipe soignante."
    if p_raw >= thr["recall90"]:
        return "Modéré", "Quelques signes de gravité : surveillance habituelle renforcée."
    return "Faible", "Peu de signes de gravité dans les données saisies (ne remplace pas l'évaluation clinique)."


def explain(X: pd.DataFrame, art: dict, top: int = 5) -> list[tuple[str, float]]:
    """Contributions SHAP natives d'XGBoost (échelle log-odds). Liste vide si indisponible."""
    try:
        import xgboost as xgb
        c = art["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        pairs = sorted(zip(X.columns, c), key=lambda t: -abs(t[1]))[:top]
        return [(LABELS.get(n, n), float(v)) for n, v in pairs]
    except Exception:  # noqa: BLE001
        return []


def predict(inp: dict, art: dict | None = None) -> dict:
    art = art or load_artifact()
    errs = validate(inp)
    if errs:
        raise ValueError("; ".join(errs))
    X = to_features(inp, art)
    p_raw = float(art["model"].predict_proba(X)[:, 1][0])
    p_cal = float(art["calibrator"].predict([p_raw])[0])
    level, msg = level_from(p_raw, art["thresholds"])
    return dict(risk=p_cal, score_raw=p_raw, level=level, message=msg, factors=explain(X, art), features=X)
