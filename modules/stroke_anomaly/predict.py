"""Logique de prédiction du module AVC · détection d'anomalies (sans Streamlit → testable).

Modèle déployé : Elliptic Envelope (meilleur AUC de la comparaison, voir Partie C du notebook).
Entrées : 10 variables saisies par le clinicien (âge, glycémie, bmi, hypertension, maladie cardiaque, genre,
          statut marital, type de travail, lieu de résidence, tabagisme).
Sorties : score d'anomalie, position parmi les patients d'entraînement, niveau d'alerte, variables les plus atypiques.

On rejoue ici le prétraitement de la phase 3 : one-hot (drop="if_binary") puis standardisation z-score,
à partir des paramètres exportés dans stroke_anomaly_deploy.joblib.
Un score élevé = profil éloigné du profil habituel. Ce n'est PAS une probabilité d'AVC.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

ARTIFACT = Path(__file__).parent / "artifacts" / "stroke_anomaly_deploy.joblib"

LABELS = {"age": "Âge", "avg_glucose_level": "Glycémie moyenne", "bmi": "Indice de masse corporelle (bmi)",
          "hypertension": "Hypertension", "heart_disease": "Maladie cardiaque"}
CAT_LABELS = {"gender": "Genre", "ever_married": "Déjà marié(e)", "work_type": "Type de travail",
              "Residence_type": "Résidence", "smoking_status": "Tabagisme"}

# plages acceptées dans le formulaire (bornes larges, pour éviter les fautes de frappe ; cf. règles métier du notebook)
RANGES = {"age": (0, 120), "avg_glucose_level": (30, 400), "bmi": (10, 100)}
BINARY = ("hypertension", "heart_disease")
CATEGORICAL = ("gender", "ever_married", "work_type", "Residence_type", "smoking_status")


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else ARTIFACT
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}. Exécuter la cellule d'export du notebook (Partie D) "
                                f"et copier stroke_anomaly_deploy.joblib dans modules/stroke_anomaly/artifacts/.")
    return joblib.load(p)


def validate(inp: dict, art: dict | None = None) -> list[str]:
    """Retourne la liste des erreurs de saisie (vide = OK)."""
    errs = []
    for k, (lo, hi) in RANGES.items():
        v = inp.get(k)
        if v is None:
            errs.append(f"{k} manquant")
        elif not lo <= v <= hi:
            errs.append(f"{k} = {v} hors de la plage [{lo}, {hi}]")
    for k in BINARY:
        if inp.get(k) not in (0, 1, True, False):
            errs.append(f"{k} doit valoir 0 ou 1")
    if art is not None:
        for k in CATEGORICAL:
            if inp.get(k) not in art["cat_choices"][k]:
                errs.append(f"{k} = {inp.get(k)!r} inconnu (choix : {art['cat_choices'][k]})")
    return errs


def to_features(inp: dict, art: dict) -> np.ndarray:
    """Variables brutes → vecteur standardisé (1 × 17), mêmes colonnes et même ordre que X_scaled du notebook."""
    raw = []
    for col in art["features"]:
        s = art["spec"][col]
        if s["type"] in ("num", "bin"):
            raw.append(float(inp[col]))
        else:
            raw.append(1.0 if inp[s["column"]] == s["value"] else 0.0)
    return ((np.array(raw) - np.array(art["mean"])) / np.array(art["scale"])).reshape(1, -1)


def grouped_factors(contrib: np.ndarray, inp: dict, art: dict) -> list[tuple[str, float]]:
    """Regroupe les colonnes one-hot par variable d'origine et affiche la valeur réellement saisie."""
    groups: dict[str, float] = {}
    for col, c in zip(art["features"], contrib):
        s = art["spec"][col]
        key = s["column"] if s["type"] == "onehot" else col
        groups[key] = groups.get(key, 0.0) + float(c)
    out = []
    for key, c in groups.items():
        if key in CAT_LABELS:
            label = f"{CAT_LABELS[key]} : {inp[key]}"
        elif key in BINARY:
            label = f"{LABELS[key]} : {'Oui' if inp[key] in (1, True) else 'Non'}"
        else:
            label = f"{LABELS[key]} : {inp[key]:g}"
        out.append((label, c))
    return out


def contributions(z: np.ndarray, art: dict) -> np.ndarray:
    """Décomposition exacte de la distance de Mahalanobis (= score) par variable : d² = Σ δᵢ·(Pδ)ᵢ."""
    m = art["model"]
    d = z.ravel() - m.location_
    return d * (m.precision_ @ d)


def level_of(score: float, th: dict) -> tuple[str, str]:
    if score > th["top5"]:
        return "Très atypique", "Ce profil fait partie des ≈ 5 % les plus éloignés du profil habituel des patients."
    if score > th["top10"]:
        return "Atypique", "Ce profil fait partie des 10 % les plus éloignés du profil habituel."
    if score > th["top20"]:
        return "Légèrement atypique", "Ce profil fait partie des 20 % les plus éloignés du profil habituel."
    return "Habituel", "Ce profil ressemble à la majorité des patients du jeu de données."


def predict(inp: dict, art: dict | None = None, top_k: int = 3) -> dict:
    art = art or load_artifact()
    errs = validate(inp, art)
    if errs:
        raise ValueError("; ".join(errs))
    z = to_features(inp, art)
    score = float(-art["model"].score_samples(z)[0])          # même règle que le notebook : grand = atypique
    level, message = level_of(score, art["thresholds"])
    percentile = float(np.interp(score, art["ref_scores"], np.linspace(0, 100, len(art["ref_scores"]))))
    factors = sorted(grouped_factors(contributions(z, art), inp, art), key=lambda t: -t[1])
    factors = [f for f in factors[:top_k] if f[1] > 0]
    return dict(score=score, percentile=percentile, level=level, message=message,
                flagged=score > art["thresholds"]["top5"], factors=factors, features=z)
