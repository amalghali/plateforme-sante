"""Affectation exploratoire à un groupe, sans réentraînement ni diagnostic."""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn

ARTIFACT = Path(__file__).parent / "artifacts" / "retine_deploy.joblib"

@lru_cache(maxsize=1)
def load_artifact():
    art = joblib.load(ARTIFACT)
    if sklearn.__version__ != art["version_sklearn"]:
        raise ValueError("Ce modèle nécessite scikit-learn " + art["version_sklearn"])
    return art

def to_features(inputs, art):
    if inputs.get("quality") != 1:
        raise ValueError("Une image de qualité suffisante (quality = 1) est requise.")
    values = []
    for name in art["features"]:
        value = inputs.get(name)
        if isinstance(value, (bool, str)) or value is None:
            raise ValueError(f"{name} : une mesure numérique est requise.")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} : mesure invalide.") from None
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} : mesure finie et non négative requise.")
        if name in ("ma1", "ma5") and not value.is_integer():
            raise ValueError(f"{name} doit être un comptage entier.")
        if name == "optic_disc_diameter" and value <= 0:
            raise ValueError("Le diamètre du disque optique doit être positif.")
        values.append(value)
    raw = pd.DataFrame([values], columns=art["features"])
    prepared = art["imputer"].transform(raw)
    prepared = pd.DataFrame(prepared, columns=art["features"])
    scaled = art["scaler"].transform(prepared)
    return pd.DataFrame(scaled, columns=art["features"])

def predict(inputs, art=None):
    art = load_artifact() if art is None else art
    x = to_features(inputs, art)
    cluster = int(art["model"].predict(x)[0])
    distances = art["model"].transform(x)[0]
    return {
        "cluster": cluster,
        "description": art["profiles"][cluster]["description"],
        "effectif": art["profiles"][cluster]["effectif"],
        "distance": float(distances[cluster]),
        "message": "Affectation descriptive à un groupe ; aucune probabilité ni gravité médicale.",
    }
