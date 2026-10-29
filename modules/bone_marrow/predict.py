"""Logique de prédiction du module Diabetes 130-US Hospitals (régression : durée de séjour en jours).
Sans Streamlit -> testable. Le pipeline sauvegardé (prétraitement + MLP) est rechargé tel quel."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd

ARTIFACT = Path(__file__).parent / "artifacts" / "diabetes_regression.joblib"


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else ARTIFACT
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}. Copier diabetes_regression.joblib "
                                f"dans modules/bone_marrow/artifacts/.")
    return joblib.load(p)


def validate(inp: dict, art: dict) -> list[str]:
    """Retourne la liste des erreurs de saisie (vide = OK)."""
    errs = []
    for f in art["features"]:
        v = inp.get(f)
        if v is None:
            errs.append(f"{f} manquant")
        elif f in art["numeric"]:
            r = art["num_ranges"][f]
            if not r["min"] <= v <= r["max"]:
                errs.append(f"{f} = {v} hors de la plage [{r['min']:g}, {r['max']:g}]")
        elif f in art["nominal"]:
            if v not in art["cat_choices"][f]:
                errs.append(f"{f} = {v} : valeur inconnue")
        elif v not in (0, 1):
            errs.append(f"{f} doit valoir 0 ou 1")
    return errs


def to_frame(inp: dict, art: dict) -> pd.DataFrame:
    """Une ligne, mêmes colonnes et même ordre que dans le notebook."""
    row = {}
    for f in art["features"]:
        if f in art["numeric"]:
            row[f] = float(inp[f])
        elif f in art["nominal"]:
            row[f] = str(inp[f])
        else:
            row[f] = int(inp[f])
    return pd.DataFrame([row], columns=art["features"])


def predict(inp: dict, art: dict | None = None, check: bool = True) -> dict:
    art = art or load_artifact()
    if check:
        errs = validate(inp, art)
        if errs:
            raise ValueError("; ".join(errs))
    X = to_frame(inp, art)
    lo_t, hi_t = art["target"]["lo"], art["target"]["hi"]
    raw = float(art["model"].predict(X)[0])
    days = min(max(raw, lo_t), hi_t)                         # même plafonnement que dans le notebook
    iv = art["interval"]                                     # quantiles empiriques des résidus de test
    low = min(max(days + iv["lo"], lo_t), hi_t)
    high = min(max(days + iv["hi"], lo_t), hi_t)
    return dict(days=days, low=low, high=high, level=iv["level"], features=X)