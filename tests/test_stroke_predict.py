"""Test du module AVC (sans Streamlit) : python tests/test_stroke_predict.py  (ou pytest).
Utilise l'artefact exporté par le notebook : il contient 6 patients de référence (parité avec le notebook)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from modules.stroke_anomaly import predict as P

TYPIQUE = dict(age=45, avg_glucose_level=90, bmi=27, hypertension=0, heart_disease=0, gender="Female",
               ever_married="Yes", work_type="Private", Residence_type="Urban", smoking_status="never smoked")
EXTREME = dict(age=82, avg_glucose_level=270, bmi=60, hypertension=1, heart_disease=1, gender="Male",
               ever_married="Yes", work_type="Self-employed", Residence_type="Rural", smoking_status="smokes")


def test_parity_with_notebook():
    art = P.load_artifact()
    for ref in art["parity"]:
        r = P.predict(ref["inputs"], art)
        assert abs(r["score"] - ref["score"]) < 1e-6, (r["score"], ref["score"])


def test_features_shape_and_order():
    art = P.load_artifact(); r = P.predict(TYPIQUE, art)
    assert r["features"].shape == (1, 17) and len(art["features"]) == 17
    assert r["level"] in {"Habituel", "Légèrement atypique", "Atypique", "Très atypique"} and 0 <= r["percentile"] <= 100


def test_extreme_profile_is_more_atypical():
    art = P.load_artifact(); a, b = P.predict(TYPIQUE, art), P.predict(EXTREME, art)
    assert b["score"] > a["score"] and b["flagged"] and not a["flagged"] and b["factors"]


def test_contributions_sum_to_score():
    art = P.load_artifact(); z = P.to_features(EXTREME, art)
    assert abs(P.contributions(z, art).sum() - (-art["model"].score_samples(z)[0])) < 1e-6


def test_validation():
    art = P.load_artifact()
    assert not P.validate(TYPIQUE, art)
    assert P.validate({**TYPIQUE, "age": 200}, art) and P.validate({**TYPIQUE, "gender": "Alien"}, art)
    try:
        P.predict({**TYPIQUE, "bmi": 5}, art); raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_parity_with_notebook(); test_features_shape_and_order(); test_extreme_profile_is_more_atypical()
    test_contributions_sum_to_score(); test_validation(); print("OK : 5 tests")
