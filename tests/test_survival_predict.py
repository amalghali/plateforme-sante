"""Test sans Streamlit avec un artefact synthétique : python tests/test_survival_predict.py  (ou pytest)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from modules.patient_survival import predict as P

FEATS = ["shock_index", "gcs_total", "d1_sysbp_min", "ventilated_apache", "sysbp_range", "age", "elective_surgery",
         "d1_sysbp_max", "icu_admit_source_Operating Room / Recovery", "apache_2_bodysystem_Metabolic"]
num = lambda m, s, lo, hi: dict(type="num", mean=m, scale=s, lo=lo, hi=hi)
SPEC = {"shock_index": num(1.2, .4, .5, 3), "gcs_total": num(12, 3, 3, 15), "d1_sysbp_min": num(96, 20, 50, 140),
        "ventilated_apache": dict(type="bin"), "sysbp_range": num(55, 25, 10, 150), "age": num(62, 16, 20, 89),
        "elective_surgery": dict(type="bin"), "d1_sysbp_max": num(150, 25, 100, 220),
        "icu_admit_source_Operating Room / Recovery": dict(type="onehot", column="icu_admit_source", value="Operating Room / Recovery"),
        "apache_2_bodysystem_Metabolic": dict(type="onehot", column="apache_2_bodysystem", value="Metabolic")}


def fake_artifact():
    rng = np.random.default_rng(0); X = rng.normal(size=(500, 10)); y = (X[:, 0] - X[:, 1] + rng.normal(size=500) > 1).astype(int)
    m = LogisticRegression().fit(X, y); iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(X)[:, 1], y)
    return dict(model=m, calibrator=iso, features=FEATS, spec=SPEC, thresholds=dict(f1=.7, recall80=.45, recall90=.3),
                cat_choices={}, metrics={})


BASE = dict(age=70, d1_heartrate_max=130, d1_sysbp_min=70, d1_sysbp_max=120, gcs_eyes_apache=2, gcs_motor_apache=3,
            gcs_verbal_apache=2, ventilated_apache=True, elective_surgery=False, icu_admit_source="Floor", apache_2_bodysystem="Metabolic")


def test_features_and_prediction():
    art = fake_artifact(); r = P.predict(BASE, art)
    assert 0 <= r["risk"] <= 1 and r["level"] in {"Faible", "Modéré", "Élevé", "Très élevé"}
    X = r["features"]; assert list(X.columns) == FEATS and X.shape == (1, 10)
    assert abs(X["shock_index"].iloc[0] - ((130 / 70) - 1.2) / .4) < 1e-9           # indice de choc puis z-score
    assert X["apache_2_bodysystem_Metabolic"].iloc[0] == 1 and X["icu_admit_source_Operating Room / Recovery"].iloc[0] == 0
    assert X["sysbp_range"].iloc[0] == (50 - 55) / 25


def test_clipping():
    art = fake_artifact(); X = P.to_features({**BASE, "d1_heartrate_max": 250, "d1_sysbp_min": 31}, art)
    assert abs(X["shock_index"].iloc[0] - (3 - 1.2) / .4) < 1e-9                    # plafonné au percentile 99


def test_validation():
    assert P.validate({**BASE, "age": 5}) and P.validate({**BASE, "d1_sysbp_max": 60}) and not P.validate(BASE)


if __name__ == "__main__":
    test_features_and_prediction(); test_clipping(); test_validation(); print("OK : 3 tests")
