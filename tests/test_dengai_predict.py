"""Test sans Streamlit avec un artefact synthétique : python tests/test_dengai_predict.py  (ou pytest)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from modules.dengai import predict as P

# 19 features used by the deployed model (climate + week_sin + week_cos)
FEATS = [
    "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw",
    "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k",
    "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c",
    "station_max_temp_c", "station_min_temp_c",
]


def fake_artifact():
    """Crée un artefact synthétique avec un modèle jouet (même structure que le vrai)."""
    rng = np.random.default_rng(42)
    n_feat = len(FEATS) + 2  # +2 for week_sin, week_cos
    all_feats = FEATS + ["week_sin", "week_cos"]
    X = rng.normal(size=(200, n_feat))
    y = np.log1p(np.abs(X[:, 0] * 10 + X[:, 5] * 5 + rng.normal(size=200) * 3))

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=42),
        n_estimators=10, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    means = X.mean(axis=0).tolist()
    scales = X.std(axis=0).tolist()

    climate_cols = FEATS[:]
    feature_ranges = {c: {"min": -3.0, "max": 3.0, "mean": 0.0, "median": 0.0} for c in climate_cols}
    importances = dict(zip(all_feats, model.feature_importances_.tolist()))

    return dict(
        models={"sj": model, "iq": model},
        scalers={
            "sj": {"scaler": None, "means": means, "scales": scales},
            "iq": {"scaler": None, "means": means, "scales": scales},
        },
        features=all_feats,
        climate_cols=climate_cols,
        feature_ranges=feature_ranges,
        metrics={
            "sj": {"mae": 0.78, "rmse": 0.94, "r2": 0.025, "mae_raw": 17.9, "n_test": 234, "n_train": 702},
            "iq": {"mae": 0.86, "rmse": 1.09, "r2": -0.054, "mae_raw": 7.9, "n_test": 130, "n_train": 390},
        },
        top_features=all_feats[:5],
        importances_sj=importances,
        importances_iq=importances,
        cities=["sj", "iq"],
        version="1.0",
        model_name="AdaBoost",
        model_params={"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3},
    )


BASE_SJ = dict(
    city="sj", weekofyear=26,
    station_avg_temp_c=27.0, station_max_temp_c=33.0, station_min_temp_c=22.0,
    reanalysis_specific_humidity_g_per_kg=16.0, reanalysis_dew_point_temp_k=295.0,
    precipitation_amt_mm=30.0, station_precip_mm=25.0,
    reanalysis_relative_humidity_percent=78.0,
    ndvi_se=0.15, ndvi_sw=0.10, ndvi_ne=0.10, ndvi_nw=0.10,
    reanalysis_air_temp_k=299.0, reanalysis_avg_temp_k=299.0,
    reanalysis_max_air_temp_k=303.0, reanalysis_min_air_temp_k=295.0,
    reanalysis_precip_amt_kg_per_m2=30.0, reanalysis_sat_precip_amt_mm=30.0,
    reanalysis_tdtr_k=3.0, station_diur_temp_rng_c=8.0,
)

BASE_IQ = {**BASE_SJ, "city": "iq"}


def test_prediction_sj():
    art = fake_artifact()
    r = P.predict(BASE_SJ, art)
    assert isinstance(r["cases"], int) and r["cases"] >= 0
    assert r["level"] in {"Faible", "Modéré", "Élevé", "Très élevé"}
    assert r["lower"] <= r["cases"] <= r["upper"] or r["lower"] == 0
    assert r["city"] == "sj"
    assert r["features"].shape == (1, len(FEATS) + 2)


def test_prediction_iq():
    art = fake_artifact()
    r = P.predict(BASE_IQ, art)
    assert isinstance(r["cases"], int) and r["cases"] >= 0
    assert r["city"] == "iq"


def test_validation_bad_city():
    errs = P.validate({"city": "xx", "weekofyear": 10})
    assert any("city" in e for e in errs)


def test_validation_missing_week():
    errs = P.validate({"city": "sj"})
    assert any("weekofyear" in e for e in errs)


def test_validation_ok():
    errs = P.validate(BASE_SJ)
    assert errs == []


def test_fill_defaults():
    art = fake_artifact()
    partial = {"city": "sj", "weekofyear": 10, "station_avg_temp_c": 25.0}
    filled = P.fill_defaults(partial, art)
    # Missing climate cols should be filled with median
    assert "ndvi_ne" in filled
    # Provided value should be kept
    assert filled["station_avg_temp_c"] == 25.0


def test_factors():
    art = fake_artifact()
    r = P.predict(BASE_SJ, art)
    assert isinstance(r["factors"], list)
    if r["factors"]:
        assert len(r["factors"]) <= 5
        assert all(isinstance(f, tuple) and len(f) == 2 for f in r["factors"])


if __name__ == "__main__":
    test_prediction_sj()
    test_prediction_iq()
    test_validation_bad_city()
    test_validation_missing_week()
    test_validation_ok()
    test_fill_defaults()
    test_factors()
    print("OK : 7 tests")
