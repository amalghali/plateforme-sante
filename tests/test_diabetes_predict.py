"""Tests du module Diabetes Readmission, sans Streamlit : python tests/test_diabetes_predict.py  (ou pytest).
Utilise le vrai artefact (artifacts/diabetes_deploy.joblib) et des patients fictifs (aucune donnée réelle)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.diabetes_readmission import predict as P

BASE = dict(age="[70-80)", admission_type="Emergency", admission_source="Emergency_Room", discharge="Home",
            time_in_hospital=3, num_lab_procedures=44, num_procedures=1, num_medications=15, number_diagnoses=9,
            diabetes_med=True, insulin="Steady", n_med_changes=0, number_outpatient=0, number_emergency=0, number_inpatient=0)


def test_features_and_prediction():
    art = P.load_artifact(); r = P.predict(BASE, art)
    X = r["features"]; assert list(X.columns) == art["features"] and X.shape == (1, 18)
    m, s = art["scaler"]["time_in_hospital"]; assert abs(X["time_in_hospital"].iloc[0] - (3 - m) / s) < 1e-9
    assert X["age"].iloc[0] == 7 and X["diabetesMed"].iloc[0] == 1 and X["insulin_No"].iloc[0] == 0
    assert X["discharge_disposition_id_Home"].iloc[0] == 1 and X["admission_source_id_Emergency_Room"].iloc[0] == 1
    assert X["admission_type_id_Elective"].iloc[0] == 0 and X["discharge_disposition_id_Hospice"].iloc[0] == 0
    assert abs(sum(r["scores"].values()) - 1) < 1e-9 and r["decision"] in {"NO", ">30", "<30"}


def test_total_visits_is_sum_of_capped_counters():
    art = P.load_artifact(); m, s = art["scaler_new"]["total_visits"]
    X = P.to_features({**BASE, "number_outpatient": 1, "number_emergency": 4, "number_inpatient": 2}, art)
    capped = 1 + min(4, art["caps"]["number_emergency"]) + 2                     # urgences plafonnées à 3
    assert abs(X["total_visits"].iloc[0] - (capped - m) / s) < 1e-9


def test_decision_rule():
    assert P.decide([0.30, 0.30, 0.40], 0.34) == "<30"        # score <30 au-dessus du seuil
    assert P.decide([0.50, 0.20, 0.30], 0.34) == "NO"
    assert P.decide([0.20, 0.50, 0.30], 0.34) == ">30"


def test_high_risk_profile_is_alerted():
    art = P.load_artifact()
    risky = {**BASE, "number_inpatient": 6, "number_emergency": 3, "number_outpatient": 3, "discharge": "Home_Health"}
    assert P.predict(risky, art)["scores"]["<30"] > P.predict(BASE, art)["scores"]["<30"]


def test_validation_and_record():
    art = P.load_artifact()
    assert P.validate({**BASE, "time_in_hospital": 20}, art) and P.validate({**BASE, "insulin": "beaucoup"}, art)
    assert not P.validate(BASE, art)
    rec = dict(age="[70-80)", admission_type_id=1, admission_source_id=7, discharge_disposition_id=1, time_in_hospital=3,
               num_lab_procedures=44, num_procedures=1, num_medications=15, number_diagnoses=9, diabetesMed="Yes",
               number_outpatient=0, number_emergency=0, number_inpatient=0, **{m: "No" for m in P.MEDICAMENTS})
    rec["insulin"] = "Steady"
    assert P.inputs_from_record(rec, art) == BASE                               # codes d'origine -> mêmes 15 champs


if __name__ == "__main__":
    test_features_and_prediction(); test_total_visits_is_sum_of_capped_counters(); test_decision_rule()
    test_high_risk_profile_is_alerted(); test_validation_and_record(); print("OK : 5 tests")
