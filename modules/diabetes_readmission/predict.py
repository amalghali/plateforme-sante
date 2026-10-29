"""Logique de prédiction du module Diabetes Readmission (sans Streamlit → testable).

Entrées : 15 champs saisis à la sortie du patient (formulaire).
Sorties : score de chaque classe (NO / >30 / <30), décision avec le seuil 0,34 pour <30, niveau d'alerte.
Le modèle (Gradient Boosting + class_weight) attend 18 variables préparées comme en phase 3 du notebook :
on rejoue ici le même prétraitement (regroupements, plafonnement au 99e percentile du train, somme des visites,
standardisation z-score, encodages) à partir des paramètres exportés dans artifacts/diabetes_deploy.joblib.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd

ARTIFACT = Path(__file__).parent / "artifacts" / "diabetes_deploy.joblib"

MEDICAMENTS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
               "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
               "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
               "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

# choix proposés dans le formulaire : valeur interne -> libellé affiché
ADMISSION_TYPE = {"Emergency": "Urgence", "Urgent": "Urgent", "Elective": "Programmée", "Other_Unknown": "Autre / inconnu"}
ADMISSION_SOURCE = {"Emergency_Room": "Service des urgences", "Referral": "Adressé par un médecin ou une clinique",
                    "Transfer": "Transfert d'un autre établissement", "Other_Unknown": "Autre / inconnu"}
DISCHARGE = {"Home": "Retour à domicile", "Home_Health": "Domicile avec soins infirmiers",
             "Post_Acute_Facility": "Établissement de soins de suite / rééducation", "Hospice": "Soins palliatifs",
             "Hospital_Transfer": "Transfert vers un autre hôpital", "Left_AMA": "Sortie contre avis médical",
             "Other_Unknown": "Autre / inconnu"}
INSULIN = {"No": "Non prescrite", "Down": "Dose diminuée", "Steady": "Dose stable", "Up": "Dose augmentée"}

# plages acceptées dans le formulaire (valeurs observées dans le dataset, pour éviter les fautes de frappe)
RANGES = {"time_in_hospital": (1, 14), "num_lab_procedures": (1, 132), "num_procedures": (0, 6),
          "num_medications": (1, 81), "number_diagnoses": (1, 16), "number_outpatient": (0, 42),
          "number_emergency": (0, 76), "number_inpatient": (0, 21), "n_med_changes": (0, 23)}

LEVELS = {
    "<30": ("Élevé", "Profil proche des patients réadmis dans les 30 jours : suivi rapproché recommandé "
                     "(appel ou rendez-vous dans les jours qui suivent la sortie)."),
    ">30": ("Modéré", "Profil proche des patients réadmis plus tard (après 30 jours) : suivi habituel renforcé."),
    "NO": ("Faible", "Pas de signe fort de réadmission dans les données saisies (ne remplace pas l'avis clinique)."),
}


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict:
    p = Path(path) if path else ARTIFACT
    if not p.exists():
        raise FileNotFoundError(f"Artefact introuvable : {p}. Copier diabetes_deploy.joblib dans "
                                f"modules/diabetes_readmission/artifacts/.")
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
    choices = {"admission_type": ADMISSION_TYPE, "admission_source": ADMISSION_SOURCE,
               "discharge": DISCHARGE, "insulin": INSULIN}
    for k, allowed in choices.items():
        if inp.get(k) not in allowed:
            errs.append(f"{k} = {inp.get(k)!r} n'est pas un choix valide")
    if art is not None and inp.get("age") not in art["age_categories"]:
        errs.append(f"age = {inp.get('age')!r} n'est pas une tranche valide")
    return errs


def to_features(inp: dict, art: dict) -> pd.DataFrame:
    """Rejoue le prétraitement de la phase 3 et renvoie les 18 variables dans l'ordre attendu par le modèle."""
    caps, sc, sn = art["caps"], art["scaler"], art["scaler_new"]
    capped = {c: min(float(inp[c]), caps[c]) for c in caps}                       # plafond = 99e percentile du train
    z = lambda c: (capped[c] - sc[c][0]) / sc[c][1]                                 # z-score (moyenne / écart-type du train)
    total_visits = capped["number_outpatient"] + capped["number_emergency"] + capped["number_inpatient"]
    row = {
        "number_diagnoses": z("number_diagnoses"), "num_medications": z("num_medications"),
        "time_in_hospital": z("time_in_hospital"), "num_lab_procedures": z("num_lab_procedures"),
        "num_procedures": z("num_procedures"),
        "total_visits": (total_visits - sn["total_visits"][0]) / sn["total_visits"][1],
        "n_med_changes": (float(inp["n_med_changes"]) - sn["n_med_changes"][0]) / sn["n_med_changes"][1],
        "age": float(art["age_categories"].index(inp["age"])),                       # Label Encoder : [0-10) = 0 … [90-100) = 9
        "diabetesMed": float(bool(inp["diabetes_med"])),                              # No = 0, Yes = 1
        "insulin_No": float(inp["insulin"] == "No"),
        "admission_type_id_Elective": float(inp["admission_type"] == "Elective"),
    }
    for g in ["Emergency_Room", "Referral", "Transfer"]:
        row[f"admission_source_id_{g}"] = float(inp["admission_source"] == g)
    for g in ["Home", "Home_Health", "Hospice", "Post_Acute_Facility"]:
        row[f"discharge_disposition_id_{g}"] = float(inp["discharge"] == g)
    return pd.DataFrame([row], columns=art["features"])


def inputs_from_record(rec: dict, art: dict) -> dict:
    """Construit les 15 champs à partir d'un séjour brut au format du dataset UCI (codes d'origine).
    Sert aux tests de parité avec le notebook, et pourrait pré-remplir le formulaire depuis le dossier patient."""
    if int(rec["discharge_disposition_id"]) in art["death_discharge_codes"]:
        raise ValueError("Patient décédé : la réadmission n'a pas de sens (séjours retirés en phase 3).")
    return dict(
        age=rec["age"],
        admission_type=art["groups_admission_type"].get(int(rec["admission_type_id"]), "Other_Unknown"),
        admission_source=art["groups_admission_source"].get(int(rec["admission_source_id"]), "Other_Unknown"),
        discharge=art["groups_discharge"].get(int(rec["discharge_disposition_id"]), "Other_Unknown"),
        time_in_hospital=int(rec["time_in_hospital"]), num_lab_procedures=int(rec["num_lab_procedures"]),
        num_procedures=int(rec["num_procedures"]), num_medications=int(rec["num_medications"]),
        number_diagnoses=int(rec["number_diagnoses"]), diabetes_med=rec["diabetesMed"] == "Yes",
        insulin=rec["insulin"], number_outpatient=int(rec["number_outpatient"]),
        number_emergency=int(rec["number_emergency"]), number_inpatient=int(rec["number_inpatient"]),
        n_med_changes=int(sum(rec[m] in ("Down", "Up") for m in MEDICAMENTS)),
    )


def decide(scores, threshold: float) -> str:
    """Règle de la phase 4 : alerte <30 si son score dépasse le seuil, sinon la classe au plus haut score entre NO et >30."""
    if scores[2] >= threshold:
        return "<30"
    return "NO" if scores[0] >= scores[1] else ">30"


def capacity_group(score_lt30: float, cutoffs: dict) -> str | None:
    """Place le séjour par rapport aux scores <30 de validation : top 10 %, 20 % ou 30 % des plus inquiétants."""
    for pct in sorted(cutoffs, key=int):
        if score_lt30 >= cutoffs[pct]:
            return f"top {pct} %"
    return None


def predict(inp: dict, art: dict | None = None) -> dict:
    art = art or load_artifact()
    errs = validate(inp, art)
    if errs:
        raise ValueError("; ".join(errs))
    X = to_features(inp, art)
    scores = [float(s) for s in art["model"].predict_proba(X)[0]]
    decision = decide(scores, art["threshold_lt30"])
    level, msg = LEVELS[decision]
    return dict(decision=decision, level=level, message=msg, scores=dict(zip(art["classes"], scores)),
                capacity=capacity_group(scores[2], art["capacity_cutoffs"]), features=X)
