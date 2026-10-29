"""Les 6 modules du projet. `id` = nom du dossier dans modules/ (à respecter exactement)."""
TEAM = [
    dict(id="patient_survival",  owner="Amal Ghali",           dataset="Patient Survival WiDS 2020",       problem="Classification binaire (survie / décès)"),
    dict(id="diabetes_readmission", owner="Malek Rahmouni",    dataset="Diabetes 130-US Hospitals",        problem="Classification multiclasse (NO / >30 / <30)"),
    dict(id="bone_marrow",       owner="Wael Ben Hadj Yahia",  dataset="Bone marrow transplant: children", problem="Régression (durée de survie)"),
    dict(id="dengai",            owner="Adem Ben Ismail",      dataset="DengAI",                           problem="Régression (cas de dengue)"),
    dict(id="heart_failure",     owner="Ahmed Jribi",          dataset="Profils rétiniens — Debrecen",   problem="Clustering (non supervisé)"),
    dict(id="stroke_anomaly",    owner="Nour Islem Ben Achour", dataset="Accident Vasculaire Cérébral (AVC)", problem="Détection d'anomalies (Isolation Forest)"),
]
