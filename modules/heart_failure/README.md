# Profils rétiniens — Ahmed Jribi

Le dossier conserve l’identifiant heart_failure réservé à Ahmed ; le dataset est
Diabetic Retinopathy Debrecen. L’intitulé dans core/team.py est corrigé.

Depuis la racine du dépôt, sous PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r modules/heart_failure/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_retine_predict.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Le modèle K-means k=3, le prétraitement et les profils agrégés sont dans
artifacts/retine_deploy.joblib. Aucune observation individuelle n’est incluse.
Les valeurs par défaut sont des médianes agrégées, pas un patient réel.
L’interface attend les 12 descripteurs calculés selon le protocole Debrecen,
jamais une photographie. Quality=1 est requis. Aucun fit lors de l’utilisation.

L’artefact a été réenregistré sous Python 3.12 / scikit-learn 1.6.1 après vérification :
les 1 147 affectations correspondent exactement au notebook (ARI=1).
Silhouette=0,3013 ; stabilité ARI moyen=0,9477, minimum=0,8709.
Effectifs=290/844/13. Séparation partielle, sensibilité au prétraitement,
absence de validation clinique et de test de généralisation : pas de diagnostic.

## Déploiement partagé

Le Dockerfile commun utilise Python 3.11. Harmoniser avec l’équipe vers Python
3.12 pour reproduire l’environnement testé. Le requirements.txt local fixe les
versions numériques testées ; vérifier leur compatibilité avec les autres modules
avant de les reporter dans les dépendances communes. Le Dockerfile et les
requirements communs ne sont pas modifiés par cette intégration.

Après fusion de la branche, utiliser le déploiement Streamlit prévu dans le
README commun, avec app.py. Cette préparation ne publie pas l’application.
