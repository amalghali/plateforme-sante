# 🏥 Plateforme web d'aide à la décision en santé
Projet de validation ML · 4CCE12 · CRISP-DM · 6 modules (un par membre de l'équipe)

## Architecture (1 minute)
```
app.py                 accueil + menu (découvre les modules tout seul)
core/                  code commun : team.py (les 6 modules), registry.py, ui.py   ← ne pas modifier sans prévenir
modules/<id>/          UN DOSSIER PAR MEMBRE
   module.py           META + render()  (interface Streamlit)
   predict.py          logique de prédiction SANS Streamlit (testable)
   artifacts/          modèle exporté (.joblib) depuis le notebook
modules/patient_survival/   ✅ module de référence (Amal) : copier son organisation
modules/_template/          squelette vide à copier
tests/                 un test par module
```
Chaque membre ne touche **que son dossier** `modules/<son_id>/` → aucun conflit Git.

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
python tests/test_survival_predict.py      # test du module de référence
```

## Ajouter mon module (membres 2 à 6)
1. `git checkout -b module-<mon_id>` puis `cp -r modules/_template modules/<mon_id>` (ids : `diabetes_readmission`, `bone_marrow`, `dengai`, `heart_failure`, `stroke_anomaly`).
2. Dans votre notebook (fin de phase 5), **exporter un seul fichier** `.joblib` contenant : le modèle, les paramètres de prétraitement (moyennes, écarts-types, bornes de plafonnement, encodages), la liste des variables, les métriques de test. Voir la cellule d'export du notebook d'Amal comme modèle. Le copier dans `modules/<mon_id>/artifacts/`.
3. Écrire `predict.py` : `predict(inputs: dict) -> dict` qui rejoue **exactement** le prétraitement de la phase 3 (mêmes colonnes, même ordre). Vérifier la parité avec le notebook (mêmes valeurs que le jeu de test).
4. Écrire `module.py` : `META` (title, icon, dataset, problem, owner, version) + `render()` = formulaire → prédiction → **interprétation** → limites du modèle.
5. Ajouter un test dans `tests/` puis ouvrir une Pull Request. L'accueil passe automatiquement le module en « ✅ disponible ».

### Particularités par type de problème
| Module | Sortie à afficher |
|---|---|
| Classification binaire / multiclasse | probabilité par classe + niveau d'alerte (seuil choisi en phase 5) |
| Régression (moelle osseuse, DengAI) | valeur prédite + intervalle d'incertitude (ex. erreur MAE/RMSE de test) |
| Clustering (insuffisance cardiaque) | cluster attribué + profil type du cluster (moyennes) |
| Détection d'anomalies (AVC) | score d'anomalie + « profil atypique / habituel » (seuil de contamination) |

## Règles communes
- **Mêmes versions que Colab** : `scikit-learn`, `xgboost`… (la cellule d'export les affiche) → à mettre dans `requirements.txt`. Un `.joblib` ne se recharge pas toujours avec une autre version.
- **Aucune donnée patient dans le dépôt** (`.gitignore` bloque les `.csv`). Pas d'enregistrement des saisies.
- Toujours afficher `disclaimer()` : aide à la décision, **pas un diagnostic**.
- Toujours afficher les **limites** du modèle (phase 5).

## Déploiement
**Streamlit Community Cloud (gratuit, le plus simple)** : pousser le dépôt sur GitHub → share.streamlit.io → « New app » → dépôt, branche `main`, fichier `app.py` → Deploy. Chaque `git push` met le site à jour.
**Alternative Docker** : `docker build -t plateforme-sante . && docker run -p 8501:8501 plateforme-sante`.
