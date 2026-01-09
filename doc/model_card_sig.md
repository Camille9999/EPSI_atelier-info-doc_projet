# Model Card: Prédiction de l'intensité (sig)

**Date de développement :** 09/01/2026

**Modèle :** Forêt Aléatoire (RandomForestRegressor)

**Données utilisées :**
- Fichier : data/earthquake_data_clean_sig.csv
- Cible : sig (numérique)
- Features : magnitude, cdi, mmi, tsunami, nst, dmin, gap, depth

**Prétraitement des données :**
- Suppression des colonnes inutiles ('title', 'date_time', 'net', 'magType', 'latitude', 'longitude', 'location', 'continent', 'country', 'alert')
- Normalisation : StandardScaler (moyenne=0, écart-type=1) sur colonnes numériques
- Export : earthquake_data_clean_sig.csv

**Architecture :**
- RandomForestRegressor (scikit-learn)

**Recherche d'hyperparamètres (GridSearchCV) :**
- Grille testée :
  - n_estimators : [50, 100]
  - max_depth : [3, 5, None]
- Meilleurs paramètres :
  - {'max_depth': None, 'n_estimators': 50}
- Résultats du gridsearch : 0.417

**Résultats sur le test :**
- MSE : 0.608

**Export du modèle :**
- data/model_sig_rf.joblib
