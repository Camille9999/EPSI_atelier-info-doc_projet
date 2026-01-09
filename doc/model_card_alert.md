
# Model Card: Prédiction de l'alerte (alert)

**Date de développement :** 09/01/2026

**Modèle :** Régression Logistique (LogisticRegression)

**Données utilisées :**
- Fichier : data/earthquake_data_clean_alert.csv
- Cible : alert (catégorielle)

**Prétraitement des données :**
- Suppression des colonnes inutiles ('title', 'date_time', 'net', 'magType', 'latitude', 'longitude', 'location', 'continent', 'country')
- Normalisation : StandardScaler (moyenne=0, écart-type=1) sur colonnes numériques
- Lignes initiales : 782
- Lignes après suppression des valeurs manquantes sur `alert` : 415
- Export : earthquake_data_clean_alert.csv

**Architecture :**
- LogisticRegression (scikit-learn)

**Recherche d'hyperparamètres (GridSearchCV) :**
- Grille testée :
  - C : [0.1, 1, 10]
  - solver : ['liblinear', 'lbfgs']
- Meilleurs paramètres :
  - {'C': 10, 'solver': 'lbfgs'}
- Résultats du gridsearch : 0.840


**Résultats sur le test :**

| Classe      | Précision | Rappel | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| green       | 0.91      | 0.98   | 0.95     | 65      |
| orange      | 0.71      | 1.00   | 0.83     | 5       |
| red         | 1.00      | 1.00   | 1.00     | 2       |
| yellow      | 1.00      | 0.36   | 0.53     | 11      |
| **accuracy**|           |        | 0.90     | 83      |
| macro avg   | 0.91      | 0.84   | 0.83     | 83      |
| weighted avg| 0.92      | 0.90   | 0.89     | 83      |


**Export du modèle :**
- data/model_alert_logreg.joblib
