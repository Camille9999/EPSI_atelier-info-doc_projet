# Plan de test – Modèle `sig` (RandomForestRegressor)

## Contexte

Le modèle prédit `sig` (indice de significance d'un séisme) à partir de 8 colonnes numériques :
`magnitude`, `cdi`, `mmi`, `tsunami`, `nst`, `dmin`, `gap`, `depth`.

Le pipeline complet enchaîne :
1. **Prétraitement** – suppression des colonnes inutiles, alignement sur `feature_order`
2. **Imputation** – `SimpleImputer(strategy="median")` sur les colonnes numériques
3. **Mise à l'échelle** – `StandardScaler`
4. **Détection d'anomalies** – `IsolationForest` avec seuil manuel (défaut : `−0.618`)
5. **Prédiction** – `RandomForestRegressor`

---

## Cas d'usage et scénarios de test

### UC-01 · Chargement des artefacts

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 01-A | Chargement depuis le répertoire par défaut | `ARTIFACTS_DIR` valide | Dictionnaire avec clés `imputer_num`, `scaler`, `isolation_forest`, `model`, `feature_order`, `numeric_cols`, `categorical_cols` |
| 01-B | Types sklearn corrects | Même artefacts | `SimpleImputer`, `StandardScaler`, `IsolationForest`, `RandomForestRegressor`,  `GradientBoosting`|
| 01-C | `feature_order` contient les 8 colonnes | Même artefacts | `["magnitude","cdi","mmi","tsunami","nst","dmin","gap","depth"]` |
| 01-D | Répertoire inexistant | Chemin invalide | Exception levée |

---

### UC-02 · Prétraitement – suppression des colonnes parasites

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 02-A | Colonnes brutes supprimées | CSV brut complet (title, alert, latitude…) | DataFrame avec exactement les 8 colonnes `feature_order` |
| 02-B | Aucune colonne parasite conservée | Idem | `title`, `alert`, `net`, `latitude`, `longitude` absents du résultat |
| 02-C | Forme du résultat | 1 ligne avec toutes les colonnes brutes | Shape `(1, 8)` |

---

### UC-03 · Prétraitement – colonne requise manquante

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 03-A | Colonne `magnitude` absente | DataFrame sans `magnitude` | `ValueError` avec message « Colonnes manquantes » |
| 03-B | Plusieurs colonnes absentes | DataFrame vide | `ValueError` |

---

### UC-04 · Imputation des valeurs manquantes

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 04-A | Aucun NaN après imputation | DataFrame avec `cdi=NaN`, `mmi=NaN` | `df.isna().sum().sum() == 0` |
| 04-B | Valeurs imputées finies | Idem | Toutes les valeurs sont finies (`np.isfinite`) |
| 04-C | Nombre de lignes inchangé | 5 lignes avec NaN | 5 lignes en sortie |
| 04-D | Valeur de `cdi` imputée cohérente | `cdi=NaN`, autres colonnes normales | Valeur imputée ∈ intervalle réaliste (non-NaN, finie) |

---

### UC-05 · Mise à l'échelle (StandardScaler)

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 05-A | Sortie de bonne forme | 1 ligne imputée | `ndarray` de shape `(1, 8)` |
| 05-B | Type numpy ndarray | Idem | `isinstance(X, np.ndarray)` |
| 05-C | Plusieurs lignes | 5 lignes imputées | Shape `(5, 8)` |

---

### UC-06 · Scores d'anomalie (IsolationForest)

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 06-A | Longueur du vecteur de scores | 1 observation mise à l'échelle | Vecteur de longueur 1 |
| 06-B | Scores finis | Observation typique | `np.isfinite(scores).all()` |

---

### UC-07 · Détection d'anomalie – observation extrême

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 07-A | Score très faible pour outlier absolu | `magnitude=999`, `depth=99 999`, … | Score < −0.5 |

---

### UC-08 · Détection d'anomalie – observation typique vs extrême

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 08-A | Score de la ligne typique > score de l'outlier | Ligne réelle vs outlier absolu | `score_typique > score_outlier` |

---

### UC-09 · Prédiction – observation normale

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 09-A | Prédiction finie pour une ligne normale | Observation non-anomalie | `np.isfinite(prediction)` |
| 09-B | Prédiction non-NaN | Idem | `not np.isnan(prediction)` |

---

### UC-10 · Anomalie → NaN par défaut

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 10-A | Anomalie → `prediction=NaN` | Outlier absolu, `keep_anomalies=False` | `np.isnan(prediction)` |

---

### UC-11 · `keep_anomalies=True` – prédiction conservée

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 11-A | Prédiction produite même sur anomalie | Outlier absolu, `keep_anomalies=True` | `not np.isnan(prediction)` |

---

### UC-12 · Seuil d'isolation personnalisé

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 12-A | Seuil strict → plus d'anomalies | `threshold=−0.1` vs `threshold=−0.9` | `n_anomalies(−0.1) ≥ n_anomalies(−0.9)` |
| 12-B | Seuil `−∞` → zéro anomalie | `threshold=−∞` | `n_anomalies == 0` |
| 12-C | Seuil `0.0` → au moins une anomalie | `threshold=0.0` | `n_anomalies ≥ 1` |
| 12-D | `is_anomaly` cohérent avec `iso_score` | Seuil quelconque | `is_anomaly ⟺ iso_score < threshold` |

---

### UC-13 · Pipeline complet sur données réelles

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 13-A | Au moins 50 % de prédictions produites | 200 premières lignes du CSV brut | `n_predicted > 0.5 × n_total` |
| 13-B | Aucune exception | 200 premières lignes | Pipeline s'exécute sans erreur |

---

### UC-14 · Schéma du résultat

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 14-A | Colonnes exactes | Toute entrée valide | `["iso_score", "is_anomaly", "prediction"]` |
| 14-B | Nombre de lignes identique | N lignes en entrée | N lignes en sortie |
| 14-C | Index préservé | Index personnalisé | Index identique entre entrée et sortie |

---

### UC-15 · Robustesse au bruit gaussien

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 15-A | Bruit σ=0.05 → corrélation ≥ 0.90 | 200 lignes + bruit faible | Corrélation de Pearson ≥ 0.90 |
| 15-B | Bruit σ=2.0 dégrade plus que σ=0.1 | Comparaison des deux bruits | `corr(σ=0.1) > corr(σ=2.0)` |

---

### UC-16 · Robustesse à l'imputation

| # | Scénario | Valeurs en entrée | Résultat attendu |
|---|----------|-------------------|-----------------|
| 16-A | 10 % de NaN → encore ≥ 50 % de prédictions | 200 lignes avec NaN artificiels (10 %) | `n_predicted > 0.5 × n_total` |
| 16-B | Types numériques préservés après imputation | DataFrame avec NaN | Toutes les colonnes restent numériques |

---

## Organisation des fichiers de test

```
tests/
├── conftest.py              # Fixtures partagées (artefacts, lignes typiques, données réelles)
├── test_artifacts.py        # UC-01
├── test_pipeline_steps.py   # UC-02 à UC-08
├── test_predictions.py      # UC-09 à UC-14
└── test_robustness.py       # UC-15 à UC-16

scripts/
└── curves_sig.py            # Courbes : sensibilité au bruit, à l'imputation, coude IF
```
