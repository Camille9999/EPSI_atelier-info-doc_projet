"""
Script d'entraînement du modèle sig.

Reproduit exactement le pipeline du notebook robustesse_imputation.ipynb :
  1. Chargement et nettoyage des données brutes
  2. Split 80/20 (test_size=0.2, random_state=42)
  3. Imputation médiane (SimpleImputer) + mise à l'échelle (StandardScaler) sur X
  4. Entraînement IsolationForest (contamination="auto", random_state=42)
  5. Entraînement du modèle choisi
     – cible : sig BRUT (non standardisé)
  6. Sauvegarde des artefacts dans models/imputation/

La cible d'entraînement est le sig BRUT, ce qui garantit que predict_sig.py
et curves_sig.py travaillent dans la même échelle que ce notebook.

Usage :
    python scripts/train_sig.py
    python scripts/train_sig.py --input data/earthquake_data.csv
    python scripts/train_sig.py --threshold -0.618
    python scripts/train_sig.py --model gradient_boosting --artifacts-dir models/imputation_gradient_boosting
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH     = PROJECT_ROOT / "data" / "earthquake_data.csv"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR"))
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Colonnes supprimées avant entraînement (identique à predict_sig.py)
COLUMNS_TO_DROP = [
    "title", "date_time", "net", "magType",
    "latitude", "longitude", "location", "continent", "country",
    "alert",
]

DEFAULT_THRESHOLD: float = -0.61
MODEL_CHOICES = ("random_forest", "gradient_boosting")


# ---------------------------------------------------------------------------
# Chargement et nettoyage
# ---------------------------------------------------------------------------


def load_and_clean(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le CSV brut et retourne (X_sig, y_sig) prêts pour le split."""
    df = pd.read_csv(data_path)
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df_sig = df.drop(columns=cols_to_drop)
    X = df_sig.drop(columns=["sig"])
    y = df_sig["sig"]
    return X, y


# ---------------------------------------------------------------------------
# Prétraitement (identique à build_preprocessor du notebook)
# ---------------------------------------------------------------------------


def build_preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """
    Impute puis met à l'échelle X_train et X_test.
    Identique à build_preprocessor() dans robustesse_imputation.ipynb.

    Returns
    -------
    X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, artifacts
    """
    numeric_cols    = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    imputer_num = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(
        imputer_num.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols,
        index=X_train.index,
    )
    X_test_num = pd.DataFrame(
        imputer_num.transform(X_test[numeric_cols]),
        columns=numeric_cols,
        index=X_test.index,
    )

    # Pas de colonnes catégorielles pour sig ; feature_order = ordre d'origine
    feature_order = X_train.columns.tolist()
    X_train_imp = X_train_num[feature_order]
    X_test_imp  = X_test_num[feature_order]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    artifacts = {
        "feature_order":   feature_order,
        "numeric_cols":    numeric_cols,
        "categorical_cols": categorical_cols,
        "imputer_num":     imputer_num,
        "scaler":          scaler,
    }
    return X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, artifacts


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------


def train_isolation_forest(X_train_scaled: np.ndarray) -> IsolationForest:
    """Entraîne l'Isolation Forest sur le jeu d'entraînement mis à l'échelle."""
    iso = IsolationForest(random_state=42, contamination="auto")
    iso.fit(X_train_scaled)
    return iso


def train_random_forest(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
) -> tuple[RandomForestRegressor, dict]:
    """
    Entraîne un RandomForestRegressor avec GridSearchCV.
    La cible y_train est le sig BRUT (valeurs de l'ordre de 0–3 000).
    """
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth":    [3, 5, None],
    }
    rf = RandomForestRegressor(random_state=42)
    gs = GridSearchCV(
        rf, param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_train_scaled, y_train)
    return gs.best_estimator_, gs.best_params_


def train_gradient_boosting(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
) -> tuple[GradientBoostingRegressor, dict]:
    """
    Entraîne un deuxième modèle sans tuning, comme demandé dans la consigne.
    Le prétraitement, l'IsolationForest et le format des artefacts restent identiques.
    """
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, {"random_state": 42}


def train_regressor(
    model_name: str,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
):
    """Entraîne le régresseur demandé et retourne (model, infos)."""
    if model_name == "random_forest":
        return train_random_forest(X_train_scaled, y_train)
    if model_name == "gradient_boosting":
        return train_gradient_boosting(X_train_scaled, y_train)
    raise ValueError(f"Modèle non supporté : {model_name}")


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------


def save_artifacts(
    preproc: dict,
    iso: IsolationForest,
    model,
    threshold: float,
    artifacts_dir: Path,
    metadata_path: Path,
    model_name: str,
) -> None:
    """
    Sauvegarde imputer, scaler, isolation forest et modèle RF.
    Met à jour uniquement la clé "sig" dans metadata.json
    (la clé "alert" est préservée si elle existe déjà).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preproc["imputer_num"], artifacts_dir / "imputer_num_sig.joblib")
    joblib.dump(preproc["scaler"],      artifacts_dir / "scaler_sig.joblib")
    joblib.dump(iso,                    artifacts_dir / "isolation_forest_sig.joblib")
    joblib.dump(model,                  artifacts_dir / "model_sig.joblib")

    # Lecture du metadata existant (pour ne pas écraser la clé "alert")
    metadata: dict = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    metadata["sig"] = {
        "feature_order":    preproc["feature_order"],
        "numeric_cols":     preproc["numeric_cols"],
        "categorical_cols": preproc["categorical_cols"],
        "threshold_manual": threshold,
        "model_name":       model_name,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  imputer_num_sig.joblib    -> {artifacts_dir}")
    print(f"  scaler_sig.joblib         -> {artifacts_dir}")
    print(f"  isolation_forest_sig.joblib -> {artifacts_dir}")
    print(f"  model_sig.joblib          -> {artifacts_dir}")
    print(f"  metadata.json             -> cle 'sig' mise a jour (seuil={threshold})")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne et sauvegarde un modèle sig."
    )
    parser.add_argument(
        "--input", default=str(DATA_PATH),
        help=f"CSV brut d'entrée (défaut : {DATA_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Seuil IF à enregistrer dans metadata.json (défaut : {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--artifacts-dir", default="/".join(str(ARTIFACTS_DIR).split("/")[:-1]),
        help=f"Répertoire de sauvegarde (défaut : {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="random_forest",
        help="Modèle à entraîner : random_forest ou gradient_boosting (défaut : random_forest).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    artifacts_dir = Path(args.artifacts_dir)/args.model
    metadata_path = artifacts_dir / "metadata.json"
    input_path    = Path(args.input)

    if not input_path.is_file():
        print(f"[ERREUR] Fichier introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/5] Chargement des donnees : {input_path}")
    X, y = load_and_clean(input_path)
    print(f"  {len(X)} observations, {X.shape[1]} features")
    print(f"  sig brut : min={y.min():.0f}  mean={y.mean():.0f}  max={y.max():.0f}")

    print("[2/5] Split 80/20 (random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train={len(X_train)}, Test={len(X_test)}")

    print("[3/5] Imputation + mise a l'echelle")
    _, _, X_train_scaled, X_test_scaled, preproc = build_preprocessor(X_train, X_test)
    print(f"  features : {preproc['feature_order']}")

    print("[4/5] Entrainement des modeles")
    iso = train_isolation_forest(X_train_scaled)
    scores_test = iso.score_samples(X_test_scaled)
    print(f"  IF scores (test) : min={scores_test.min():.4f}  max={scores_test.max():.4f}")

    model, model_info = train_regressor(args.model, X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"  Modele : {args.model}")
    print(f"  Infos modele : {model_info}")
    print(f"  RMSE (test, sig brut) : {rmse:.2f}")

    print(f"[5/5] Sauvegarde dans {artifacts_dir}")
    save_artifacts(
        preproc,
        iso,
        model,
        args.threshold,
        artifacts_dir,
        metadata_path,
        args.model,
    )

    print("\nEntrainement termine.")


if __name__ == "__main__":
    main()
