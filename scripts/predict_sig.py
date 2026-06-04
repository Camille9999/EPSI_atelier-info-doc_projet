"""
Pipeline de prédiction pour le modèle sig (RandomForestRegressor).

Étapes :
  1. Imputation      – SimpleImputer(strategy="median") sur les colonnes numériques
  2. Scaling         – StandardScaler
  3. Isolation Forest – détection d'anomalies ; les observations sous le seuil sont
                        marquées comme anomalies (prediction = NaN par défaut)
  4. Prédiction      – RandomForestRegressor

Usage :
    python scripts/predict_sig.py --input data/earthquake_data.csv
    python scripts/predict_sig.py --input data/earthquake_data.csv --threshold -0.618
    python scripts/predict_sig.py --input data/earthquake_data.csv --keep-anomalies
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()


# ---------------------------------------------------------------------------
# Chemins par défaut (relatifs à la racine du projet)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR"))
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Seuil d'isolation retenu manuellement (cf. robustesse_imputation.ipynb)
DEFAULT_THRESHOLD: float = -0.61

# Colonnes supprimées lors du pré-traitement initial
COLUMNS_TO_DROP = [
    "title", "date_time", "net", "magType",
    "latitude", "longitude", "location", "continent", "country",
    "alert",           # cible du modèle alert, absente du jeu sig
]


# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------

def load_artifacts(artifacts_dir: Path) -> dict:
    """Charge imputer, scaler, isolation forest, modèle et métadonnées."""
    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    sig_meta = metadata["sig"]

    artifacts = {
        "feature_order":    sig_meta["feature_order"],
        "numeric_cols":     sig_meta["numeric_cols"],
        "categorical_cols": sig_meta["categorical_cols"],
        "imputer_num":      joblib.load(artifacts_dir / "imputer_num_sig.joblib"),
        "scaler":           joblib.load(artifacts_dir / "scaler_sig.joblib"),
        "isolation_forest": joblib.load(artifacts_dir / "isolation_forest_sig.joblib"),
        "model":            joblib.load(artifacts_dir / "model_sig.joblib"),
    }
    return artifacts


# ---------------------------------------------------------------------------
# Étapes du pipeline
# ---------------------------------------------------------------------------

def preprocess(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Supprime les colonnes inutiles et aligne sur feature_order."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_raw.columns]
    df = df_raw.drop(columns=cols_to_drop)

    feature_order = artifacts["feature_order"]
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les données d'entrée : {missing}")

    return df[feature_order]


def impute(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Applique l'imputation médiane sur les colonnes numériques."""
    numeric_cols = artifacts["numeric_cols"]
    imputer_num  = artifacts["imputer_num"]

    X_num = pd.DataFrame(
        imputer_num.transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )

    categorical_cols = artifacts["categorical_cols"]
    if categorical_cols:
        # L'imputation catégorielle n'est pas utilisée pour sig (liste vide)
        return pd.concat([X_num, df[categorical_cols]], axis=1)[artifacts["feature_order"]]

    return X_num[artifacts["feature_order"]]


def scale(df_imp: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Applique le StandardScaler."""
    return artifacts["scaler"].transform(df_imp)


def isolation_forest_scores(X_scaled: np.ndarray, artifacts: dict) -> np.ndarray:
    """Retourne les scores d'anomalie (score_samples) de l'Isolation Forest."""
    return artifacts["isolation_forest"].score_samples(X_scaled)


def predict(
    df_raw: pd.DataFrame,
    artifacts: dict,
    threshold: float = DEFAULT_THRESHOLD,
    keep_anomalies: bool = False,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet et retourne un DataFrame avec :
      - iso_score     : score d'isolation (plus faible = plus anormal)
      - is_anomaly    : True si iso_score < threshold
      - prediction    : valeur prédite de `sig` (NaN pour les anomalies si keep_anomalies=False)

    Parameters
    ----------
    df_raw : DataFrame brut (peut contenir toutes les colonnes du CSV d'origine)
    artifacts : dict chargé par load_artifacts()
    threshold : seuil d'isolation (défaut -0.618)
    keep_anomalies : si True, prédit quand même les anomalies
    """
    # 1. Prétraitement
    df_feat = preprocess(df_raw, artifacts)

    # 2. Imputation
    df_imp = impute(df_feat, artifacts)

    # 3. Scaling
    X_scaled = scale(df_imp, artifacts)

    # 4. Isolation Forest
    scores = isolation_forest_scores(X_scaled, artifacts)
    is_anomaly = scores < threshold

    # 5. Prédiction
    predictions = np.full(len(df_raw), np.nan)
    mask = ~is_anomaly if not keep_anomalies else np.ones(len(df_raw), dtype=bool)

    if mask.any():
        predictions[mask] = artifacts["model"].predict(X_scaled[mask])

    results = pd.DataFrame(
        {
            "iso_score":  scores,
            "is_anomaly": is_anomaly,
            "prediction": predictions,
        },
        index=df_raw.index,
    )
    return results


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de prédiction sig : imputation → scaling → isolation forest → prédiction"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Chemin vers le CSV d'entrée (peut être le CSV brut ou pré-traité)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Chemin de sortie CSV (optionnel, affiche dans stdout si absent)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"Seuil d'isolation Forest (défaut : {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--keep-anomalies", action="store_true",
        help="Prédit aussi les observations détectées comme anomalies"
    )
    parser.add_argument(
        "--artifacts-dir", default=str(ARTIFACTS_DIR),
        help=f"Répertoire des artefacts (défaut : {ARTIFACTS_DIR})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        print(f"[ERREUR] Répertoire d'artefacts introuvable : {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERREUR] Fichier d'entrée introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Chargement des artefacts depuis : {artifacts_dir}")
    artifacts = load_artifacts(artifacts_dir)

    print(f"Chargement des données depuis : {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"  → {len(df_raw)} lignes, {df_raw.shape[1]} colonnes")

    print(f"Exécution du pipeline (seuil isolation = {args.threshold}) …")
    results = predict(
        df_raw,
        artifacts,
        threshold=args.threshold,
        keep_anomalies=args.keep_anomalies,
    )

    n_anomalies = results["is_anomaly"].sum()
    n_predicted = results["prediction"].notna().sum()
    print(f"  → Anomalies détectées : {n_anomalies} / {len(results)}")
    print(f"  → Prédictions produites : {n_predicted} / {len(results)}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Résultats exportés vers : {output_path}")
    else:
        print("\n--- Résultats (10 premières lignes) ---")
        print(results.head(10).to_string())


if __name__ == "__main__":
    main()
