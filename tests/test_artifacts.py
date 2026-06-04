"""
Tests UC-01 – Chargement des artefacts du modèle sig.

Cas couverts :
  01-A  Toutes les clés attendues sont présentes dans le dict retourné
  01-B  Chaque artefact a le type sklearn attendu
  01-C  feature_order contient exactement les 8 colonnes du modèle sig
  01-D  Lancer load_artifacts sur un répertoire inexistant lève une exception
"""

from pathlib import Path

import pytest
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts.predict_sig import ARTIFACTS_DIR, load_artifacts
from dotenv import load_dotenv
import os
load_dotenv()

# ---------------------------------------------------------------------------
# UC-01-A : Clés du dictionnaire
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR"))
EXPECTED_KEYS = {
    "feature_order",
    "numeric_cols",
    "categorical_cols",
    "imputer_num",
    "scaler",
    "isolation_forest",
    "model",
}


def test_load_artifacts_has_all_keys(artifacts):
    """UC-01-A : Le dict d'artefacts contient toutes les clés attendues."""
    assert EXPECTED_KEYS.issubset(set(artifacts.keys()))


# ---------------------------------------------------------------------------
# UC-01-B : Types sklearn
# ---------------------------------------------------------------------------


def test_imputer_type(artifacts):
    """UC-01-B : imputer_num est un SimpleImputer."""
    assert isinstance(artifacts["imputer_num"], SimpleImputer)


def test_scaler_type(artifacts):
    """UC-01-B : scaler est un StandardScaler."""
    assert isinstance(artifacts["scaler"], StandardScaler)


def test_isolation_forest_type(artifacts):
    """UC-01-B : isolation_forest est un IsolationForest."""
    assert isinstance(artifacts["isolation_forest"], IsolationForest)


def test_model_type(artifacts):
    """UC-01-B : model est un RandomForestRegressor."""
    assert isinstance(artifacts["model"], RandomForestRegressor) or isinstance(artifacts["model"], GradientBoostingRegressor)


# ---------------------------------------------------------------------------
# UC-01-C : feature_order
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_ORDER = [
    "magnitude", "cdi", "mmi", "tsunami",
    "nst", "dmin", "gap", "depth",
]


def test_feature_order_content(artifacts):
    """UC-01-C : feature_order contient exactement les 8 colonnes attendues."""
    assert artifacts["feature_order"] == EXPECTED_FEATURE_ORDER


def test_feature_order_length(artifacts):
    """UC-01-C : feature_order compte 8 colonnes."""
    assert len(artifacts["feature_order"]) == 8


def test_numeric_cols_subset_of_feature_order(artifacts):
    """UC-01-C : numeric_cols est un sous-ensemble de feature_order."""
    assert set(artifacts["numeric_cols"]).issubset(set(artifacts["feature_order"]))


# ---------------------------------------------------------------------------
# UC-01-D : Répertoire invalide
# ---------------------------------------------------------------------------


def test_load_artifacts_missing_dir_raises(tmp_path):
    """UC-01-D : Charger depuis un répertoire inexistant lève une exception."""
    with pytest.raises(Exception):
        load_artifacts(tmp_path / "does_not_exist")


def test_load_artifacts_empty_dir_raises(tmp_path):
    """UC-01-D : Charger depuis un répertoire vide (sans fichiers) lève une exception."""
    with pytest.raises(Exception):
        load_artifacts(tmp_path)
