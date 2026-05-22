"""
Fixtures partagées pour les tests du modèle sig.

Hiérarchie des fixtures :
  - artifacts (scope=session)  : artefacts sklearn chargés une seule fois
  - raw_df    (scope=session)  : 200 premières lignes du CSV brut
  - typical_row                : 1 ligne avec valeurs réalistes (format feature uniquement)
  - typical_row_raw            : idem + colonnes brutes supplémentaires (title, alert, …)
  - anomaly_row                : 1 ligne avec valeurs extrêmes (outlier absolu)
  - df_with_nans               : 5 lignes avec NaN artificiels sur cdi / mmi
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.predict_sig import ARTIFACTS_DIR, load_artifacts

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "earthquake_data.csv"

# Colonnes caractéristiques utilisées par le modèle sig
FEATURE_COLS = ["magnitude", "cdi", "mmi", "tsunami", "nst", "dmin", "gap", "depth"]

# Valeurs d'une observation réelle tirée du CSV brut (ligne 1)
_TYPICAL_VALUES = {
    "magnitude": 7.0,
    "cdi": 8.0,
    "mmi": 7.0,
    "tsunami": 1.0,
    "nst": 117.0,
    "dmin": 0.509,
    "gap": 17.0,
    "depth": 14.0,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def artifacts():
    """Charge les artefacts du modèle sig une seule fois pour toute la session."""
    return load_artifacts(ARTIFACTS_DIR)


@pytest.fixture(scope="session")
def raw_df():
    """200 premières lignes du CSV brut (tel que livré par l'acquisition)."""
    return pd.read_csv(RAW_CSV_PATH).head(200)


@pytest.fixture
def typical_row():
    """
    DataFrame à 1 ligne représentant un séisme typique.
    Contient uniquement les colonnes du feature_order (sans colonnes brutes parasites).
    """
    return pd.DataFrame([_TYPICAL_VALUES])


@pytest.fixture
def typical_row_raw():
    """
    DataFrame à 1 ligne représentant un séisme typique avec TOUTES les colonnes brutes,
    tel qu'il apparaîtrait dans le CSV d'origine (avant prétraitement).
    """
    row = dict(_TYPICAL_VALUES)
    row.update(
        {
            "title": "M 7.0 - Test séisme",
            "date_time": "01-01-2020 00:00",
            "net": "us",
            "magType": "mww",
            "latitude": -9.7963,
            "longitude": 159.596,
            "location": "Test location",
            "continent": "Oceania",
            "country": "Solomon Islands",
            "alert": "green",
            "sig": 768,  # cible ; présente dans le CSV brut mais pas dans feature_order
        }
    )
    return pd.DataFrame([row])


@pytest.fixture
def anomaly_row():
    """
    DataFrame à 1 ligne avec des valeurs absurdes (outlier absolu).
    Doit systématiquement être détecté comme anomalie par l'Isolation Forest.
    """
    return pd.DataFrame(
        [
            {
                "magnitude": 999.0,
                "cdi": 999.0,
                "mmi": 999.0,
                "tsunami": 999.0,
                "nst": 99_999.0,
                "dmin": 99_999.0,
                "gap": 99_999.0,
                "depth": 99_999.0,
            }
        ]
    )


@pytest.fixture
def df_with_nans():
    """
    DataFrame à 5 lignes dérivées de typical_row avec des NaN introduits
    sur cdi (systématiquement) et mmi (lignes paires).
    """
    rows = []
    for i in range(5):
        row = dict(_TYPICAL_VALUES)
        row["cdi"] = np.nan
        if i % 2 == 0:
            row["mmi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
