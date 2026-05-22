"""
Tests UC-02 à UC-08 – Étapes individuelles du pipeline.

Classes :
  TestPreprocess          UC-02 et UC-03
  TestImpute              UC-04
  TestScale               UC-05
  TestIsolationForest     UC-06, UC-07 et UC-08
"""

import numpy as np
import pandas as pd
import pytest

from scripts.predict_sig import (
    impute,
    isolation_forest_scores,
    preprocess,
    scale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_pipeline_up_to_scale(df, artifacts):
    """Enchaîne impute → scale et retourne X_scaled."""
    df_imp = impute(df, artifacts)
    return scale(df_imp, artifacts)


# ---------------------------------------------------------------------------
# UC-02 / UC-03 · Prétraitement
# ---------------------------------------------------------------------------


class TestPreprocess:
    """UC-02 : Suppression des colonnes parasites et alignement sur feature_order."""

    def test_output_columns_match_feature_order(self, artifacts, typical_row_raw):
        """UC-02-A : Seules les colonnes de feature_order sont conservées."""
        df = preprocess(typical_row_raw, artifacts)
        assert list(df.columns) == artifacts["feature_order"]

    def test_extra_columns_removed(self, artifacts, typical_row_raw):
        """UC-02-B : Les colonnes brutes parasites disparaissent du résultat."""
        df = preprocess(typical_row_raw, artifacts)
        for col in ("title", "alert", "net", "latitude", "longitude", "date_time",
                    "magType", "location", "continent", "country", "sig"):
            assert col not in df.columns

    def test_output_shape(self, artifacts, typical_row_raw):
        """UC-02-C : La forme de sortie est (n_lignes, 8)."""
        df = preprocess(typical_row_raw, artifacts)
        assert df.shape == (1, len(artifacts["feature_order"]))

    def test_feature_values_preserved(self, artifacts, typical_row_raw):
        """UC-02 : Les valeurs des colonnes feature_order sont inchangées."""
        df = preprocess(typical_row_raw, artifacts)
        assert df["magnitude"].iloc[0] == pytest.approx(
            typical_row_raw["magnitude"].iloc[0]
        )

# ------------------------------------------------------------------
# UC-03 : Colonne requise manquante
# ------------------------------------------------------------------

    def test_missing_magnitude_raises_value_error(self, artifacts, typical_row):
        """UC-03-A : ValueError si magnitude est absente."""
        df_missing = typical_row.drop(columns=["magnitude"])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            preprocess(df_missing, artifacts)

    def test_missing_multiple_columns_raises_value_error(self, artifacts):
        """UC-03-B : ValueError si plusieurs colonnes requises sont absentes."""
        df_empty = pd.DataFrame([{"irrelevant": 0}])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            preprocess(df_empty, artifacts)


# ---------------------------------------------------------------------------
# UC-04 · Imputation
# ---------------------------------------------------------------------------


class TestImpute:
    """UC-04 : Imputation des valeurs manquantes par la médiane."""

    def test_no_nan_after_imputation(self, artifacts, df_with_nans):
        """UC-04-A : Plus aucun NaN après imputation."""
        df_imp = impute(df_with_nans, artifacts)
        assert df_imp.isna().sum().sum() == 0

    def test_imputed_values_are_finite(self, artifacts, df_with_nans):
        """UC-04-B : Toutes les valeurs imputées sont finies."""
        df_imp = impute(df_with_nans, artifacts)
        assert np.isfinite(df_imp.values).all()

    def test_row_count_preserved(self, artifacts, df_with_nans):
        """UC-04-C : Le nombre de lignes ne change pas."""
        df_imp = impute(df_with_nans, artifacts)
        assert len(df_imp) == len(df_with_nans)

    def test_nan_cdi_imputed_to_finite_value(self, artifacts, typical_row):
        """UC-04-D : cdi=NaN est remplacé par une valeur finie."""
        row_nan = typical_row.copy()
        row_nan["cdi"] = np.nan
        df_imp = impute(row_nan, artifacts)
        assert np.isfinite(df_imp["cdi"].iloc[0])

    def test_non_nan_values_unchanged(self, artifacts, typical_row):
        """UC-04 : Les valeurs non-NaN ne sont pas modifiées."""
        df_imp = impute(typical_row, artifacts)
        assert df_imp["magnitude"].iloc[0] == pytest.approx(
            typical_row["magnitude"].iloc[0]
        )

    def test_output_columns_preserved(self, artifacts, df_with_nans):
        """UC-04 : Les colonnes de sortie correspondent à feature_order."""
        df_imp = impute(df_with_nans, artifacts)
        assert list(df_imp.columns) == artifacts["feature_order"]


# ---------------------------------------------------------------------------
# UC-05 · Mise à l'échelle (StandardScaler)
# ---------------------------------------------------------------------------


class TestScale:
    """UC-05 : Le StandardScaler produit un ndarray de bonne forme."""

    def test_output_shape_single_row(self, artifacts, typical_row):
        """UC-05-A : Shape (1, 8) pour 1 ligne."""
        df_imp = impute(typical_row, artifacts)
        X_scaled = scale(df_imp, artifacts)
        assert X_scaled.shape == (1, len(artifacts["feature_order"]))

    def test_output_is_ndarray(self, artifacts, typical_row):
        """UC-05-B : Le résultat est un numpy ndarray."""
        df_imp = impute(typical_row, artifacts)
        X_scaled = scale(df_imp, artifacts)
        assert isinstance(X_scaled, np.ndarray)

    def test_output_shape_multiple_rows(self, artifacts, df_with_nans):
        """UC-05-C : Shape (n, 8) pour n lignes."""
        df_imp = impute(df_with_nans, artifacts)
        X_scaled = scale(df_imp, artifacts)
        assert X_scaled.shape == (len(df_with_nans), len(artifacts["feature_order"]))

    def test_output_values_finite(self, artifacts, typical_row):
        """UC-05 : Toutes les valeurs après scaling sont finies."""
        df_imp = impute(typical_row, artifacts)
        X_scaled = scale(df_imp, artifacts)
        assert np.isfinite(X_scaled).all()


# ---------------------------------------------------------------------------
# UC-06 / UC-07 / UC-08 · Isolation Forest
# ---------------------------------------------------------------------------


class TestIsolationForestScores:
    """UC-06 à UC-08 : Scores d'anomalie retournés par l'Isolation Forest."""

    def test_scores_length_matches_input(self, artifacts, typical_row):
        """UC-06-A : Le vecteur de scores a la même longueur que l'entrée."""
        X_scaled = _full_pipeline_up_to_scale(typical_row, artifacts)
        scores = isolation_forest_scores(X_scaled, artifacts)
        assert len(scores) == len(typical_row)

    def test_scores_are_finite(self, artifacts, typical_row):
        """UC-06-B : Tous les scores sont des valeurs finies."""
        X_scaled = _full_pipeline_up_to_scale(typical_row, artifacts)
        scores = isolation_forest_scores(X_scaled, artifacts)
        assert np.isfinite(scores).all()

    def test_scores_are_negative(self, artifacts, typical_row):
        """UC-06 : Les scores IsolationForest sont négatifs (convention sklearn)."""
        X_scaled = _full_pipeline_up_to_scale(typical_row, artifacts)
        scores = isolation_forest_scores(X_scaled, artifacts)
        assert (scores < 0).all()

    def test_extreme_row_has_very_low_score(self, artifacts, anomaly_row):
        """UC-07-A : Un outlier absolu obtient un score nettement inférieur à −0.5."""
        X_scaled = _full_pipeline_up_to_scale(anomaly_row, artifacts)
        scores = isolation_forest_scores(X_scaled, artifacts)
        assert scores[0] < -0.5

    def test_typical_row_score_higher_than_extreme(self, artifacts, typical_row, anomaly_row):
        """UC-08-A : La ligne typique a un score supérieur à l'outlier absolu."""
        X_typ = _full_pipeline_up_to_scale(typical_row, artifacts)
        X_ano = _full_pipeline_up_to_scale(anomaly_row, artifacts)
        score_typ = isolation_forest_scores(X_typ, artifacts)[0]
        score_ano = isolation_forest_scores(X_ano, artifacts)[0]
        assert score_typ > score_ano

    def test_scores_for_multiple_rows(self, artifacts, raw_df):
        """UC-06 : Le nombre de scores correspond bien au nombre de lignes d'entrée."""
        df_feat = raw_df[["magnitude", "cdi", "mmi", "tsunami",
                           "nst", "dmin", "gap", "depth"]].head(20)
        X_scaled = _full_pipeline_up_to_scale(df_feat, artifacts)
        scores = isolation_forest_scores(X_scaled, artifacts)
        assert len(scores) == 20
