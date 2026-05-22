"""
Tests UC-09 à UC-14 – Comportement des prédictions (pipeline complet).

Classes :
  TestOutputSchema       UC-14 : structure du DataFrame de résultats
  TestNormalPrediction   UC-09 : prédiction valide pour une observation normale
  TestAnomalyBehavior    UC-10 et UC-11 : NaN par défaut, keep_anomalies
  TestThreshold          UC-12 : seuil personnalisé
  TestEndToEnd           UC-13 : pipeline complet sur données réelles
"""

import numpy as np
import pandas as pd
import pytest

from scripts.predict_sig import DEFAULT_THRESHOLD, predict


# ---------------------------------------------------------------------------
# UC-14 · Schéma du résultat
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """UC-14 : Le DataFrame retourné par predict() a la structure attendue."""

    def test_output_columns_exact(self, artifacts, typical_row):
        """UC-14-A : Les colonnes sont exactement ['iso_score','is_anomaly','prediction']."""
        results = predict(typical_row, artifacts)
        assert list(results.columns) == ["iso_score", "is_anomaly", "prediction"]

    def test_output_row_count_matches_input(self, artifacts, raw_df):
        """UC-14-B : Le nombre de lignes en sortie égale celui de l'entrée."""
        subset = raw_df.head(10)
        results = predict(subset, artifacts)
        assert len(results) == len(subset)

    def test_index_preserved(self, artifacts, typical_row):
        """UC-14-C : L'index du DataFrame d'entrée est conservé dans le résultat."""
        df_indexed = typical_row.copy()
        df_indexed.index = [42]
        results = predict(df_indexed, artifacts)
        assert list(results.index) == [42]

    def test_is_anomaly_is_boolean(self, artifacts, typical_row):
        """UC-14 : La colonne is_anomaly est de type bool."""
        results = predict(typical_row, artifacts)
        assert results["is_anomaly"].dtype == bool

    def test_iso_score_is_float(self, artifacts, typical_row):
        """UC-14 : La colonne iso_score contient des flottants."""
        results = predict(typical_row, artifacts)
        assert pd.api.types.is_float_dtype(results["iso_score"])


# ---------------------------------------------------------------------------
# UC-09 · Prédiction normale
# ---------------------------------------------------------------------------


class TestNormalPrediction:
    """UC-09 : Une observation non anomalie produit une prédiction finie."""

    def test_prediction_not_nan_for_non_anomaly(self, artifacts, typical_row):
        """UC-09-B : Prédiction non-NaN si la ligne n'est pas une anomalie."""
        results = predict(typical_row, artifacts)
        if not results["is_anomaly"].iloc[0]:
            assert not np.isnan(results["prediction"].iloc[0])

    def test_prediction_is_finite_for_non_anomaly(self, artifacts, typical_row):
        """UC-09-A : Prédiction finie si la ligne n'est pas une anomalie."""
        results = predict(typical_row, artifacts)
        if not results["is_anomaly"].iloc[0]:
            assert np.isfinite(results["prediction"].iloc[0])

    def test_iso_score_is_finite(self, artifacts, typical_row):
        """UC-09 : iso_score est toujours une valeur finie (même pour une anomalie)."""
        results = predict(typical_row, artifacts)
        assert np.isfinite(results["iso_score"].iloc[0])


# ---------------------------------------------------------------------------
# UC-10 / UC-11 · Comportement des anomalies
# ---------------------------------------------------------------------------


class TestAnomalyBehavior:
    """UC-10 et UC-11 : Gestion des observations anomalies."""

    def test_anomaly_prediction_is_nan_by_default(self, artifacts, anomaly_row):
        """UC-10-A : Un outlier absolu obtient prediction=NaN par défaut."""
        results = predict(anomaly_row, artifacts, threshold=DEFAULT_THRESHOLD)
        if results["is_anomaly"].iloc[0]:
            assert np.isnan(results["prediction"].iloc[0])

    def test_anomaly_detected_at_low_threshold(self, artifacts, anomaly_row):
        """UC-10 : L'outlier absolu est toujours une anomalie avec threshold=0."""
        results = predict(anomaly_row, artifacts, threshold=0.0)
        assert results["is_anomaly"].iloc[0]

    def test_keep_anomalies_produces_non_nan_prediction(self, artifacts, anomaly_row):
        """UC-11-A : keep_anomalies=True → prediction non-NaN même pour un outlier."""
        results = predict(anomaly_row, artifacts, keep_anomalies=True)
        assert not np.isnan(results["prediction"].iloc[0])

    def test_keep_anomalies_prediction_is_finite(self, artifacts, anomaly_row):
        """UC-11 : keep_anomalies=True → prediction finie."""
        results = predict(anomaly_row, artifacts, keep_anomalies=True)
        assert np.isfinite(results["prediction"].iloc[0])

    def test_is_anomaly_not_affected_by_keep_anomalies(self, artifacts, anomaly_row):
        """UC-11 : Le flag is_anomaly reste identique que keep_anomalies soit True ou False."""
        r_default = predict(anomaly_row, artifacts, keep_anomalies=False)
        r_kept = predict(anomaly_row, artifacts, keep_anomalies=True)
        assert r_default["is_anomaly"].iloc[0] == r_kept["is_anomaly"].iloc[0]


# ---------------------------------------------------------------------------
# UC-12 · Seuil d'isolation personnalisé
# ---------------------------------------------------------------------------


class TestThreshold:
    """UC-12 : Le seuil d'isolation forest influence correctement les anomalies."""

    def test_strict_threshold_produces_more_anomalies(self, artifacts, raw_df):
        """UC-12-A : threshold=−0.1 (strict) ≥ anomalies que threshold=−0.9 (permissif)."""
        r_strict = predict(raw_df, artifacts, threshold=-0.1)
        r_loose = predict(raw_df, artifacts, threshold=-0.9)
        assert r_strict["is_anomaly"].sum() >= r_loose["is_anomaly"].sum()

    def test_threshold_minus_infinity_yields_no_anomaly(self, artifacts, raw_df):
        """UC-12-B : threshold=−∞ → aucune anomalie détectée."""
        results = predict(raw_df, artifacts, threshold=-np.inf)
        assert results["is_anomaly"].sum() == 0

    def test_threshold_zero_yields_at_least_one_anomaly(self, artifacts, raw_df):
        """UC-12-C : threshold=0.0 → au moins une anomalie détectée."""
        results = predict(raw_df, artifacts, threshold=0.0)
        assert results["is_anomaly"].sum() >= 1

    def test_is_anomaly_consistent_with_iso_score(self, artifacts, raw_df):
        """UC-12-D : is_anomaly ⟺ iso_score < threshold (cohérence interne)."""
        threshold = DEFAULT_THRESHOLD
        results = predict(raw_df, artifacts, threshold=threshold)
        expected_anomaly = results["iso_score"] < threshold
        pd.testing.assert_series_equal(
            expected_anomaly, results["is_anomaly"], check_names=False
        )

    def test_all_predictions_present_with_minus_inf_threshold(self, artifacts, raw_df):
        """UC-12 : threshold=−∞ → toutes les prédictions sont produites."""
        results = predict(raw_df, artifacts, threshold=-np.inf)
        assert results["prediction"].isna().sum() == 0


# ---------------------------------------------------------------------------
# UC-13 · Pipeline complet sur données réelles
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """UC-13 : Exécution complète sur les données brutes réelles."""

    def test_pipeline_runs_without_exception(self, artifacts, raw_df):
        """UC-13-B : Le pipeline complet s'exécute sans lever d'exception."""
        predict(raw_df, artifacts)

    def test_more_than_half_predictions_produced(self, artifacts, raw_df):
        """UC-13-A : Au moins 50 % des prédictions sont non-NaN."""
        results = predict(raw_df, artifacts)
        n_predicted = results["prediction"].notna().sum()
        assert n_predicted > len(raw_df) * 0.5

    def test_output_has_correct_columns(self, artifacts, raw_df):
        """UC-13 : Le résultat sur données réelles a les colonnes attendues."""
        results = predict(raw_df, artifacts)
        assert list(results.columns) == ["iso_score", "is_anomaly", "prediction"]

    def test_all_iso_scores_are_finite(self, artifacts, raw_df):
        """UC-13 : Tous les iso_score sont finis, même avec des NaN dans l'entrée."""
        results = predict(raw_df, artifacts)
        assert np.isfinite(results["iso_score"].values).all()
