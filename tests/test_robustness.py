"""
Tests UC-15 à UC-16 – Robustesse du pipeline.

Classes :
  TestNoiseSensitivity       UC-15 : robustesse au bruit gaussien
  TestImputationRobustness   UC-16 : robustesse à l'introduction artificielle de NaN
"""

import numpy as np
import pandas as pd
import pytest

from scripts.predict_sig import DEFAULT_THRESHOLD, predict


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

# Colonnes numériques à perturber (toutes sauf tsunami qui est binaire)
_NUMERIC_PERTURBABLE = ["magnitude", "cdi", "mmi", "nst", "dmin", "gap", "depth"]


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Corrélation de Pearson sans dépendance à scipy."""
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _add_gaussian_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    """Ajoute un bruit gaussien N(0, sigma) aux colonnes numériques perturbables."""
    df_noisy = df.copy()
    for col in _NUMERIC_PERTURBABLE:
        if col in df_noisy.columns:
            # Remplace les NaN existants par la médiane avant bruitage
            median_val = df_noisy[col].median()
            df_noisy[col] = df_noisy[col].fillna(median_val)
            df_noisy[col] = df_noisy[col] + rng.normal(0.0, sigma, size=len(df_noisy))
    return df_noisy


def _introduce_nans(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Introduit des NaN au taux `rate` dans les colonnes numériques."""
    df_missing = df.copy()
    for col in _NUMERIC_PERTURBABLE:
        if col in df_missing.columns:
            mask = rng.random(len(df_missing)) < rate
            df_missing.loc[mask, col] = np.nan
    return df_missing


def _correlation_with_noise(df: pd.DataFrame, artifacts: dict,
                             sigma: float, seed: int = 42) -> float:
    """
    Retourne la corrélation de Pearson entre prédictions sans bruit et
    prédictions avec bruit gaussien d'écart-type sigma.
    Utilise keep_anomalies=True pour maximiser le nombre de prédictions valides.
    """
    rng = np.random.default_rng(seed)
    results_clean = predict(df, artifacts, keep_anomalies=True)
    df_noisy = _add_gaussian_noise(df, sigma, rng)
    results_noisy = predict(df_noisy, artifacts, keep_anomalies=True)

    clean_pred = results_clean["prediction"].values
    noisy_pred = results_noisy["prediction"].values
    valid = np.isfinite(clean_pred) & np.isfinite(noisy_pred)
    if valid.sum() < 10:
        pytest.skip("Pas assez de prédictions valides pour calculer la corrélation.")
    return _pearsonr(clean_pred[valid], noisy_pred[valid])


# ---------------------------------------------------------------------------
# UC-15 · Sensibilité au bruit gaussien
# ---------------------------------------------------------------------------


class TestNoiseSensitivity:
    """UC-15 : Le modèle reste stable face à un bruit gaussien faible."""

    def test_small_noise_high_correlation(self, artifacts, raw_df):
        """UC-15-A : Bruit σ=0.05 → corrélation de Pearson ≥ 0.90."""
        corr = _correlation_with_noise(raw_df, artifacts, sigma=0.05)
        assert corr >= 0.90, f"Corrélation insuffisante sous bruit σ=0.05 : {corr:.3f}"

    def test_large_noise_lower_correlation_than_small(self, artifacts, raw_df):
        """UC-15-B : Bruit σ=2.0 dégrade davantage les prédictions que σ=0.1."""
        corr_small = _correlation_with_noise(raw_df, artifacts, sigma=0.1, seed=1)
        corr_large = _correlation_with_noise(raw_df, artifacts, sigma=2.0, seed=1)
        assert corr_small > corr_large, (
            f"Attendu corr(σ=0.1)={corr_small:.3f} > corr(σ=2.0)={corr_large:.3f}"
        )

    def test_zero_noise_perfect_correlation(self, artifacts, raw_df):
        """UC-15 : Bruit σ=0 → corrélation parfaite (r=1.0)."""
        corr = _correlation_with_noise(raw_df, artifacts, sigma=0.0)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_noisy_predictions_finite(self, artifacts, raw_df):
        """UC-15 : Les prédictions restent finies après ajout de bruit σ=0.5."""
        rng = np.random.default_rng(0)
        df_noisy = _add_gaussian_noise(raw_df, sigma=0.5, rng=rng)
        results = predict(df_noisy, artifacts, keep_anomalies=True)
        assert np.isfinite(results["prediction"].values).all()


# ---------------------------------------------------------------------------
# UC-16 · Robustesse à l'imputation (NaN artificiels)
# ---------------------------------------------------------------------------


class TestImputationRobustness:
    """UC-16 : Le pipeline impute correctement les données partiellement manquantes."""

    def test_10pct_missing_still_over_50pct_predicted(self, artifacts, raw_df):
        """UC-16-A : 10 % de NaN introduits → encore ≥ 50 % de prédictions valides."""
        rng = np.random.default_rng(0)
        df_missing = _introduce_nans(raw_df, rate=0.10, rng=rng)
        results = predict(df_missing, artifacts)
        n_predicted = results["prediction"].notna().sum()
        assert n_predicted > len(df_missing) * 0.50

    def test_30pct_missing_still_produces_predictions(self, artifacts, raw_df):
        """UC-16 : 30 % de NaN → au moins 1 prédiction produite."""
        rng = np.random.default_rng(7)
        df_missing = _introduce_nans(raw_df, rate=0.30, rng=rng)
        results = predict(df_missing, artifacts)
        assert results["prediction"].notna().sum() >= 1

    def test_imputation_preserves_numeric_dtypes(self, artifacts, df_with_nans):
        """UC-16-B : Après imputation, toutes les colonnes sont bien numériques."""
        from scripts.predict_sig import impute
        df_imp = impute(df_with_nans, artifacts)
        for col in df_imp.columns:
            assert pd.api.types.is_numeric_dtype(df_imp[col]), (
                f"Colonne '{col}' n'est pas numérique après imputation."
            )

    def test_all_nan_column_still_imputed(self, artifacts, typical_row):
        """UC-16 : Une colonne entièrement NaN est imputée sans erreur."""
        from scripts.predict_sig import impute
        df_all_nan = typical_row.copy()
        df_all_nan["cdi"] = np.nan
        df_all_nan["mmi"] = np.nan
        # Pas d'exception attendue, résultat fini
        df_imp = impute(df_all_nan, artifacts)
        assert np.isfinite(df_imp["cdi"].iloc[0])
        assert np.isfinite(df_imp["mmi"].iloc[0])
