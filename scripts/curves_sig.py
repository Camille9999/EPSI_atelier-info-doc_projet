"""
Tracé des courbes d'analyse pour le modèle sig (RandomForestRegressor).

Courbes produites :
  1. Sensibilité au bruit      – corrélation de Pearson vs σ du bruit gaussien
  2. Sensibilité à l'imputation – % de prédictions valides vs taux de NaN introduits
  3. Méthode du coude           – nombre d'anomalies vs seuil d'Isolation Forest
  4. Impact du bruit par feature – variation RMSE (%) vs intensité du bruit (%) par feature
  5. Impact imputation par feature – RMSE (abs. et %) vs taux de NaN par feature
  6. Robustesse IF               – RMSE et coverage (%) vs seuil d'isolation

Usage :
    python scripts/curves_sig.py
    python scripts/curves_sig.py --output figures/
    python scripts/curves_sig.py --no-show --output figures/
    python scripts/curves_sig.py --n-rows 1000
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Rendre le module predict_sig importable depuis n'importe quel répertoire
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

from scripts.predict_sig import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_THRESHOLD,
    load_artifacts,
    predict,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "earthquake_data.csv"

# Colonnes numériques perturbables pour les courbes de sensibilité globale
_NUMERIC_COLS = ["magnitude", "cdi", "mmi", "nst", "dmin", "gap", "depth"]

# Colonnes analysées dans les courbes d'impact du bruit par feature
# (aligné sur le notebook analyse_bruit.ipynb qui exclut cdi, mmi, tsunami)
_NOISE_ANALYSIS_COLS = ["magnitude", "nst", "dmin", "gap", "depth"]


# ---------------------------------------------------------------------------
# Utilitaires partagés
# ---------------------------------------------------------------------------


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Corrélation de Pearson sans dépendance externe à scipy."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    if denom == 0:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _add_gaussian_noise(
    df: pd.DataFrame, sigma: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Ajoute un bruit gaussien N(0, sigma) aux colonnes numériques."""
    df_noisy = df.copy()
    for col in _NUMERIC_COLS:
        if col in df_noisy.columns:
            median_val = df_noisy[col].median()
            df_noisy[col] = df_noisy[col].fillna(median_val)
            df_noisy[col] += rng.normal(0.0, sigma, size=len(df_noisy))
    return df_noisy


def _introduce_nans(
    df: pd.DataFrame, rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Introduit des NaN au taux `rate` (∈ [0, 1]) dans les colonnes numériques."""
    df_missing = df.copy()
    for col in _NUMERIC_COLS:
        if col in df_missing.columns:
            mask = rng.random(len(df_missing)) < rate
            df_missing.loc[mask, col] = np.nan
    return df_missing


def _save_or_show(
    fig: plt.Figure,
    filename: str,
    output_dir: Path | None,
    show: bool,
) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  → Sauvegardé : {path}")
    if show:
        plt.show()
    plt.close(fig)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE en ignorant les NaN dans y_pred."""
    valid = np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


# ---------------------------------------------------------------------------
# Courbe 1 : Sensibilité au bruit
# ---------------------------------------------------------------------------


def compute_noise_sensitivity(
    df: pd.DataFrame,
    artifacts: dict,
    sigmas: np.ndarray,
    seed: int = 42,
) -> list[float]:
    """
    Pour chaque σ dans `sigmas`, calcule la corrélation de Pearson entre
    les prédictions sans bruit et les prédictions avec bruit gaussien d'écart-type σ.

    Parameters
    ----------
    df       : données brutes d'entrée
    artifacts: artefacts du modèle chargés par load_artifacts()
    sigmas   : vecteur de valeurs σ à tester
    seed     : graine aléatoire pour la reproductibilité

    Returns
    -------
    Liste de corrélations (NaN si pas assez de prédictions valides).
    """
    rng = np.random.default_rng(seed)
    results_clean = predict(df, artifacts, keep_anomalies=True)
    clean_pred = results_clean["prediction"].values

    correlations = []
    for sigma in sigmas:
        df_noisy = _add_gaussian_noise(df, sigma=sigma, rng=rng)
        results_noisy = predict(df_noisy, artifacts, keep_anomalies=True)
        noisy_pred = results_noisy["prediction"].values

        valid = np.isfinite(clean_pred) & np.isfinite(noisy_pred)
        if valid.sum() < 5:
            correlations.append(float("nan"))
        else:
            correlations.append(_pearsonr(clean_pred[valid], noisy_pred[valid]))

    return correlations


def plot_noise_sensitivity(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace la courbe de sensibilité au bruit et sauvegarde/affiche la figure."""
    sigmas = np.linspace(0.0, 2.0, 21)
    print("  Calcul de la sensibilité au bruit …")
    correlations = compute_noise_sensitivity(df, artifacts, sigmas)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, correlations, marker="o", color="steelblue", label="Corrélation mesurée")
    ax.axhline(0.9, linestyle="--", color="tomato", label="Seuil critique (r = 0.90)")
    ax.axvline(DEFAULT_THRESHOLD * -1, linestyle=":", color="gray", alpha=0.6)
    ax.set_xlabel("Écart-type du bruit gaussien (σ)")
    ax.set_ylabel("Corrélation de Pearson (prédictions bruitées vs propres)")
    ax.set_title("Courbe de sensibilité au bruit – Modèle sig")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    _save_or_show(fig, "noise_sensitivity.png", output_dir, show)


# ---------------------------------------------------------------------------
# Courbe 2 : Sensibilité à l'imputation
# ---------------------------------------------------------------------------


def compute_imputation_sensitivity(
    df: pd.DataFrame,
    artifacts: dict,
    missing_rates: np.ndarray,
    seed: int = 0,
) -> list[float]:
    """
    Pour chaque taux de NaN dans `missing_rates`, calcule le pourcentage de
    prédictions valides (non-NaN) retournées par le pipeline.

    Parameters
    ----------
    df            : données brutes d'entrée
    artifacts     : artefacts du modèle
    missing_rates : vecteur de taux ∈ [0, 1]
    seed          : graine aléatoire

    Returns
    -------
    Liste de pourcentages (0–100).
    """
    rng = np.random.default_rng(seed)
    pct_predicted = []
    for rate in missing_rates:
        df_missing = _introduce_nans(df, rate=rate, rng=rng)
        results = predict(df_missing, artifacts)
        pct = results["prediction"].notna().sum() / len(results) * 100
        pct_predicted.append(pct)
    return pct_predicted


def plot_imputation_sensitivity(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace la courbe de sensibilité à l'imputation et sauvegarde/affiche la figure."""
    missing_rates = np.linspace(0.0, 0.5, 11)
    print("  Calcul de la sensibilité à l'imputation …")
    pct_predicted = compute_imputation_sensitivity(df, artifacts, missing_rates)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        missing_rates * 100,
        pct_predicted,
        marker="s",
        color="darkorange",
        label="% prédictions produites",
    )
    ax.axhline(50, linestyle="--", color="tomato", label="Seuil d'acceptabilité (50 %)")
    ax.set_xlabel("Taux de valeurs manquantes introduites (%)")
    ax.set_ylabel("Prédictions produites (%)")
    ax.set_title("Courbe de sensibilité à l'imputation – Modèle sig")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    _save_or_show(fig, "imputation_sensitivity.png", output_dir, show)


# ---------------------------------------------------------------------------
# Courbe 3 : Méthode du coude – score d'anomalie
# ---------------------------------------------------------------------------


def compute_elbow_anomalies(
    df: pd.DataFrame,
    artifacts: dict,
    thresholds: np.ndarray,
) -> list[int]:
    """
    Pour chaque seuil d'Isolation Forest dans `thresholds`, compte le nombre
    d'anomalies détectées dans `df`.

    Parameters
    ----------
    df         : données brutes d'entrée
    artifacts  : artefacts du modèle
    thresholds : vecteur de seuils à tester

    Returns
    -------
    Liste d'entiers : nombre d'anomalies détectées pour chaque seuil.
    """
    n_anomalies = []
    for thr in thresholds:
        results = predict(df, artifacts, threshold=float(thr))
        n_anomalies.append(int(results["is_anomaly"].sum()))
    return n_anomalies


def plot_elbow_anomaly_score(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace la courbe « méthode du coude » pour le seuil IF et sauvegarde/affiche."""
    thresholds = np.linspace(-0.9, -0.3, 31)
    print("  Calcul de la méthode du coude …")
    n_anomalies = compute_elbow_anomalies(df, artifacts, thresholds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, n_anomalies, marker="D", color="forestgreen",
            label="Nombre d'anomalies")
    ax.axvline(
        DEFAULT_THRESHOLD,
        linestyle="--",
        color="tomato",
        label=f"Seuil retenu ({DEFAULT_THRESHOLD})",
    )
    ax.set_xlabel("Seuil d'Isolation Forest")
    ax.set_ylabel("Nombre d'anomalies détectées")
    ax.set_title("Méthode du coude – Score d'anomalie (modèle sig)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, "elbow_anomaly_score.png", output_dir, show)


# ---------------------------------------------------------------------------
# Courbe 4 : Impact du bruit par feature
# ---------------------------------------------------------------------------


def compute_noise_impact_per_feature(
    df: pd.DataFrame,
    artifacts: dict,
    intensities: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """
    Pour chaque feature et chaque intensité de bruit (exprimée en % de l'écart-type
    de la feature), calcule la variation moyenne du RMSE par rapport au RMSE sans bruit.

    Évalue uniquement sur le jeu de TEST (split identique au notebook analyse_bruit)
    pour éviter que les observations d'entraînement, mieux apprises par le RF,
    ne masquent l'impact réel du bruit.

    Parameters
    ----------
    df           : données brutes (doit contenir la colonne "sig" comme cible)
    artifacts    : artefacts du modèle
    intensities  : intensités à tester (en %, ex. [1, 5, 10, 20, 50])
    n_repeats    : nombre de répétitions par point (le bruit est aléatoire)
    seed         : graine aléatoire pour le bruit
    test_size    : proportion du jeu de test (défaut 0.2, comme le notebook)
    random_state : graine du split (défaut 42, comme le notebook)

    Returns
    -------
    Dictionnaire feature → liste de variations RMSE (%).
    """
    _, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    rng = np.random.default_rng(seed)
    y_true = df_test["sig"].values
    results_clean = predict(df_test, artifacts, keep_anomalies=True)
    rmse_clean = _rmse(y_true, results_clean["prediction"].values)

    rmse_variations: dict[str, list[float]] = {}
    for feat in _NOISE_ANALYSIS_COLS:
        if feat not in df_test.columns:
            continue
        std_feat = float(df_test[feat].std())
        variations = []
        for intensity in intensities:
            sigma = std_feat * intensity / 100.0
            trial_vars = []
            for _ in range(n_repeats):
                df_noisy = df_test.copy()
                df_noisy[feat] = df_noisy[feat].fillna(df_noisy[feat].median())
                df_noisy[feat] += rng.normal(0.0, sigma, size=len(df_noisy))
                results_noisy = predict(df_noisy, artifacts, keep_anomalies=True)
                rmse_noisy = _rmse(y_true, results_noisy["prediction"].values)
                trial_vars.append(100.0 * (rmse_noisy - rmse_clean) / rmse_clean)
            variations.append(float(np.mean(trial_vars)))
        rmse_variations[feat] = variations

    return rmse_variations


def plot_noise_impact_per_feature(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace l'impact du bruit par feature (variation RMSE % vs intensité %)."""
    intensities = np.array([1, 3, 5, 10, 15, 20, 30, 50], dtype=float)
    print("  Calcul de l'impact du bruit par feature …")
    rmse_variations = compute_noise_impact_per_feature(df, artifacts, intensities)

    fig, ax = plt.subplots(figsize=(11, 6))
    for feat, variations in rmse_variations.items():
        ax.plot(intensities, variations, marker="o", label=feat)
    ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
    ax.set_xlabel("Intensité du bruit (% de l'écart-type de la feature)")
    ax.set_ylabel("Variation du RMSE (%)")
    ax.set_title("Impact du bruit sur les features – Modèle sig")
    ax.legend(ncol=2, title="Features")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, "noise_impact_per_feature.png", output_dir, show)


# ---------------------------------------------------------------------------
# Courbe 5 : Impact de l'imputation par feature
# ---------------------------------------------------------------------------


def compute_imputation_impact_per_feature(
    df: pd.DataFrame,
    artifacts: dict,
    missing_rates: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, list[float]], float]:
    """
    Pour chaque feature et chaque taux de NaN, supprime aléatoirement des valeurs
    de CETTE SEULE feature, impute, puis calcule la RMSE.

    Évalue uniquement sur le jeu de TEST (split identique au notebook
    robustesse_imputation) pour éviter que les observations d'entraînement
    ne masquent l'impact réel de l'imputation.

    Parameters
    ----------
    df            : données brutes (doit contenir la colonne "sig")
    artifacts     : artefacts du modèle
    missing_rates : vecteur de taux ∈ [0, 1]
    n_repeats     : nombre de répétitions par point
    seed          : graine aléatoire
    test_size     : proportion du jeu de test (défaut 0.2)
    random_state  : graine du split (défaut 42)

    Returns
    -------
    (rmse_per_feature, rmse_baseline)
      rmse_per_feature : feature → liste de RMSE absolus
      rmse_baseline    : RMSE sans données manquantes
    """
    _, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    rng = np.random.default_rng(seed)
    y_true = df_test["sig"].values
    results_clean = predict(df_test, artifacts, keep_anomalies=True)
    rmse_baseline = _rmse(y_true, results_clean["prediction"].values)

    rmse_per_feature: dict[str, list[float]] = {}
    for feat in artifacts["feature_order"]:
        if feat not in df_test.columns:
            continue
        feature_curve = []
        for rate in missing_rates:
            n_drop = int(np.floor(rate * len(df_test)))
            trial_rmses = []
            for _ in range(n_repeats):
                df_missing = df_test.copy()
                if n_drop > 0:
                    idx_missing = rng.choice(len(df_missing), size=n_drop, replace=False)
                    df_missing.iloc[idx_missing, df_missing.columns.get_loc(feat)] = np.nan
                results = predict(df_missing, artifacts, keep_anomalies=True)
                trial_rmses.append(_rmse(y_true, results["prediction"].values))
            feature_curve.append(float(np.mean(trial_rmses)))
        rmse_per_feature[feat] = feature_curve

    return rmse_per_feature, rmse_baseline


def plot_imputation_impact_per_feature(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace l'impact de l'imputation par feature (RMSE absolu gauche, variation % droite)."""
    missing_rates = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    print("  Calcul de l'impact de l'imputation par feature …")
    rmse_per_feature, rmse_baseline = compute_imputation_impact_per_feature(
        df, artifacts, missing_rates
    )

    fig, ax_rmse = plt.subplots(figsize=(12, 7))
    ax_pct = ax_rmse.twinx()

    for feat, rmse_curve in rmse_per_feature.items():
        rmse_arr = np.asarray(rmse_curve, dtype=float)
        line, = ax_rmse.plot(
            missing_rates * 100, rmse_arr, marker="o", label=feat
        )
        if not np.isclose(rmse_baseline, 0.0):
            pct_change = (rmse_arr - rmse_baseline) / rmse_baseline * 100.0
            ax_pct.plot(
                missing_rates * 100, pct_change,
                linestyle="--", alpha=0.45, color=line.get_color(),
            )

    ax_rmse.set_xlabel("Taux de valeurs supprimées dans la feature (%)")
    ax_rmse.set_ylabel("RMSE (absolu)")
    ax_pct.set_ylabel("Variation RMSE (%)")
    ax_rmse.set_title("Impact de l'imputation par feature – Modèle sig")
    ax_rmse.legend(ncol=2, title="Features (lignes pleines = RMSE, tirets = variation %)")
    ax_rmse.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, "imputation_impact_per_feature.png", output_dir, show)


# ---------------------------------------------------------------------------
# Courbe 6 : Robustesse – RMSE et coverage selon le seuil d'isolation
# ---------------------------------------------------------------------------


def compute_robustness_curve(
    df: pd.DataFrame,
    artifacts: dict,
    thresholds: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Pour chaque seuil d'Isolation Forest, calcule la RMSE et le coverage
    sur le jeu de TEST uniquement (split identique au notebook robustesse_imputation).

    L'IF ayant été entraîné sur les données d'entraînement, évaluer les scores
    sur toutes les données biaiserait la courbe (les observations d'entraînement
    ont des scores IF systématiquement plus élevés). On isole donc le jeu de test
    pour reproduire fidèlement le graphique du notebook.

    Parameters
    ----------
    df           : données brutes (doit contenir la colonne "sig")
    artifacts    : artefacts du modèle
    thresholds   : vecteur de seuils à tester
    test_size    : proportion du jeu de test (défaut 0.2, comme le notebook)
    random_state : graine du split (défaut 42, comme le notebook)

    Returns
    -------
    DataFrame avec colonnes ["threshold", "rmse", "coverage"].
    """
    _, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    y_true = df_test["sig"].values
    # keep_anomalies=True pour obtenir un score sur toutes les lignes du test
    results_all = predict(df_test, artifacts, keep_anomalies=True)
    scores = results_all["iso_score"].values
    y_pred_all = results_all["prediction"].values

    rows = []
    for th in thresholds:
        mask = scores >= th
        coverage = float(mask.mean())
        if mask.sum() == 0:
            rmse = float("nan")
        else:
            rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred_all[mask]) ** 2)))
        rows.append({"threshold": float(th), "rmse": rmse, "coverage": coverage * 100})

    return pd.DataFrame(rows)


def plot_robustness_curve(
    df: pd.DataFrame,
    artifacts: dict,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Trace la courbe de robustesse : RMSE (gauche) et coverage % (droite) vs seuil IF."""
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    scores_all = predict(df_test, artifacts, keep_anomalies=True)["iso_score"].values
    thresholds = np.linspace(scores_all.min(), scores_all.max(), 120)
    print("  Calcul de la courbe de robustesse …")
    rob = compute_robustness_curve(df, artifacts, thresholds)

    fig, ax_rmse = plt.subplots(figsize=(11, 6))
    ax_cov = ax_rmse.twinx()

    ax_rmse.plot(
        rob["threshold"], rob["rmse"],
        color="#d62728", linewidth=2.0, label="RMSE (sig)",
    )
    ax_cov.plot(
        rob["threshold"], rob["coverage"],
        color="#1f77b4", linewidth=2.0, linestyle="--", label="Coverage (%)",
    )
    ax_rmse.axvline(
        DEFAULT_THRESHOLD, linestyle=":", color="gray",
        label=f"Seuil retenu ({DEFAULT_THRESHOLD})",
    )

    ax_rmse.set_xlabel("Seuil d'isolation (score_samples)")
    ax_rmse.set_ylabel("RMSE", color="#d62728")
    ax_cov.set_ylabel("Coverage (%)", color="#1f77b4")
    ax_rmse.set_title("Robustesse sig : RMSE et coverage selon le seuil d'isolation")
    ax_rmse.grid(True, alpha=0.3)

    lines1, labels1 = ax_rmse.get_legend_handles_labels()
    lines2, labels2 = ax_cov.get_legend_handles_labels()
    ax_rmse.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    _save_or_show(fig, "robustness_curve.png", output_dir, show)


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère les courbes d'analyse pour le modèle sig."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_PATH),
        help=f"CSV d'entrée (défaut : {DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Répertoire de sortie des figures (ex. figures/). Optionnel.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Ne pas afficher les figures (utile en mode non-interactif / CI).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=500,
        help="Nombre de lignes du CSV à utiliser (défaut : 500).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"Répertoire des artefacts (défaut : {ARTIFACTS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output) if args.output else None
    show = not args.no_show

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

    print(f"Chargement des données depuis : {input_path} ({args.n_rows} lignes max)")
    df = pd.read_csv(input_path).head(args.n_rows)
    print(f"  → {len(df)} lignes, {df.shape[1]} colonnes")

    print("\n[1/6] Courbe de sensibilité au bruit")
    plot_noise_sensitivity(df, artifacts, output_dir=output_dir, show=show)

    print("\n[2/6] Courbe de sensibilité à l'imputation")
    plot_imputation_sensitivity(df, artifacts, output_dir=output_dir, show=show)

    print("\n[3/6] Méthode du coude – score d'anomalie")
    plot_elbow_anomaly_score(df, artifacts, output_dir=output_dir, show=show)

    print("\n[4/6] Impact du bruit par feature")
    plot_noise_impact_per_feature(df, artifacts, output_dir=output_dir, show=show)

    print("\n[5/6] Impact de l'imputation par feature")
    plot_imputation_impact_per_feature(df, artifacts, output_dir=output_dir, show=show)

    print("\n[6/6] Robustesse – RMSE et coverage selon le seuil d'isolation")
    plot_robustness_curve(df, artifacts, output_dir=output_dir, show=show)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
