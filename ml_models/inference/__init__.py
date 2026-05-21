"""
ML Alpha Engine — Inference Pipeline.

Loads trained models and generates fresh alpha scores.
Integrates with the feature store and regime engine.

Output: date | ticker | alpha_score | downside_probability |
        model_confidence | rank
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


ALPHA_SCORES_PATH = Path("data/processed/alpha_scores.parquet")


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


def load_alpha_scores(path: Path = ALPHA_SCORES_PATH) -> pd.DataFrame:
    """Load saved alpha scores."""
    if not path.exists():
        logger.warning(f"Alpha scores not found at {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_alpha_scores(
    alpha_scores: pd.DataFrame,
    path: Path = ALPHA_SCORES_PATH,
) -> Path:
    """Persist alpha scores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha_scores.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Alpha scores saved → {path} ({len(alpha_scores)} rows)")
    return path


def get_latest_alpha_scores() -> pd.DataFrame:
    """Get the most recent alpha scores for each ticker."""
    scores = load_alpha_scores()
    if scores.empty:
        return scores

    latest_date = scores["date"].max()
    return scores[scores["date"] == latest_date].copy()


def run_inference_pipeline(
    inr_prices: pd.DataFrame | None = None,
    feature_store: pd.DataFrame | None = None,
    regime_states: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full inference pipeline: features → ML → alpha scores.

    Steps:
    1. Load/build extended feature store
    2. Build ML dataset
    3. Run ensemble alpha generation
    4. Compute composite confidence
    5. Save results

    Parameters
    ----------
    inr_prices : Raw price data (loaded from disk if None)
    feature_store : Base feature store (loaded if None)
    regime_states : Regime data (loaded if None)
    save : Whether to persist alpha scores

    Returns
    -------
    Alpha scores DataFrame
    """
    logger.info("▶ Running ML inference pipeline...")

    # Load data if not provided
    if inr_prices is None:
        path = Path("data/processed/inr_prices.parquet")
        if not path.exists():
            logger.error("No price data found")
            return pd.DataFrame()
        inr_prices = pd.read_parquet(path)
        inr_prices["date"] = pd.to_datetime(inr_prices["date"])

    if feature_store is None:
        from features.feature_store import load_feature_store
        try:
            feature_store = load_feature_store()
        except FileNotFoundError:
            logger.error("Feature store not found — run main pipeline first")
            return pd.DataFrame()

    if regime_states is None:
        rs_path = Path("data/processed/regime_states.parquet")
        if rs_path.exists():
            regime_states = pd.read_parquet(rs_path)
            regime_states["date"] = pd.to_datetime(regime_states["date"])

    # Step 1: Build extended feature store
    from ml_models.features import build_extended_feature_store
    extended_store = build_extended_feature_store(feature_store, inr_prices, regime_states)

    # Step 2: Build ML dataset
    from ml_models.datasets import build_ml_dataset
    dataset = build_ml_dataset(inr_prices, extended_store, regime_states)

    if dataset.empty:
        logger.error("ML dataset is empty")
        return pd.DataFrame()

    # Step 3: Generate alpha scores
    from ml_models.ensembles import generate_alpha_scores
    alpha_scores = generate_alpha_scores(dataset)

    if alpha_scores.empty:
        logger.error("Alpha score generation failed")
        return pd.DataFrame()

    # Step 4: Compute composite confidence
    from ml_models.confidence import compute_composite_confidence
    from ml_models.quality import detect_feature_drift

    # Feature drift for confidence
    features_wide = extended_store.pivot_table(
        index=["date", "ticker"], columns="feature", values="value", aggfunc="first"
    ).reset_index()
    features_wide.columns.name = None
    drift_report = detect_feature_drift(features_wide)

    alpha_scores = compute_composite_confidence(
        alpha_scores, regime_states, drift_report
    )

    # Step 5: Save
    if save:
        save_alpha_scores(alpha_scores)

    logger.info(f"✓ Inference pipeline complete: {len(alpha_scores)} alpha scores, "
                f"mean confidence={alpha_scores['composite_confidence'].mean():.3f}")

    return alpha_scores
