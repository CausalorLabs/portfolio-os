"""
ML Alpha Engine — Main Orchestrator.

Portfolio-aware ML pipeline:
  features → quality → training → ensemble → confidence → scores

The ML system answers: "Which assets deserve higher portfolio weight?"
NOT: "What exact price will happen tomorrow?"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from ml_models.datasets import build_ml_dataset
from ml_models.features import build_extended_feature_store
from ml_models.quality import run_feature_quality_pipeline
from ml_models.ensembles import generate_alpha_scores
from ml_models.confidence import compute_composite_confidence
from ml_models.evaluation import evaluate_alpha_model
from ml_models.tracking import get_tracker
from ml_models.inference import save_alpha_scores, load_alpha_scores


def run_alpha_pipeline(
    inr_prices: pd.DataFrame | None = None,
    feature_store: pd.DataFrame | None = None,
    regime_states: pd.DataFrame | None = None,
    save: bool = True,
    track: bool = True,
) -> dict:
    """
    Full ML alpha pipeline.

    Steps:
    1. Build extended feature store
    2. Run feature quality pipeline
    3. Build ML dataset with targets
    4. Generate alpha scores (walk-forward ensemble)
    5. Compute composite confidence
    6. Evaluate model
    7. Track experiment
    8. Save results

    Returns
    -------
    dict with keys: alpha_scores, evaluation, quality_summary,
                    feature_importance, dataset
    """
    logger.info("=" * 60)
    logger.info("▶ ML ALPHA PIPELINE")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    if inr_prices is None:
        path = Path("data/processed/inr_prices.parquet")
        if not path.exists():
            logger.error("No price data available")
            return {}
        inr_prices = pd.read_parquet(path)
        inr_prices["date"] = pd.to_datetime(inr_prices["date"])

    if feature_store is None:
        from features.feature_store import load_feature_store
        try:
            feature_store = load_feature_store()
        except FileNotFoundError:
            logger.error("Feature store not available — run main pipeline first")
            return {}

    if regime_states is None:
        rs_path = Path("data/processed/regime_states.parquet")
        if rs_path.exists():
            regime_states = pd.read_parquet(rs_path)
            regime_states["date"] = pd.to_datetime(regime_states["date"])

    # ── Step 1: Extended features ────────────────────────────────────────
    logger.info("▸ Step 1 — Building extended feature store")
    extended_store = build_extended_feature_store(feature_store, inr_prices, regime_states)

    # ── Step 2: Feature quality ──────────────────────────────────────────
    logger.info("▸ Step 2 — Feature quality analysis")
    features_wide = extended_store.pivot_table(
        index=["date", "ticker"], columns="feature", values="value", aggfunc="first"
    ).reset_index()
    features_wide.columns.name = None

    quality_result = run_feature_quality_pipeline(features_wide)
    cleaned_features = quality_result["cleaned_features"]

    # ── Step 3: ML dataset ───────────────────────────────────────────────
    logger.info("▸ Step 3 — Building ML dataset")
    dataset = build_ml_dataset(inr_prices, extended_store, regime_states)

    if dataset.empty:
        logger.error("ML dataset is empty — aborting")
        return {"quality_summary": quality_result["quality_summary"]}

    # ── Step 4: Alpha scores ─────────────────────────────────────────────
    logger.info("▸ Step 4 — Generating alpha scores (walk-forward ensemble)")
    alpha_scores = generate_alpha_scores(dataset)

    if alpha_scores.empty:
        logger.error("Alpha generation failed — aborting")
        return {"quality_summary": quality_result["quality_summary"], "dataset": dataset}

    # ── Step 5: Composite confidence ─────────────────────────────────────
    logger.info("▸ Step 5 — Computing composite confidence")
    alpha_scores = compute_composite_confidence(
        alpha_scores, regime_states, quality_result.get("drift_report")
    )

    # ── Step 6: Evaluation ───────────────────────────────────────────────
    logger.info("▸ Step 6 — Evaluating model")
    target_col = "forward_rank_5d"
    if target_col in dataset.columns:
        pred_df = alpha_scores[["date", "ticker", "alpha_score"]].merge(
            dataset[["date", "ticker", target_col]], on=["date", "ticker"], how="inner"
        ).rename(columns={"alpha_score": "prediction", target_col: "actual"})

        evaluation = evaluate_alpha_model(pred_df, alpha_scores)
    else:
        evaluation = {"rank_ic": 0, "grade": "N/A"}

    # ── Step 7: Track ────────────────────────────────────────────────────
    if track:
        logger.info("▸ Step 7 — Tracking experiment")
        try:
            tracker = get_tracker()
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(
                OmegaConf.load("configs/ml_alpha.yaml"), resolve=True
            ) if Path("configs/ml_alpha.yaml").exists() else {}

            tracker.log_full_experiment(
                params=cfg.get("models", {}),
                metrics=evaluation,
                feature_importance=pd.DataFrame(),
                alpha_scores=alpha_scores,
                run_name=f"alpha_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
            )
        except Exception as exc:
            logger.warning(f"Experiment tracking failed: {exc}")

    # ── Step 8: Save ─────────────────────────────────────────────────────
    if save:
        save_alpha_scores(alpha_scores)

    logger.info("=" * 60)
    logger.info(f"✓ ML ALPHA PIPELINE COMPLETE")
    logger.info(f"  Alpha scores: {len(alpha_scores)} rows")
    logger.info(f"  IC: {evaluation.get('rank_ic', 'N/A')}")
    logger.info(f"  Grade: {evaluation.get('grade', 'N/A')}")
    logger.info(f"  Features: {quality_result['quality_summary']['features_retained']}")
    logger.info("=" * 60)

    return {
        "alpha_scores": alpha_scores,
        "evaluation": evaluation,
        "quality_summary": quality_result["quality_summary"],
        "dataset": dataset,
    }
