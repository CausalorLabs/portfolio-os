"""
ML Alpha Engine — Ensemble Alpha Engine.

Combines LightGBM + CatBoost + momentum baseline into a single
alpha score per asset per date.

Final output:
    date | ticker | alpha_score | downside_probability | model_confidence | rank

IMPORTANT: This engine is portfolio-aware, NOT price-prediction obsessed.
The goal is cross-sectional ranking for allocation tilts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── Model factories ─────────────────────────────────────────────────────────


def create_lightgbm_model(params: dict | None = None):
    """Create a LightGBM regressor with portfolio-tuned defaults."""
    import lightgbm as lgb

    cfg = _load_config().get("models", {}).get("lightgbm", {})
    defaults = {
        "n_estimators": cfg.get("n_estimators", 500),
        "max_depth": cfg.get("max_depth", 6),
        "learning_rate": cfg.get("learning_rate", 0.05),
        "subsample": cfg.get("subsample", 0.8),
        "colsample_bytree": cfg.get("colsample_bytree", 0.8),
        "min_child_samples": cfg.get("min_child_samples", 20),
        "reg_alpha": cfg.get("reg_alpha", 0.1),
        "reg_lambda": cfg.get("reg_lambda", 1.0),
        "random_state": cfg.get("random_state", 42),
        "verbose": -1,
    }
    if params:
        defaults.update(params)

    return lgb.LGBMRegressor(**defaults)


def create_catboost_model(params: dict | None = None):
    """Create a CatBoost regressor with portfolio-tuned defaults."""
    from catboost import CatBoostRegressor

    cfg = _load_config().get("models", {}).get("catboost", {})
    defaults = {
        "iterations": cfg.get("iterations", 500),
        "depth": cfg.get("depth", 6),
        "learning_rate": cfg.get("learning_rate", 0.05),
        "l2_leaf_reg": cfg.get("l2_leaf_reg", 3.0),
        "subsample": cfg.get("subsample", 0.8),
        "random_seed": cfg.get("random_seed", 42),
        "verbose": cfg.get("verbose", 0),
    }
    if params:
        defaults.update(params)

    return CatBoostRegressor(**defaults)


def create_momentum_baseline(lookback: int = 60):
    """
    Momentum baseline — simple cross-sectional momentum rank.

    Not a fitted model; computed directly from returns.
    """
    return MomentumBaseline(lookback=lookback)


class MomentumBaseline:
    """Simple momentum-based ranking as an ensemble member."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumBaseline":
        """No training needed; momentum is a direct computation."""
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return cross-sectional rank based on momentum features."""
        # Use the momentum_60d column if present (index 1 typically)
        # Fall back to mean of all features as rank proxy
        if X.shape[1] > 1:
            scores = X[:, 1]  # momentum_60d
        else:
            scores = X[:, 0]

        # Handle NaNs
        valid = ~np.isnan(scores)
        ranks = np.full(len(scores), 0.5)
        if valid.sum() > 0:
            from scipy.stats import rankdata
            ranks[valid] = rankdata(scores[valid]) / valid.sum()
        return ranks

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array([])


# ── Ensemble engine ─────────────────────────────────────────────────────────


class AlphaEnsemble:
    """
    Ensemble alpha engine combining multiple models.

    Produces: alpha_score, model_confidence (from dispersion)
    """

    def __init__(self):
        self.models: dict[str, object] = {}
        self.weights: dict[str, float] = {}
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def build(self, feature_cols: list[str]) -> "AlphaEnsemble":
        """Initialize ensemble models from config."""
        cfg = _load_config()
        ensemble_cfg = cfg.get("ensemble", {})
        weights = ensemble_cfg.get("weights", {
            "lightgbm": 0.40,
            "catboost": 0.40,
            "momentum_baseline": 0.20,
        })

        mom_lookback = cfg.get("models", {}).get("momentum_baseline", {}).get("lookback_days", 60)

        self.models = {
            "lightgbm": create_lightgbm_model(),
            "catboost": create_catboost_model(),
            "momentum_baseline": create_momentum_baseline(mom_lookback),
        }
        self.weights = weights
        self.feature_cols = feature_cols

        logger.info(f"Ensemble initialized: {list(self.models.keys())} "
                     f"weights={self.weights}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AlphaEnsemble":
        """Train all ensemble models."""
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                logger.info(f"  {name}: trained on {len(X)} samples")
            except Exception as exc:
                logger.error(f"  {name}: training failed — {exc}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Generate predictions from all models."""
        predictions = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                predictions[name] = preds
            except Exception as exc:
                logger.error(f"  {name}: prediction failed — {exc}")
                predictions[name] = np.full(len(X), 0.5)
        return predictions

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate ensemble alpha scores and confidence estimates.

        Returns
        -------
        (alpha_scores, model_confidence)
        alpha_scores: Weighted average of model predictions
        model_confidence: Based on prediction dispersion (agreement)
        """
        predictions = self.predict(X)

        if not predictions:
            return np.full(len(X), 0.5), np.full(len(X), 0.5)

        # Weighted average
        alpha = np.zeros(len(X))
        total_weight = 0
        for name, preds in predictions.items():
            w = self.weights.get(name, 0)
            alpha += preds * w
            total_weight += w

        if total_weight > 0:
            alpha /= total_weight

        # Confidence from prediction dispersion
        pred_stack = np.column_stack(list(predictions.values()))
        dispersion = np.std(pred_stack, axis=1)

        # Normalize dispersion to confidence: low dispersion → high confidence
        max_disp = np.percentile(dispersion, 95) if len(dispersion) > 10 else 1.0
        confidence = 1.0 - np.clip(dispersion / max(max_disp, 1e-6), 0, 1)

        # Floor and ceiling
        cfg = _load_config().get("confidence", {})
        min_conf = cfg.get("min_confidence", 0.20)
        max_conf = cfg.get("max_confidence", 0.95)
        confidence = np.clip(confidence, min_conf, max_conf)

        return alpha, confidence

    def get_feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importance across models."""
        frames = []
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_") and len(model.feature_importances_) > 0:
                imp = model.feature_importances_
                df = pd.DataFrame({
                    "feature": self.feature_cols[:len(imp)],
                    "importance": imp,
                    "model": name,
                })
                frames.append(df)

        if frames:
            all_imp = pd.concat(frames, ignore_index=True)
            avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(
                ascending=False).reset_index()
            return avg_imp
        return pd.DataFrame(columns=["feature", "importance"])


# ── Alpha score generator ───────────────────────────────────────────────────


def generate_alpha_scores(
    dataset: pd.DataFrame,
    target_col: str = "forward_rank_5d",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Full alpha pipeline: walk-forward train → ensemble → alpha scores.

    Returns
    -------
    DataFrame: date | ticker | alpha_score | downside_probability |
               model_confidence | rank
    """
    from ml_models.training import (
        generate_walk_forward_splits,
        walk_forward_train,
        aggregate_walk_forward_results,
    )

    cfg = _load_config()

    # Determine feature columns
    if feature_cols is None:
        exclude = {"date", "ticker", "forward_rank_5d", "forward_rank_20d",
                    "risk_adjusted_20d", "risk_adjusted_rank_20d",
                    "downside_prob_20d", "prediction", "actual"}
        feature_cols = [c for c in dataset.columns if c not in exclude
                        and not c.startswith("forward_") and not c.startswith("risk_adjusted_")]

    logger.info(f"Alpha pipeline: {len(feature_cols)} features, target={target_col}")

    ensemble = AlphaEnsemble().build(feature_cols)

    # Walk-forward with each model
    all_test_preds = []
    for model_name in ["lightgbm", "catboost"]:
        logger.info(f"Walk-forward training: {model_name}")

        if model_name == "lightgbm":
            factory = create_lightgbm_model
        else:
            factory = create_catboost_model

        results = walk_forward_train(
            dataset=dataset,
            target_col=target_col,
            feature_cols=feature_cols,
            model_factory=factory,
        )

        if results:
            agg = aggregate_walk_forward_results(results)
            preds = agg["all_predictions"].copy()
            preds["model"] = model_name
            all_test_preds.append(preds)
            logger.info(f"  {model_name}: avg IC={agg['avg_test_ic']:.4f}, "
                         f"stability={agg['ic_stability']:.4f}")

    # Momentum baseline — direct computation
    mom_preds = dataset[["date", "ticker"]].copy()
    momentum_col = "momentum_60d" if "momentum_60d" in dataset.columns else None
    if momentum_col:
        mom_preds["prediction"] = dataset.groupby("date")[momentum_col].rank(pct=True)
    else:
        mom_preds["prediction"] = 0.5
    mom_preds["actual"] = dataset.get(target_col, 0.5)
    mom_preds["model"] = "momentum_baseline"
    all_test_preds.append(mom_preds)

    if not all_test_preds:
        logger.error("No predictions generated")
        return pd.DataFrame()

    # Combine all predictions
    combined = pd.concat(all_test_preds, ignore_index=True)

    # Ensemble: weighted average per (date, ticker)
    weights = cfg.get("ensemble", {}).get("weights", {
        "lightgbm": 0.40, "catboost": 0.40, "momentum_baseline": 0.20
    })

    combined["weight"] = combined["model"].map(weights).fillna(0)
    combined["weighted_pred"] = combined["prediction"] * combined["weight"]

    alpha_scores = combined.groupby(["date", "ticker"]).agg(
        alpha_score=("weighted_pred", "sum"),
        total_weight=("weight", "sum"),
        n_models=("model", "count"),
        pred_std=("prediction", "std"),
    ).reset_index()

    alpha_scores["alpha_score"] = alpha_scores["alpha_score"] / alpha_scores["total_weight"].clip(lower=1e-6)

    # Confidence from model dispersion
    confidence_cfg = cfg.get("confidence", {})
    min_conf = confidence_cfg.get("min_confidence", 0.20)
    max_conf = confidence_cfg.get("max_confidence", 0.95)

    max_std = alpha_scores["pred_std"].quantile(0.95) if len(alpha_scores) > 10 else 1.0
    alpha_scores["model_confidence"] = np.clip(
        1.0 - alpha_scores["pred_std"].fillna(0.5) / max(max_std, 1e-6),
        min_conf, max_conf
    )

    # Rank per date
    alpha_scores["rank"] = alpha_scores.groupby("date")["alpha_score"].rank(pct=True)

    # Downside probability placeholder (from downside target if available)
    alpha_scores["downside_probability"] = 0.5  # Will be overridden by confidence layer

    result = alpha_scores[["date", "ticker", "alpha_score", "downside_probability",
                           "model_confidence", "rank"]].sort_values(["date", "ticker"])

    logger.info(f"Alpha scores generated: {len(result)} rows, "
                f"{result['ticker'].nunique()} tickers")

    return result
