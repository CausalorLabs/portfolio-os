"""
ML Alpha Engine — Confidence Layer (V1).

Estimates when the model should be trusted LESS.
A defining feature of the platform.

Inputs:
  - Prediction dispersion (model disagreement)
  - Regime stability (from Sprint 2)
  - Rolling IC (rank prediction quality over time)
  - Feature drift (distribution shifts)

Logic: low confidence → reduce alpha influence on portfolio.
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


# ── Confidence components ───────────────────────────────────────────────────


def compute_prediction_dispersion(
    model_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Measure disagreement across ensemble models.

    High disagreement → lower confidence.
    Returns: date | ticker | dispersion_score (0-1, higher = more agreement)
    """
    if len(model_predictions) < 2:
        return pd.DataFrame(columns=["date", "ticker", "dispersion_score"])

    # Merge predictions from all models on (date, ticker)
    merged = None
    for name, df in model_predictions.items():
        pred_df = df[["date", "ticker", "prediction"]].rename(
            columns={"prediction": f"pred_{name}"}
        )
        if merged is None:
            merged = pred_df
        else:
            merged = merged.merge(pred_df, on=["date", "ticker"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["date", "ticker", "dispersion_score"])

    pred_cols = [c for c in merged.columns if c.startswith("pred_")]
    merged["std"] = merged[pred_cols].std(axis=1)

    # Normalize: low std → high agreement → high score
    max_std = merged["std"].quantile(0.95) if len(merged) > 10 else 1.0
    merged["dispersion_score"] = np.clip(1.0 - merged["std"] / max(max_std, 1e-6), 0, 1)

    return merged[["date", "ticker", "dispersion_score"]]


def compute_regime_confidence(
    regime_states: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Map regime stability to confidence.

    Unstable regime (many transitions) → lower confidence.
    Returns: date | regime_stability_score (0-1)
    """
    if regime_states is None or regime_states.empty:
        return pd.DataFrame(columns=["date", "regime_stability_score"])

    rs = regime_states.copy()
    rs["date"] = pd.to_datetime(rs["date"])
    rs = rs.sort_values("date")

    # Rolling transition count over 20-day window
    rs["changed"] = (rs["regime"] != rs["regime"].shift()).astype(int)
    rs["transitions_20d"] = rs["changed"].rolling(20, min_periods=1).sum()

    # 0 transitions → 1.0, many → lower
    max_trans = max(rs["transitions_20d"].max(), 1)
    rs["regime_stability_score"] = np.clip(1.0 - rs["transitions_20d"] / max_trans, 0, 1)

    # Also factor in regime confidence from detector
    if "confidence" in rs.columns:
        rs["regime_stability_score"] = 0.6 * rs["regime_stability_score"] + 0.4 * rs["confidence"]

    return rs[["date", "regime_stability_score"]]


def compute_rolling_ic(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Measure rank prediction quality over time using rolling IC.

    IC = Spearman(predicted_rank, actual_rank) computed daily,
    then smoothed over window.

    Returns: date | rolling_ic_score (0-1)
    """
    if predictions.empty or actuals is None or actuals.empty:
        return pd.DataFrame(columns=["date", "rolling_ic_score"])

    merged = predictions.merge(actuals[["date", "ticker", "actual"]], on=["date", "ticker"], how="inner")

    daily_ics = []
    for dt, grp in merged.groupby("date"):
        if len(grp) < 3:
            continue
        ic, _ = spearmanr(grp["prediction"], grp["actual"])
        daily_ics.append({"date": dt, "ic": ic})

    if not daily_ics:
        return pd.DataFrame(columns=["date", "rolling_ic_score"])

    ic_df = pd.DataFrame(daily_ics).sort_values("date")

    # EWMA smoothing
    cfg = _load_config().get("confidence", {})
    halflife = cfg.get("decay_halflife", 60)
    ic_df["rolling_ic"] = ic_df["ic"].ewm(halflife=halflife, min_periods=5).mean()

    # Normalize to 0-1 (IC ranges from -1 to +1)
    ic_df["rolling_ic_score"] = np.clip((ic_df["rolling_ic"] + 1) / 2, 0, 1)

    return ic_df[["date", "rolling_ic_score"]]


def compute_feature_drift_confidence(drift_report: pd.DataFrame | None) -> float:
    """
    Scalar confidence from feature drift analysis.

    Many drifted features → lower confidence.
    Returns: float (0-1)
    """
    if drift_report is None or drift_report.empty:
        return 0.8  # default moderate confidence

    n_features = len(drift_report)
    n_drifted = drift_report["drifted"].sum()
    drift_pct = n_drifted / max(n_features, 1)

    # 0% drifted → 1.0, 50%+ drifted → 0.3
    return float(np.clip(1.0 - drift_pct * 1.4, 0.3, 1.0))


# ── Composite confidence ────────────────────────────────────────────────────


def compute_composite_confidence(
    alpha_scores: pd.DataFrame,
    regime_states: pd.DataFrame | None = None,
    drift_report: pd.DataFrame | None = None,
    model_predictions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Compute composite confidence for each alpha score.

    Combines:
      - Model dispersion (40%)
      - Regime stability (30%)
      - Feature drift (15%)
      - Base model confidence (15%)

    Returns
    -------
    DataFrame: date | ticker | alpha_score | model_confidence |
               regime_confidence | composite_confidence | rank
    """
    cfg = _load_config().get("confidence", {})
    min_conf = cfg.get("min_confidence", 0.20)
    max_conf = cfg.get("max_confidence", 0.95)

    result = alpha_scores.copy()
    result["date"] = pd.to_datetime(result["date"])

    # Component 1: Model dispersion (already in alpha_scores as model_confidence)
    if "model_confidence" not in result.columns:
        result["model_confidence"] = 0.5

    # Component 2: Regime stability
    regime_conf = compute_regime_confidence(regime_states)
    if not regime_conf.empty:
        result = result.merge(regime_conf, on="date", how="left")
        result["regime_stability_score"] = result["regime_stability_score"].fillna(0.5)
    else:
        result["regime_stability_score"] = 0.5

    # Component 3: Feature drift (scalar, applied uniformly)
    drift_conf = compute_feature_drift_confidence(drift_report)

    # Component 4: Composite
    result["composite_confidence"] = np.clip(
        0.40 * result["model_confidence"]
        + 0.30 * result["regime_stability_score"]
        + 0.15 * drift_conf
        + 0.15 * result["model_confidence"],  # reinforce model confidence
        min_conf, max_conf
    )

    logger.info(f"  Composite confidence: "
                f"mean={result['composite_confidence'].mean():.3f}, "
                f"std={result['composite_confidence'].std():.3f}")

    return result
