"""
ML Alpha Engine — SHAP Explainability Layer.

Per-asset feature attribution:
  - Why did weight increase?
  - Why did confidence drop?
  - Why did downside probability rise?

CRITICAL for trust. An opaque ML system has no place in
portfolio construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ── SHAP computation ────────────────────────────────────────────────────────


def compute_shap_values(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    model_name: str = "unknown",
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values for a trained model.

    Parameters
    ----------
    model : Trained model (LightGBM, CatBoost, or sklearn-compatible)
    X : Feature matrix
    feature_names : Column names
    model_name : Name for logging
    max_samples : Max samples for SHAP computation (performance)

    Returns
    -------
    DataFrame: sample_idx | feature | shap_value | feature_value
    """
    import shap

    # Subsample for performance
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
    else:
        X_sample = X
        idx = np.arange(len(X))

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
        except Exception as exc:
            logger.warning(f"SHAP failed for {model_name}: {exc}")
            return pd.DataFrame(columns=["sample_idx", "feature", "shap_value", "feature_value"])

    if isinstance(X_sample, pd.DataFrame):
        X_vals = X_sample.values
    else:
        X_vals = X_sample

    rows = []
    for i in range(len(X_sample)):
        for j, feat in enumerate(feature_names[:shap_values.shape[1]]):
            rows.append({
                "sample_idx": int(idx[i]) if i < len(idx) else i,
                "feature": feat,
                "shap_value": float(shap_values[i, j]),
                "feature_value": float(X_vals[i, j]) if not np.isnan(X_vals[i, j]) else None,
            })

    result = pd.DataFrame(rows)
    logger.info(f"  SHAP values computed for {model_name}: "
                f"{len(X_sample)} samples × {len(feature_names)} features")
    return result


# ── Per-asset explanations ──────────────────────────────────────────────────


def explain_asset_score(
    shap_df: pd.DataFrame,
    sample_idx: int,
    feature_names: list[str],
    top_k: int = 5,
) -> dict:
    """
    Explain a single asset's alpha score.

    Returns
    -------
    dict with:
      - top_positive: features pushing score UP
      - top_negative: features pushing score DOWN
      - net_shap: total SHAP value
    """
    asset_shap = shap_df[shap_df["sample_idx"] == sample_idx].copy()
    if asset_shap.empty:
        return {"top_positive": [], "top_negative": [], "net_shap": 0}

    asset_shap = asset_shap.sort_values("shap_value", ascending=False)

    positive = asset_shap[asset_shap["shap_value"] > 0].head(top_k)
    negative = asset_shap[asset_shap["shap_value"] < 0].tail(top_k)

    return {
        "top_positive": [
            {"feature": row["feature"], "shap": round(row["shap_value"], 4),
             "value": row["feature_value"]}
            for _, row in positive.iterrows()
        ],
        "top_negative": [
            {"feature": row["feature"], "shap": round(row["shap_value"], 4),
             "value": row["feature_value"]}
            for _, row in negative.iterrows()
        ],
        "net_shap": round(float(asset_shap["shap_value"].sum()), 4),
    }


def explain_portfolio(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    tickers: list[str],
    model_name: str = "ensemble",
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Generate human-readable explanations for all assets.

    Returns
    -------
    DataFrame: ticker | direction | feature | contribution | feature_value
    """
    shap_df = compute_shap_values(model, X, feature_names, model_name)
    if shap_df.empty:
        return pd.DataFrame()

    rows = []
    for i, ticker in enumerate(tickers):
        explanation = explain_asset_score(shap_df, i, feature_names, top_k)

        for entry in explanation["top_positive"]:
            rows.append({
                "ticker": ticker,
                "direction": "positive",
                "feature": entry["feature"],
                "contribution": entry["shap"],
                "feature_value": entry["value"],
            })
        for entry in explanation["top_negative"]:
            rows.append({
                "ticker": ticker,
                "direction": "negative",
                "feature": entry["feature"],
                "contribution": entry["shap"],
                "feature_value": entry["value"],
            })

    return pd.DataFrame(rows)


# ── Global feature importance ───────────────────────────────────────────────


def compute_global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Uses mean absolute SHAP value per feature.
    """
    if shap_df.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "rank"])

    importance = shap_df.groupby("feature")["shap_value"].agg(
        mean_abs_shap=lambda x: x.abs().mean(),
        mean_shap="mean",
        std_shap="std",
    ).sort_values("mean_abs_shap", ascending=False).reset_index()

    importance["rank"] = range(1, len(importance) + 1)
    return importance


def format_explanation_text(explanation: dict, ticker: str) -> str:
    """Format explanation as human-readable text."""
    lines = [f"Asset: {ticker}"]

    if explanation["top_positive"]:
        lines.append("  Overweight drivers:")
        for e in explanation["top_positive"]:
            lines.append(f"    + {e['feature']}: {e['shap']:+.4f}")

    if explanation["top_negative"]:
        lines.append("  Underweight drivers:")
        for e in explanation["top_negative"]:
            lines.append(f"    − {e['feature']}: {e['shap']:+.4f}")

    lines.append(f"  Net SHAP: {explanation['net_shap']:+.4f}")
    return "\n".join(lines)
