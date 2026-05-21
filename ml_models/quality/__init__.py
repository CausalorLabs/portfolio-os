"""
ML Alpha Engine — Feature Quality System.

Critical for ML reliability:
  - Drift detection (PSI — population stability index)
  - Correlation analysis (redundancy removal)
  - Missing data policies (imputation, validity windows)

Most ML systems fail at feature quality. This module prevents that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── Population Stability Index (drift detection) ────────────────────────────


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index between reference and current distributions.

    PSI < 0.10 → no significant change
    PSI 0.10–0.20 → moderate change
    PSI > 0.20 → significant drift
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(ref_clean) < 20 or len(cur_clean) < 20:
        return 0.0

    # Use reference quantiles as bin edges
    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Ensure unique breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    # Proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def detect_feature_drift(
    feature_store_wide: pd.DataFrame,
    reference_window: int = 252,
    current_window: int = 60,
    psi_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Detect distribution drift across all features.

    Parameters
    ----------
    feature_store_wide : Wide-format features (date, ticker, feat1, feat2, ...)
    reference_window : Days of reference data (lookback baseline)
    current_window : Days of current data to compare
    psi_threshold : Threshold for flagging drift

    Returns
    -------
    DataFrame: feature | psi | drifted | reference_mean | current_mean
    """
    cfg = _load_config().get("quality", {}).get("drift", {})
    psi_threshold = cfg.get("threshold", psi_threshold)
    reference_window = cfg.get("reference_window", reference_window)

    df = feature_store_wide.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    dates = df["date"].unique()
    if len(dates) < reference_window + current_window:
        logger.warning("Insufficient data for drift detection")
        return pd.DataFrame(columns=["feature", "psi", "drifted"])

    cutoff = dates[-current_window]
    ref_start = dates[-(reference_window + current_window)]

    ref_df = df[(df["date"] >= ref_start) & (df["date"] < cutoff)]
    cur_df = df[df["date"] >= cutoff]

    # Get numeric feature columns (exclude date, ticker)
    feat_cols = [c for c in df.columns if c not in ("date", "ticker")]

    results = []
    for col in feat_cols:
        ref_vals = ref_df[col].dropna()
        cur_vals = cur_df[col].dropna()
        psi_val = compute_psi(ref_vals, cur_vals)
        results.append({
            "feature": col,
            "psi": round(psi_val, 4),
            "drifted": psi_val > psi_threshold,
            "reference_mean": round(float(ref_vals.mean()), 6) if len(ref_vals) > 0 else None,
            "current_mean": round(float(cur_vals.mean()), 6) if len(cur_vals) > 0 else None,
        })

    result_df = pd.DataFrame(results).sort_values("psi", ascending=False)
    n_drifted = result_df["drifted"].sum()
    logger.info(f"  Drift detection: {n_drifted}/{len(feat_cols)} features drifted (PSI > {psi_threshold})")

    return result_df


# ── Correlation analysis ────────────────────────────────────────────────────


def analyze_feature_correlations(
    feature_store_wide: pd.DataFrame,
    max_corr: float = 0.85,
    method: str = "spearman",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Analyze pairwise feature correlations and flag redundant pairs.

    Parameters
    ----------
    feature_store_wide : Wide-format features
    max_corr : Maximum allowed pairwise correlation
    method : Correlation method ('spearman' or 'pearson')

    Returns
    -------
    (correlation_matrix, features_to_drop)
    """
    cfg = _load_config().get("quality", {}).get("correlation", {})
    max_corr = cfg.get("max_pairwise", max_corr)
    method = cfg.get("method", method)

    feat_cols = [c for c in feature_store_wide.columns if c not in ("date", "ticker")]
    corr_matrix = feature_store_wide[feat_cols].corr(method=method)

    # Find highly correlated pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    high_pairs = []

    for col in upper.columns:
        correlated = upper.index[upper[col].abs() > max_corr].tolist()
        for row in correlated:
            if row not in to_drop:
                to_drop.add(col)
                high_pairs.append((row, col, float(corr_matrix.loc[row, col])))

    to_drop = sorted(to_drop)
    logger.info(f"  Correlation analysis: {len(high_pairs)} redundant pairs, "
                f"dropping {len(to_drop)} features (|corr| > {max_corr})")

    if high_pairs:
        for f1, f2, corr_val in high_pairs[:5]:
            logger.debug(f"    {f1} ↔ {f2}: {corr_val:.3f}")

    return corr_matrix, to_drop


# ── Missing data policies ───────────────────────────────────────────────────


def apply_missing_data_policy(
    feature_store_wide: pd.DataFrame,
    max_missing_pct: float = 0.30,
    imputation: str = "forward_fill",
    validity_window: int = 252,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply standardized missing data handling.

    Steps:
    1. Drop features with > max_missing_pct missing values
    2. Drop features that don't exist for validity_window days
    3. Apply imputation to remaining features

    Returns
    -------
    (cleaned DataFrame, dropped feature names)
    """
    cfg = _load_config().get("quality", {}).get("missing", {})
    max_missing_pct = cfg.get("max_missing_pct", max_missing_pct)
    imputation = cfg.get("imputation", imputation)
    validity_window = cfg.get("validity_window", validity_window)

    df = feature_store_wide.copy()
    feat_cols = [c for c in df.columns if c not in ("date", "ticker")]

    dropped = []

    # 1. Drop features with too many missing values
    for col in feat_cols:
        missing_pct = df[col].isna().mean()
        if missing_pct > max_missing_pct:
            dropped.append(col)

    # 2. Drop features without enough history
    dates = pd.to_datetime(df["date"])
    for col in feat_cols:
        if col in dropped:
            continue
        valid_dates = dates[df[col].notna()].nunique()
        if valid_dates < validity_window:
            dropped.append(col)

    if dropped:
        logger.info(f"  Missing policy: dropping {len(dropped)} features")
        for f in dropped[:5]:
            logger.debug(f"    Dropped: {f}")
        df = df.drop(columns=dropped)

    # 3. Apply imputation
    remaining_feats = [c for c in df.columns if c not in ("date", "ticker")]
    if imputation == "forward_fill":
        for col in remaining_feats:
            df[col] = df.groupby("ticker")[col].ffill()
    elif imputation == "median":
        for col in remaining_feats:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"  After missing policy: {len(remaining_feats)} features retained")

    return df, dropped


# ── Full quality pipeline ───────────────────────────────────────────────────


def run_feature_quality_pipeline(
    feature_store_wide: pd.DataFrame,
) -> dict:
    """
    Run the complete feature quality pipeline.

    Returns
    -------
    dict with keys: cleaned_features, drift_report, corr_matrix,
                    dropped_features, quality_summary
    """
    logger.info("Running feature quality pipeline...")

    # 1. Missing data
    cleaned, dropped_missing = apply_missing_data_policy(feature_store_wide)

    # 2. Drift detection
    drift_report = detect_feature_drift(cleaned)

    # 3. Correlation analysis
    corr_matrix, dropped_corr = analyze_feature_correlations(cleaned)

    # Drop redundant features
    cleaned = cleaned.drop(columns=[c for c in dropped_corr if c in cleaned.columns])

    all_dropped = dropped_missing + dropped_corr
    feat_cols = [c for c in cleaned.columns if c not in ("date", "ticker")]

    summary = {
        "total_features_input": len(feature_store_wide.columns) - 2,
        "dropped_missing": len(dropped_missing),
        "dropped_correlation": len(dropped_corr),
        "drifted_features": int(drift_report["drifted"].sum()) if not drift_report.empty else 0,
        "features_retained": len(feat_cols),
    }
    logger.info(f"  Quality summary: {summary['features_retained']} features retained "
                f"(dropped {summary['dropped_missing']} missing, {summary['dropped_correlation']} correlated)")

    return {
        "cleaned_features": cleaned,
        "drift_report": drift_report,
        "corr_matrix": corr_matrix,
        "dropped_features": all_dropped,
        "quality_summary": summary,
    }
