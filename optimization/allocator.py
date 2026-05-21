"""
Signal-tilted allocator — blends risk-based allocation (HRP)
with signal strength (composite score) to produce a final
target portfolio.

Tilts are SMALL and CONTROLLED — this is NOT aggressive betting.
"""

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path


def build_signal_tilted_portfolio(
    base_weights: pd.DataFrame,
    signal_scores: pd.DataFrame,
    tilt_strength: float = 0.20,
) -> pd.DataFrame:
    """
    Tilt base weights by signal strength.

    Logic:
        adjusted_weight = base_weight * (1 + tilt_strength * (rank - 0.5))
        Then re-normalize to sum to 1.0.

    This means a high-ranked asset gets a mild overweight, and a
    low-ranked asset gets a mild underweight. With 4 assets and
    tilt_strength=0.20, the max tilt is ±10% of the base weight.

    Parameters
    ----------
    base_weights : pd.DataFrame
        Columns: ticker, target_weight (from HRP or baseline).
    signal_scores : pd.DataFrame
        Columns: date, ticker, composite_score, composite_rank.
    tilt_strength : float
        How much signals can move weights (0.0 = no tilt, 1.0 = full).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, base_weight, signal_rank, tilt_multiplier,
                 target_weight, strategy
    """
    # Use the latest signal scores
    latest_date = signal_scores["date"].max()
    latest_signals = signal_scores[signal_scores["date"] == latest_date][
        ["ticker", "composite_rank"]
    ].copy()

    df = base_weights[["ticker", "target_weight"]].merge(
        latest_signals, on="ticker", how="left"
    )

    # Assets without signals keep base weight (rank = 0.5 → no tilt)
    df["composite_rank"] = df["composite_rank"].fillna(0.5)

    # Tilt multiplier: rank 1.0 → (1 + tilt*0.5), rank 0.0 → (1 - tilt*0.5)
    df["tilt_multiplier"] = 1.0 + tilt_strength * (df["composite_rank"] - 0.5)

    df["base_weight"] = df["target_weight"]
    df["target_weight"] = df["base_weight"] * df["tilt_multiplier"]

    # Re-normalize
    total = df["target_weight"].sum()
    if total > 0:
        df["target_weight"] = df["target_weight"] / total

    df["strategy"] = "signal_tilted_hrp"

    logger.info(f"Signal-tilted allocation (tilt={tilt_strength:.0%}):")
    for _, row in df.sort_values("target_weight", ascending=False).iterrows():
        logger.info(
            f"  {row['ticker']:15s}  "
            f"base={row['base_weight']:.2%}  "
            f"rank={row['composite_rank']:.2f}  "
            f"tilt={row['tilt_multiplier']:.3f}  "
            f"final={row['target_weight']:.2%}"
        )

    return df[["ticker", "base_weight", "composite_rank", "tilt_multiplier",
               "target_weight", "strategy"]]


# ── ML Alpha Integration ────────────────────────────────────────────────────


def build_alpha_tilted_portfolio(
    base_weights: pd.DataFrame,
    alpha_scores: pd.DataFrame,
    regime_behavior=None,
    max_tilt_strength: float = 0.30,
    min_confidence_to_tilt: float = 0.40,
) -> pd.DataFrame:
    """
    Blend HRP baseline with ML alpha scores and regime behavior.

    Formula:
        final_weight = hrp_weight × (1 + tilt_strength × alpha_score × confidence)

    Where:
        - alpha_score: cross-sectional rank from ML ensemble (0-1)
        - confidence: composite confidence from confidence layer (0-1)
        - tilt_strength: from regime behavior or config

    The ML TILTS allocations — it does NOT determine them.

    Parameters
    ----------
    base_weights : DataFrame with ticker, target_weight (from HRP)
    alpha_scores : DataFrame with ticker, alpha_score, composite_confidence, rank
    regime_behavior : Optional RegimeBehavior from regime engine
    max_tilt_strength : Maximum tilt strength cap
    min_confidence_to_tilt : Don't tilt if confidence below this

    Returns
    -------
    DataFrame: ticker | base_weight | alpha_score | confidence |
               tilt_multiplier | target_weight | strategy
    """
    # Use latest alpha scores
    alpha = alpha_scores.copy()
    if "date" in alpha.columns:
        latest = alpha["date"].max()
        alpha = alpha[alpha["date"] == latest]

    # Determine tilt strength
    if regime_behavior is not None:
        tilt_strength = min(regime_behavior.tilt_strength, max_tilt_strength)
    else:
        tilt_strength = 0.20

    # Merge base weights with alpha scores
    confidence_col = "composite_confidence" if "composite_confidence" in alpha.columns else "model_confidence"
    alpha_merge = alpha[["ticker", "alpha_score", confidence_col, "rank"]].copy()
    alpha_merge = alpha_merge.rename(columns={confidence_col: "confidence"})

    df = base_weights[["ticker", "target_weight"]].merge(
        alpha_merge, on="ticker", how="left"
    )

    # Fill missing: no alpha = no tilt
    df["alpha_score"] = df["alpha_score"].fillna(0.5)
    df["confidence"] = df["confidence"].fillna(0.5)
    df["rank"] = df["rank"].fillna(0.5)

    # Apply confidence gate
    df["effective_alpha"] = np.where(
        df["confidence"] >= min_confidence_to_tilt,
        df["alpha_score"] - 0.5,  # center around 0
        0.0,
    )

    # Tilt multiplier: alpha × confidence × strength
    df["tilt_multiplier"] = 1.0 + tilt_strength * df["effective_alpha"] * df["confidence"]

    df["base_weight"] = df["target_weight"]
    df["target_weight"] = df["base_weight"] * df["tilt_multiplier"]

    # Re-normalize
    total = df["target_weight"].sum()
    if total > 0:
        df["target_weight"] = df["target_weight"] / total

    # Apply regime constraints if available
    if regime_behavior is not None:
        from regimes.behavior import apply_regime_constraints
        try:
            asset_types_path = Path("configs/asset_master.csv")
            if asset_types_path.exists():
                import csv
                asset_types = {}
                with open(asset_types_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        asset_types[row.get("ticker", "")] = row.get("asset_type", "equity")
            else:
                asset_types = {t: "equity" for t in df["ticker"]}

            weights_dict = dict(zip(df["ticker"], df["target_weight"]))
            constrained = apply_regime_constraints(weights_dict, regime_behavior, asset_types)
            df["target_weight"] = df["ticker"].map(constrained)
        except Exception as exc:
            logger.warning(f"Regime constraint application failed: {exc}")

    df["strategy"] = "alpha_tilted_hrp"

    logger.info(f"Alpha-tilted allocation (tilt={tilt_strength:.0%}, "
                f"min_conf={min_confidence_to_tilt:.0%}):")
    for _, row in df.sort_values("target_weight", ascending=False).iterrows():
        logger.info(
            f"  {row['ticker']:15s}  "
            f"base={row['base_weight']:.2%}  "
            f"alpha={row['alpha_score']:.2f}  "
            f"conf={row['confidence']:.2f}  "
            f"tilt={row['tilt_multiplier']:.3f}  "
            f"final={row['target_weight']:.2%}"
        )

    return df[["ticker", "base_weight", "alpha_score", "confidence",
               "tilt_multiplier", "target_weight", "strategy"]]
