"""
Signal-tilted allocator — blends risk-based allocation (HRP)
with signal strength (composite score) to produce a final
target portfolio.

Tilts are SMALL and CONTROLLED — this is NOT aggressive betting.
"""

import pandas as pd
from loguru import logger


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
