"""
Signal ranker — simple composite ranking score for asset selection.

This is NOT ML. This is interpretable, weighted factor scoring.
"""

import pandas as pd
from loguru import logger


DEFAULT_WEIGHTS = {
    "factor_momentum": 0.4,
    "factor_trend": 0.3,
    "factor_low_vol": 0.3,
}


def calculate_composite_score(
    store: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute a weighted composite ranking score per asset per date.

    Parameters
    ----------
    store : pd.DataFrame
        Feature store (long format: date, ticker, feature, value).
    weights : dict
        Mapping of feature name → weight. Must sum to ~1.0.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, composite_score
        Plus individual factor columns for transparency.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    factor_names = list(weights.keys())
    factors = store[store["feature"].isin(factor_names)].copy()

    if factors.empty:
        logger.warning("No factor features found in store")
        return pd.DataFrame(columns=["date", "ticker", "composite_score"])

    # Pivot to wide
    wide = factors.pivot_table(
        index=["date", "ticker"],
        columns="feature",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Drop dates with missing factors
    wide = wide.dropna(subset=factor_names)

    # Compute weighted composite
    wide["composite_score"] = sum(
        wide[f] * w for f, w in weights.items() if f in wide.columns
    )

    # Rank per date (higher = better)
    wide["composite_rank"] = wide.groupby("date")["composite_score"].rank(
        ascending=True, pct=True
    )

    result = wide[["date", "ticker"] + factor_names + ["composite_score", "composite_rank"]]
    result = result.sort_values(["date", "composite_rank"], ascending=[True, False])

    _log_latest(result)
    return result


def _log_latest(scores: pd.DataFrame) -> None:
    """Log the most recent ranking."""
    latest_date = scores["date"].max()
    latest = scores[scores["date"] == latest_date].sort_values(
        "composite_rank", ascending=False
    )

    logger.info(f"Signal Ranking ({latest_date.date()}):")
    for _, row in latest.iterrows():
        logger.info(
            f"  {row['ticker']:15s}  "
            f"score={row['composite_score']:.3f}  "
            f"rank={row['composite_rank']:.2f}"
        )
