"""
Allocation reporting — explains WHY allocations look the way they do.

Produces a portfolio recommendation CSV and human-readable summary.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


REPORTS_DIR = Path("reports")


def generate_allocation_report(
    strategies: dict[str, pd.DataFrame],
    final_weights: pd.DataFrame,
    turnover_df: pd.DataFrame | None = None,
    rebalance_decision: dict | None = None,
) -> pd.DataFrame:
    """
    Build a comprehensive allocation report comparing strategies
    and explaining the final target portfolio.

    Parameters
    ----------
    strategies : dict
        Strategy name → weights DataFrame (ticker, target_weight).
    final_weights : pd.DataFrame
        The chosen portfolio (ticker, target_weight, strategy).
    turnover_df : pd.DataFrame, optional
        Turnover analysis output.
    rebalance_decision : dict, optional
        Output from should_rebalance().

    Returns
    -------
    pd.DataFrame
        Comparison table across all strategies.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build strategy comparison
    comparison = None
    for name, wdf in strategies.items():
        col = wdf[["ticker", "target_weight"]].rename(
            columns={"target_weight": name}
        )
        if comparison is None:
            comparison = col
        else:
            comparison = comparison.merge(col, on="ticker", how="outer")

    if comparison is not None:
        comparison = comparison.fillna(0.0)
        comparison = comparison.sort_values("ticker")

        # Save comparison
        path = REPORTS_DIR / "strategy_comparison.csv"
        comparison.to_csv(path, index=False)
        logger.info(f"Strategy comparison → {path}")

    # Build portfolio recommendation
    rec = _build_recommendation(final_weights, turnover_df)
    rec_path = REPORTS_DIR / "portfolio_recommendation.csv"
    rec.to_csv(rec_path, index=False)
    logger.info(f"Portfolio recommendation → {rec_path}")

    # Log summary
    _log_report(strategies, final_weights, turnover_df, rebalance_decision)

    return comparison if comparison is not None else pd.DataFrame()


def _build_recommendation(
    final_weights: pd.DataFrame,
    turnover_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the portfolio_recommendation.csv."""
    rec = final_weights[["ticker", "target_weight"]].copy()

    if turnover_df is not None and "current_weight" in turnover_df.columns:
        rec = rec.merge(
            turnover_df[["ticker", "current_weight", "direction"]],
            on="ticker",
            how="left",
        )
        rec["current_weight"] = rec["current_weight"].fillna(0.0)
        rec["action"] = rec["direction"].fillna("HOLD")
        rec = rec.drop(columns=["direction"])
    else:
        rec["current_weight"] = 0.0
        rec["action"] = "NEW"

    rec["weight_change"] = rec["target_weight"] - rec["current_weight"]

    # Format as percentages for readability
    for col in ["current_weight", "target_weight", "weight_change"]:
        rec[col] = rec[col].round(4)

    return rec[["ticker", "current_weight", "target_weight", "weight_change", "action"]]


def _log_report(
    strategies: dict[str, pd.DataFrame],
    final_weights: pd.DataFrame,
    turnover_df: pd.DataFrame | None,
    rebalance_decision: dict | None,
) -> None:
    """Log a human-readable allocation report."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALLOCATION REPORT")
    logger.info("=" * 60)

    # Strategy comparison
    logger.info("\n  Strategy Comparison:")
    logger.info(f"  {'Ticker':15s}", end="")
    for name in strategies:
        logger.info(f"  {name:>12s}", end="")
    logger.info("")

    tickers = set()
    for wdf in strategies.values():
        tickers.update(wdf["ticker"].tolist())

    for t in sorted(tickers):
        line = f"  {t:15s}"
        for name, wdf in strategies.items():
            w = wdf[wdf["ticker"] == t]["target_weight"]
            val = w.iloc[0] if len(w) > 0 else 0.0
            line += f"  {val:>11.2%}"
        logger.info(line)

    # Final portfolio
    logger.info("\n  Final Target Portfolio:")
    strategy_name = final_weights["strategy"].iloc[0] if "strategy" in final_weights.columns else "unknown"
    logger.info(f"  Strategy: {strategy_name}")
    for _, row in final_weights.sort_values("target_weight", ascending=False).iterrows():
        logger.info(f"    {row['ticker']:15s}  {row['target_weight']:.2%}")

    total = final_weights["target_weight"].sum()
    logger.info(f"    {'TOTAL':15s}  {total:.2%}")

    # Turnover
    if turnover_df is not None and "total_turnover" in turnover_df.attrs:
        logger.info(f"\n  Turnover: {turnover_df.attrs['total_turnover']:.2%} (one-way)")

    # Rebalance
    if rebalance_decision is not None:
        status = "REBALANCE" if rebalance_decision["should_rebalance"] else "HOLD"
        logger.info(f"  Rebalance: {status} — {rebalance_decision['reason']}")
