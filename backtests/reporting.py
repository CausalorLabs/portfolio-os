"""
Backtest reporting — generates professional strategy comparison
reports with friction metrics.
"""

from pathlib import Path

import pandas as pd
from loguru import logger


REPORTS_DIR = Path("reports")


def generate_backtest_report(
    comparison: pd.DataFrame,
    attribution: dict,
    ledger_summary: dict,
) -> None:
    """
    Generate backtest reports: CSV comparison + summary log.

    Parameters
    ----------
    comparison : pd.DataFrame
        Strategy comparison table from compare_backtest_results().
    attribution : dict
        Performance attribution for the primary strategy.
    ledger_summary : dict
        Aggregate ledger stats.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save strategy comparison
    if not comparison.empty:
        path = REPORTS_DIR / "backtest_comparison.csv"
        comparison.to_csv(path)
        logger.info(f"Backtest comparison → {path}")

    # Save attribution
    attr_df = pd.DataFrame([attribution])
    attr_path = REPORTS_DIR / "backtest_attribution.csv"
    attr_df.to_csv(attr_path, index=False)
    logger.info(f"Attribution report → {attr_path}")

    _log_backtest_summary(comparison, attribution, ledger_summary)


def _log_backtest_summary(
    comparison: pd.DataFrame,
    attribution: dict,
    ledger_summary: dict,
) -> None:
    """Log a comprehensive backtest summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTEST REPORT")
    logger.info("=" * 60)

    if not comparison.empty:
        logger.info("\n  Strategy Comparison:")
        logger.info(
            f"  {'Strategy':20s} {'CAGR':>8s} {'Sharpe':>8s} "
            f"{'MaxDD':>8s} {'Friction':>10s} {'Final NAV':>14s}"
        )
        logger.info("  " + "-" * 72)
        for name, row in comparison.iterrows():
            logger.info(
                f"  {name:20s} {row['cagr']:>+7.2%} {row['sharpe']:>8.3f} "
                f"{row['max_drawdown']:>+7.2%} "
                f"₹{row['total_friction']:>9,.0f} "
                f"₹{row['final_nav']:>13,.0f}"
            )

    if attribution:
        logger.info("\n  Primary Strategy Attribution:")
        logger.info(f"    Gross CAGR:     {attribution.get('gross_cagr', 0):+.2%}")
        logger.info(f"    Net CAGR:       {attribution.get('net_cagr', 0):+.2%}")
        logger.info(f"    Friction drag:  {attribution.get('friction_cagr_drag', 0):.2%}")

    if ledger_summary:
        logger.info(f"\n  Trade Ledger:")
        logger.info(f"    Total trades:   {ledger_summary.get('n_trades', 0)}")
        logger.info(f"    Total slippage: ₹{ledger_summary.get('total_slippage', 0):,.0f}")
        logger.info(f"    Total costs:    ₹{ledger_summary.get('total_costs', 0):,.0f}")
        logger.info(f"    Total taxes:    ₹{ledger_summary.get('total_taxes', 0):,.0f}")
        logger.info(f"    Net realized:   ₹{ledger_summary.get('total_realized_pnl', 0):,.0f}")
