"""
Signal decay analysis — measure how long signals remain predictive.

Computes information coefficient (IC) and forward-return-based metrics
at multiple horizons to detect alpha half-life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def evaluate_forward_returns(
    signal_scores: pd.DataFrame,
    wide_prices: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute forward returns at multiple horizons for each signal date.

    Parameters
    ----------
    signal_scores : pd.DataFrame
        Columns: date, ticker, composite_score, composite_rank.
    wide_prices : pd.DataFrame
        Wide-format daily prices (columns = tickers, index = dates).
    horizons : list of int
        Forward-looking horizons in trading days. Default: [5, 10, 20, 40, 60].

    Returns
    -------
    pd.DataFrame
        Signal scores merged with forward returns at each horizon.
    """
    if horizons is None:
        horizons = [5, 10, 20, 40, 60]

    wide_returns = wide_prices.pct_change()
    signal_dates = sorted(signal_scores["date"].unique())

    records = []
    for date in signal_dates:
        day_signals = signal_scores[signal_scores["date"] == date]
        if date not in wide_prices.index:
            continue

        date_idx = wide_prices.index.get_loc(date)

        for _, row in day_signals.iterrows():
            ticker = row["ticker"]
            if ticker not in wide_prices.columns:
                continue

            record = {
                "date": date,
                "ticker": ticker,
                "composite_score": row["composite_score"],
                "composite_rank": row["composite_rank"],
            }

            for h in horizons:
                end_idx = date_idx + h
                if end_idx >= len(wide_prices):
                    record[f"fwd_return_{h}d"] = np.nan
                else:
                    start_price = wide_prices.iloc[date_idx][ticker]
                    end_price = wide_prices.iloc[end_idx][ticker]
                    if start_price > 0:
                        record[f"fwd_return_{h}d"] = (end_price / start_price) - 1
                    else:
                        record[f"fwd_return_{h}d"] = np.nan

            records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Forward returns: {len(df)} signal-date observations, "
                f"horizons={horizons}")
    return df


def calculate_signal_decay(
    forward_returns: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute information coefficient (IC) at each horizon.

    IC = rank correlation between signal rank and forward return.
    Higher IC = more predictive signal. IC decay shows alpha half-life.

    Parameters
    ----------
    forward_returns : pd.DataFrame
        Output of evaluate_forward_returns().
    horizons : list of int, optional
        Horizons to analyze.

    Returns
    -------
    pd.DataFrame
        One row per horizon: horizon, ic_mean, ic_std, ic_ir, hit_rate, avg_spread.
    """
    if horizons is None:
        horizons = [5, 10, 20, 40, 60]

    results = []
    for h in horizons:
        col = f"fwd_return_{h}d"
        if col not in forward_returns.columns:
            continue

        valid = forward_returns[["date", "composite_rank", col]].dropna()
        if len(valid) < 20:
            continue

        # IC per date, then aggregate
        ic_by_date = valid.groupby("date").apply(
            lambda g: g["composite_rank"].corr(g[col], method="spearman")
            if len(g) > 2 else np.nan,
            include_groups=False,
        ).dropna()

        if ic_by_date.empty:
            continue

        ic_mean = ic_by_date.mean()
        ic_std = ic_by_date.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Hit rate: fraction of dates where IC > 0
        hit_rate = (ic_by_date > 0).mean()

        # Avg long-short spread
        top_half = valid[valid["composite_rank"] >= 0.5][col].mean()
        bot_half = valid[valid["composite_rank"] < 0.5][col].mean()
        avg_spread = top_half - bot_half if not (np.isnan(top_half) or np.isnan(bot_half)) else 0.0

        results.append({
            "horizon": h,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "hit_rate": hit_rate,
            "avg_spread": avg_spread,
            "n_observations": len(ic_by_date),
        })

    df = pd.DataFrame(results)
    _log_decay(df)
    return df


def _log_decay(df: pd.DataFrame) -> None:
    """Log signal decay analysis."""
    if df.empty:
        logger.warning("No signal decay results")
        return

    logger.info("\nSignal Decay Analysis:")
    logger.info(f"  {'Horizon':>8s}  {'IC Mean':>8s}  {'IC IR':>8s}  "
                f"{'Hit Rate':>8s}  {'Spread':>8s}")
    logger.info("  " + "-" * 50)

    for _, row in df.iterrows():
        logger.info(
            f"  {row['horizon']:>5}D    {row['ic_mean']:>+8.4f}  "
            f"{row['ic_ir']:>8.3f}  {row['hit_rate']:>7.0%}  "
            f"{row['avg_spread']:>+8.4f}"
        )

    # Identify decay point
    if len(df) >= 2:
        peak_ic = df["ic_mean"].abs().max()
        peak_h = df.loc[df["ic_mean"].abs().idxmax(), "horizon"]
        last_positive = df[df["ic_mean"] > 0]["horizon"].max() if (df["ic_mean"] > 0).any() else 0
        logger.info(f"\n  Peak IC at {peak_h}D ({peak_ic:.4f})")
        logger.info(f"  Signal positive through {last_positive}D")

        if last_positive <= 20:
            logger.warning("  ⚠ Short signal half-life — frequent rebalancing needed")
        elif last_positive >= 60:
            logger.info("  ✓ Signal persistence is good — supports quarterly rebalancing")
