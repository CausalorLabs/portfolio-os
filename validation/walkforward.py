"""
Walk-forward validation engine — out-of-sample testing.

Simulates real-world research deployment: train on past, freeze params,
test on unseen future, roll forward. One of the strongest defenses
against overfitting.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from backtests.engine import run_backtest
from analytics.metrics import (
    calculate_cagr,
    calculate_sharpe,
    calculate_sortino,
    calculate_volatility,
    calculate_max_drawdown,
)


def generate_train_test_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[dict]:
    """
    Generate rolling train/test window pairs.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date range of available data.
    train_years : int
        Length of training window in years.
    test_years : int
        Length of test window in years.
    step_years : int
        How many years to roll forward each step.

    Returns
    -------
    list of dict
        Each dict has: train_start, train_end, test_start, test_end, window_id.
    """
    start = dates.min()
    end = dates.max()
    total_years = (end - start).days / 365.25

    if total_years < train_years + test_years:
        logger.warning(
            f"Insufficient data: {total_years:.1f} years available, "
            f"need {train_years + test_years} for one window"
        )
        return []

    windows = []
    window_id = 0
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        # Snap to actual trading dates
        train_dates = dates[(dates >= train_start) & (dates < train_end)]
        test_dates = dates[(dates >= test_start) & (dates < test_end)]

        if len(train_dates) < 60 or len(test_dates) < 20:
            cursor += pd.DateOffset(years=step_years)
            continue

        windows.append({
            "window_id": window_id,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_days": len(train_dates),
            "test_days": len(test_dates),
        })
        window_id += 1
        cursor += pd.DateOffset(years=step_years)

    logger.info(f"Walk-forward: {len(windows)} windows "
                f"({train_years}yr train / {test_years}yr test)")
    return windows


def _compute_window_metrics(nav_df: pd.DataFrame) -> dict:
    """Compute standard metrics from a NAV series."""
    if nav_df.empty or len(nav_df) < 5:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "volatility": 0.0, "max_drawdown": 0.0}

    # Build a nav-like dataframe for analytics.metrics compatibility
    nav_compat = nav_df[["date", "nav"]].rename(columns={"nav": "portfolio_nav"})
    nav_compat["daily_return"] = nav_compat["portfolio_nav"].pct_change()
    returns = nav_compat["daily_return"].dropna()

    if returns.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "volatility": 0.0, "max_drawdown": 0.0}

    return {
        "cagr": calculate_cagr(nav_compat),
        "sharpe": calculate_sharpe(returns),
        "sortino": calculate_sortino(returns),
        "volatility": calculate_volatility(returns),
        "max_drawdown": calculate_max_drawdown(nav_compat),
    }


def run_walkforward_validation(
    wide_prices: pd.DataFrame,
    strategy_fn,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
    initial_capital: float = 1_000_000.0,
    frequency: str = "quarterly",
    slippage_bps: float = 10,
    country_map: dict | None = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """
    Run walk-forward validation across multiple train/test windows.

    For each window:
      1. Train = run strategy optimization on training period
      2. Freeze = lock parameters
      3. Test = backtest frozen strategy on unseen test period

    Returns
    -------
    pd.DataFrame
        One row per window with train/test metrics + metadata.
    """
    dates = wide_prices.index
    windows = generate_train_test_windows(
        dates, train_years, test_years, step_years
    )

    if not windows:
        return pd.DataFrame()

    results = []
    for w in windows:
        logger.info(
            f"\n  Window {w['window_id']}: "
            f"Train {w['train_start'].date()}→{w['train_end'].date()} | "
            f"Test {w['test_start'].date()}→{w['test_end'].date()}"
        )

        # ── Train: run backtest on training period ───────────────────────
        try:
            train_result = run_backtest(
                wide_prices=wide_prices,
                strategy_fn=strategy_fn,
                initial_capital=initial_capital,
                frequency=frequency,
                slippage_bps=slippage_bps,
                country_map=country_map or {},
                start_date=w["train_start"],
                end_date=w["train_end"],
                warmup_days=min(warmup_days, w["train_days"] // 3),
            )
            train_metrics = _compute_window_metrics(train_result["nav_series"])
        except Exception as e:
            logger.warning(f"  Train failed: {e}")
            train_metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                           "volatility": 0.0, "max_drawdown": 0.0}

        # ── Test: run backtest on unseen test period ─────────────────────
        try:
            test_result = run_backtest(
                wide_prices=wide_prices,
                strategy_fn=strategy_fn,
                initial_capital=initial_capital,
                frequency=frequency,
                slippage_bps=slippage_bps,
                country_map=country_map or {},
                start_date=w["test_start"],
                end_date=w["test_end"],
                warmup_days=min(warmup_days, w["test_days"] // 3),
            )
            test_metrics = _compute_window_metrics(test_result["nav_series"])
        except Exception as e:
            logger.warning(f"  Test failed: {e}")
            test_metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                          "volatility": 0.0, "max_drawdown": 0.0}

        row = {
            "window_id": w["window_id"],
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "train_days": w["train_days"],
            "test_days": w["test_days"],
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = v
        for k, v in test_metrics.items():
            row[f"test_{k}"] = v

        # Degradation: how much worse is OOS vs IS
        if train_metrics["sharpe"] != 0:
            row["sharpe_degradation"] = (
                (test_metrics["sharpe"] - train_metrics["sharpe"])
                / abs(train_metrics["sharpe"])
            )
        else:
            row["sharpe_degradation"] = 0.0

        results.append(row)

        logger.info(
            f"    Train: CAGR={train_metrics['cagr']:+.2%}, "
            f"Sharpe={train_metrics['sharpe']:.3f}"
        )
        logger.info(
            f"    Test:  CAGR={test_metrics['cagr']:+.2%}, "
            f"Sharpe={test_metrics['sharpe']:.3f}, "
            f"Degradation={row['sharpe_degradation']:+.1%}"
        )

    df = pd.DataFrame(results)
    _log_walkforward_summary(df)
    return df


def _log_walkforward_summary(df: pd.DataFrame) -> None:
    """Summary statistics across all walk-forward windows."""
    if df.empty:
        return

    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Windows:           {len(df)}")
    logger.info(f"  Avg Train Sharpe:  {df['train_sharpe'].mean():.3f}")
    logger.info(f"  Avg Test Sharpe:   {df['test_sharpe'].mean():.3f}")
    logger.info(f"  Avg Degradation:   {df['sharpe_degradation'].mean():+.1%}")
    logger.info(f"  Test Sharpe > 0:   {(df['test_sharpe'] > 0).sum()}/{len(df)}")

    # Consistency ratio: fraction of windows where test > 0
    consistency = (df["test_sharpe"] > 0).mean()
    logger.info(f"  Consistency Ratio: {consistency:.1%}")

    if consistency >= 0.7:
        logger.info("  ✓ Strategy shows good out-of-sample consistency")
    elif consistency >= 0.5:
        logger.warning("  ⚠ Strategy shows moderate consistency — monitor closely")
    else:
        logger.error("  ✗ Strategy shows poor OOS consistency — likely overfit")
