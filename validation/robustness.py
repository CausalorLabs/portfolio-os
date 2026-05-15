"""
Parameter robustness testing — sweep key strategy parameters
and measure sensitivity.

Good strategies degrade gracefully when parameters change.
Fragile strategies collapse.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from loguru import logger

from backtests.engine import run_backtest
from validation.walkforward import _compute_window_metrics


def run_parameter_sensitivity(
    wide_prices: pd.DataFrame,
    strategy_fn_factory,
    param_grid: dict[str, list],
    initial_capital: float = 1_000_000.0,
    frequency: str = "quarterly",
    slippage_bps: float = 10,
    country_map: dict | None = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """
    Sweep parameters and evaluate strategy performance for each combination.

    Parameters
    ----------
    wide_prices : pd.DataFrame
        Wide-format daily prices.
    strategy_fn_factory : callable
        Function(param_dict) → strategy_fn.
        The strategy_fn is the same signature as run_backtest expects:
        strategy_fn(returns, tickers) → dict[str, float].
    param_grid : dict
        Parameter name → list of values to test.
        Example: {"momentum_window": [20, 40, 60, 120]}
    initial_capital, frequency, slippage_bps, country_map, warmup_days
        Passed through to run_backtest.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination with performance metrics.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    logger.info(f"Parameter sensitivity: {len(combinations)} combinations")
    logger.info(f"  Parameters: {param_names}")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"  [{i + 1}/{len(combinations)}] {label}")

        try:
            strategy_fn = strategy_fn_factory(params)
            bt = run_backtest(
                wide_prices=wide_prices,
                strategy_fn=strategy_fn,
                initial_capital=initial_capital,
                frequency=frequency,
                slippage_bps=slippage_bps,
                country_map=country_map or {},
                warmup_days=warmup_days,
            )
            metrics = _compute_window_metrics(bt["nav_series"])
            n_trades = len(bt["rebalance_log"])
        except Exception as e:
            logger.warning(f"    Failed: {e}")
            metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                       "volatility": 0.0, "max_drawdown": 0.0}
            n_trades = 0

        row = {**params, **metrics, "n_rebalances": n_trades}
        results.append(row)

    df = pd.DataFrame(results)
    _log_sensitivity_summary(df, param_names)
    return df


def evaluate_stability_surface(
    sensitivity_results: pd.DataFrame,
    metric: str = "sharpe",
) -> dict:
    """
    Analyze the stability of a metric across parameter combinations.

    Returns
    -------
    dict
        stability_score (0–1), best_params, worst_params,
        mean, std, range, cv (coefficient of variation).
    """
    values = sensitivity_results[metric].dropna()

    if values.empty:
        return {"stability_score": 0.0}

    mean_val = values.mean()
    std_val = values.std()
    range_val = values.max() - values.min()
    cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")

    # Stability score: lower CV = more stable (cap at 1.0)
    stability = max(0.0, 1.0 - cv)

    best_idx = values.idxmax()
    worst_idx = values.idxmin()

    result = {
        "metric": metric,
        "mean": mean_val,
        "std": std_val,
        "range": range_val,
        "cv": cv,
        "stability_score": stability,
        "best_value": values.max(),
        "worst_value": values.min(),
        "best_params": sensitivity_results.iloc[best_idx].to_dict() if best_idx is not None else {},
        "worst_params": sensitivity_results.iloc[worst_idx].to_dict() if worst_idx is not None else {},
        "positive_pct": (values > 0).mean(),
    }

    logger.info(f"\nStability Surface ({metric}):")
    logger.info(f"  Mean:     {mean_val:.4f}")
    logger.info(f"  Std:      {std_val:.4f}")
    logger.info(f"  CV:       {cv:.2f}")
    logger.info(f"  Score:    {stability:.2f}/1.00")
    logger.info(f"  Positive: {result['positive_pct']:.0%}")

    return result


def _log_sensitivity_summary(df: pd.DataFrame, param_names: list[str]) -> None:
    """Log parameter sensitivity summary."""
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER SENSITIVITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Combinations tested: {len(df)}")

    for metric in ["sharpe", "cagr", "max_drawdown"]:
        if metric in df.columns:
            logger.info(
                f"  {metric:15s}: mean={df[metric].mean():.4f}, "
                f"std={df[metric].std():.4f}, "
                f"range=[{df[metric].min():.4f}, {df[metric].max():.4f}]"
            )
