"""
Stress testing engine — simulate extreme market conditions.

Scenarios: COVID crash, INR depreciation, tech collapse, liquidity crunch.
Stresses: slippage, volatility, correlations, FX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from backtests.engine import run_backtest
from validation.walkforward import _compute_window_metrics


# ── Pre-defined stress scenarios ─────────────────────────────────────────────

STRESS_SCENARIOS = {
    "covid_crash": {
        "description": "Rapid drawdown (COVID-style, ~35% drop)",
        "return_shock": -0.03,  # Applied daily for duration
        "vol_multiplier": 3.0,
        "correlation_boost": 0.3,
        "duration_days": 25,
    },
    "inr_depreciation": {
        "description": "FX shock — INR depreciates 10% over 2 months",
        "return_shock": -0.002,  # Mild equity drag
        "vol_multiplier": 1.5,
        "correlation_boost": 0.1,
        "duration_days": 40,
    },
    "tech_collapse": {
        "description": "Sector rotation — tech-heavy assets drop 25%",
        "return_shock": -0.015,
        "vol_multiplier": 2.0,
        "correlation_boost": 0.2,
        "duration_days": 30,
    },
    "liquidity_crunch": {
        "description": "Spread widening, slippage 5x normal",
        "return_shock": -0.005,
        "vol_multiplier": 2.5,
        "correlation_boost": 0.4,
        "duration_days": 15,
    },
}


def run_stress_scenarios(
    wide_prices: pd.DataFrame,
    strategy_fn,
    scenarios: dict | None = None,
    initial_capital: float = 1_000_000.0,
    frequency: str = "quarterly",
    base_slippage_bps: float = 10,
    country_map: dict | None = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """
    Run stress test scenarios by applying synthetic shocks to price data.

    For each scenario, we perturb the last N days of the price history
    and re-run the backtest to measure impact.

    Returns
    -------
    pd.DataFrame
        One row per scenario with: baseline metrics, stressed metrics, impact.
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    # ── Baseline (no stress) ─────────────────────────────────────────────
    logger.info("Running baseline backtest...")
    baseline_bt = run_backtest(
        wide_prices=wide_prices,
        strategy_fn=strategy_fn,
        initial_capital=initial_capital,
        frequency=frequency,
        slippage_bps=base_slippage_bps,
        country_map=country_map or {},
        warmup_days=warmup_days,
    )
    baseline_metrics = _compute_window_metrics(baseline_bt["nav_series"])

    results = []
    for name, params in scenarios.items():
        logger.info(f"\nStress scenario: {name} — {params['description']}")

        stressed_prices = _apply_stress(wide_prices, params)
        stress_slippage = base_slippage_bps * (
            5 if name == "liquidity_crunch" else 2
        )

        try:
            stressed_bt = run_backtest(
                wide_prices=stressed_prices,
                strategy_fn=strategy_fn,
                initial_capital=initial_capital,
                frequency=frequency,
                slippage_bps=stress_slippage,
                country_map=country_map or {},
                warmup_days=warmup_days,
            )
            stressed_metrics = _compute_window_metrics(stressed_bt["nav_series"])
        except Exception as e:
            logger.warning(f"  Stress test failed: {e}")
            stressed_metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                               "volatility": 0.0, "max_drawdown": 0.0}

        row = {
            "scenario": name,
            "description": params["description"],
            "baseline_cagr": baseline_metrics["cagr"],
            "stressed_cagr": stressed_metrics["cagr"],
            "cagr_impact": stressed_metrics["cagr"] - baseline_metrics["cagr"],
            "baseline_sharpe": baseline_metrics["sharpe"],
            "stressed_sharpe": stressed_metrics["sharpe"],
            "sharpe_impact": stressed_metrics["sharpe"] - baseline_metrics["sharpe"],
            "baseline_max_dd": baseline_metrics["max_drawdown"],
            "stressed_max_dd": stressed_metrics["max_drawdown"],
            "dd_impact": stressed_metrics["max_drawdown"] - baseline_metrics["max_drawdown"],
            "stressed_vol": stressed_metrics["volatility"],
        }
        results.append(row)

        logger.info(
            f"  CAGR: {baseline_metrics['cagr']:+.2%} → {stressed_metrics['cagr']:+.2%} "
            f"({row['cagr_impact']:+.2%})"
        )
        logger.info(
            f"  MaxDD: {baseline_metrics['max_drawdown']:+.2%} → "
            f"{stressed_metrics['max_drawdown']:+.2%}"
        )

    df = pd.DataFrame(results)
    _log_stress_summary(df)
    return df


def simulate_liquidity_stress(
    wide_prices: pd.DataFrame,
    strategy_fn,
    slippage_levels: list[float] | None = None,
    initial_capital: float = 1_000_000.0,
    frequency: str = "quarterly",
    country_map: dict | None = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """
    Test strategy sensitivity to varying slippage levels.

    Returns
    -------
    pd.DataFrame
        One row per slippage level with performance metrics.
    """
    if slippage_levels is None:
        slippage_levels = [0, 5, 10, 20, 50, 100]

    results = []
    for bps in slippage_levels:
        logger.info(f"  Slippage={bps}bps")
        try:
            bt = run_backtest(
                wide_prices=wide_prices,
                strategy_fn=strategy_fn,
                initial_capital=initial_capital,
                frequency=frequency,
                slippage_bps=bps,
                country_map=country_map or {},
                warmup_days=warmup_days,
            )
            metrics = _compute_window_metrics(bt["nav_series"])
        except Exception as e:
            logger.warning(f"    Failed: {e}")
            metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                       "volatility": 0.0, "max_drawdown": 0.0}

        results.append({"slippage_bps": bps, **metrics})

    return pd.DataFrame(results)


def _apply_stress(
    wide_prices: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Apply a stress scenario to the price data."""
    stressed = wide_prices.copy()
    duration = params["duration_days"]
    shock = params["return_shock"]
    vol_mult = params["vol_multiplier"]

    # Apply shock to the last `duration` days
    n = len(stressed)
    stress_start = max(0, n - duration)

    for col in stressed.columns:
        prices = stressed[col].values.copy()
        for i in range(stress_start, n):
            noise = np.random.normal(0, abs(shock) * (vol_mult - 1))
            daily_shock = 1 + shock + noise
            prices[i] = prices[i - 1] * max(daily_shock, 0.8)  # Floor at -20%/day
        stressed[col] = prices

    return stressed


def _log_stress_summary(df: pd.DataFrame) -> None:
    """Log stress test summary."""
    logger.info("\n" + "=" * 60)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 60)
    for _, row in df.iterrows():
        logger.info(
            f"  {row['scenario']:20s}: "
            f"CAGR Δ={row['cagr_impact']:+.2%}, "
            f"Sharpe Δ={row['sharpe_impact']:+.3f}, "
            f"MaxDD={row['stressed_max_dd']:+.2%}"
        )
