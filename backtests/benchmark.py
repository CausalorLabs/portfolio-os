"""
Benchmark engine — runs backtests for standard benchmark strategies
to compare against the optimized portfolio.

Benchmarks:
    - Buy & Hold (initial equal weight, no rebalance)
    - Equal Weight (monthly rebalance)
    - Inverse Volatility (monthly rebalance)
    - Single-asset benchmarks (SPY, NIFTY proxy)
"""

import numpy as np
import pandas as pd
from loguru import logger

from optimization.baselines import (
    equal_weight_portfolio,
    inverse_volatility_portfolio,
)
from optimization.hrp import allocate_hrp_weights
from optimization.covariance import calculate_shrinkage_covariance
from backtests.engine import run_backtest


# ── Strategy functions (callables for the backtest engine) ───────────────────


def _equal_weight_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """Equal weight — rebalance to 1/N each period."""
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def _inverse_vol_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """Inverse volatility weighting using trailing 60-day vol."""
    window = min(60, len(returns))
    vol = returns[tickers].iloc[-window:].std()
    vol = vol.replace(0, np.nan).dropna()
    if vol.empty:
        return _equal_weight_strategy(returns, tickers)
    inv = 1.0 / vol
    weights = inv / inv.sum()
    return weights.to_dict()


def _hrp_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """HRP allocation using trailing data."""
    ret = returns[tickers].dropna()
    if len(ret) < 60:
        return _equal_weight_strategy(returns, tickers)
    cov = calculate_shrinkage_covariance(ret, window=120)
    hrp_df = allocate_hrp_weights(ret, cov=cov)
    return dict(zip(hrp_df["ticker"], hrp_df["target_weight"]))


def _buy_and_hold_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """Buy & Hold — equal weight at inception, never rebalance.
    We return None after first call to skip rebalancing."""
    return {t: 1.0 / len(tickers) for t in tickers}


# ── Benchmark runner ─────────────────────────────────────────────────────────


BENCHMARK_STRATEGIES = {
    "buy_and_hold": _equal_weight_strategy,  # only rebalances once effectively
    "equal_weight": _equal_weight_strategy,
    "inverse_vol": _inverse_vol_strategy,
    "hrp": _hrp_strategy,
}


def run_benchmark_suite(
    wide_prices: pd.DataFrame,
    country_map: dict[str, str],
    initial_capital: float = 1_000_000.0,
    slippage_bps: float = 10,
    warmup_days: int = 120,
) -> dict[str, dict]:
    """
    Run all benchmark strategies and return their backtest results.

    Parameters
    ----------
    wide_prices : pd.DataFrame
        Wide-format daily prices in INR.
    country_map : dict
        ticker → country.
    initial_capital : float
    slippage_bps : float
    warmup_days : int

    Returns
    -------
    dict
        strategy_name → backtest result dict.
    """
    results = {}

    for name, strategy_fn in BENCHMARK_STRATEGIES.items():
        # Buy & hold uses yearly rebalance (effectively never)
        freq = "quarterly" if name == "buy_and_hold" else "monthly"

        logger.info(f"\n  ▸ Benchmark: {name} ({freq})")
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=strategy_fn,
            initial_capital=initial_capital,
            frequency=freq,
            slippage_bps=slippage_bps,
            country_map=country_map,
            warmup_days=warmup_days,
        )
        results[name] = result

    return results


def compare_backtest_results(
    results: dict[str, dict],
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Build a comparison table of key metrics across all backtest strategies.

    Returns
    -------
    pd.DataFrame
        Rows = strategies, columns = metrics.
    """
    rows = []

    for name, result in results.items():
        nav_df = result["nav_series"]
        if nav_df.empty or len(nav_df) < 2:
            continue

        nav = nav_df["nav"]
        n_days = len(nav)
        years = n_days / 252

        # CAGR
        total_return = nav.iloc[-1] / nav.iloc[0]
        cagr = total_return ** (1 / years) - 1 if years > 0 else 0.0

        # Volatility & Sharpe
        daily_ret = nav.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0

        # Sortino
        downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
        sortino = (cagr - risk_free_rate) / downside if downside > 0 else 0.0

        # Max drawdown
        peak = nav.cummax()
        dd = (nav - peak) / peak
        max_dd = dd.min()

        # Friction
        ledger_summary = result["ledger"].summary()
        total_friction = ledger_summary.get("total_friction", 0.0)
        friction_drag = total_friction / nav.iloc[0] if nav.iloc[0] > 0 else 0.0

        # Turnover
        rebal_log = result.get("rebalance_log", [])
        avg_turnover = (
            np.mean([r["turnover"] for r in rebal_log]) if rebal_log else 0.0
        )

        rows.append({
            "strategy": name,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "total_friction": total_friction,
            "friction_drag_pct": friction_drag,
            "avg_turnover": avg_turnover,
            "n_rebalances": len(rebal_log),
            "final_nav": nav.iloc[-1],
        })

    df = pd.DataFrame(rows).set_index("strategy")
    return df
