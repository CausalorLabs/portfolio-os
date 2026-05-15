"""
Monte Carlo simulation engine — stress portfolio uncertainty
through randomized return paths and bootstrapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def generate_bootstrap_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    n_days: int = 252,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate bootstrap return paths using block bootstrapping.

    Block bootstrap preserves autocorrelation structure better than
    i.i.d. sampling.

    Parameters
    ----------
    returns : pd.Series
        Historical daily returns.
    n_paths : int
        Number of simulated paths.
    n_days : int
        Length of each path (trading days).
    block_size : int
        Size of contiguous blocks to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_days) — each row is a simulated return path.
    """
    rng = np.random.default_rng(seed)
    ret_arr = returns.dropna().values
    n = len(ret_arr)

    if n < block_size:
        block_size = max(1, n // 2)

    n_blocks = int(np.ceil(n_days / block_size))
    paths = np.zeros((n_paths, n_days))

    for i in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size)
            blocks.append(ret_arr[start:start + block_size])
        path = np.concatenate(blocks)[:n_days]
        paths[i] = path

    logger.info(f"Monte Carlo: {n_paths} paths × {n_days} days "
                f"(block_size={block_size})")
    return paths


def run_monte_carlo_simulation(
    returns: pd.Series,
    initial_value: float = 1_000_000.0,
    n_paths: int = 1000,
    n_days: int = 252,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation and compute distribution statistics.

    Returns
    -------
    dict
        Keys: paths (NAV paths), return_distribution, drawdown_distribution,
              percentiles, summary stats.
    """
    bootstrap_returns = generate_bootstrap_paths(
        returns, n_paths, n_days, block_size, seed
    )

    # Convert returns to NAV paths
    cum_returns = np.cumprod(1 + bootstrap_returns, axis=1)
    nav_paths = initial_value * cum_returns

    # Terminal values
    terminal_values = nav_paths[:, -1]
    terminal_returns = terminal_values / initial_value - 1

    # Max drawdowns per path
    max_drawdowns = np.zeros(n_paths)
    for i in range(n_paths):
        running_max = np.maximum.accumulate(nav_paths[i])
        drawdowns = (nav_paths[i] - running_max) / running_max
        max_drawdowns[i] = drawdowns.min()

    # Percentiles
    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return_pctiles = {f"p{p}": np.percentile(terminal_returns, p) for p in pctiles}
    dd_pctiles = {f"p{p}": np.percentile(max_drawdowns, p) for p in pctiles}

    # Summary
    summary = {
        "n_paths": n_paths,
        "n_days": n_days,
        "initial_value": initial_value,
        "mean_return": terminal_returns.mean(),
        "median_return": np.median(terminal_returns),
        "std_return": terminal_returns.std(),
        "mean_terminal": terminal_values.mean(),
        "worst_terminal": terminal_values.min(),
        "best_terminal": terminal_values.max(),
        "prob_loss": (terminal_returns < 0).mean(),
        "prob_gt_10pct": (terminal_returns > 0.10).mean(),
        "mean_max_dd": max_drawdowns.mean(),
        "worst_max_dd": max_drawdowns.min(),
        "cvar_5pct": terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)].mean(),
    }

    result = {
        "nav_paths": nav_paths,
        "terminal_returns": terminal_returns,
        "max_drawdowns": max_drawdowns,
        "return_percentiles": return_pctiles,
        "drawdown_percentiles": dd_pctiles,
        "summary": summary,
    }

    _log_mc_summary(summary, return_pctiles, dd_pctiles)
    return result


def _log_mc_summary(summary: dict, ret_pct: dict, dd_pct: dict) -> None:
    """Log Monte Carlo results."""
    logger.info("\n" + "=" * 60)
    logger.info("MONTE CARLO SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Paths:              {summary['n_paths']}")
    logger.info(f"  Horizon:            {summary['n_days']} days")
    logger.info(f"  Mean Return:        {summary['mean_return']:+.2%}")
    logger.info(f"  Median Return:      {summary['median_return']:+.2%}")
    logger.info(f"  Prob of Loss:       {summary['prob_loss']:.1%}")
    logger.info(f"  CVaR (5%):          {summary['cvar_5pct']:+.2%}")
    logger.info(f"  Mean Max DD:        {summary['mean_max_dd']:+.2%}")
    logger.info(f"  Worst Max DD:       {summary['worst_max_dd']:+.2%}")

    logger.info("\n  Return Distribution:")
    for k, v in ret_pct.items():
        logger.info(f"    {k:>4s}: {v:+.2%}")

    logger.info("\n  Drawdown Distribution:")
    for k, v in dd_pct.items():
        logger.info(f"    {k:>4s}: {v:+.2%}")
