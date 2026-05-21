"""
Deployment — Long-Horizon Walk-Forward Evaluation.

Sprint 8: Multi-year survivability testing.
  - Walk-forward over extended periods
  - Comparison vs baselines (buy & hold, equal weight, risk parity)
  - Crisis-period performance analysis
  - Portfolio survivability metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class WalkForwardResult:
    """Walk-forward evaluation result."""
    strategy: str
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float
    total_return: float
    n_years: float
    n_trades: int = 0


class WalkForwardEvaluator:
    """
    Long-horizon walk-forward evaluator.

    Tests portfolio strategy survivability over extended time periods
    and compares against standard baselines.
    """

    def __init__(self):
        cfg = _load_config().get("walkforward", {})
        self._min_years = cfg.get("min_years", 3)
        self._benchmarks = cfg.get("benchmarks", ["buy_and_hold", "equal_weight", "risk_parity"])
        self._results: list[WalkForwardResult] = []

    def evaluate_nav(
        self,
        nav_series: pd.Series,
        strategy_name: str = "portfolio",
        trading_days_per_year: int = 252,
    ) -> WalkForwardResult:
        """
        Evaluate a NAV series for long-horizon performance.

        Args:
            nav_series: Daily NAV values (indexed by date)
            strategy_name: Name of the strategy
        """
        if nav_series.empty or len(nav_series) < 20:
            return WalkForwardResult(
                strategy=strategy_name,
                cagr=0, sharpe=0, max_drawdown=0, calmar=0,
                total_return=0, n_years=0,
            )

        returns = nav_series.pct_change().dropna()
        n_days = len(returns)
        n_years = n_days / trading_days_per_year

        # CAGR
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0

        # Sharpe
        mean_ret = returns.mean() * trading_days_per_year
        std_ret = returns.std() * np.sqrt(trading_days_per_year)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0

        # Max drawdown
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = abs(drawdown.min())

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else 0

        result = WalkForwardResult(
            strategy=strategy_name,
            cagr=cagr,
            sharpe=sharpe,
            max_drawdown=max_dd,
            calmar=calmar,
            total_return=total_return,
            n_years=n_years,
        )
        self._results.append(result)
        return result

    def generate_baseline_buy_and_hold(
        self,
        prices_wide: pd.DataFrame,
    ) -> pd.Series:
        """Generate buy-and-hold equal-weight baseline NAV."""
        if prices_wide.empty:
            return pd.Series(dtype=float)

        returns = prices_wide.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
        nav = (1 + portfolio_returns).cumprod() * 1_000_000
        return nav

    def generate_baseline_equal_weight(
        self,
        prices_wide: pd.DataFrame,
        rebalance_freq: str = "ME",
    ) -> pd.Series:
        """Generate equal-weight rebalanced baseline NAV."""
        if prices_wide.empty:
            return pd.Series(dtype=float)

        returns = prices_wide.pct_change().dropna()
        n_assets = returns.shape[1]

        # Simple: equal-weight returns
        portfolio_returns = returns.mean(axis=1)
        nav = (1 + portfolio_returns).cumprod() * 1_000_000
        return nav

    def generate_baseline_risk_parity(
        self,
        prices_wide: pd.DataFrame,
        lookback: int = 60,
    ) -> pd.Series:
        """Generate inverse-volatility weighted baseline NAV."""
        if prices_wide.empty:
            return pd.Series(dtype=float)

        returns = prices_wide.pct_change().dropna()

        # Rolling inverse vol weights
        nav_values = [1_000_000.0]
        for i in range(lookback, len(returns)):
            window = returns.iloc[i - lookback:i]
            vols = window.std()
            vols = vols.replace(0, vols.mean())  # avoid division by zero
            inv_vols = 1 / vols
            weights = inv_vols / inv_vols.sum()
            day_return = (returns.iloc[i] * weights).sum()
            nav_values.append(nav_values[-1] * (1 + day_return))

        idx = returns.index[lookback - 1:]
        return pd.Series(nav_values, index=idx)

    def run_comparison(
        self,
        nav_series: pd.Series,
        prices_wide: pd.DataFrame | None = None,
    ) -> list[WalkForwardResult]:
        """
        Run walk-forward evaluation of portfolio vs baselines.
        """
        self._results.clear()

        # Portfolio strategy
        self.evaluate_nav(nav_series, "portfolio")

        # Baselines
        if prices_wide is not None and not prices_wide.empty:
            if "buy_and_hold" in self._benchmarks:
                bh_nav = self.generate_baseline_buy_and_hold(prices_wide)
                self.evaluate_nav(bh_nav, "buy_and_hold")

            if "equal_weight" in self._benchmarks:
                ew_nav = self.generate_baseline_equal_weight(prices_wide)
                self.evaluate_nav(ew_nav, "equal_weight")

            if "risk_parity" in self._benchmarks:
                rp_nav = self.generate_baseline_risk_parity(prices_wide)
                if not rp_nav.empty:
                    self.evaluate_nav(rp_nav, "risk_parity")

        return list(self._results)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "strategy": r.strategy,
                "cagr": r.cagr,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "calmar": r.calmar,
                "total_return": r.total_return,
                "n_years": r.n_years,
            }
            for r in self._results
        ])

    def summary(self) -> dict:
        portfolio = next((r for r in self._results if r.strategy == "portfolio"), None)
        return {
            "portfolio_cagr": portfolio.cagr if portfolio else None,
            "portfolio_sharpe": portfolio.sharpe if portfolio else None,
            "n_strategies": len(self._results),
            "sufficient_data": (portfolio.n_years >= self._min_years) if portfolio else False,
        }
