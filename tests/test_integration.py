"""
Integration tests — verify multi-module workflows work end-to-end.

These tests use synthetic data and exercise real cross-module paths
without hitting the network.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.metrics import calculate_cagr, calculate_sharpe, calculate_max_drawdown
from analytics.returns import build_returns_table
from analytics.drawdown import calculate_drawdown_series, calculate_drawdown_periods
from optimization.hrp import allocate_hrp_weights
from optimization.constraints import apply_weight_caps
from optimization.covariance import calculate_shrinkage_covariance
from backtests.engine import run_backtest
from validation.regimes import identify_market_regimes, evaluate_regime_performance
from validation.overfitting import detect_overfitting
from validation.monte_carlo import run_monte_carlo_simulation
from validation.diagnostics import generate_diagnostics
from validation.research_score import calculate_research_score


class TestAnalyticsPipeline:
    """Analytics chain: NAV → returns → metrics → drawdown."""

    def test_returns_to_metrics(self, nav_df):
        nav_df["daily_return"] = nav_df["portfolio_nav"].pct_change()
        returns_table = build_returns_table(nav_df)

        # Metrics should work on the output
        cagr = calculate_cagr(returns_table)
        assert isinstance(cagr, float)

        dd_series = calculate_drawdown_series(returns_table)
        assert "drawdown" in dd_series.columns
        assert (dd_series["drawdown"] <= 0).all()

        dd_periods = calculate_drawdown_periods(returns_table)
        assert isinstance(dd_periods, pd.DataFrame)


class TestOptimizationPipeline:
    """Optimization: returns → covariance → HRP → constraints → weights."""

    def test_full_allocation_chain(self, wide_returns):
        cov = calculate_shrinkage_covariance(wide_returns, window=120)
        hrp = allocate_hrp_weights(wide_returns, cov=cov)
        capped = apply_weight_caps(hrp, max_weight=0.40, min_weight=0.05)

        assert abs(capped["target_weight"].sum() - 1.0) < 1e-6
        assert capped["target_weight"].max() <= 0.40 + 1e-6
        assert capped["target_weight"].min() >= 0.05 - 1e-6


class TestBacktestPipeline:
    """Backtesting: prices → strategy → backtest → metrics."""

    def test_backtest_to_metrics(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )

        nav = result["nav_series"]
        assert not nav.empty

        # Convert backtest NAV to analytics format
        nav_col = "nav" if "nav" in nav.columns else "portfolio_nav"
        analytics_nav = nav.rename(columns={nav_col: "portfolio_nav"})
        analytics_nav["daily_return"] = analytics_nav["portfolio_nav"].pct_change()

        cagr = calculate_cagr(analytics_nav)
        max_dd = calculate_max_drawdown(analytics_nav)
        sharpe = calculate_sharpe(analytics_nav["daily_return"].dropna())

        assert isinstance(cagr, float)
        assert max_dd <= 0
        assert isinstance(sharpe, float)

    def test_hrp_strategy_in_backtest(self, wide_prices, wide_returns, country_map):
        """HRP weights → backtest engine integration."""
        def hrp_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
            ret = returns[tickers].dropna()
            if len(ret) < 60:
                n = len(tickers)
                return {t: 1.0 / n for t in tickers}
            hrp = allocate_hrp_weights(ret)
            capped = apply_weight_caps(hrp, max_weight=0.40, min_weight=0.05)
            return dict(zip(capped["ticker"], capped["target_weight"]))

        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=hrp_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )

        nav = result["nav_series"]
        nav_col = "nav" if "nav" in nav.columns else "portfolio_nav"
        final_nav = nav[nav_col].iloc[-1]
        assert final_nav > 0
        assert len(result["rebalance_log"]) > 0


class TestValidationPipeline:
    """Validation: backtest → regimes → overfitting → diagnostics → score."""

    def test_regime_to_overfitting(self, wide_prices, equal_weight_strategy, country_map):
        regimes = identify_market_regimes(
            wide_prices,
            benchmark_ticker=wide_prices.columns[0],
        )
        assert "regime" in regimes.columns

        bt = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        regime_perf = evaluate_regime_performance(bt["nav_series"], regimes)

        # Feed into overfitting detector with synthetic walk-forward
        wf = pd.DataFrame({
            "window_id": [1, 2],
            "train_sharpe": [1.0, 0.9],
            "test_sharpe": [0.7, 0.5],
            "train_cagr": [0.12, 0.10],
            "test_cagr": [0.08, 0.05],
            "train_max_drawdown": [-0.10, -0.12],
            "test_max_drawdown": [-0.15, -0.18],
            "sharpe_degradation": [-0.30, -0.44],
        })
        report = detect_overfitting(wf, regime_results=regime_perf)
        assert report["assessment"] in {"LIKELY_OVERFIT", "POSSIBLE_OVERFIT", "MONITOR", "ACCEPTABLE"}

    def test_diagnostics_to_score(self, wide_returns):
        """diagnostics + research score with Monte Carlo input."""
        returns = wide_returns.mean(axis=1)
        mc = run_monte_carlo_simulation(returns, n_paths=50, n_days=100)

        diag = generate_diagnostics(monte_carlo_summary=mc.get("summary"))
        assert isinstance(diag, dict)
        assert "tail_risk" in diag

        score = calculate_research_score()
        assert 0 <= score["total_score"] <= 100

    def test_stress_feeds_diagnostics(self, wide_prices, equal_weight_strategy, country_map):
        from validation.stress_tests import run_stress_scenarios

        stress = run_stress_scenarios(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            base_slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )

        diag = generate_diagnostics(stress_results=stress)
        assert isinstance(diag, dict)
        if not stress.empty:
            assert diag["friction_sensitivity"]["grade"] in {"RESILIENT", "MODERATE", "FRAGILE"}
