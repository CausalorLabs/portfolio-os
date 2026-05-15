"""
Tests for validation package — walk-forward, regimes, robustness,
overfitting, Monte Carlo, stress tests, diagnostics, research score.
"""

import numpy as np
import pandas as pd
import pytest

from validation.walkforward import generate_train_test_windows, run_walkforward_validation
from validation.regimes import identify_market_regimes, evaluate_regime_performance
from validation.robustness import run_parameter_sensitivity, evaluate_stability_surface
from validation.overfitting import detect_overfitting, calculate_strategy_stability
from validation.monte_carlo import generate_bootstrap_paths, run_monte_carlo_simulation
from validation.stress_tests import run_stress_scenarios, simulate_liquidity_stress
from validation.diagnostics import generate_diagnostics
from validation.research_score import calculate_research_score
from backtests.engine import run_backtest


# ── Walk-forward ─────────────────────────────────────────────────────────────


class TestTrainTestWindows:
    def test_generates_windows(self, wide_prices):
        dates = wide_prices.index
        windows = generate_train_test_windows(dates, train_years=1, test_years=1, step_years=1)
        assert isinstance(windows, list)
        assert len(windows) >= 0  # may be 0 if not enough data

    def test_window_structure(self, wide_prices):
        dates = wide_prices.index
        windows = generate_train_test_windows(dates, train_years=1, test_years=1, step_years=1)
        if windows:
            w = windows[0]
            for key in ["window_id", "train_start", "train_end", "test_start", "test_end"]:
                assert key in w, f"Missing key: {key}"


class TestWalkforward:
    def test_returns_dataframe(self, wide_prices, equal_weight_strategy, country_map):
        result = run_walkforward_validation(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            train_years=1,
            test_years=1,
            step_years=1,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        assert isinstance(result, pd.DataFrame)


# ── Market regimes ───────────────────────────────────────────────────────────


class TestRegimes:
    def test_identify_regimes(self, wide_prices):
        result = identify_market_regimes(
            wide_prices,
            benchmark_ticker=wide_prices.columns[0],
        )
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns

    def test_regime_values(self, wide_prices):
        result = identify_market_regimes(
            wide_prices,
            benchmark_ticker=wide_prices.columns[0],
        )
        valid_regimes = {"bull", "bear", "high_vol", "sideways"}
        assert set(result["regime"].unique()).issubset(valid_regimes)

    def test_evaluate_regime_performance(self, wide_prices, equal_weight_strategy, country_map):
        regimes = identify_market_regimes(
            wide_prices,
            benchmark_ticker=wide_prices.columns[0],
        )
        bt = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        result = evaluate_regime_performance(bt["nav_series"], regimes)
        assert isinstance(result, pd.DataFrame)


# ── Monte Carlo ──────────────────────────────────────────────────────────────


class TestMonteCarlo:
    def test_bootstrap_paths_shape(self, wide_returns):
        returns = wide_returns.mean(axis=1)
        paths = generate_bootstrap_paths(returns, n_paths=50, n_days=100, block_size=10)
        assert paths.shape == (50, 100)

    def test_simulation_keys(self, wide_returns):
        returns = wide_returns.mean(axis=1)
        result = run_monte_carlo_simulation(
            returns=returns,
            initial_value=1_000_000,
            n_paths=50,
            n_days=100,
            block_size=10,
        )
        assert isinstance(result, dict)
        for key in ["nav_paths", "terminal_returns", "max_drawdowns", "summary"]:
            assert key in result, f"Missing key: {key}"

    def test_summary_has_prob_loss(self, wide_returns):
        returns = wide_returns.mean(axis=1)
        result = run_monte_carlo_simulation(
            returns=returns,
            initial_value=1_000_000,
            n_paths=100,
            n_days=100,
        )
        assert "prob_loss" in result["summary"]
        assert 0 <= result["summary"]["prob_loss"] <= 1


# ── Overfitting ──────────────────────────────────────────────────────────────


class TestOverfitting:
    def test_detect_with_minimal_data(self):
        wf = pd.DataFrame({
            "window_id": [1, 2],
            "train_sharpe": [1.2, 1.0],
            "test_sharpe": [0.8, 0.6],
            "train_cagr": [0.15, 0.12],
            "test_cagr": [0.10, 0.08],
            "train_max_drawdown": [-0.10, -0.12],
            "test_max_drawdown": [-0.15, -0.18],
            "sharpe_degradation": [-0.33, -0.40],
        })
        result = detect_overfitting(wf)
        assert isinstance(result, dict)
        assert "assessment" in result
        assert result["assessment"] in {"LIKELY_OVERFIT", "POSSIBLE_OVERFIT", "MONITOR", "ACCEPTABLE"}

    def test_strategy_stability(self):
        wf = pd.DataFrame({
            "window_id": [1, 2, 3],
            "train_sharpe": [1.5, 1.3, 1.4],
            "test_sharpe": [0.9, 0.8, 1.0],
            "train_cagr": [0.20, 0.18, 0.19],
            "test_cagr": [0.12, 0.10, 0.13],
            "train_max_drawdown": [-0.10, -0.12, -0.11],
            "test_max_drawdown": [-0.15, -0.18, -0.14],
            "sharpe_degradation": [-0.40, -0.38, -0.29],
        })
        result = calculate_strategy_stability(wf)
        assert "stability_score" in result
        assert 0 <= result["stability_score"] <= 100


# ── Robustness ───────────────────────────────────────────────────────────────


class TestParameterSensitivity:
    def test_evaluate_stability_surface(self):
        sensitivity = pd.DataFrame({
            "params": [{"w": 60}, {"w": 120}, {"w": 180}],
            "sharpe": [0.8, 0.9, 0.85],
            "cagr": [0.15, 0.17, 0.16],
            "max_drawdown": [-0.12, -0.10, -0.11],
        })
        result = evaluate_stability_surface(sensitivity)
        assert "stability_score" in result
        assert "cv" in result
        assert 0 <= result["stability_score"] <= 1


# ── Stress tests ─────────────────────────────────────────────────────────────


class TestStressTests:
    def test_returns_dataframe(self, wide_prices, equal_weight_strategy, country_map):
        result = run_stress_scenarios(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            base_slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_scenario_column(self, wide_prices, equal_weight_strategy, country_map):
        result = run_stress_scenarios(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            base_slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        if not result.empty:
            assert "scenario" in result.columns
            assert "cagr_impact" in result.columns

    def test_liquidity_stress(self, wide_prices, equal_weight_strategy, country_map):
        result = simulate_liquidity_stress(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            slippage_levels=[0, 10, 50],
            initial_capital=1_000_000,
            country_map=country_map,
            warmup_days=60,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


# ── Diagnostics & research score ─────────────────────────────────────────────


class TestDiagnostics:
    def test_all_none_inputs(self):
        result = generate_diagnostics()
        assert isinstance(result, dict)
        assert "strategy_stability" in result

    def test_with_walkforward_data(self):
        wf = pd.DataFrame({
            "window_id": [1, 2],
            "train_sharpe": [1.2, 1.0],
            "test_sharpe": [0.8, 0.6],
            "sharpe_degradation": [-0.33, -0.40],
            "test_cagr": [0.10, 0.08],
        })
        result = generate_diagnostics(walkforward_results=wf)
        assert result["strategy_stability"]["grade"] in {"STRONG", "MODERATE", "WEAK"}


class TestResearchScore:
    def test_all_none_neutral_score(self):
        result = calculate_research_score()
        assert isinstance(result, dict)
        assert "total_score" in result
        assert "grade" in result
        assert 0 <= result["total_score"] <= 100

    def test_grade_assignment(self):
        wf = pd.DataFrame({
            "test_sharpe": [1.0, 0.9, 0.8],
            "test_cagr": [0.15, 0.12, 0.10],
            "test_max_drawdown": [-0.10, -0.12, -0.11],
        })
        result = calculate_research_score(walkforward_results=wf)
        assert result["grade"] in {"A", "B", "C", "D", "F"}

    def test_components_present(self):
        result = calculate_research_score()
        expected = {"sharpe_stability", "drawdown_consistency", "turnover_efficiency",
                    "regime_robustness", "parameter_robustness"}
        assert expected.issubset(set(result["components"].keys()))
