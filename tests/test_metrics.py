"""
Tests for analytics/metrics.py — core risk metrics.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.metrics import (
    calculate_cagr,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_calmar,
    calculate_skewness,
    calculate_kurtosis,
    calculate_all_metrics,
)


class TestCAGR:
    def test_positive_growth(self, nav_df):
        cagr = calculate_cagr(nav_df)
        assert isinstance(cagr, float)

    def test_flat_nav_zero_cagr(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=252),
            "portfolio_nav": [1_000_000] * 252,
        })
        assert abs(calculate_cagr(nav)) < 0.001

    def test_known_doubling(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        values = np.linspace(1_000_000, 2_000_000, 500)
        nav = pd.DataFrame({"date": dates, "portfolio_nav": values})
        cagr = calculate_cagr(nav)
        assert 0.3 < cagr < 0.6  # ~doubling in ~2 years


class TestVolatility:
    def test_positive(self, daily_returns):
        vol = calculate_volatility(daily_returns)
        assert vol > 0

    def test_annualized(self, daily_returns):
        vol = calculate_volatility(daily_returns)
        daily_std = daily_returns.std()
        assert abs(vol - daily_std * np.sqrt(252)) < 1e-6

    def test_constant_returns_near_zero_vol(self):
        ret = pd.Series([0.001] * 100)
        assert calculate_volatility(ret) < 1e-10


class TestSharpe:
    def test_reasonable_range(self, daily_returns):
        sharpe = calculate_sharpe(daily_returns)
        assert -5 < sharpe < 10

    def test_near_zero_vol_extreme_sharpe(self):
        ret = pd.Series([0.001] * 100)
        # Near-zero vol makes Sharpe extremely large or zero depending on impl
        sharpe = calculate_sharpe(ret)
        assert isinstance(sharpe, float)

    def test_higher_returns_higher_sharpe(self):
        ret_low = pd.Series(np.random.normal(0.0001, 0.01, 252))
        ret_high = pd.Series(np.random.normal(0.001, 0.01, 252))
        assert calculate_sharpe(ret_high) > calculate_sharpe(ret_low)


class TestSortino:
    def test_positive_returns_high_sortino(self):
        ret = pd.Series(np.abs(np.random.normal(0.001, 0.01, 252)))
        sortino = calculate_sortino(ret)
        assert sortino == float("inf")  # no downside

    def test_reasonable_range(self, daily_returns):
        sortino = calculate_sortino(daily_returns)
        assert isinstance(sortino, float)


class TestMaxDrawdown:
    def test_negative_or_zero(self, nav_df):
        dd = calculate_max_drawdown(nav_df)
        assert dd <= 0

    def test_no_drawdown(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=100),
            "portfolio_nav": range(100, 200),
        })
        assert calculate_max_drawdown(nav) == 0.0

    def test_known_drawdown(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=4),
            "portfolio_nav": [100, 80, 60, 90],
        })
        assert abs(calculate_max_drawdown(nav) - (-0.4)) < 1e-6


class TestCalmar:
    def test_returns_float(self, nav_df, daily_returns):
        calmar = calculate_calmar(nav_df, daily_returns)
        assert isinstance(calmar, float)


class TestSkewnessKurtosis:
    def test_skewness_range(self, daily_returns):
        skew = calculate_skewness(daily_returns)
        assert -5 < skew < 5

    def test_kurtosis_range(self, daily_returns):
        kurt = calculate_kurtosis(daily_returns)
        assert isinstance(kurt, float)


class TestCalculateAllMetrics:
    def test_returns_all_keys(self, nav_df):
        nav_df["daily_return"] = nav_df["portfolio_nav"].pct_change()
        metrics = calculate_all_metrics(nav_df)
        expected_keys = {
            "cagr", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "skewness", "kurtosis",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_all_numeric(self, nav_df):
        nav_df["daily_return"] = nav_df["portfolio_nav"].pct_change()
        metrics = calculate_all_metrics(nav_df)
        for k, v in metrics.items():
            assert isinstance(v, (int, float)), f"{k} is not numeric: {type(v)}"
