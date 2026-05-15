"""
Tests for analytics/returns.py and analytics/drawdown.py.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.returns import (
    calculate_daily_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
    calculate_rolling_returns,
    build_returns_table,
)
from analytics.drawdown import (
    calculate_drawdown_series,
    calculate_drawdown_periods,
)


class TestDailyReturns:
    def test_columns(self, nav_df):
        result = calculate_daily_returns(nav_df)
        assert "daily_return" in result.columns
        assert "date" in result.columns

    def test_first_row_nan(self, nav_df):
        result = calculate_daily_returns(nav_df)
        assert pd.isna(result["daily_return"].iloc[0])

    def test_length_preserved(self, nav_df):
        result = calculate_daily_returns(nav_df)
        assert len(result) == len(nav_df)


class TestLogReturns:
    def test_columns(self, nav_df):
        result = calculate_log_returns(nav_df)
        assert "log_return" in result.columns

    def test_close_to_simple_for_small(self, nav_df):
        daily = calculate_daily_returns(nav_df)
        log = calculate_log_returns(nav_df)
        # For small returns, log ≈ simple
        simple = daily["daily_return"].dropna()
        log_r = log["log_return"].dropna()
        diff = (simple - log_r).abs().mean()
        assert diff < 0.01


class TestCumulativeReturns:
    def test_starts_at_zero(self, nav_df):
        result = calculate_cumulative_returns(nav_df)
        assert result["cumulative_return"].iloc[0] == 0.0

    def test_positive_for_growing_nav(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=10),
            "portfolio_nav": range(100, 110),
        })
        result = calculate_cumulative_returns(nav)
        assert result["cumulative_return"].iloc[-1] > 0


class TestRollingReturns:
    def test_window_nans(self, nav_df):
        result = calculate_rolling_returns(nav_df, window=20)
        col = "rolling_20d_return"
        assert col in result.columns
        assert result[col].iloc[:20].isna().all()

    def test_non_nan_after_window(self, nav_df):
        result = calculate_rolling_returns(nav_df, window=20)
        assert result["rolling_20d_return"].iloc[25:].notna().all()


class TestBuildReturnsTable:
    def test_all_columns_present(self, nav_df):
        result = build_returns_table(nav_df)
        for col in ["daily_return", "log_return", "cumulative_return",
                     "rolling_20d_return", "rolling_60d_return"]:
            assert col in result.columns

    def test_length(self, nav_df):
        result = build_returns_table(nav_df)
        assert len(result) == len(nav_df)


class TestDrawdownSeries:
    def test_columns(self, nav_df):
        dd = calculate_drawdown_series(nav_df)
        assert "drawdown" in dd.columns
        assert "rolling_peak" in dd.columns

    def test_drawdown_non_positive(self, nav_df):
        dd = calculate_drawdown_series(nav_df)
        assert (dd["drawdown"] <= 0).all()

    def test_monotonic_nav_zero_dd(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=100),
            "portfolio_nav": range(100, 200),
        })
        dd = calculate_drawdown_series(nav)
        assert (dd["drawdown"] == 0).all()


class TestDrawdownPeriods:
    def test_returns_dataframe(self, nav_df):
        result = calculate_drawdown_periods(nav_df)
        assert isinstance(result, pd.DataFrame)

    def test_depth_negative(self, nav_df):
        result = calculate_drawdown_periods(nav_df)
        if not result.empty:
            assert (result["depth"] < 0).all()

    def test_no_dd_for_monotonic(self):
        nav = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=100),
            "portfolio_nav": range(100, 200),
        })
        result = calculate_drawdown_periods(nav)
        assert result.empty
