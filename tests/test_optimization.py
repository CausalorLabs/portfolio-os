"""
Tests for optimization package — HRP, baselines, constraints, covariance.
"""

import numpy as np
import pandas as pd
import pytest

from optimization.hrp import allocate_hrp_weights
from optimization.baselines import (
    equal_weight_portfolio,
    inverse_volatility_portfolio,
    risk_parity_portfolio,
)
from optimization.constraints import apply_weight_caps
from optimization.covariance import (
    calculate_covariance_matrix,
    calculate_shrinkage_covariance,
)
from optimization.allocator import build_signal_tilted_portfolio
from optimization.turnover import calculate_turnover


class TestEqualWeight:
    def test_sums_to_one(self):
        result = equal_weight_portfolio(["A", "B", "C", "D"])
        assert abs(result["target_weight"].sum() - 1.0) < 1e-9

    def test_all_equal(self):
        result = equal_weight_portfolio(["A", "B", "C"])
        assert all(abs(w - 1 / 3) < 1e-9 for w in result["target_weight"])

    def test_columns(self):
        result = equal_weight_portfolio(["A"])
        assert "ticker" in result.columns
        assert "target_weight" in result.columns
        assert "strategy" in result.columns


class TestInverseVol:
    def test_sums_to_one(self, wide_returns):
        result = inverse_volatility_portfolio(wide_returns)
        assert abs(result["target_weight"].sum() - 1.0) < 1e-6

    def test_lower_vol_higher_weight(self, wide_returns):
        result = inverse_volatility_portfolio(wide_returns)
        # lower vol asset should get higher weight
        result = result.sort_values("volatility")
        # First row = lowest vol → should have highest weight
        weights_sorted_by_vol = result["target_weight"].values
        assert weights_sorted_by_vol[0] >= weights_sorted_by_vol[-1]


class TestHRP:
    def test_returns_dataframe(self, wide_returns):
        result = allocate_hrp_weights(wide_returns)
        assert isinstance(result, pd.DataFrame)

    def test_weights_sum_to_one(self, wide_returns):
        result = allocate_hrp_weights(wide_returns)
        assert abs(result["target_weight"].sum() - 1.0) < 1e-6

    def test_all_positive(self, wide_returns):
        result = allocate_hrp_weights(wide_returns)
        assert (result["target_weight"] > 0).all()

    def test_required_columns(self, wide_returns):
        result = allocate_hrp_weights(wide_returns)
        assert "ticker" in result.columns
        assert "target_weight" in result.columns

    def test_with_custom_cov(self, wide_returns):
        cov = calculate_shrinkage_covariance(wide_returns)
        result = allocate_hrp_weights(wide_returns, cov=cov)
        assert abs(result["target_weight"].sum() - 1.0) < 1e-6


class TestWeightCaps:
    def test_max_cap_enforced(self, wide_returns):
        hrp = allocate_hrp_weights(wide_returns)
        capped = apply_weight_caps(hrp, max_weight=0.30)
        assert capped["target_weight"].max() <= 0.30 + 1e-6

    def test_sums_to_one_after_cap(self, wide_returns):
        hrp = allocate_hrp_weights(wide_returns)
        capped = apply_weight_caps(hrp, max_weight=0.30)
        assert abs(capped["target_weight"].sum() - 1.0) < 1e-6


class TestCovariance:
    def test_sample_cov_shape(self, wide_returns):
        cov = calculate_covariance_matrix(wide_returns)
        n = wide_returns.shape[1]
        assert cov.shape == (n, n)

    def test_symmetric(self, wide_returns):
        cov = calculate_covariance_matrix(wide_returns)
        assert np.allclose(cov.values, cov.values.T)

    def test_shrinkage_cov(self, wide_returns):
        cov = calculate_shrinkage_covariance(wide_returns)
        assert cov.shape[0] == cov.shape[1]
        assert np.allclose(cov.values, cov.values.T)

    def test_windowed_cov(self, wide_returns):
        cov = calculate_covariance_matrix(wide_returns, window=60)
        assert cov.shape[0] == wide_returns.shape[1]


class TestSignalTilt:
    def test_tilt_shifts_weights(self, wide_returns):
        base = allocate_hrp_weights(wide_returns)
        # Create fake signal scores with required columns
        signal = pd.DataFrame({
            "date": [pd.Timestamp("2021-06-01")] * len(base),
            "ticker": base["ticker"].values,
            "composite_score": np.random.uniform(0, 1, len(base)),
            "composite_rank": np.linspace(0, 1, len(base)),
        })
        tilted = build_signal_tilted_portfolio(base, signal, tilt_strength=0.20)
        assert abs(tilted["target_weight"].sum() - 1.0) < 1e-6
        # Weights should differ from base
        assert not np.allclose(
            base.sort_values("ticker")["target_weight"].values,
            tilted.sort_values("ticker")["target_weight"].values,
        )


class TestTurnover:
    def test_zero_turnover_same_weights(self):
        current = pd.DataFrame({"ticker": ["A", "B"], "current_weight": [0.5, 0.5]})
        target = pd.DataFrame({"ticker": ["A", "B"], "target_weight": [0.5, 0.5]})
        result = calculate_turnover(current, target)
        assert result.attrs["total_turnover"] == 0.0

    def test_full_turnover(self):
        current = pd.DataFrame({"ticker": ["A"], "current_weight": [1.0]})
        target = pd.DataFrame({"ticker": ["B"], "target_weight": [1.0]})
        result = calculate_turnover(current, target)
        assert result.attrs["total_turnover"] == 1.0
