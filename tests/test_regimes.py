"""Tests for the regime intelligence engine."""

from datetime import date

import numpy as np
import pandas as pd
import pytest


# ── Feature pipeline tests ───────────────────────────────────────────────────


class TestRegimeFeatures:
    @pytest.fixture()
    def sample_inr_prices(self):
        """Create sample INR prices for 3 tickers over 300 days."""
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=300)
        tickers = ["SPY", "RELIANCE.NS", "AAPL"]
        rows = []
        for ticker in tickers:
            base = 1000 + np.random.randn() * 100
            prices = base * np.cumprod(1 + np.random.randn(300) * 0.01)
            for d, p in zip(dates, prices):
                rows.append({"date": d, "ticker": ticker, "inr_price": p, "fx_rate": 83.5})
        return pd.DataFrame(rows)

    def test_build_regime_features(self, sample_inr_prices):
        from regimes.features import build_regime_features
        features = build_regime_features(sample_inr_prices)
        assert not features.empty
        assert "date" in features.columns
        assert "vix" in features.columns
        assert "realized_vol" in features.columns

    def test_vix_features_synthetic(self, sample_inr_prices):
        from regimes.features import compute_vix_features
        vix = compute_vix_features(None)
        # Should synthesize from SPY even without real VIX
        assert "vix" in vix.columns
        assert "vix_zscore" in vix.columns

    def test_momentum_features(self, sample_inr_prices):
        from regimes.features import compute_momentum_features
        mom = compute_momentum_features(sample_inr_prices)
        assert "spy_momentum" in mom.columns

    def test_breadth_score(self, sample_inr_prices):
        from regimes.features import compute_breadth_score
        breadth = compute_breadth_score(sample_inr_prices, ma_window=50)
        if not breadth.empty:
            assert "breadth_score" in breadth.columns
            assert breadth["breadth_score"].between(0, 1).all()

    def test_cross_asset_correlation(self, sample_inr_prices):
        from regimes.features import compute_cross_asset_correlation
        corr = compute_cross_asset_correlation(sample_inr_prices, window=30)
        if not corr.empty:
            assert "cross_asset_corr" in corr.columns

    def test_realized_vol(self, sample_inr_prices):
        from regimes.features import compute_realized_vol_features
        vol = compute_realized_vol_features(sample_inr_prices, short_window=10, long_window=60)
        assert "realized_vol" in vol.columns
        assert "vol_regime_ratio" in vol.columns

    def test_liquidity_stress(self, sample_inr_prices):
        from regimes.features import compute_liquidity_stress
        liq = compute_liquidity_stress(sample_inr_prices, atr_window=10)
        assert "liquidity_stress" in liq.columns


# ── Detector tests ───────────────────────────────────────────────────────────


class TestRegimeDetector:
    @pytest.fixture()
    def sample_features(self):
        """Create 200 days of regime features with varying conditions."""
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        return pd.DataFrame({
            "date": dates,
            "vix": np.concatenate([
                np.full(60, 15),    # calm
                np.full(40, 35),    # panic spike
                np.full(50, 22),    # high vol
                np.full(50, 14),    # calm again
            ]),
            "vix_sma20": np.full(n, 20),
            "vix_zscore": np.concatenate([
                np.full(60, -0.5),
                np.full(40, 2.5),
                np.full(50, 0.8),
                np.full(50, -0.3),
            ]),
            "vix_percentile": np.full(n, 0.5),
            "spy_momentum": np.concatenate([
                np.full(60, 0.10),    # strong
                np.full(40, -0.15),   # crash
                np.full(50, -0.03),   # weak
                np.full(50, 0.08),    # recovery
            ]),
            "nifty_momentum": np.concatenate([
                np.full(60, 0.08),
                np.full(40, -0.12),
                np.full(50, -0.01),
                np.full(50, 0.06),
            ]),
            "breadth_score": np.concatenate([
                np.full(60, 0.70),
                np.full(40, 0.20),
                np.full(50, 0.40),
                np.full(50, 0.65),
            ]),
            "breadth_advancing": np.full(n, 10),
            "cross_asset_corr": np.concatenate([
                np.full(60, 0.30),
                np.full(40, 0.80),   # correlation spike
                np.full(50, 0.50),
                np.full(50, 0.25),
            ]),
            "realized_vol": np.concatenate([
                np.full(60, 0.12),
                np.full(40, 0.40),
                np.full(50, 0.25),
                np.full(50, 0.11),
            ]),
            "vol_regime_ratio": np.concatenate([
                np.full(60, 0.8),
                np.full(40, 2.5),
                np.full(50, 1.4),
                np.full(50, 0.7),
            ]),
            "atr_pct": np.full(n, 0.01),
            "liquidity_stress": np.concatenate([
                np.full(60, 0.2),
                np.full(40, 0.9),
                np.full(50, 0.5),
                np.full(50, 0.2),
            ]),
        })

    def test_detect_regimes(self, sample_features):
        from regimes.detectors import detect_regimes
        regimes = detect_regimes(sample_features)
        assert "regime" in regimes.columns
        assert "confidence" in regimes.columns
        assert "transition_score" in regimes.columns
        assert len(regimes) == len(sample_features)

    def test_panic_detected_during_crisis(self, sample_features):
        from regimes.detectors import detect_regimes
        regimes = detect_regimes(sample_features, apply_persistence=False)
        # Days 60-100 have VIX=35, vol spike, breadth collapse — should be panic
        panic_days = regimes.iloc[60:100]
        assert (panic_days["raw_regime"] == "panic").sum() > 30

    def test_risk_on_during_calm(self, sample_features):
        from regimes.detectors import detect_regimes
        regimes = detect_regimes(sample_features, apply_persistence=False)
        # Days 0-60 have strong momentum, healthy breadth, low vol
        calm_days = regimes.iloc[0:60]
        assert (calm_days["raw_regime"] == "risk_on").sum() > 40

    def test_persistence_prevents_flapping(self, sample_features):
        from regimes.detectors import detect_regimes
        raw = detect_regimes(sample_features, apply_persistence=False)
        persisted = detect_regimes(sample_features, apply_persistence=True)
        # Persisted should have fewer transitions
        raw_transitions = (raw["raw_regime"] != raw["raw_regime"].shift()).sum()
        pers_transitions = (persisted["regime"] != persisted["regime"].shift()).sum()
        assert pers_transitions <= raw_transitions

    def test_confidence_bounded(self, sample_features):
        from regimes.detectors import detect_regimes
        regimes = detect_regimes(sample_features)
        assert regimes["confidence"].between(0, 1).all()

    def test_all_regimes_valid(self, sample_features):
        from regimes.detectors import detect_regimes
        regimes = detect_regimes(sample_features)
        valid = {"risk_on", "risk_off", "panic", "high_vol"}
        assert set(regimes["regime"].unique()).issubset(valid)


# ── Transitions tests ────────────────────────────────────────────────────────


class TestTransitions:
    def test_transition_matrix(self):
        from regimes.transitions import compute_transition_matrix
        series = pd.Series(["risk_on", "risk_on", "risk_off", "risk_off", "panic", "risk_on"])
        matrix = compute_transition_matrix(series)
        assert matrix.shape[0] == matrix.shape[1]  # square
        # Each row should sum to ~1
        assert all(abs(matrix.sum(axis=1) - 1.0) < 0.01)

    def test_regime_durations(self):
        from regimes.transitions import compute_regime_durations
        dates = pd.date_range("2024-01-01", periods=10)
        regimes = pd.DataFrame({
            "date": dates,
            "regime": ["risk_on"] * 5 + ["panic"] * 3 + ["risk_on"] * 2,
        })
        durations = compute_regime_durations(regimes)
        assert len(durations) == 3
        assert durations.iloc[0]["regime"] == "risk_on"
        assert durations.iloc[1]["regime"] == "panic"

    def test_stability_metrics(self):
        from regimes.transitions import compute_stability_metrics
        dates = pd.date_range("2023-01-01", periods=365)
        regimes = pd.DataFrame({
            "date": dates,
            "regime": (["risk_on"] * 180 + ["risk_off"] * 100 + ["risk_on"] * 85),
        })
        metrics = compute_stability_metrics(regimes)
        assert "transitions_per_year" in metrics
        assert "avg_duration_days" in metrics
        assert metrics["transitions_per_year"] > 0


# ── Behavior mapping tests ───────────────────────────────────────────────────


class TestBehavior:
    def test_get_regime_behavior(self):
        from regimes.behavior import get_regime_behavior
        b = get_regime_behavior("panic")
        assert b.max_equity_weight < 0.50  # defensive
        assert b.tilt_strength == 0.0      # no tilts in panic
        assert b.rebalance_drift_threshold > 0.05  # higher to reduce turnover

    def test_risk_on_more_aggressive(self):
        from regimes.behavior import get_regime_behavior
        ron = get_regime_behavior("risk_on")
        roff = get_regime_behavior("risk_off")
        assert ron.max_equity_weight > roff.max_equity_weight
        assert ron.tilt_strength > roff.tilt_strength

    def test_apply_regime_constraints(self):
        from regimes.behavior import apply_regime_constraints, get_regime_behavior
        weights = {"AAPL": 0.30, "MSFT": 0.30, "GOLDIETF.NS": 0.20, "PPF": 0.20}
        types = {"AAPL": "equity", "MSFT": "equity", "GOLDIETF.NS": "etf", "PPF": "fixed_income"}

        # Panic: max equity 35%
        panic_behavior = get_regime_behavior("panic")
        adjusted = apply_regime_constraints(weights, panic_behavior, types)

        equity_total = adjusted["AAPL"] + adjusted["MSFT"] + adjusted["GOLDIETF.NS"]
        assert equity_total <= panic_behavior.max_equity_weight + 0.01


# ── Evaluation tests ─────────────────────────────────────────────────────────


class TestEvaluation:
    def test_crisis_alignment(self):
        from regimes.evaluation import evaluate_crisis_alignment
        dates = pd.date_range("2020-01-01", periods=100)
        regimes = pd.DataFrame({
            "date": dates,
            "regime": ["risk_on"] * 30 + ["panic"] * 30 + ["risk_off"] * 20 + ["risk_on"] * 20,
        })
        crises = [{"name": "Test Crisis", "start": date(2020, 1, 31), "end": date(2020, 3, 10)}]
        result = evaluate_crisis_alignment(regimes, crises)
        assert len(result) == 1
        assert result.iloc[0]["pct_defensive"] > 0

    def test_evaluate_predictive_value(self):
        from regimes.evaluation import evaluate_predictive_value
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        regimes = pd.DataFrame({
            "date": dates,
            "regime": (["risk_on"] * 100 + ["risk_off"] * 100),
        })
        nav = pd.DataFrame({
            "date": dates,
            "portfolio_nav": np.cumprod(1 + np.random.randn(n) * 0.005) * 1e6,
        })
        result = evaluate_predictive_value(regimes, nav)
        assert not result.empty
        assert "fwd_return_5d" in result.columns

    def test_regime_quality_score(self):
        from regimes.evaluation import evaluate_regime_quality
        n = 365
        dates = pd.bdate_range("2023-01-01", periods=n)
        regimes = pd.DataFrame({
            "date": dates,
            "regime": (["risk_on"] * 200 + ["panic"] * 30 + ["risk_off"] * 135),
        })
        nav = pd.DataFrame({
            "date": dates,
            "portfolio_nav": np.cumprod(1 + np.random.randn(n) * 0.005) * 1e6,
        })
        quality = evaluate_regime_quality(regimes, nav)
        assert "total_score" in quality
        assert 0 <= quality["total_score"] <= 100
        assert quality["grade"] in ("A", "B", "C", "D")


# ── Full pipeline test ───────────────────────────────────────────────────────


class TestRegimePipeline:
    def test_run_pipeline_with_data(self):
        """Integration test: run full regime pipeline against real processed data."""
        from pathlib import Path
        if not Path("data/processed/inr_prices.parquet").exists():
            pytest.skip("Processed data not available")

        from regimes import run_regime_pipeline
        result = run_regime_pipeline(save=False)

        assert "regimes" in result
        assert "features" in result
        assert "transition_matrix" in result
        assert "current_regime" in result
        assert "behavior" in result

        assert result["current_regime"] in {"risk_on", "risk_off", "panic", "high_vol"}
        assert result["behavior"].max_equity_weight > 0
        assert len(result["regimes"]) > 100
