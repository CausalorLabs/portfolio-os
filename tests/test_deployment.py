"""
Sprint 8 — Deployment Engine Tests.

Tests for:
  - Validation framework (schema, staleness, allocation, NaN/Inf)
  - Failure simulation (missing prices, stale FX, NaN predictions, etc.)
  - Trust calibration (score computation, mode mapping, penalties)
  - Walk-forward evaluation (NAV metrics, baselines, comparison)
  - Security layer (rate limiting, token auth, CORS)
  - Hardening engine (backup, restore)
  - Stabilization report (assessment, readiness)
  - Deployment engine (override, approval, readiness check)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Framework
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidationFramework:
    """Tests for E2E validation."""

    def test_schema_prices_valid(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "price": [150]})
        results = vf.check_schema_consistency(prices=df)
        assert all(r.passed for r in results)

    def test_schema_prices_invalid(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        df = pd.DataFrame({"price": [150]})  # Missing date/ticker
        results = vf.check_schema_consistency(prices=df)
        assert any(not r.passed for r in results)

    def test_allocation_sanity_valid(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        vf._max_leverage = 1.1  # Allow small tolerance for fully-invested portfolio
        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        results = vf.check_allocation_sanity(weights)
        assert all(r.passed for r in results)

    def test_allocation_negative_weights(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        weights = {"A": 0.5, "B": -0.2, "C": 0.7}
        results = vf.check_allocation_sanity(weights)
        neg_check = [r for r in results if r.check_name == "negative_weights"]
        assert not neg_check[0].passed

    def test_allocation_over_concentrated(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        weights = {"A": 0.8, "B": 0.2}
        results = vf.check_allocation_sanity(weights)
        conc_check = [r for r in results if r.check_name == "concentration"]
        assert not conc_check[0].passed

    def test_nan_inf_check(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        results = vf.check_nan_inf(df, "test")
        nan_check = [r for r in results if r.check_name == "nan_test"]
        assert not nan_check[0].passed

    def test_nan_inf_clean(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        results = vf.check_nan_inf(df, "clean")
        assert all(r.passed for r in results)

    def test_run_all_checks(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        results = vf.run_all_checks(allocation={"A": 0.5, "B": 0.5})
        assert len(results) > 0
        assert isinstance(vf.to_dataframe(), pd.DataFrame)

    def test_critical_failures(self):
        from deployment.validation import ValidationFramework
        vf = ValidationFramework()
        vf.check_schema_consistency(prices=pd.DataFrame({"bad": [1]}))
        assert len(vf.critical_failures) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Failure Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureSimulation:
    """Tests for failure simulation."""

    def test_missing_prices(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "price": np.random.randn(100) + 100,
        })
        result = sim.simulate_missing_prices(prices, drop_fraction=0.2)
        assert result.scenario == "missing_prices"
        assert bool(result.survived) is True

    def test_stale_fx(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        result = sim.simulate_stale_fx({"USDINR": 83.5})
        assert result.survived is True
        assert result.degradation_mode == "fallback"

    def test_nan_predictions(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        scores = {"A": 0.5, "B": 0.3, "C": 0.7, "D": 0.2}
        result = sim.simulate_nan_predictions(scores, nan_fraction=0.25)
        assert result.scenario == "nan_predictions"

    def test_confidence_collapse(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        result = sim.simulate_confidence_collapse(0.1)
        assert result.survived is True

    def test_partial_pipeline_non_critical(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        result = sim.simulate_partial_pipeline(["ml_inference"])
        assert result.survived is True

    def test_partial_pipeline_critical(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        result = sim.simulate_partial_pipeline(["ingestion"])
        assert result.survived is False

    def test_run_all_scenarios(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        prices = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
            "price": np.random.randn(50) + 100,
        })
        features = pd.DataFrame({"momentum": np.random.randn(50)})
        scores = {"A": 0.5, "B": 0.3}
        results = sim.run_all_scenarios(prices, features, scores)
        assert len(results) >= 5

    def test_to_dataframe(self):
        from deployment.failure_sim import FailureSimulator
        sim = FailureSimulator()
        sim.simulate_stale_fx({})
        df = sim.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Trust Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrustCalibration:
    """Tests for trust calibration."""

    def test_high_trust_autonomous(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        result = tc.calibrate(
            model_health=0.9, data_quality=0.95,
            regime_stability=0.85, execution_reliability=0.9,
            operational_health=0.9,
        )
        assert result.overall_trust >= 0.80
        assert result.recommended_mode == "autonomous"

    def test_low_trust_advisory(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        result = tc.calibrate(
            model_health=0.2, data_quality=0.3,
            regime_stability=0.2, execution_reliability=0.3,
            operational_health=0.2,
        )
        assert result.overall_trust < 0.50
        assert result.recommended_mode == "advisory"

    def test_medium_trust_assisted(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        result = tc.calibrate(
            model_health=0.6, data_quality=0.7,
            regime_stability=0.6, execution_reliability=0.6,
            operational_health=0.6,
        )
        assert result.recommended_mode == "assisted"

    def test_penalties_reduce_trust(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        normal = tc.calibrate(
            model_health=0.7, data_quality=0.7,
            regime_stability=0.7, execution_reliability=0.7,
            operational_health=0.7,
        )
        penalized = tc.calibrate(
            model_health=0.7, data_quality=0.7,
            regime_stability=0.7, execution_reliability=0.7,
            operational_health=0.7,
            penalties={"feature_drift": True, "unstable_covariance": True},
        )
        assert penalized.overall_trust < normal.overall_trust

    def test_history_tracking(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        tc.calibrate(model_health=0.5, data_quality=0.5)
        tc.calibrate(model_health=0.8, data_quality=0.8)
        df = tc.get_history()
        assert len(df) == 2

    def test_latest(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        tc.calibrate(model_health=0.5, data_quality=0.5)
        assert tc.latest() is not None

    def test_clamp_values(self):
        from deployment.trust import TrustCalibrator
        tc = TrustCalibrator()
        result = tc.calibrate(
            model_health=1.5, data_quality=-0.5,
        )
        assert 0 <= result.model_health <= 1
        assert 0 <= result.data_quality <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardEvaluation:
    """Tests for walk-forward evaluation."""

    def _make_nav(self, n_days=500, growth_rate=0.0003):
        dates = pd.date_range("2020-01-01", periods=n_days)
        nav = 1_000_000 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01 + growth_rate))
        return pd.Series(nav, index=dates)

    def test_evaluate_nav(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        nav = self._make_nav()
        result = ev.evaluate_nav(nav, "test_strategy")
        assert result.strategy == "test_strategy"
        assert result.n_years > 1
        assert result.cagr != 0

    def test_evaluate_empty_nav(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        result = ev.evaluate_nav(pd.Series(dtype=float), "empty")
        assert result.cagr == 0

    def test_baseline_buy_and_hold(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        prices = pd.DataFrame({
            "A": np.random.randn(200).cumsum() + 100,
            "B": np.random.randn(200).cumsum() + 50,
        })
        nav = ev.generate_baseline_buy_and_hold(prices)
        assert len(nav) > 0

    def test_baseline_risk_parity(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        prices = pd.DataFrame({
            "A": np.random.randn(200).cumsum() + 100,
            "B": np.random.randn(200).cumsum() + 50,
        })
        nav = ev.generate_baseline_risk_parity(prices, lookback=30)
        assert len(nav) > 0

    def test_run_comparison(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        nav = self._make_nav()
        prices = pd.DataFrame({
            "A": np.random.randn(500).cumsum() + 100,
            "B": np.random.randn(500).cumsum() + 50,
        })
        results = ev.run_comparison(nav, prices)
        assert len(results) >= 2  # portfolio + at least one baseline

    def test_to_dataframe(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        nav = self._make_nav(100)
        ev.evaluate_nav(nav, "test")
        df = ev.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "cagr" in df.columns

    def test_summary(self):
        from deployment.walkforward import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        nav = self._make_nav()
        ev.evaluate_nav(nav, "portfolio")
        s = ev.summary()
        assert s["portfolio_cagr"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Security Layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityLayer:
    """Tests for security layer."""

    def test_rate_limiter_allows(self):
        from deployment.security import RateLimiter
        rl = RateLimiter(max_rpm=10)
        assert rl.check("client1") is True

    def test_rate_limiter_blocks(self):
        from deployment.security import RateLimiter
        rl = RateLimiter(max_rpm=3)
        for _ in range(3):
            rl.check("client1")
        assert rl.check("client1") is False

    def test_rate_limiter_remaining(self):
        from deployment.security import RateLimiter
        rl = RateLimiter(max_rpm=5)
        rl.check("c1")
        rl.check("c1")
        assert rl.remaining("c1") == 3

    def test_security_layer_no_auth(self):
        from deployment.security import SecurityLayer
        sec = SecurityLayer()
        assert sec.verify_token("anything") is True  # Auth disabled

    def test_security_layer_cors(self):
        from deployment.security import SecurityLayer
        sec = SecurityLayer()
        origins = sec.get_cors_origins()
        assert isinstance(origins, list)

    def test_summary(self):
        from deployment.security import SecurityLayer
        sec = SecurityLayer()
        s = sec.summary()
        assert "auth_enabled" in s
        assert "rate_limit_rpm" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Hardening Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardeningEngine:
    """Tests for operational hardening."""

    def test_create_backup(self, tmp_path):
        from deployment.hardening import HardeningEngine
        engine = HardeningEngine()
        engine._backup_dir = tmp_path / "backups"
        meta = engine.create_backup(label="test")
        assert meta is not None
        assert "timestamp" in meta

    def test_list_backups(self, tmp_path):
        from deployment.hardening import HardeningEngine
        engine = HardeningEngine()
        engine._backup_dir = tmp_path / "backups"
        engine.create_backup("b1")
        backups = engine.list_backups()
        assert len(backups) == 1

    def test_snapshot_environment(self):
        from deployment.hardening import HardeningEngine
        engine = HardeningEngine()
        snap = engine.snapshot_environment()
        assert "python_version" in snap
        assert "timestamp" in snap

    def test_summary(self, tmp_path):
        from deployment.hardening import HardeningEngine
        engine = HardeningEngine()
        engine._backup_dir = tmp_path / "backups"
        s = engine.summary()
        assert s["enabled"] is True
        assert s["total_backups"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Stabilization Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestStabilizationReport:
    """Tests for MVP stabilization report."""

    def test_model_health_assessment(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        result = report.assess_model_health(
            rank_ic=0.05, grade="B",
            confidence_mean=0.6, feature_drift_pct=5.0,
        )
        assert result["model_ready"] is True
        assert len(result["concerns"]) == 0

    def test_model_health_concerns(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        result = report.assess_model_health(
            rank_ic=0.01, grade="D",
            confidence_mean=0.2, feature_drift_pct=25.0,
        )
        assert result["model_ready"] is False
        assert len(result["concerns"]) > 0

    def test_operations_assessment(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        result = report.assess_operations(
            pipeline_success_rate=0.98,
            sla_compliance_pct=95.0,
            uptime_pct=99.5,
            avg_pipeline_minutes=5.0,
        )
        assert result["ops_ready"] is True

    def test_full_report_ready(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        report.assess_model_health(0.05, "A", 0.7, 2.0)
        report.assess_operations(0.99, 98.0, 99.9, 3.0)
        report.assess_validation(10, 10, 0, 6, 7)
        report.assess_trust(0.85, "autonomous")
        result = report.generate()
        assert result["deployment_ready"] is True

    def test_full_report_not_ready(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        report.assess_model_health(0.01, "F", 0.1, 30.0)
        report.assess_operations(0.80, 70.0, 90.0, 15.0)
        report.assess_validation(5, 10, 3, 2, 7)
        report.assess_trust(0.3, "advisory")
        result = report.generate()
        assert result["deployment_ready"] is False
        assert len(result["concerns"]) > 0

    def test_to_dataframe(self):
        from deployment.report import StabilizationReport
        report = StabilizationReport()
        report.assess_model_health(0.05, "B", 0.6, 5.0)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# Deployment Engine (Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeploymentEngine:
    """Integration tests for the deployment engine."""

    def test_default_mode(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        assert engine.get_approval_mode() == "assisted"

    def test_set_mode(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        engine.set_approval_mode("autonomous")
        assert engine.get_approval_mode() == "autonomous"

    def test_invalid_mode_raises(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        with pytest.raises(ValueError):
            engine.set_approval_mode("invalid")

    def test_advisory_no_execute(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        engine.set_approval_mode("advisory")
        result = engine.check_execution_approval(trust_score=0.9, trade_count=5)
        assert result["approved"] is False
        assert result["action"] == "display_recommendation"

    def test_autonomous_high_trust(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        engine.set_approval_mode("autonomous")
        result = engine.check_execution_approval(trust_score=0.9, trade_count=5)
        assert result["approved"] is True

    def test_autonomous_low_trust_blocked(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        engine.set_approval_mode("autonomous")
        result = engine.check_execution_approval(trust_score=0.5, trade_count=5)
        assert result["approved"] is False

    def test_assisted_requires_approval(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        engine.set_approval_mode("assisted")
        result = engine.check_execution_approval(trust_score=0.7, trade_count=3)
        assert result["approved"] is False
        assert result["action"] == "require_approval"

    def test_readiness_check(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        report = engine.run_readiness_check(
            trust_score=0.8,
            rank_ic=0.05,
            grade="B",
            confidence_mean=0.6,
        )
        assert "deployment_ready" in report
        assert "sections" in report

    def test_summary(self):
        from deployment import DeploymentEngine
        engine = DeploymentEngine()
        s = engine.summary()
        assert "mode" in s
        assert s["mode"] == "assisted"
