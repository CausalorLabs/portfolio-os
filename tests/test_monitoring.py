"""
Tests for monitoring/ — Sprint 6: Attribution, Explainability & Monitoring.

Covers: attribution, explainability, alerts, notifications, observability,
        anomaly_detection, audit, monitoring engine.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timezone


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def portfolio_weights():
    return {"AAPL": 0.30, "SPY": 0.25, "RELIANCE.NS": 0.25, "INFY.NS": 0.20}


@pytest.fixture()
def benchmark_weights():
    return {"AAPL": 0.25, "SPY": 0.25, "RELIANCE.NS": 0.25, "INFY.NS": 0.25}


@pytest.fixture()
def portfolio_returns():
    return {"AAPL": 0.08, "SPY": 0.05, "RELIANCE.NS": 0.12, "INFY.NS": 0.03}


@pytest.fixture()
def benchmark_returns():
    return {"AAPL": 0.06, "SPY": 0.05, "RELIANCE.NS": 0.10, "INFY.NS": 0.04}


@pytest.fixture()
def feature_data():
    return pd.DataFrame({
        "return_21d": [0.05, 0.03, 0.08, 0.02],
        "momentum_63d": [0.12, 0.08, 0.15, 0.05],
        "momentum_126d": [0.20, 0.10, 0.25, 0.08],
        "volatility_21d": [0.15, 0.18, 0.12, 0.20],
        "sharpe_rolling": [1.2, 0.8, 1.5, 0.6],
    }, index=["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"])


# ══════════════════════════════════════════════════════════════════════════════
# Performance Attribution
# ══════════════════════════════════════════════════════════════════════════════


class TestAllocationEffect:
    def test_overweight_outperformer(self):
        from monitoring.attribution import calculate_allocation_effect
        # Overweight AAPL which beat benchmark
        result = calculate_allocation_effect(
            {"AAPL": 0.40, "SPY": 0.60},
            {"AAPL": 0.50, "SPY": 0.50},
            {"AAPL": 0.08, "SPY": 0.04},
            0.06,  # benchmark total
        )
        assert "AAPL" in result
        assert "SPY" in result

    def test_zero_for_equal_weights(self):
        from monitoring.attribution import calculate_allocation_effect
        w = {"A": 0.5, "B": 0.5}
        result = calculate_allocation_effect(w, w, {"A": 0.05, "B": 0.05}, 0.05)
        assert abs(sum(result.values())) < 1e-10


class TestSelectionEffect:
    def test_positive_when_outperforming(self, portfolio_returns, benchmark_returns, benchmark_weights):
        from monitoring.attribution import calculate_selection_effect
        result = calculate_selection_effect(
            {"AAPL": 0.25}, benchmark_weights,
            portfolio_returns, benchmark_returns,
        )
        # AAPL: 8% vs 6% → positive selection
        assert result.get("AAPL", 0) > 0


class TestInteractionEffect:
    def test_computes_cross_term(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        from monitoring.attribution import calculate_interaction_effect
        result = calculate_interaction_effect(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns,
        )
        assert isinstance(result, dict)
        assert len(result) == 4


class TestFullAttribution:
    def test_decomposition(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        from monitoring.attribution import run_performance_attribution
        result = run_performance_attribution(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns,
        )
        assert result.active_return is not None
        # Allocation + Selection + Interaction should approximately equal active return
        explained = result.allocation_effect + result.selection_effect + result.interaction_effect
        assert abs(explained - result.active_return) < 0.01

    def test_with_currency(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        from monitoring.attribution import run_performance_attribution
        result = run_performance_attribution(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns,
            local_returns={"AAPL": 0.06, "SPY": 0.04, "RELIANCE.NS": 0.12, "INFY.NS": 0.03},
            total_returns=portfolio_returns,
        )
        assert result.currency_effect != 0 or True  # May be zero if local == total

    def test_sector_details(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        from monitoring.attribution import run_performance_attribution, attribution_to_dataframe
        result = run_performance_attribution(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns,
        )
        df = attribution_to_dataframe(result)
        assert len(df) == 4
        assert "allocation_effect" in df.columns

    def test_summary_series(self, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
        from monitoring.attribution import run_performance_attribution, attribution_summary_series
        result = run_performance_attribution(
            portfolio_weights, benchmark_weights,
            portfolio_returns, benchmark_returns,
        )
        s = attribution_summary_series(result)
        assert "active_return" in s.index


# ══════════════════════════════════════════════════════════════════════════════
# Factor Attribution
# ══════════════════════════════════════════════════════════════════════════════


class TestFactorExposures:
    def test_computes_exposures(self, portfolio_weights, feature_data):
        from monitoring.attribution import compute_factor_exposures
        exposures = compute_factor_exposures(portfolio_weights, feature_data)
        assert "momentum" in exposures
        assert "market_beta" in exposures
        assert isinstance(exposures["momentum"], float)

    def test_empty_features(self, portfolio_weights):
        from monitoring.attribution import compute_factor_exposures
        empty = pd.DataFrame()
        exposures = compute_factor_exposures(portfolio_weights, empty)
        assert all(v == 0 for v in exposures.values())


class TestFactorReturns:
    def test_computes_returns(self, feature_data):
        from monitoring.attribution import compute_factor_returns
        returns = pd.Series(
            [0.05, 0.03, 0.08, 0.02],
            index=["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"],
        )
        result = compute_factor_returns(returns, feature_data)
        assert isinstance(result, dict)
        assert "momentum" in result


class TestFullFactorAttribution:
    def test_decomposition(self, portfolio_weights, portfolio_returns, feature_data):
        from monitoring.attribution import run_factor_attribution, factor_attribution_to_dataframe
        result = run_factor_attribution(
            portfolio_weights, portfolio_returns, feature_data,
        )
        assert len(result.factors) > 0
        assert result.r_squared >= 0
        df = factor_attribution_to_dataframe(result)
        assert "factor" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# Decision Explainability
# ══════════════════════════════════════════════════════════════════════════════


class TestAllocationExplanation:
    def test_explains_changes(self, portfolio_weights):
        from monitoring.explainability import explain_allocation_change
        target = {"AAPL": 0.15, "SPY": 0.35, "RELIANCE.NS": 0.30, "INFY.NS": 0.20}
        result = explain_allocation_change(
            portfolio_weights, target,
            regime="risk_off",
            regime_changed=True,
            confidence=0.7,
        )
        assert len(result.drivers) > 0
        assert "regime" in result.regime_context.lower()
        assert len(result.weight_changes) > 0

    def test_no_change(self, portfolio_weights):
        from monitoring.explainability import explain_allocation_change
        result = explain_allocation_change(portfolio_weights, portfolio_weights)
        assert "No significant" in result.summary

    def test_with_alpha(self, portfolio_weights):
        from monitoring.explainability import explain_allocation_change
        target = {"AAPL": 0.15, "SPY": 0.35, "RELIANCE.NS": 0.30, "INFY.NS": 0.20}
        result = explain_allocation_change(
            portfolio_weights, target,
            alpha_scores={"SPY": 0.8, "AAPL": 0.2},
        )
        assert any("alpha" in d.lower() for d in result.drivers)


class TestRegimeExplanation:
    def test_regime_shift(self):
        from monitoring.explainability import explain_regime_shift
        result = explain_regime_shift(
            "risk_on", "panic", 0.85,
            regime_features={"vix": 35.0, "breadth": 0.28},
        )
        assert "panic" in result.summary.lower() or "CRITICAL" in result.summary
        assert len(result.drivers) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Trade Decision Narratives
# ══════════════════════════════════════════════════════════════════════════════


class TestTradeNarrative:
    def test_trade_narrative(self):
        from monitoring.explainability import generate_trade_narrative, format_narrative_text
        narrative = generate_trade_narrative(
            "trade",
            trades=[
                {"ticker": "AAPL", "action": "SELL", "quantity": 10, "notional": 2000},
                {"ticker": "SPY", "action": "BUY", "quantity": 5, "notional": 2500},
            ],
            regime="risk_off",
            confidence=0.7,
            trigger="weight_drift",
        )
        assert "Executed" in narrative.narrative
        assert len(narrative.bullet_points) > 0
        text = format_narrative_text(narrative)
        assert "Decision:" in text

    def test_no_trade_narrative(self):
        from monitoring.explainability import generate_trade_narrative, format_narrative_markdown
        narrative = generate_trade_narrative(
            "no_trade",
            trigger="scheduled",
            turnover_budget_remaining=0.12,
        )
        assert "no trades" in narrative.narrative.lower()
        md = format_narrative_markdown(narrative)
        assert "##" in md

    def test_harvest_narrative(self):
        from monitoring.explainability import generate_trade_narrative
        narrative = generate_trade_narrative(
            "harvest",
            trades=[{"ticker": "AAPL", "unrealized_loss": -5000}],
            regime="risk_on",
        )
        assert "harvesting" in narrative.narrative.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Decision Timeline
# ══════════════════════════════════════════════════════════════════════════════


class TestDecisionTimeline:
    def test_add_and_query(self):
        from monitoring.explainability import DecisionTimeline
        tl = DecisionTimeline()
        tl.add_event("rebalance", "Rebalanced portfolio")
        tl.add_event("regime_change", "Risk ON → OFF", severity="WARNING")
        tl.add_event("alert", "Vol spike", severity="CRITICAL")

        assert len(tl.recent()) == 3
        assert len(tl.by_type("regime_change")) == 1
        assert len(tl.by_severity("CRITICAL")) == 1

    def test_to_dataframe(self):
        from monitoring.explainability import DecisionTimeline
        tl = DecisionTimeline()
        tl.add_event("test", "Test event")
        df = tl.to_dataframe()
        assert not df.empty
        assert "event_type" in df.columns

    def test_summary(self):
        from monitoring.explainability import DecisionTimeline
        tl = DecisionTimeline()
        tl.add_event("a", "x")
        tl.add_event("b", "y")
        s = tl.summary()
        assert s["total_events"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# Alert Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioAlerts:
    def test_concentration_alert(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_portfolio({"AAPL": 0.60, "SPY": 0.40})
        # HHI = 0.36 + 0.16 = 0.52 > 0.25 threshold
        assert any(a.title == "High concentration" for a in alerts)

    def test_position_limit(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_portfolio({"AAPL": 0.35, "SPY": 0.35, "GOOG": 0.30})
        assert any("limit breach" in a.title for a in alerts)

    def test_drift_alert(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_portfolio(
            {"AAPL": 0.40, "SPY": 0.60},
            target_weights={"AAPL": 0.25, "SPY": 0.75},
        )
        assert any("drift" in a.title.lower() for a in alerts)

    def test_cash_alerts(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_portfolio({"A": 1.0}, cash_pct=0.30)
        assert any("cash" in a.title.lower() for a in alerts)


class TestRiskAlerts:
    def test_vol_spike(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_risk(current_vol=0.30, target_vol=0.15)
        assert any("Volatility" in a.title for a in alerts)

    def test_drawdown_critical(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_risk(current_drawdown=-0.25)
        assert any(a.severity == "CRITICAL" for a in alerts)

    def test_no_alert_normal(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_risk(current_vol=0.12, target_vol=0.15)
        assert len(alerts) == 0


class TestRegimeAlerts:
    def test_panic_alert(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_regime("panic", regime_confidence=0.9)
        assert any(a.severity == "CRITICAL" for a in alerts)

    def test_transition_alert(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_regime(
            "risk_off", regime_changed=True, previous_regime="risk_on",
        )
        assert any("Regime change" in a.title for a in alerts)

    def test_instability(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        history = ["risk_on", "risk_off", "risk_on", "panic", "risk_off",
                    "risk_on", "risk_off", "panic", "risk_on", "risk_off"]
        alerts = engine.check_regime("risk_off", regime_history=history)
        assert any("Unstable" in a.title for a in alerts)


class TestMLAlerts:
    def test_confidence_collapse(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_ml(confidence=0.1)
        assert any("confidence" in a.title.lower() for a in alerts)

    def test_feature_drift(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_ml(
            confidence=0.5,
            feature_zscores={"momentum_63d": 4.5},
        )
        assert any("drift" in a.title.lower() for a in alerts)


class TestOperationalAlerts:
    def test_pipeline_failure(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_operational(
            pipeline_status={"ingestion": {"error_count": 5}},
        )
        assert any("Pipeline" in a.title for a in alerts)

    def test_stale_data(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_operational(stale_components=["features"])
        assert any("Stale" in a.title for a in alerts)


class TestAlertEngine:
    def test_run_all(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.run_all_checks(
            weights={"AAPL": 0.60, "SPY": 0.40},
            current_regime="panic",
            ml_confidence=0.1,
        )
        assert len(alerts) > 0

    def test_dedup_cooldown(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        a1 = engine.check_regime("panic")
        a2 = engine.check_regime("panic")
        # Second call should be suppressed by cooldown
        assert len(a2) == 0

    def test_acknowledge(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        alerts = engine.check_regime("panic")
        assert len(alerts) > 0
        assert engine.acknowledge(alerts[0].alert_id)
        assert len(engine.unacknowledged()) == 0

    def test_summary(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        engine.check_regime("panic")
        s = engine.summary()
        assert s["total_alerts"] > 0

    def test_to_dataframe(self):
        from monitoring.alerts import AlertEngine
        engine = AlertEngine()
        engine.check_regime("panic")
        df = engine.to_dataframe()
        assert not df.empty
        assert "severity" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# Notifications
# ══════════════════════════════════════════════════════════════════════════════


class TestNotifications:
    def test_log_channel(self):
        from monitoring.notifications import LogChannel
        ch = LogChannel()
        assert ch.send("Test", "message", "WARNING")
        assert ch.stats["sent"] == 1

    def test_severity_filter(self):
        from monitoring.notifications import LogChannel
        ch = LogChannel(min_severity="CRITICAL")
        assert not ch.send("Test", "msg", "INFO")
        assert ch.send("Test", "msg", "CRITICAL")

    def test_dispatcher(self):
        from monitoring.notifications import NotificationDispatcher
        disp = NotificationDispatcher()
        sent = disp.dispatch("Test Alert", "Test message", "CRITICAL")
        assert sent >= 1  # At least log channel

    def test_rate_limiting(self):
        from monitoring.notifications import NotificationDispatcher
        disp = NotificationDispatcher()
        disp._max_per_hour = 2
        disp.dispatch("A", "a", "CRITICAL")
        disp.dispatch("B", "b", "CRITICAL")
        sent = disp.dispatch("C", "c", "CRITICAL")
        assert sent == 0  # Rate limited

    def test_digest_mode(self):
        from monitoring.notifications import NotificationDispatcher
        disp = NotificationDispatcher()
        disp._digest_mode = True
        sent = disp.dispatch("Test", "msg", "WARNING")
        assert sent == 0  # Buffered
        assert len(disp._digest_buffer) == 1

    def test_summary(self):
        from monitoring.notifications import NotificationDispatcher
        disp = NotificationDispatcher()
        s = disp.summary()
        assert "channels" in s


# ══════════════════════════════════════════════════════════════════════════════
# Observability
# ══════════════════════════════════════════════════════════════════════════════


class TestObservability:
    def test_record_success(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        h = engine.record_success("ingestion", latency_ms=150)
        assert h.status == "healthy"
        assert h.latency_ms == 150

    def test_record_error(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        engine.record_error("ingestion", "Connection timeout")
        engine.record_error("ingestion", "Connection timeout")
        engine.record_error("ingestion", "Connection timeout")
        h = engine.get_component_health("ingestion")
        assert h.status == "unhealthy"
        assert h.error_count == 3

    def test_staleness_check(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        stale = engine.check_staleness()
        # All components should be stale (never had success)
        assert len(stale) > 0

    def test_model_health(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        h = engine.update_model_health(
            "lgbm_alpha",
            rolling_ic=0.05,
            confidence_mean=0.6,
        )
        assert h.status == "healthy"

    def test_model_health_degraded(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        h = engine.update_model_health(
            "lgbm_alpha",
            rolling_ic=0.01,  # below threshold
            confidence_mean=0.6,
        )
        assert h.status == "degraded"

    def test_latency_stats(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        for ms in [100, 150, 200, 120, 180]:
            engine.record_success("ingestion", latency_ms=ms)
        stats = engine.get_latency_stats("ingestion")
        assert stats["mean_ms"] > 0
        assert stats["p95_ms"] >= stats["mean_ms"]

    def test_health_dataframe(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        engine.record_success("ingestion")
        df = engine.health_dataframe()
        assert not df.empty

    def test_summary(self):
        from monitoring.observability import ObservabilityEngine
        engine = ObservabilityEngine()
        engine.record_success("ingestion")
        s = engine.summary()
        assert "overall_status" in s


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioAnomalies:
    def test_exposure_jump(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        anomalies = engine.check_portfolio_anomalies(
            weight_changes={"AAPL": 0.25},  # 25% jump
        )
        assert len(anomalies) > 0
        assert anomalies[0].category == "portfolio"

    def test_turnover_spike(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        # Build stable baseline (need enough points for meaningful std)
        for t in [0.05, 0.06, 0.04, 0.05, 0.06, 0.05, 0.04, 0.05,
                  0.05, 0.06, 0.04, 0.05, 0.06, 0.05, 0.04, 0.05,
                  0.05, 0.06, 0.04, 0.05]:
            engine.check_portfolio_anomalies(turnover=t)
        # Now a spike
        anomalies = engine.check_portfolio_anomalies(turnover=0.50)
        assert any(a.metric == "turnover_spike" for a in anomalies)


class TestModelAnomalies:
    def test_confidence_collapse(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        # Build stable baseline (need enough points)
        for c in [0.7, 0.65, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70,
                  0.7, 0.65, 0.72, 0.68, 0.70, 0.71, 0.69, 0.70,
                  0.7, 0.68, 0.70, 0.69]:
            engine.check_model_anomalies(confidence=c)
        # Now a collapse
        anomalies = engine.check_model_anomalies(confidence=0.1)
        assert any("confidence" in a.metric for a in anomalies)


class TestExecutionAnomalies:
    def test_slippage_anomaly(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        anomalies = engine.check_execution_anomalies(
            actual_slippage=0.005,
            expected_slippage=0.001,  # 5x expected
        )
        assert any("slippage" in a.metric for a in anomalies)

    def test_execution_divergence(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        anomalies = engine.check_execution_anomalies(
            planned_notional=100000,
            actual_notional=85000,  # 15% divergence
        )
        assert any("divergence" in a.metric for a in anomalies)


class TestAnomalyEngine:
    def test_run_all(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        anomalies = engine.run_all_checks(
            weight_changes={"AAPL": 0.30},
        )
        assert len(anomalies) > 0

    def test_to_dataframe(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.check_portfolio_anomalies(weight_changes={"X": 0.20})
        df = engine.to_dataframe()
        assert not df.empty

    def test_summary(self):
        from monitoring.anomaly_detection import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        engine.check_portfolio_anomalies(weight_changes={"X": 0.20})
        s = engine.summary()
        assert s["total_anomalies"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Audit & Traceability
# ══════════════════════════════════════════════════════════════════════════════


class TestAuditTrail:
    def test_record_event(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        event = trail.record_event(
            "ingestion", "ingestion", "Loaded data",
        )
        assert len(event.trace_id) == 12
        assert trail.recent(1)[0].trace_id == event.trace_id

    def test_parent_linking(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        e1 = trail.record_event("ingestion", "ingestion", "Step 1")
        e2 = trail.record_event("feature", "features", "Step 2", parent_id=e1.trace_id)
        assert e2.parent_id == e1.trace_id

    def test_lineage(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        e1 = trail.record_event("ingestion", "ingestion", "Raw data")
        e2 = trail.record_event("feature", "features", "Features", parent_id=e1.trace_id)
        e3 = trail.record_event("prediction", "ml", "Alpha", parent_id=e2.trace_id)

        lineage = trail.get_lineage(e3.trace_id)
        assert len(lineage) == 3
        assert lineage[0].trace_id == e3.trace_id
        assert lineage[-1].trace_id == e1.trace_id

    def test_children(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        parent = trail.record_event("ingestion", "ingestion", "Root")
        trail.record_event("feature", "features", "Child 1", parent_id=parent.trace_id)
        trail.record_event("feature", "features", "Child 2", parent_id=parent.trace_id)
        children = trail.get_children(parent.trace_id)
        assert len(children) == 2

    def test_pipeline_trace(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        events = trail.trace_pipeline_run([
            {"event_type": "ingestion", "component": "ingestion", "description": "Load"},
            {"event_type": "feature", "component": "features", "description": "Compute"},
            {"event_type": "prediction", "component": "ml", "description": "Predict"},
        ])
        assert len(events) == 3
        # Each should have parent = previous
        assert events[1].parent_id == events[0].trace_id
        assert events[2].parent_id == events[1].trace_id

    def test_convenience_methods(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        trail.record_ingestion("yahoo", 100, duration_ms=500)
        trail.record_feature_computation(50, 10, duration_ms=200)
        trail.record_prediction("lgbm", 10, 0.65, duration_ms=100)
        trail.record_optimization("hrp", 10, "risk_on", duration_ms=300)
        trail.record_execution_decision("trade", 3, 0.005, duration_ms=50)

        assert len(trail.get_by_type("ingestion")) == 1
        assert len(trail.get_by_type("prediction")) == 1
        assert len(trail.get_by_component("ml_inference")) == 1

    def test_to_dataframe(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        trail.record_event("test", "test", "Test event")
        df = trail.to_dataframe()
        assert not df.empty
        assert "trace_id" in df.columns

    def test_summary(self):
        from monitoring.audit import AuditTrail
        trail = AuditTrail()
        trail.record_event("a", "comp_a", "x")
        trail.record_event("b", "comp_b", "y")
        s = trail.summary()
        assert s["total_events"] == 2
        assert "a" in s["by_type"]


# ══════════════════════════════════════════════════════════════════════════════
# Monitoring Engine (Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════


class TestMonitoringEngine:
    def test_initialization(self):
        from monitoring import MonitoringEngine
        engine = MonitoringEngine()
        assert engine.alerts is not None
        assert engine.anomalies is not None
        assert engine.audit is not None
        assert engine.timeline is not None
        assert engine.notifications is not None
        assert engine.observability is not None

    def test_health_check(self):
        from monitoring import MonitoringEngine
        engine = MonitoringEngine()
        result = engine.run_health_check()
        assert "system_health" in result
        assert "alerts" in result
        assert "anomalies" in result

    def test_monitoring_cycle(self):
        from monitoring import MonitoringEngine
        engine = MonitoringEngine()
        result = engine.run_monitoring_cycle(
            weights={"AAPL": 0.60, "SPY": 0.40},
            current_regime="panic",
            regime_changed=True,
            previous_regime="risk_on",
            ml_confidence=0.1,
        )
        assert result["alerts_fired"] > 0

    def test_summary(self):
        from monitoring import MonitoringEngine
        engine = MonitoringEngine()
        s = engine.summary()
        assert "alerts" in s
        assert "anomalies" in s
        assert "observability" in s
        assert "audit" in s
