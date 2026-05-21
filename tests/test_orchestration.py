"""
Sprint 7 — Orchestration Engine Tests.

Tests for:
  - Event bus (publish/subscribe, history, filtering)
  - Dependency graph (topological sort, cycle detection, readiness)
  - Retry engine (exponential backoff, fallbacks, circuit breaker)
  - Scheduler (cadences, due tasks)
  - MLOps engine (retraining triggers, model registry, shadow deployment)
  - State coordinator (state updates, persistence)
  - SLA tracker (compliance, staleness)
  - Governance (config snapshots)
  - Orchestration engine (full pipeline run, stage registration)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventBus:
    """Tests for the event bus."""

    def test_publish_event(self):
        from orchestration.events import EventBus
        bus = EventBus()
        event = bus.publish("test_event", "test_source", {"key": "value"})
        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.payload == {"key": "value"}
        assert event.event_id  # non-empty

    def test_subscribe_and_handle(self):
        from orchestration.events import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("data_updated", lambda e: received.append(e))
        bus.publish("data_updated", "ingestion", {"rows": 100})
        assert len(received) == 1
        assert received[0].event_type == "data_updated"

    def test_recent_events(self):
        from orchestration.events import EventBus
        bus = EventBus()
        for i in range(5):
            bus.publish(f"event_{i}", "test")
        recent = bus.recent(n=3)
        assert len(recent) == 3

    def test_filter_by_type(self):
        from orchestration.events import EventBus
        bus = EventBus()
        bus.publish("type_a", "src")
        bus.publish("type_b", "src")
        bus.publish("type_a", "src")
        filtered = bus.recent(n=10, event_type="type_a")
        assert len(filtered) == 2

    def test_event_summary(self):
        from orchestration.events import EventBus
        bus = EventBus()
        bus.publish("alert", "monitor")
        bus.publish("alert", "monitor")
        bus.publish("data", "ingestion")
        summary = bus.summary()
        assert summary["total_events"] == 3
        assert summary["by_type"]["alert"] == 2

    def test_to_dataframe(self):
        from orchestration.events import EventBus
        bus = EventBus()
        bus.publish("test", "src")
        df = bus.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "event_type" in df.columns

    def test_parent_event(self):
        from orchestration.events import EventBus
        bus = EventBus()
        parent = bus.publish("parent", "src")
        child = bus.publish("child", "src", parent_id=parent.event_id)
        assert child.parent_id == parent.event_id

    def test_handler_failure_marks_event_failed(self):
        from orchestration.events import EventBus
        bus = EventBus()
        bus.subscribe("bad_event", lambda e: 1/0)
        event = bus.publish("bad_event", "src")
        assert event.status == "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency Graph
# ═══════════════════════════════════════════════════════════════════════════════

class TestDependencyGraph:
    """Tests for the dependency graph."""

    def test_add_dependency(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("optimization", "risk")
        assert "risk" in g.get_dependencies("optimization")

    def test_is_ready(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("B", "A")
        assert not g.is_ready("B")
        g.mark_completed("A")
        assert g.is_ready("B")

    def test_topological_sort(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("B", "A")
        g.add_dependency("C", "B")
        order = g.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_cycle_detection(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "A")
        assert g.has_cycle()

    def test_no_cycle(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("B", "A")
        assert not g.has_cycle()

    def test_get_runnable(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("B", "A")
        g.add_dependency("C", "A")
        g.mark_completed("A")
        runnable = g.get_runnable()
        assert "B" in runnable
        assert "C" in runnable

    def test_unmet_dependencies(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("C", "A")
        g.add_dependency("C", "B")
        g.mark_completed("A")
        unmet = g.unmet("C")
        assert unmet == ["B"]

    def test_reset(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g.add_dependency("B", "A")
        g.mark_completed("A")
        g.reset()
        assert not g.is_ready("B")

    def test_summary(self):
        from orchestration.dependencies import DependencyGraph
        g = DependencyGraph()
        g._deps.clear()  # Clear config-loaded deps
        g.add_dependency("B", "A")
        s = g.summary()
        assert s["stages"] == 2
        assert s["edges"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Retry Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryEngine:
    """Tests for the retry engine."""

    def test_success_no_retry(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        success, result = engine.execute_with_retry("test", lambda: 42)
        assert success is True
        assert result == 42

    def test_retry_on_failure(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        # Override to minimal backoff for testing
        engine._default_base = 0.01
        engine._default_max = 2
        engine._stage_overrides = {}

        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Transient error")
            return "ok"

        success, result = engine.execute_with_retry("test", flaky)
        assert success is True
        assert result == "ok"
        assert call_count[0] == 2

    def test_all_retries_exhausted(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        engine._default_base = 0.01
        engine._default_max = 2
        engine._stage_overrides = {}

        success, result = engine.execute_with_retry(
            "test", lambda: 1/0
        )
        assert success is False
        assert result is None

    def test_fallback_on_failure(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        engine._default_base = 0.01
        engine._default_max = 1
        engine._stage_overrides = {}

        success, result = engine.execute_with_retry(
            "test", lambda: 1/0,
            fallback_fn=lambda: "fallback_result"
        )
        assert success is True
        assert result == "fallback_result"

    def test_circuit_breaker(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        engine._default_base = 0.01
        engine._default_max = 1
        engine._stage_overrides = {}

        # Exhaust retries to open circuit
        engine.execute_with_retry("test", lambda: 1/0)
        assert engine.is_circuit_open("test")

        # Second attempt should be blocked
        success, _ = engine.execute_with_retry("test", lambda: 42)
        assert success is False

    def test_retry_stats(self):
        from orchestration.retries import RetryEngine
        engine = RetryEngine()
        engine.execute_with_retry("test", lambda: 42)
        stats = engine.get_retry_stats("test")
        assert stats["successes"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class TestScheduler:
    """Tests for the scheduler."""

    def test_register_and_should_run(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("daily_task", "daily")
        assert sched.should_run("daily_task")  # Never run → should run

    def test_mark_completed_resets(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("task", "daily")
        sched.mark_completed("task")
        assert not sched.should_run("task")  # Just ran

    def test_daily_cadence(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("task", "daily")
        past = datetime.now(timezone.utc) - timedelta(hours=25)
        sched.mark_completed("task", now=past)
        assert sched.should_run("task")

    def test_weekly_cadence(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("task", "weekly")
        past = datetime.now(timezone.utc) - timedelta(days=3)
        sched.mark_completed("task", now=past)
        assert not sched.should_run("task")  # Only 3 days

    def test_get_due_tasks(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("a", "daily")
        sched.register("b", "daily")
        sched.mark_completed("a")
        due = sched.get_due_tasks()
        assert "b" in due
        assert "a" not in due

    def test_summary(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("t1", "daily")
        sched.register("t2", "weekly")
        s = sched.summary()
        assert s["total_tasks"] == 2
        assert s["enabled"] == 2

    def test_disabled_task(self):
        from orchestration.scheduling import Scheduler
        sched = Scheduler()
        sched.register("t1", "daily")
        sched._tasks["t1"].enabled = False
        assert not sched.should_run("t1")


# ═══════════════════════════════════════════════════════════════════════════════
# MLOps Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLOpsEngine:
    """Tests for the MLOps engine."""

    def test_no_retraining_needed(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        decision = engine.check_retraining_needed(
            current_ic=0.05, mean_confidence=0.6,
            feature_drift_zscore=1.0, days_since_training=30,
        )
        assert not decision.should_retrain

    def test_retraining_low_ic(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        decision = engine.check_retraining_needed(
            current_ic=0.01, mean_confidence=0.6,
            feature_drift_zscore=1.0, days_since_training=30,
        )
        assert decision.should_retrain
        assert decision.urgency == "urgent"

    def test_retraining_confidence_collapse(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        decision = engine.check_retraining_needed(
            current_ic=0.05, mean_confidence=0.1,
            feature_drift_zscore=1.0, days_since_training=30,
        )
        assert decision.should_retrain
        assert decision.urgency == "critical"

    def test_retraining_old_model(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        decision = engine.check_retraining_needed(
            current_ic=0.05, mean_confidence=0.6,
            feature_drift_zscore=1.0, days_since_training=100,
        )
        assert decision.should_retrain

    def test_register_and_promote_model(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        engine.register_model("v1", 0.05, "B", 20, as_shadow=False)
        engine.register_model("v2", 0.07, "A", 25, as_shadow=True)
        engine.promote_model("v2")
        champion = engine.get_champion()
        assert champion.model_id == "v2"
        assert champion.is_champion

    def test_shadow_comparison(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        result = engine.record_shadow_comparison(
            champion_ic=0.04, shadow_ic=0.06, shadow_model_id="v2"
        )
        assert result["improvement"] == pytest.approx(0.02)
        assert result["should_promote"] is True

    def test_shadow_no_promote(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        result = engine.record_shadow_comparison(
            champion_ic=0.04, shadow_ic=0.041, shadow_model_id="v2"
        )
        assert result["should_promote"] is False

    def test_summary(self):
        from orchestration.mlops import MLOpsEngine
        engine = MLOpsEngine()
        engine.register_model("v1", 0.05, "B", 20)
        s = engine.summary()
        assert s["total_models"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# State Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateCoordinator:
    """Tests for the state coordinator."""

    def test_update_state(self):
        from orchestration.state import StateCoordinator
        coord = StateCoordinator()
        coord._state_file = Path("/tmp/test_state.json")
        coord.update(regime="risk_off", portfolio_nav=1_000_000)
        state = coord.get()
        assert state["regime"] == "risk_off"
        assert state["portfolio_nav"] == 1_000_000

    def test_no_update_on_same_value(self):
        from orchestration.state import StateCoordinator
        coord = StateCoordinator()
        coord._state_file = Path("/tmp/test_state2.json")
        coord._snapshot_on_change = False
        coord.update(regime="risk_on")
        state1 = coord.get()
        coord.update(regime="risk_on")  # Same value
        state2 = coord.get()
        assert state1["regime"] == state2["regime"]

    def test_summary(self):
        from orchestration.state import StateCoordinator
        coord = StateCoordinator()
        coord._state_file = Path("/tmp/test_state3.json")
        coord.update(regime="crisis", pipeline_status="healthy")
        s = coord.summary()
        assert s["regime"] == "crisis"
        assert s["pipeline_status"] == "healthy"


# ═══════════════════════════════════════════════════════════════════════════════
# SLA Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLATracker:
    """Tests for the SLA tracker."""

    def test_record_stage_met(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        check = tracker.record_stage("ingestion", duration_seconds=30)
        assert check.met is True

    def test_record_stage_breach(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        check = tracker.record_stage("ingestion", duration_seconds=700)
        assert check.met is False

    def test_staleness_check(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        assert tracker.check_staleness(old_time) is True
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        assert tracker.check_staleness(recent) is False

    def test_uptime_calculation(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        tracker.record_run(success=True, duration_minutes=5)
        tracker.record_run(success=True, duration_minutes=5)
        tracker.record_run(success=False, duration_minutes=10)
        pct = tracker.get_uptime_pct()
        assert pct == pytest.approx(66.67, abs=0.1)

    def test_compliance_report(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        tracker.record_stage("ingestion", 30)
        tracker.record_stage("risk", 60)
        df = tracker.get_compliance_report()
        assert len(df) == 2

    def test_summary(self):
        from orchestration.sla import SLATracker
        tracker = SLATracker()
        tracker.record_stage("ingestion", 30)
        s = tracker.summary()
        assert s["total_checks"] == 1
        assert s["sla_met"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Governance
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernance:
    """Tests for governance engine."""

    def test_snapshot_configs(self, tmp_path):
        from orchestration.governance import GovernanceEngine
        engine = GovernanceEngine()
        engine._snapshot_dir = tmp_path / "snapshots"
        result = engine.snapshot_configs(run_id="test123")
        assert result is not None
        assert "hash" in result
        assert "file" in result

    def test_list_snapshots(self, tmp_path):
        from orchestration.governance import GovernanceEngine
        engine = GovernanceEngine()
        engine._snapshot_dir = tmp_path / "snapshots"
        engine.snapshot_configs(run_id="r1")
        snapshots = engine.list_snapshots()
        assert len(snapshots) >= 1

    def test_config_change_detection(self, tmp_path):
        from orchestration.governance import GovernanceEngine
        engine = GovernanceEngine()
        engine._snapshot_dir = tmp_path / "snapshots"
        s1 = engine.snapshot_configs()
        # Same config → no change
        assert not engine.config_changed_since(s1["hash"])


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration Engine (Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestrationEngine:
    """Integration tests for the orchestration engine."""

    def test_register_and_run_stage(self):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        engine.register_stage("test_stage", lambda: {"result": "ok"})
        success, result = engine.run_stage("test_stage")
        assert success is True
        assert result == {"result": "ok"}

    def test_unregistered_stage_skipped(self):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        success, result = engine.run_stage("nonexistent")
        assert success is False

    def test_pipeline_run_context(self, tmp_path):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        engine.governance._snapshot_dir = tmp_path / "gov"
        engine.register_stage("step1", lambda: "done")

        with engine.pipeline_run(run_id="test_run") as run_id:
            assert run_id == "test_run"
            success, _ = engine.run_stage("step1")
            assert success is True

    def test_pipeline_run_stages(self, tmp_path):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        engine._stages = ["a", "b"]
        engine.governance._snapshot_dir = tmp_path / "gov"
        engine.register_stage("a", lambda: 1)
        engine.register_stage("b", lambda: 2)
        result = engine.run_pipeline()
        assert result["completed"] == 2
        assert result["failed"] == 0

    def test_failed_stage_continues(self, tmp_path):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        engine._stages = ["a", "b"]
        engine.governance._snapshot_dir = tmp_path / "gov"
        engine.retries._default_base = 0.01
        engine.retries._default_max = 1
        engine.retries._stage_overrides = {}
        engine.register_stage("a", lambda: 1/0)
        engine.register_stage("b", lambda: 2)
        result = engine.run_pipeline()
        assert result["failed"] == 1
        assert result["completed"] == 1

    def test_summary(self):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        s = engine.summary()
        assert "events" in s
        assert "deps" in s
        assert "sla" in s

    def test_stage_with_dependency_blocked(self, tmp_path):
        from orchestration import OrchestrationEngine
        engine = OrchestrationEngine()
        engine.governance._snapshot_dir = tmp_path / "gov"
        engine.deps.add_dependency("b", "a")
        engine.register_stage("a", lambda: 1)
        engine.register_stage("b", lambda: 2)

        # b should be blocked (a not completed)
        success, _ = engine.run_stage("b")
        assert success is False

        # Run a first
        engine.run_stage("a")
        success, result = engine.run_stage("b")
        assert success is True
        assert result == 2


# Need Path import for state tests
from pathlib import Path
