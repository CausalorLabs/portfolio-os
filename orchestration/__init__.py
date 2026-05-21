"""
Orchestration Engine — Workflow Orchestration for Portfolio OS.

Sprint 7: Automation, Orchestration & Continuous Operations.

Central orchestrator that manages the daily portfolio lifecycle:
  1. Ingestion → 2. Features → 3. Regime Detection → 4. ML Inference
  5. Risk Calculation → 6. Optimization → 7. Utility Evaluation
  8. Execution Simulation → 9. Attribution → 10. Alerts → 11. Monitoring

Subsystems:
  - Event Bus: publish/subscribe event-driven architecture
  - Dependency Graph: DAG-based stage dependencies
  - Retry Engine: exponential backoff with fallback strategies
  - Scheduler: daily/weekly/monthly operational cadences
  - MLOps Engine: continuous ML operations & shadow deployment
  - State Coordinator: global system state management
  - SLA Tracker: pipeline performance vs targets
  - Governance: config snapshots for reproducibility
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator

from loguru import logger
from omegaconf import OmegaConf

from orchestration.events import EventBus
from orchestration.dependencies import DependencyGraph
from orchestration.retries import RetryEngine
from orchestration.scheduling import Scheduler
from orchestration.mlops import MLOpsEngine
from orchestration.state import StateCoordinator
from orchestration.sla import SLATracker
from orchestration.governance import GovernanceEngine


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class OrchestrationEngine:
    """
    Central workflow orchestrator for the portfolio operating system.

    Coordinates the daily pipeline lifecycle with:
      - Event-driven stage transitions
      - Dependency-aware execution order
      - Retry logic with fallbacks
      - SLA compliance tracking
      - Config governance snapshots
    """

    def __init__(self):
        cfg = _load_config()
        self._pipeline_name = cfg.get("orchestration", {}).get("pipeline_name", "portfolio_daily")
        self._stages = cfg.get("orchestration", {}).get("stages", [])
        self._timeout_minutes = cfg.get("orchestration", {}).get("timeout_minutes", 30)

        # Subsystems
        self.events = EventBus()
        self.deps = DependencyGraph()
        self.retries = RetryEngine()
        self.scheduler = Scheduler()
        self.mlops = MLOpsEngine()
        self.state = StateCoordinator()
        self.sla = SLATracker()
        self.governance = GovernanceEngine()

        # Stage registry: name → callable
        self._stage_registry: dict[str, Callable] = {}
        self._fallback_registry: dict[str, Callable] = {}

        # Run tracking
        self._current_run_id: str | None = None
        self._stage_results: dict[str, dict] = {}

    def register_stage(
        self,
        name: str,
        fn: Callable,
        fallback_fn: Callable | None = None,
    ) -> None:
        """Register a pipeline stage with optional fallback."""
        self._stage_registry[name] = fn
        if fallback_fn:
            self._fallback_registry[name] = fallback_fn

    @contextmanager
    def pipeline_run(self, run_id: str | None = None) -> Generator[str, None, None]:
        """Context manager for a full pipeline run."""
        self._current_run_id = run_id or uuid.uuid4().hex[:12]
        start = time.time()

        # Governance snapshot
        self.governance.snapshot_configs(run_id=self._current_run_id)

        # Reset dependency tracking
        self.deps.reset()
        self._stage_results.clear()

        self.state.update(pipeline_status="running")
        self.events.publish("pipeline_started", self._pipeline_name, {
            "run_id": self._current_run_id,
        })

        logger.info(f"═══ Pipeline {self._pipeline_name} [{self._current_run_id}] ═══")

        try:
            yield self._current_run_id

            duration = time.time() - start
            self.sla.record_run(success=True, duration_minutes=duration / 60)
            self.state.update(pipeline_status="healthy")
            self.events.publish("pipeline_completed", self._pipeline_name, {
                "run_id": self._current_run_id,
                "duration_seconds": duration,
            })
            logger.info(f"═══ Pipeline completed in {duration:.1f}s ═══")
        except Exception as e:
            duration = time.time() - start
            self.sla.record_run(success=False, duration_minutes=duration / 60)
            self.state.update(pipeline_status="unhealthy")
            self.events.publish("pipeline_failed", self._pipeline_name, {
                "run_id": self._current_run_id,
                "error": str(e),
            })
            logger.error(f"═══ Pipeline FAILED after {duration:.1f}s: {e} ═══")
            raise

    def run_stage(self, name: str, *args, **kwargs) -> tuple[bool, object]:
        """
        Run a single pipeline stage with retry logic and SLA tracking.

        Returns (success, result).
        """
        fn = self._stage_registry.get(name)
        if fn is None:
            logger.warning(f"  Stage {name} not registered, skipping")
            return False, None

        # Check dependencies
        if not self.deps.is_ready(name):
            unmet = self.deps.unmet(name)
            logger.warning(f"  Stage {name} blocked — unmet deps: {unmet}")
            return False, None

        self.events.publish(f"stage_{name}_started", name)
        start = time.time()

        fallback = self._fallback_registry.get(name)
        success, result = self.retries.execute_with_retry(
            name, fn, *args, fallback_fn=fallback, **kwargs
        )

        duration = time.time() - start
        self.sla.record_stage(name, duration)

        if success:
            self.deps.mark_completed(name)
            self._stage_results[name] = {"status": "completed", "duration": duration}
            self.events.publish(f"stage_{name}_completed", name, {
                "duration_seconds": duration,
            })
        else:
            self._stage_results[name] = {"status": "failed", "duration": duration}
            self.events.publish(f"stage_{name}_failed", name)

        return success, result

    def run_pipeline(self, stage_args: dict[str, dict] | None = None) -> dict:
        """
        Run the full pipeline in dependency order.

        Args:
            stage_args: optional per-stage kwargs: {"ingestion": {...}, ...}

        Returns dict with run results.
        """
        stage_args = stage_args or {}

        with self.pipeline_run() as run_id:
            results = {}
            for stage in self._stages:
                if stage not in self._stage_registry:
                    logger.debug(f"  Stage {stage} not registered, skipping")
                    continue

                kwargs = stage_args.get(stage, {})
                success, result = self.run_stage(stage, **kwargs)
                results[stage] = {
                    "success": success,
                    "result": result,
                }

                if not success:
                    logger.warning(f"  Stage {stage} failed — continuing pipeline")

            return {
                "run_id": run_id,
                "stages": results,
                "completed": sum(1 for r in results.values() if r["success"]),
                "failed": sum(1 for r in results.values() if not r["success"]),
                "total": len(results),
            }

    def summary(self) -> dict:
        return {
            "pipeline": self._pipeline_name,
            "registered_stages": len(self._stage_registry),
            "events": self.events.summary(),
            "deps": self.deps.summary(),
            "retries": self.retries.get_retry_stats(),
            "state": self.state.summary(),
            "sla": self.sla.summary(),
            "mlops": self.mlops.summary(),
            "governance": self.governance.summary(),
        }
