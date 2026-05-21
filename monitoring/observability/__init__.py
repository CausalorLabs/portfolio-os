"""
Monitoring — Operational Observability Layer.

Tracks health of all system components:
  - Pipeline health (ingestion, features, ML, optimization, execution)
  - Model health (rolling IC, confidence degradation, drift metrics)
  - API health (uptime, latency, request failures)
  - System health dashboard data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Component Health
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ComponentHealth:
    """Health status of a single system component."""
    component: str
    status: str          # healthy | degraded | unhealthy | unknown
    latency_ms: float
    last_success: datetime | None
    error_count: int
    message: str
    last_checked: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelHealth:
    """Health metrics for ML models."""
    model_name: str
    rolling_ic: float
    confidence_mean: float
    confidence_std: float
    prediction_dispersion: float
    feature_drift_detected: bool
    status: str           # healthy | degraded | unhealthy
    details: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Observability Engine
# ══════════════════════════════════════════════════════════════════════════════


class ObservabilityEngine:
    """
    Tracks health and performance of all system components.

    Provides the operational monitoring backbone for the platform.
    """

    def __init__(self):
        cfg = _load_config().get("observability", {})
        self._components: dict[str, ComponentHealth] = {}
        self._model_health: dict[str, ModelHealth] = {}
        self._latency_history: dict[str, list[tuple[datetime, float]]] = {}
        self._error_log: list[dict] = []

        self._stale_threshold_hours = cfg.get("stale_threshold_hours", 24)
        self._min_ic = cfg.get("min_acceptable_ic", 0.02)
        self._confidence_floor = cfg.get("confidence_floor", 0.3)

        # Register standard components
        for comp in cfg.get("pipeline_components", [
            "ingestion", "features", "ml_inference",
            "optimization", "risk_engine", "execution", "monitoring",
        ]):
            self._components[comp] = ComponentHealth(
                component=comp,
                status="unknown",
                latency_ms=0,
                last_success=None,
                error_count=0,
                message="Not yet checked",
            )

    # ── Component Registration ───────────────────────────────────────────

    def record_success(
        self,
        component: str,
        latency_ms: float = 0,
        message: str = "OK",
        metadata: dict | None = None,
    ) -> ComponentHealth:
        """Record a successful operation for a component."""
        now = datetime.now(timezone.utc)
        health = self._components.get(component, ComponentHealth(
            component=component, status="unknown",
            latency_ms=0, last_success=None, error_count=0, message="",
        ))

        health.status = "healthy"
        health.latency_ms = latency_ms
        health.last_success = now
        health.message = message
        health.last_checked = now
        health.metadata = metadata or {}

        self._components[component] = health

        # Track latency history
        if component not in self._latency_history:
            self._latency_history[component] = []
        self._latency_history[component].append((now, latency_ms))
        # Keep last 100
        self._latency_history[component] = self._latency_history[component][-100:]

        return health

    def record_error(
        self,
        component: str,
        error: str,
        latency_ms: float = 0,
    ) -> ComponentHealth:
        """Record an error for a component."""
        now = datetime.now(timezone.utc)
        health = self._components.get(component, ComponentHealth(
            component=component, status="unknown",
            latency_ms=0, last_success=None, error_count=0, message="",
        ))

        health.error_count += 1
        health.latency_ms = latency_ms
        health.message = f"Error: {error}"
        health.last_checked = now
        health.status = "unhealthy" if health.error_count >= 3 else "degraded"

        self._components[component] = health
        self._error_log.append({
            "timestamp": now,
            "component": component,
            "error": error,
            "error_count": health.error_count,
        })

        logger.error(f"  Component error [{component}]: {error}")
        return health

    # ── Health Checks ────────────────────────────────────────────────────

    def check_staleness(self) -> list[str]:
        """Check which components have stale data."""
        stale = []
        now = datetime.now(timezone.utc)
        threshold = timedelta(hours=self._stale_threshold_hours)

        for name, health in self._components.items():
            if health.last_success is None:
                stale.append(name)
            elif (now - health.last_success) > threshold:
                stale.append(name)
                health.status = "degraded"
                health.message = f"Stale: >{self._stale_threshold_hours}h since last success"

        return stale

    def get_component_health(self, component: str) -> ComponentHealth | None:
        """Get health status for a specific component."""
        return self._components.get(component)

    def get_all_health(self) -> dict[str, ComponentHealth]:
        """Get all component health statuses."""
        return dict(self._components)

    # ── Model Health ─────────────────────────────────────────────────────

    def update_model_health(
        self,
        model_name: str,
        rolling_ic: float = 0.0,
        confidence_mean: float = 0.5,
        confidence_std: float = 0.1,
        prediction_dispersion: float = 0.0,
        feature_drift_detected: bool = False,
    ) -> ModelHealth:
        """Update health metrics for an ML model."""
        # Determine status
        status = "healthy"
        if rolling_ic < self._min_ic:
            status = "degraded"
        if confidence_mean < self._confidence_floor:
            status = "unhealthy"
        if feature_drift_detected:
            status = "degraded" if status == "healthy" else "unhealthy"

        health = ModelHealth(
            model_name=model_name,
            rolling_ic=rolling_ic,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            prediction_dispersion=prediction_dispersion,
            feature_drift_detected=feature_drift_detected,
            status=status,
            details={
                "ic_threshold": self._min_ic,
                "confidence_floor": self._confidence_floor,
            },
        )
        self._model_health[model_name] = health

        logger.info(
            f"  Model health [{model_name}]: {status} "
            f"(IC={rolling_ic:.4f}, conf={confidence_mean:.2f})"
        )
        return health

    def get_model_health(self, model_name: str) -> ModelHealth | None:
        """Get model health for a specific model."""
        return self._model_health.get(model_name)

    # ── Latency Analysis ─────────────────────────────────────────────────

    def get_latency_stats(self, component: str) -> dict:
        """Get latency statistics for a component."""
        history = self._latency_history.get(component, [])
        if not history:
            return {"component": component, "status": "no_data"}

        latencies = [lat for _, lat in history]
        import numpy as np
        return {
            "component": component,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "max_ms": float(np.max(latencies)),
            "n_samples": len(latencies),
        }

    # ── Export ────────────────────────────────────────────────────────────

    def health_dataframe(self) -> pd.DataFrame:
        """Export component health as DataFrame."""
        rows = []
        for name, h in self._components.items():
            rows.append({
                "component": name,
                "status": h.status,
                "latency_ms": h.latency_ms,
                "last_success": h.last_success,
                "error_count": h.error_count,
                "message": h.message,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def model_health_dataframe(self) -> pd.DataFrame:
        """Export model health as DataFrame."""
        rows = []
        for name, h in self._model_health.items():
            rows.append({
                "model": name,
                "status": h.status,
                "rolling_ic": h.rolling_ic,
                "confidence_mean": h.confidence_mean,
                "confidence_std": h.confidence_std,
                "prediction_dispersion": h.prediction_dispersion,
                "feature_drift": h.feature_drift_detected,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def error_dataframe(self) -> pd.DataFrame:
        """Export error log as DataFrame."""
        return pd.DataFrame(self._error_log) if self._error_log else pd.DataFrame()

    def summary(self) -> dict:
        """Overall system health summary."""
        statuses = [h.status for h in self._components.values()]
        from collections import Counter
        status_counts = Counter(statuses)

        # Overall status: worst component
        if "unhealthy" in statuses:
            overall = "unhealthy"
        elif "degraded" in statuses:
            overall = "degraded"
        elif "unknown" in statuses:
            overall = "unknown"
        else:
            overall = "healthy"

        model_statuses = [h.status for h in self._model_health.values()]

        return {
            "overall_status": overall,
            "components": dict(status_counts),
            "total_components": len(self._components),
            "stale_components": len(self.check_staleness()),
            "total_errors": sum(h.error_count for h in self._components.values()),
            "model_health": {
                "total_models": len(self._model_health),
                "healthy": model_statuses.count("healthy"),
                "degraded": model_statuses.count("degraded"),
                "unhealthy": model_statuses.count("unhealthy"),
            },
        }
