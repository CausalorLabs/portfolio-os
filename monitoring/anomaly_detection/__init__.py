"""
Monitoring — Anomaly Detection Engine.

Detects unexpected behavior across:
  - Portfolio anomalies (turnover spikes, exposure jumps, covariance changes)
  - Model anomalies (unstable predictions, confidence collapse, feature explosions)
  - Execution anomalies (unrealistic slippage, execution divergence, tax spikes)

Uses z-score based detection with configurable thresholds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str
    timestamp: datetime
    category: str      # portfolio | model | execution
    metric: str
    observed_value: float
    expected_value: float
    zscore: float
    severity: str      # INFO | WARNING | CRITICAL
    description: str
    metadata: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Detection Engine
# ══════════════════════════════════════════════════════════════════════════════


class AnomalyDetectionEngine:
    """
    Detects unexpected portfolio, model, and execution behavior.

    Maintains rolling baselines for z-score comparison.
    """

    def __init__(self):
        cfg = _load_config().get("anomaly_detection", {})
        self._zscore_threshold = cfg.get("zscore_threshold", 3.0)
        self._turnover_zscore = cfg.get("turnover_spike_zscore", 3.0)
        self._exposure_jump = cfg.get("exposure_jump_threshold", 0.15)
        self._feature_explosion = cfg.get("feature_explosion_zscore", 5.0)
        self._slippage_mult = cfg.get("slippage_anomaly_mult", 3.0)
        self._exec_divergence = cfg.get("execution_divergence_pct", 0.05)
        self._lookback = cfg.get("lookback_window", 60)

        self._baselines: dict[str, list[float]] = {}
        self._anomalies: list[Anomaly] = []

    def _make_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _update_baseline(self, metric: str, value: float) -> None:
        """Update rolling baseline for a metric."""
        if metric not in self._baselines:
            self._baselines[metric] = []
        self._baselines[metric].append(value)
        # Keep only lookback window
        self._baselines[metric] = self._baselines[metric][-self._lookback:]

    def _get_zscore(self, metric: str, value: float) -> float:
        """Compute z-score against rolling baseline."""
        history = self._baselines.get(metric, [])
        if len(history) < 5:
            return 0.0
        mean = np.mean(history)
        std = np.std(history)
        if std < 1e-10:
            return 0.0
        return (value - mean) / std

    def _severity_from_zscore(self, zscore: float) -> str:
        """Map z-score to severity."""
        z = abs(zscore)
        if z >= 5:
            return "CRITICAL"
        elif z >= 3:
            return "WARNING"
        else:
            return "INFO"

    def _record_anomaly(
        self,
        category: str,
        metric: str,
        observed: float,
        expected: float,
        zscore: float,
        description: str,
        metadata: dict | None = None,
    ) -> Anomaly:
        """Record a detected anomaly."""
        severity = self._severity_from_zscore(zscore)
        anomaly = Anomaly(
            anomaly_id=self._make_id(),
            timestamp=datetime.now(timezone.utc),
            category=category,
            metric=metric,
            observed_value=observed,
            expected_value=expected,
            zscore=zscore,
            severity=severity,
            description=description,
            metadata=metadata or {},
        )
        self._anomalies.append(anomaly)
        logger.warning(
            f"  ANOMALY [{category}/{severity}] {metric}: "
            f"z={zscore:.2f} ({description})"
        )
        return anomaly

    # ── Portfolio Anomalies ──────────────────────────────────────────────

    def check_portfolio_anomalies(
        self,
        turnover: float = 0.0,
        weight_changes: dict[str, float] | None = None,
        nav_return: float | None = None,
    ) -> list[Anomaly]:
        """Detect portfolio behavior anomalies."""
        anomalies = []

        # Turnover spike
        self._update_baseline("turnover", turnover)
        z = self._get_zscore("turnover", turnover)
        if abs(z) > self._turnover_zscore:
            mean = np.mean(self._baselines.get("turnover", [turnover]))
            a = self._record_anomaly(
                "portfolio", "turnover_spike", turnover, float(mean), z,
                f"Turnover {turnover:.1%} is {z:.1f}σ from baseline {mean:.1%}",
            )
            anomalies.append(a)

        # Exposure jumps
        if weight_changes:
            for asset, change in weight_changes.items():
                if abs(change) > self._exposure_jump:
                    a = self._record_anomaly(
                        "portfolio", f"exposure_jump_{asset}",
                        change, 0.0, change / self._exposure_jump,
                        f"{asset} weight changed by {change:.1%} "
                        f"(threshold: ±{self._exposure_jump:.1%})",
                        {"asset": asset},
                    )
                    anomalies.append(a)

        # Abnormal return
        if nav_return is not None:
            self._update_baseline("daily_return", nav_return)
            z = self._get_zscore("daily_return", nav_return)
            if abs(z) > self._zscore_threshold:
                mean = np.mean(self._baselines.get("daily_return", [nav_return]))
                a = self._record_anomaly(
                    "portfolio", "abnormal_return",
                    nav_return, float(mean), z,
                    f"Return {nav_return:.2%} is {z:.1f}σ from baseline",
                )
                anomalies.append(a)

        return anomalies

    # ── Model Anomalies ──────────────────────────────────────────────────

    def check_model_anomalies(
        self,
        predictions: dict[str, float] | None = None,
        confidence: float | None = None,
        feature_values: dict[str, float] | None = None,
    ) -> list[Anomaly]:
        """Detect ML model behavior anomalies."""
        anomalies = []

        # Prediction instability
        if predictions:
            pred_values = list(predictions.values())
            pred_std = np.std(pred_values) if len(pred_values) > 1 else 0
            self._update_baseline("prediction_dispersion", pred_std)
            z = self._get_zscore("prediction_dispersion", pred_std)
            if abs(z) > self._zscore_threshold:
                mean = np.mean(self._baselines.get("prediction_dispersion", [pred_std]))
                a = self._record_anomaly(
                    "model", "prediction_instability",
                    pred_std, float(mean), z,
                    f"Prediction dispersion {pred_std:.4f} is {z:.1f}σ from baseline",
                )
                anomalies.append(a)

        # Confidence collapse
        if confidence is not None:
            self._update_baseline("confidence", confidence)
            z = self._get_zscore("confidence", confidence)
            if z < -self._zscore_threshold:  # Only care about drops
                mean = np.mean(self._baselines.get("confidence", [confidence]))
                a = self._record_anomaly(
                    "model", "confidence_collapse",
                    confidence, float(mean), z,
                    f"Confidence {confidence:.2%} dropped {z:.1f}σ below baseline",
                )
                anomalies.append(a)

        # Feature explosions
        if feature_values:
            for feature, value in feature_values.items():
                metric = f"feature_{feature}"
                self._update_baseline(metric, value)
                z = self._get_zscore(metric, value)
                if abs(z) > self._feature_explosion:
                    mean = np.mean(self._baselines.get(metric, [value]))
                    a = self._record_anomaly(
                        "model", f"feature_explosion_{feature}",
                        value, float(mean), z,
                        f"Feature {feature}={value:.4f} is {z:.1f}σ from baseline",
                        {"feature": feature},
                    )
                    anomalies.append(a)

        return anomalies

    # ── Execution Anomalies ──────────────────────────────────────────────

    def check_execution_anomalies(
        self,
        actual_slippage: float = 0.0,
        expected_slippage: float = 0.0,
        planned_notional: float = 0.0,
        actual_notional: float = 0.0,
        tax_cost_pct: float = 0.0,
    ) -> list[Anomaly]:
        """Detect execution behavior anomalies."""
        anomalies = []

        # Unrealistic slippage
        if expected_slippage > 0:
            slippage_ratio = actual_slippage / expected_slippage
            if slippage_ratio > self._slippage_mult:
                a = self._record_anomaly(
                    "execution", "slippage_anomaly",
                    actual_slippage, expected_slippage, slippage_ratio,
                    f"Actual slippage ({actual_slippage:.4f}) is "
                    f"{slippage_ratio:.1f}x expected ({expected_slippage:.4f})",
                )
                anomalies.append(a)

        # Execution divergence
        if planned_notional > 0:
            divergence = abs(actual_notional - planned_notional) / planned_notional
            if divergence > self._exec_divergence:
                a = self._record_anomaly(
                    "execution", "execution_divergence",
                    actual_notional, planned_notional,
                    divergence / self._exec_divergence,
                    f"Executed notional diverged {divergence:.1%} from plan",
                )
                anomalies.append(a)

        # Tax cost spike
        self._update_baseline("tax_cost_pct", tax_cost_pct)
        z = self._get_zscore("tax_cost_pct", tax_cost_pct)
        if abs(z) > self._zscore_threshold and tax_cost_pct > 0:
            mean = np.mean(self._baselines.get("tax_cost_pct", [tax_cost_pct]))
            a = self._record_anomaly(
                "execution", "tax_spike",
                tax_cost_pct, float(mean), z,
                f"Tax cost {tax_cost_pct:.2%} is {z:.1f}σ above baseline",
            )
            anomalies.append(a)

        return anomalies

    # ── Run All Checks ───────────────────────────────────────────────────

    def run_all_checks(
        self,
        turnover: float = 0.0,
        weight_changes: dict[str, float] | None = None,
        nav_return: float | None = None,
        predictions: dict[str, float] | None = None,
        confidence: float | None = None,
        actual_slippage: float = 0.0,
        expected_slippage: float = 0.0,
    ) -> list[Anomaly]:
        """Run all anomaly detection checks."""
        all_anomalies = []

        all_anomalies.extend(
            self.check_portfolio_anomalies(turnover, weight_changes, nav_return)
        )
        all_anomalies.extend(
            self.check_model_anomalies(predictions, confidence)
        )
        all_anomalies.extend(
            self.check_execution_anomalies(actual_slippage, expected_slippage)
        )

        if all_anomalies:
            logger.info(f"  Anomaly detection: {len(all_anomalies)} anomalies found")
        return all_anomalies

    # ── Query & Export ───────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[Anomaly]:
        """Get most recent anomalies."""
        return sorted(self._anomalies, key=lambda a: a.timestamp, reverse=True)[:n]

    def by_category(self, category: str) -> list[Anomaly]:
        """Filter anomalies by category."""
        return [a for a in self._anomalies if a.category == category]

    def to_dataframe(self) -> pd.DataFrame:
        """Export anomalies as DataFrame."""
        if not self._anomalies:
            return pd.DataFrame()
        rows = [
            {
                "anomaly_id": a.anomaly_id,
                "timestamp": a.timestamp,
                "category": a.category,
                "metric": a.metric,
                "observed_value": a.observed_value,
                "expected_value": a.expected_value,
                "zscore": a.zscore,
                "severity": a.severity,
                "description": a.description,
            }
            for a in self._anomalies
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Anomaly detection summary."""
        from collections import Counter
        cats = Counter(a.category for a in self._anomalies)
        sevs = Counter(a.severity for a in self._anomalies)
        return {
            "total_anomalies": len(self._anomalies),
            "by_category": dict(cats),
            "by_severity": dict(sevs),
            "baseline_metrics": len(self._baselines),
        }
