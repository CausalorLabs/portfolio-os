"""
Deployment — Portfolio Trust Calibration Engine.

Sprint 8: Trust scoring that controls automation authority.
  - Composite trust score from 5 dimensions
  - Trust → approval mode mapping
  - Trust degradation on anomalies
  - Trust history tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class TrustAssessment:
    """Trust calibration result."""
    model_health: float
    data_quality: float
    regime_stability: float
    execution_reliability: float
    operational_health: float
    overall_trust: float
    recommended_mode: str  # advisory | assisted | autonomous
    timestamp: datetime


class TrustCalibrator:
    """
    Portfolio trust calibration engine.

    Computes a composite trust score from 5 dimensions:
    model health, data quality, regime stability, execution reliability,
    and operational health.

    Trust score determines automation authority:
      - High (≥0.80): autonomous execution
      - Medium (0.50-0.80): assisted (human approval)
      - Low (<0.50): advisory only
    """

    def __init__(self):
        cfg = _load_config().get("trust", {})
        self._weights = cfg.get("weights", {
            "model_health": 0.25,
            "data_quality": 0.25,
            "regime_stability": 0.20,
            "execution_reliability": 0.15,
            "operational_health": 0.15,
        })
        thresholds = cfg.get("thresholds", {})
        self._high_threshold = thresholds.get("high_trust", 0.80)
        self._medium_threshold = thresholds.get("medium_trust", 0.50)
        self._low_threshold = thresholds.get("low_trust", 0.30)

        self._reductions = cfg.get("reduce_on", {})
        self._history: list[TrustAssessment] = []

    def calibrate(
        self,
        model_health: float = 0.5,
        data_quality: float = 0.5,
        regime_stability: float = 0.5,
        execution_reliability: float = 0.5,
        operational_health: float = 0.5,
        penalties: dict[str, bool] | None = None,
    ) -> TrustAssessment:
        """
        Compute composite trust score.

        Args:
            model_health: ML model quality (IC, confidence, grade)
            data_quality: Data freshness and completeness
            regime_stability: Regime consistency (no flapping)
            execution_reliability: Trade execution success rate
            operational_health: Pipeline success rate
            penalties: Dict of penalty flags {feature_drift, unstable_covariance, etc.}
        """
        # Base scores
        scores = {
            "model_health": max(0, min(1, model_health)),
            "data_quality": max(0, min(1, data_quality)),
            "regime_stability": max(0, min(1, regime_stability)),
            "execution_reliability": max(0, min(1, execution_reliability)),
            "operational_health": max(0, min(1, operational_health)),
        }

        # Apply penalties
        if penalties:
            for penalty_name, is_active in penalties.items():
                if is_active:
                    reduction = self._reductions.get(penalty_name, 0.05)
                    # Apply penalty to overall score later
                    for key in scores:
                        scores[key] = max(0, scores[key] - reduction * 0.5)

        # Weighted composite
        overall = sum(
            scores[k] * self._weights.get(k, 0.2) for k in scores
        )
        overall = max(0, min(1, overall))

        # Determine mode
        if overall >= self._high_threshold:
            mode = "autonomous"
        elif overall >= self._medium_threshold:
            mode = "assisted"
        else:
            mode = "advisory"

        assessment = TrustAssessment(
            model_health=scores["model_health"],
            data_quality=scores["data_quality"],
            regime_stability=scores["regime_stability"],
            execution_reliability=scores["execution_reliability"],
            operational_health=scores["operational_health"],
            overall_trust=overall,
            recommended_mode=mode,
            timestamp=datetime.now(timezone.utc),
        )
        self._history.append(assessment)

        logger.info(f"  Trust: {overall:.3f} → {mode}")
        return assessment

    def get_history(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "model_health": a.model_health,
                "data_quality": a.data_quality,
                "regime_stability": a.regime_stability,
                "execution_reliability": a.execution_reliability,
                "operational_health": a.operational_health,
                "overall_trust": a.overall_trust,
                "recommended_mode": a.recommended_mode,
            }
            for a in self._history
        ])

    def latest(self) -> TrustAssessment | None:
        return self._history[-1] if self._history else None

    def summary(self) -> dict:
        latest = self.latest()
        return {
            "overall_trust": latest.overall_trust if latest else None,
            "recommended_mode": latest.recommended_mode if latest else "assisted",
            "assessments": len(self._history),
        }
