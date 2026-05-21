"""
Orchestration — Continuous ML Operations (MLOps).

Automated ML lifecycle:
  - Retraining triggers (IC degradation, confidence collapse, drift)
  - Shadow model deployment & comparison
  - Model registry integration (MLflow)
  - Champion/challenger model promotion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class ModelRecord:
    """Record of a model version."""
    model_id: str
    trained_at: datetime
    rank_ic: float
    grade: str
    n_features: int
    is_champion: bool = False
    is_shadow: bool = False
    days_in_production: int = 0


@dataclass
class RetrainingDecision:
    """Decision about whether to retrain."""
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    urgency: str = "routine"  # routine | urgent | critical


class MLOpsEngine:
    """
    Continuous ML operations engine.

    Monitors model health and triggers retraining when needed.
    Supports shadow deployment for safe model promotion.
    """

    def __init__(self):
        cfg = _load_config().get("mlops", {})
        self._retrain_cfg = cfg.get("retraining", {})
        self._shadow_cfg = cfg.get("shadow_deployment", {})
        self._triggers = self._retrain_cfg.get("trigger_conditions", {})
        self._models: list[ModelRecord] = []
        self._shadow_results: list[dict] = []

    def check_retraining_needed(
        self,
        current_ic: float,
        mean_confidence: float,
        feature_drift_zscore: float,
        days_since_training: int,
    ) -> RetrainingDecision:
        """
        Evaluate whether model retraining is needed.

        Checks IC degradation, confidence collapse, feature drift,
        and time since last training.
        """
        reasons = []
        urgency = "routine"

        ic_threshold = self._triggers.get("ic_threshold", 0.02)
        conf_floor = self._triggers.get("confidence_floor", 0.3)
        drift_z = self._triggers.get("feature_drift_zscore", 3.0)
        max_days = self._triggers.get("max_days_since_training", 90)

        if current_ic < ic_threshold:
            reasons.append(f"IC degraded to {current_ic:.4f} (threshold: {ic_threshold})")
            urgency = "urgent"

        if mean_confidence < conf_floor:
            reasons.append(f"Mean confidence {mean_confidence:.3f} below floor {conf_floor}")
            urgency = "critical" if mean_confidence < conf_floor * 0.5 else "urgent"

        if feature_drift_zscore > drift_z:
            reasons.append(f"Feature drift z-score {feature_drift_zscore:.2f} exceeds {drift_z}")

        if days_since_training > max_days:
            reasons.append(f"Last training {days_since_training} days ago (max: {max_days})")

        should = len(reasons) > 0
        if should:
            logger.info(f"  Retraining recommended ({urgency}): {'; '.join(reasons)}")
        else:
            logger.debug("  Model health OK — no retraining needed")

        return RetrainingDecision(
            should_retrain=should,
            reasons=reasons,
            urgency=urgency,
        )

    def register_model(
        self,
        model_id: str,
        rank_ic: float,
        grade: str,
        n_features: int,
        as_shadow: bool = False,
    ) -> ModelRecord:
        """Register a new model version."""
        record = ModelRecord(
            model_id=model_id,
            trained_at=datetime.now(timezone.utc),
            rank_ic=rank_ic,
            grade=grade,
            n_features=n_features,
            is_shadow=as_shadow,
        )
        self._models.append(record)
        logger.info(f"  Registered model {model_id} (IC={rank_ic:.4f}, grade={grade}, shadow={as_shadow})")
        return record

    def get_champion(self) -> ModelRecord | None:
        """Get the current champion model."""
        champions = [m for m in self._models if m.is_champion]
        return champions[-1] if champions else None

    def promote_model(self, model_id: str) -> bool:
        """Promote a shadow model to champion."""
        target = None
        for m in self._models:
            if m.model_id == model_id:
                target = m
            else:
                m.is_champion = False

        if target:
            target.is_champion = True
            target.is_shadow = False
            logger.info(f"  ★ Promoted model {model_id} to champion")
            return True
        return False

    def record_shadow_comparison(
        self,
        champion_ic: float,
        shadow_ic: float,
        shadow_model_id: str,
    ) -> dict:
        """Record a shadow vs champion comparison."""
        min_improvement = self._shadow_cfg.get("min_improvement_ic", 0.005)
        improvement = shadow_ic - champion_ic
        should_promote = improvement >= min_improvement

        result = {
            "timestamp": datetime.now(timezone.utc),
            "shadow_model_id": shadow_model_id,
            "champion_ic": champion_ic,
            "shadow_ic": shadow_ic,
            "improvement": improvement,
            "min_required": min_improvement,
            "should_promote": should_promote,
        }
        self._shadow_results.append(result)

        if should_promote:
            logger.info(f"  Shadow model {shadow_model_id} outperforms champion by {improvement:.4f}")
        return result

    def get_models(self) -> list[ModelRecord]:
        return list(self._models)

    def summary(self) -> dict:
        champion = self.get_champion()
        return {
            "total_models": len(self._models),
            "champion": champion.model_id if champion else None,
            "champion_ic": champion.rank_ic if champion else None,
            "shadows": sum(1 for m in self._models if m.is_shadow),
            "comparisons": len(self._shadow_results),
        }
