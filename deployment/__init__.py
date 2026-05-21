"""
Deployment Engine — MVP Hardening, Validation & Personal Deployment.

Sprint 8: Personally deployable portfolio operating system.

Subsystems:
  - Validation Framework: E2E pipeline integrity checks
  - Failure Simulator: Chaos testing for graceful degradation
  - Trust Calibrator: Trust scoring → automation authority
  - Walk-Forward Evaluator: Long-horizon survivability
  - Security Layer: API auth, rate limiting, CORS
  - Hardening Engine: Backup, recovery, reproducibility
  - Stabilization Report: MVP readiness assessment
  - Human Override: Advisory → Assisted → Autonomous modes
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from deployment.validation import ValidationFramework
from deployment.failure_sim import FailureSimulator
from deployment.trust import TrustCalibrator
from deployment.walkforward import WalkForwardEvaluator
from deployment.security import SecurityLayer
from deployment.hardening import HardeningEngine
from deployment.report import StabilizationReport


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class DeploymentEngine:
    """
    Central deployment and hardening engine.

    Coordinates validation, trust calibration, and operational
    readiness for personal portfolio deployment.
    """

    def __init__(self):
        cfg = _load_config()

        # Override config
        override_cfg = cfg.get("override", {})
        self._default_mode = override_cfg.get("default_mode", "assisted")
        self._approval_modes = override_cfg.get("approval_modes", {})

        # Shadow config
        shadow_cfg = cfg.get("shadow", {})
        self._shadow_enabled = shadow_cfg.get("enabled", True)
        self._shadow_mode = shadow_cfg.get("mode", "assisted")

        # Subsystems
        self.validation = ValidationFramework()
        self.failure_sim = FailureSimulator()
        self.trust = TrustCalibrator()
        self.walkforward = WalkForwardEvaluator()
        self.security = SecurityLayer()
        self.hardening = HardeningEngine()
        self.report = StabilizationReport()

        # Current mode
        self._current_mode = self._default_mode

    # ── Human Override Layer ─────────────────────────────────────────────

    def get_approval_mode(self) -> str:
        """Get current approval mode."""
        return self._current_mode

    def set_approval_mode(self, mode: str) -> None:
        """Set approval mode (advisory, assisted, autonomous)."""
        if mode not in ("advisory", "assisted", "autonomous"):
            raise ValueError(f"Invalid mode: {mode}")
        self._current_mode = mode
        logger.info(f"  Approval mode set to: {mode}")

    def should_auto_execute(self) -> bool:
        """Check if auto-execution is allowed in current mode."""
        mode_cfg = self._approval_modes.get(self._current_mode, {})
        return mode_cfg.get("auto_execute", False)

    def check_execution_approval(
        self,
        trust_score: float,
        trade_count: int,
    ) -> dict:
        """
        Check if execution should proceed based on mode and trust.

        Returns approval decision with reasoning.
        """
        mode = self._current_mode

        if mode == "advisory":
            return {
                "approved": False,
                "mode": mode,
                "reason": "Advisory mode — recommendations only",
                "action": "display_recommendation",
            }
        elif mode == "autonomous":
            min_trust = self._approval_modes.get("autonomous", {}).get("min_trust_score", 0.80)
            if trust_score >= min_trust:
                return {
                    "approved": True,
                    "mode": mode,
                    "reason": f"Autonomous mode — trust {trust_score:.3f} ≥ {min_trust}",
                    "action": "auto_execute",
                }
            else:
                return {
                    "approved": False,
                    "mode": mode,
                    "reason": f"Trust {trust_score:.3f} below autonomous threshold {min_trust}",
                    "action": "require_approval",
                }
        else:  # assisted
            return {
                "approved": False,
                "mode": mode,
                "reason": f"Assisted mode — {trade_count} trades require approval",
                "action": "require_approval",
            }

    # ── Deployment Readiness ─────────────────────────────────────────────

    def run_readiness_check(
        self,
        trust_score: float = 0.5,
        rank_ic: float = 0.0,
        grade: str = "C",
        confidence_mean: float = 0.5,
        pipeline_success_rate: float = 1.0,
        sla_compliance_pct: float = 100.0,
    ) -> dict:
        """
        Run comprehensive deployment readiness assessment.

        Returns the stabilization report.
        """
        self.report.assess_model_health(
            rank_ic=rank_ic,
            grade=grade,
            confidence_mean=confidence_mean,
            feature_drift_pct=0.0,
        )
        self.report.assess_operations(
            pipeline_success_rate=pipeline_success_rate,
            sla_compliance_pct=sla_compliance_pct,
            uptime_pct=100.0,
            avg_pipeline_minutes=5.0,
        )

        val_results = self.validation.run_all_checks()
        passed = sum(1 for c in val_results if c.passed)
        critical = len(self.validation.critical_failures)
        self.report.assess_validation(
            checks_passed=passed,
            checks_total=len(val_results),
            critical_failures=critical,
            failure_sim_survived=0,
            failure_sim_total=0,
        )
        self.report.assess_trust(trust_score=trust_score, recommended_mode=self._current_mode)

        return self.report.generate()

    def summary(self) -> dict:
        trust_latest = self.trust.latest()
        return {
            "mode": self._current_mode,
            "shadow_enabled": self._shadow_enabled,
            "trust_score": trust_latest.overall_trust if trust_latest else None,
            "validation_passed": self.validation.all_passed,
            "backups": self.hardening.summary()["total_backups"],
            "security": self.security.summary(),
        }
