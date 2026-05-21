"""
Deployment — MVP Stabilization Report Generator.

Sprint 8: Comprehensive system health report.
  - Architecture summary
  - Model performance evaluation
  - Operational evaluation
  - Trust assessment
  - Deployment readiness
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger


class StabilizationReport:
    """
    MVP Stabilization Report generator.

    Produces a comprehensive assessment of system readiness
    for personal deployment.
    """

    def __init__(self):
        self._sections: dict[str, dict] = {}

    def add_section(self, name: str, data: dict) -> None:
        """Add a section to the report."""
        self._sections[name] = data

    def assess_model_health(
        self,
        rank_ic: float,
        grade: str,
        confidence_mean: float,
        feature_drift_pct: float,
    ) -> dict:
        """Assess ML model readiness."""
        assessment = {
            "rank_ic": rank_ic,
            "grade": grade,
            "confidence_mean": confidence_mean,
            "feature_drift_pct": feature_drift_pct,
            "model_ready": grade in ("A", "B") and rank_ic > 0.02,
            "concerns": [],
        }
        if rank_ic < 0.02:
            assessment["concerns"].append(f"Low Rank IC: {rank_ic:.4f}")
        if grade in ("D", "F"):
            assessment["concerns"].append(f"Poor grade: {grade}")
        if confidence_mean < 0.3:
            assessment["concerns"].append(f"Low confidence: {confidence_mean:.3f}")
        if feature_drift_pct > 20:
            assessment["concerns"].append(f"High feature drift: {feature_drift_pct:.1f}%")

        self._sections["model_health"] = assessment
        return assessment

    def assess_operations(
        self,
        pipeline_success_rate: float,
        sla_compliance_pct: float,
        uptime_pct: float,
        avg_pipeline_minutes: float,
    ) -> dict:
        """Assess operational readiness."""
        assessment = {
            "pipeline_success_rate": pipeline_success_rate,
            "sla_compliance_pct": sla_compliance_pct,
            "uptime_pct": uptime_pct,
            "avg_pipeline_minutes": avg_pipeline_minutes,
            "ops_ready": pipeline_success_rate > 0.95 and sla_compliance_pct > 90,
            "concerns": [],
        }
        if pipeline_success_rate < 0.95:
            assessment["concerns"].append(f"Low success rate: {pipeline_success_rate*100:.1f}%")
        if sla_compliance_pct < 90:
            assessment["concerns"].append(f"SLA compliance: {sla_compliance_pct:.1f}%")

        self._sections["operations"] = assessment
        return assessment

    def assess_validation(
        self,
        checks_passed: int,
        checks_total: int,
        critical_failures: int,
        failure_sim_survived: int,
        failure_sim_total: int,
    ) -> dict:
        """Assess validation results."""
        assessment = {
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "pass_rate": checks_passed / checks_total if checks_total else 1.0,
            "critical_failures": critical_failures,
            "failure_sim_survived": failure_sim_survived,
            "failure_sim_total": failure_sim_total,
            "validation_ready": critical_failures == 0,
            "concerns": [],
        }
        if critical_failures > 0:
            assessment["concerns"].append(f"{critical_failures} critical validation failures")

        self._sections["validation"] = assessment
        return assessment

    def assess_trust(
        self,
        trust_score: float,
        recommended_mode: str,
    ) -> dict:
        """Assess trust level."""
        assessment = {
            "trust_score": trust_score,
            "recommended_mode": recommended_mode,
            "concerns": [],
        }
        if trust_score < 0.5:
            assessment["concerns"].append(f"Low trust score: {trust_score:.3f}")

        self._sections["trust"] = assessment
        return assessment

    def generate(self) -> dict:
        """Generate the full stabilization report."""
        all_concerns = []
        for section in self._sections.values():
            all_concerns.extend(section.get("concerns", []))

        readiness_flags = {
            "model": self._sections.get("model_health", {}).get("model_ready", False),
            "operations": self._sections.get("operations", {}).get("ops_ready", False),
            "validation": self._sections.get("validation", {}).get("validation_ready", False),
        }
        deployment_ready = all(readiness_flags.values()) and len(all_concerns) <= 2

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sections": dict(self._sections),
            "readiness": readiness_flags,
            "deployment_ready": deployment_ready,
            "total_concerns": len(all_concerns),
            "concerns": all_concerns,
            "recommendation": (
                "READY for personal deployment"
                if deployment_ready
                else f"NOT READY — {len(all_concerns)} concerns to address"
            ),
        }

        logger.info(f"  Stabilization: {'READY' if deployment_ready else 'NOT READY'} ({len(all_concerns)} concerns)")
        return report

    def to_dataframe(self) -> pd.DataFrame:
        report = self.generate()
        rows = []
        for section_name, section_data in report.get("sections", {}).items():
            for key, value in section_data.items():
                if key != "concerns":
                    rows.append({"section": section_name, "metric": key, "value": str(value)})
        return pd.DataFrame(rows)
