"""
Validation reporting — generate institutional-style research reports.

Outputs: CSV summaries + consolidated validation report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


REPORTS_DIR = Path("reports")


def generate_validation_report(
    walkforward_results: pd.DataFrame | None = None,
    regime_results: pd.DataFrame | None = None,
    sensitivity_results: pd.DataFrame | None = None,
    stress_results: pd.DataFrame | None = None,
    signal_decay: pd.DataFrame | None = None,
    monte_carlo_summary: dict | None = None,
    overfitting_report: dict | None = None,
    diagnostics: dict | None = None,
    research_score: dict | None = None,
) -> dict[str, Path]:
    """
    Save all validation artifacts and generate consolidated report.

    Returns
    -------
    dict
        Mapping of report name → file path.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Walk-forward results
    if walkforward_results is not None and not walkforward_results.empty:
        path = REPORTS_DIR / "walkforward_results.csv"
        walkforward_results.to_csv(path, index=False)
        outputs["walkforward"] = path
        logger.info(f"  Saved → {path}")

    # Regime performance
    if regime_results is not None and not regime_results.empty:
        path = REPORTS_DIR / "regime_performance.csv"
        regime_results.to_csv(path, index=False)
        outputs["regimes"] = path
        logger.info(f"  Saved → {path}")

    # Parameter sensitivity
    if sensitivity_results is not None and not sensitivity_results.empty:
        path = REPORTS_DIR / "parameter_sensitivity.csv"
        sensitivity_results.to_csv(path, index=False)
        outputs["sensitivity"] = path
        logger.info(f"  Saved → {path}")

    # Stress tests
    if stress_results is not None and not stress_results.empty:
        path = REPORTS_DIR / "stress_test_results.csv"
        stress_results.to_csv(path, index=False)
        outputs["stress"] = path
        logger.info(f"  Saved → {path}")

    # Signal decay
    if signal_decay is not None and not signal_decay.empty:
        path = REPORTS_DIR / "signal_decay.csv"
        signal_decay.to_csv(path, index=False)
        outputs["signal_decay"] = path
        logger.info(f"  Saved → {path}")

    # Monte Carlo summary
    if monte_carlo_summary is not None:
        mc_df = pd.DataFrame([monte_carlo_summary])
        path = REPORTS_DIR / "monte_carlo_summary.csv"
        mc_df.to_csv(path, index=False)
        outputs["monte_carlo"] = path
        logger.info(f"  Saved → {path}")

    # Research score
    if research_score is not None:
        score_df = pd.DataFrame([research_score])
        path = REPORTS_DIR / "research_score.csv"
        score_df.to_csv(path, index=False)
        outputs["research_score"] = path
        logger.info(f"  Saved → {path}")

    # Overfitting report
    if overfitting_report is not None:
        # Flatten flags for CSV
        flags_data = []
        for f in overfitting_report.get("flags", []):
            flags_data.append(f)
        if flags_data:
            flags_df = pd.DataFrame(flags_data)
            path = REPORTS_DIR / "overfitting_flags.csv"
            flags_df.to_csv(path, index=False)
            outputs["overfitting"] = path
            logger.info(f"  Saved → {path}")

    # Diagnostics summary
    if diagnostics is not None:
        diag_rows = []
        for key, val in diagnostics.items():
            if isinstance(val, dict):
                grade = val.get("grade", val.get("score", str(val)))
                diag_rows.append({"metric": key, "grade": grade, "detail": str(val)})
            else:
                diag_rows.append({"metric": key, "grade": str(val), "detail": ""})
        diag_df = pd.DataFrame(diag_rows)
        path = REPORTS_DIR / "diagnostics_summary.csv"
        diag_df.to_csv(path, index=False)
        outputs["diagnostics"] = path
        logger.info(f"  Saved → {path}")

    logger.info(f"\nValidation reports: {len(outputs)} files saved to reports/")
    return outputs
