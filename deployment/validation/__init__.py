"""
Deployment — End-to-End Validation Framework.

Sprint 8: Pipeline integrity and portfolio consistency checks.
  - Schema validation across pipeline stages
  - Data freshness checks
  - Allocation sanity checks
  - Constraint violation detection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class CheckResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    details: str
    severity: str = "INFO"  # INFO | WARNING | CRITICAL


class ValidationFramework:
    """
    End-to-end pipeline and portfolio validation.

    Runs integrity checks across all pipeline stages to ensure
    data quality, schema consistency, and allocation sanity.
    """

    def __init__(self):
        cfg = _load_config().get("validation", {})
        self._max_alloc_sum = cfg.get("max_allocation_sum", 1.05)
        self._min_alloc_sum = cfg.get("min_allocation_sum", 0.90)
        self._max_leverage = cfg.get("max_leverage", 1.0)
        self._max_single_pos = cfg.get("max_single_position", 0.30)
        self._checks: list[CheckResult] = []

    def check_schema_consistency(
        self,
        prices: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        weights: pd.DataFrame | None = None,
    ) -> list[CheckResult]:
        """Validate schema consistency across pipeline artifacts."""
        results = []

        if prices is not None:
            has_cols = {"date", "ticker"}.issubset(set(prices.columns))
            results.append(CheckResult(
                "schema_prices",
                has_cols,
                "Prices schema valid" if has_cols else "Missing date/ticker columns",
                "CRITICAL" if not has_cols else "INFO",
            ))

        if features is not None:
            has_cols = "date" in features.columns
            results.append(CheckResult(
                "schema_features",
                has_cols,
                "Features schema valid" if has_cols else "Missing date column",
                "CRITICAL" if not has_cols else "INFO",
            ))

        if weights is not None:
            has_cols = "ticker" in weights.columns and "target_weight" in weights.columns
            results.append(CheckResult(
                "schema_weights",
                has_cols,
                "Weights schema valid" if has_cols else "Missing ticker/target_weight",
                "CRITICAL" if not has_cols else "INFO",
            ))

        self._checks.extend(results)
        return results

    def check_stale_data(
        self,
        file_paths: list[str] | None = None,
        max_hours: float = 24,
    ) -> list[CheckResult]:
        """Check for stale data files."""
        results = []
        paths = file_paths or [
            "data/processed/inr_prices.parquet",
            "data/processed/features.parquet",
            "data/processed/target_weights.parquet",
        ]

        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                results.append(CheckResult(
                    f"stale_{path.stem}", False,
                    f"File missing: {path_str}", "WARNING",
                ))
                continue

            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
            fresh = age_hours <= max_hours
            results.append(CheckResult(
                f"stale_{path.stem}", fresh,
                f"{path.name}: {age_hours:.1f}h old" + (" (stale)" if not fresh else ""),
                "WARNING" if not fresh else "INFO",
            ))

        self._checks.extend(results)
        return results

    def check_allocation_sanity(
        self,
        weights: dict[str, float] | pd.Series | None = None,
    ) -> list[CheckResult]:
        """Check allocation for impossible values."""
        results = []
        if weights is None:
            return results

        if isinstance(weights, pd.Series):
            w = weights.to_dict()
        else:
            w = dict(weights)

        total = sum(w.values())

        # Sum check
        sum_ok = self._min_alloc_sum <= total <= self._max_alloc_sum
        results.append(CheckResult(
            "allocation_sum", sum_ok,
            f"Weight sum: {total:.4f} (range: {self._min_alloc_sum}-{self._max_alloc_sum})",
            "CRITICAL" if not sum_ok else "INFO",
        ))

        # Negative weights
        neg = {k: v for k, v in w.items() if v < 0}
        results.append(CheckResult(
            "negative_weights", len(neg) == 0,
            "No negative weights" if not neg else f"Negative weights: {neg}",
            "CRITICAL" if neg else "INFO",
        ))

        # Concentration
        concentrated = {k: v for k, v in w.items() if v > self._max_single_pos}
        results.append(CheckResult(
            "concentration", len(concentrated) == 0,
            "No over-concentrated positions" if not concentrated else f"Over-concentrated: {concentrated}",
            "WARNING" if concentrated else "INFO",
        ))

        # Leverage
        leverage = sum(abs(v) for v in w.values())
        lev_ok = leverage <= self._max_leverage + 0.05
        results.append(CheckResult(
            "leverage", lev_ok,
            f"Leverage: {leverage:.4f} (max: {self._max_leverage})",
            "CRITICAL" if not lev_ok else "INFO",
        ))

        self._checks.extend(results)
        return results

    def check_nan_inf(
        self,
        df: pd.DataFrame | None = None,
        name: str = "data",
    ) -> list[CheckResult]:
        """Check for NaN/Inf values in a DataFrame."""
        results = []
        if df is None:
            return results

        nan_cols = df.columns[df.isna().any()].tolist()
        inf_cols = df.columns[df.apply(lambda s: np.isinf(s) if s.dtype in ["float64", "float32"] else False).any()].tolist()

        results.append(CheckResult(
            f"nan_{name}", len(nan_cols) == 0,
            f"No NaN in {name}" if not nan_cols else f"NaN in: {nan_cols[:5]}",
            "WARNING" if nan_cols else "INFO",
        ))
        results.append(CheckResult(
            f"inf_{name}", len(inf_cols) == 0,
            f"No Inf in {name}" if not inf_cols else f"Inf in: {inf_cols[:5]}",
            "CRITICAL" if inf_cols else "INFO",
        ))

        self._checks.extend(results)
        return results

    def run_all_checks(
        self,
        prices: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        weights: pd.DataFrame | None = None,
        allocation: dict[str, float] | None = None,
    ) -> list[CheckResult]:
        """Run all validation checks."""
        self._checks.clear()
        self.check_schema_consistency(prices, features, weights)
        self.check_stale_data()
        self.check_allocation_sanity(allocation)
        if prices is not None:
            self.check_nan_inf(prices, "prices")
        if features is not None:
            self.check_nan_inf(features, "features")

        passed = sum(1 for c in self._checks if c.passed)
        total = len(self._checks)
        critical = sum(1 for c in self._checks if not c.passed and c.severity == "CRITICAL")

        logger.info(f"  Validation: {passed}/{total} checks passed, {critical} critical failures")
        return list(self._checks)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._checks:
            return pd.DataFrame()
        return pd.DataFrame([
            {"check": c.check_name, "passed": c.passed, "details": c.details, "severity": c.severity}
            for c in self._checks
        ])

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self._checks)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self._checks if not c.passed and c.severity == "CRITICAL"]
