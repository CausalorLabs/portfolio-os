"""
Orchestration — Retry & Self-Healing Engine.

Provides exponential backoff retry logic with:
  - Per-stage retry policies
  - Fallback strategies (use_cached, use_previous, skip)
  - Circuit breaker pattern
  - Retry history tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class RetryRecord:
    """Record of a retry attempt."""
    stage: str
    attempt: int
    timestamp: datetime
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    fallback_used: str | None = None


class RetryEngine:
    """
    Retry engine with exponential backoff and fallback strategies.
    """

    def __init__(self):
        cfg = _load_config().get("retries", {})
        self._default_max = cfg.get("default_max_retries", 3)
        self._default_base = cfg.get("default_backoff_base", 2.0)
        self._default_max_backoff = cfg.get("default_backoff_max", 300)
        self._stage_overrides = cfg.get("stage_overrides", {})
        self._history: list[RetryRecord] = []
        self._circuit_open: dict[str, datetime] = {}  # stage → opened time

    def _get_policy(self, stage: str) -> dict:
        override = self._stage_overrides.get(stage, {})
        return {
            "max_retries": override.get("max_retries", self._default_max),
            "backoff_base": override.get("backoff_base", self._default_base),
            "fallback": override.get("fallback", "skip"),
        }

    def execute_with_retry(
        self,
        stage: str,
        fn: Callable,
        *args,
        fallback_fn: Callable | None = None,
        **kwargs,
    ) -> tuple[bool, object]:
        """
        Execute function with retry logic.

        Returns (success, result).
        If all retries fail and fallback exists, runs fallback.
        """
        # Check circuit breaker
        if stage in self._circuit_open:
            opened = self._circuit_open[stage]
            elapsed = (datetime.now(timezone.utc) - opened).total_seconds()
            if elapsed < 60:
                logger.warning(f"  Circuit open for {stage}, skipping")
                return False, None
            else:
                del self._circuit_open[stage]

        policy = self._get_policy(stage)
        max_retries = policy["max_retries"]
        backoff_base = policy["backoff_base"]

        last_error = None
        for attempt in range(1, max_retries + 1):
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                duration = time.time() - start
                self._history.append(RetryRecord(
                    stage=stage, attempt=attempt,
                    timestamp=datetime.now(timezone.utc),
                    success=True, duration_seconds=duration,
                ))
                if attempt > 1:
                    logger.info(f"  ✓ {stage} succeeded on attempt {attempt}")
                return True, result
            except Exception as e:
                duration = time.time() - start
                last_error = str(e)
                self._history.append(RetryRecord(
                    stage=stage, attempt=attempt,
                    timestamp=datetime.now(timezone.utc),
                    success=False, error=last_error,
                    duration_seconds=duration,
                ))
                logger.warning(f"  ✗ {stage} attempt {attempt}/{max_retries}: {e}")

                if attempt < max_retries:
                    wait = min(
                        backoff_base ** attempt,
                        self._default_max_backoff,
                    )
                    logger.debug(f"    Backing off {wait:.1f}s")
                    time.sleep(wait)

        # All retries exhausted
        logger.error(f"  ✗ {stage} failed after {max_retries} attempts")

        # Try fallback
        fallback_name = policy["fallback"]
        if fallback_fn is not None:
            try:
                logger.info(f"  → Fallback ({fallback_name}) for {stage}")
                result = fallback_fn(*args, **kwargs)
                self._history.append(RetryRecord(
                    stage=stage, attempt=max_retries + 1,
                    timestamp=datetime.now(timezone.utc),
                    success=True, fallback_used=fallback_name,
                ))
                return True, result
            except Exception as e:
                logger.error(f"  ✗ Fallback for {stage} also failed: {e}")

        # Open circuit breaker
        self._circuit_open[stage] = datetime.now(timezone.utc)
        return False, None

    def get_retry_stats(self, stage: str | None = None) -> dict:
        """Get retry statistics."""
        records = self._history
        if stage:
            records = [r for r in records if r.stage == stage]

        if not records:
            return {"total": 0, "success_rate": 1.0, "avg_attempts": 0}

        successes = [r for r in records if r.success]
        return {
            "total": len(records),
            "successes": len(successes),
            "failures": len(records) - len(successes),
            "success_rate": len(successes) / len(records) if records else 1.0,
            "fallbacks_used": sum(1 for r in records if r.fallback_used),
        }

    def is_circuit_open(self, stage: str) -> bool:
        """Check if circuit breaker is open for a stage."""
        return stage in self._circuit_open

    @property
    def history(self) -> list[RetryRecord]:
        return list(self._history)
