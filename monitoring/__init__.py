"""
Structured logging & observability.

Provides:
  - Execution ID tracking across pipeline runs
  - Structured log format with JSON output option
  - Pipeline timing context manager
  - Step-level instrumentation
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from loguru import logger

# ── Execution context ────────────────────────────────────────────────────────

_current_execution_id: str | None = None
_current_pipeline: str | None = None


def new_execution_id() -> str:
    """Generate a new execution ID."""
    return uuid.uuid4().hex[:12]


def get_execution_id() -> str:
    """Get the current execution ID, or create one."""
    global _current_execution_id
    if _current_execution_id is None:
        _current_execution_id = new_execution_id()
    return _current_execution_id


def set_execution_context(execution_id: str, pipeline: str) -> None:
    """Set the current execution context."""
    global _current_execution_id, _current_pipeline
    _current_execution_id = execution_id
    _current_pipeline = pipeline


# ── Logger setup ─────────────────────────────────────────────────────────────


def _format_record(record: dict) -> str:
    """Custom log format with execution ID."""
    exec_id = _current_execution_id or "------"
    pipeline = _current_pipeline or "system"
    elapsed = record.get("elapsed", "")
    return (
        "<green>{time:HH:mm:ss}</green> "
        f"<cyan>[{exec_id}]</cyan> "
        f"<blue>{pipeline}</blue> | "
        "<level>{level: <8}</level> | "
        "{message}\n"
    )


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure loguru with structured format."""
    logger.remove()

    if json_output:
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=_format_record,
            colorize=True,
        )

    # File output (always JSON for machine parsing)
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        serialize=True,
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )


# ── Pipeline timing ─────────────────────────────────────────────────────────


@contextmanager
def pipeline_context(name: str) -> Generator[str, None, None]:
    """Context manager that sets execution context and times the pipeline."""
    exec_id = new_execution_id()
    set_execution_context(exec_id, name)

    logger.info(f"Pipeline started: {name}")
    logger.info(f"Execution ID: {exec_id}")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    start = time.perf_counter()
    try:
        yield exec_id
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"Pipeline FAILED after {elapsed:.1f}s: {name}")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"Pipeline completed in {elapsed:.1f}s: {name}")
    finally:
        global _current_execution_id, _current_pipeline
        _current_execution_id = None
        _current_pipeline = None


@contextmanager
def step_timer(step_name: str) -> Generator[None, None, None]:
    """Time an individual pipeline step."""
    logger.info(f"▸ {step_name}")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error(f"  ✗ {step_name} FAILED ({elapsed:.1f}s)")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"  ✓ {step_name} ({elapsed:.1f}s)")
