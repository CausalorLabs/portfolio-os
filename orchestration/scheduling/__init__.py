"""
Orchestration — Scheduling Framework.

Lightweight cron-like scheduler for portfolio operations:
  - Daily / weekly / monthly cadences
  - Next-run computation
  - Schedule-aware task planning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time
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
class ScheduledTask:
    """A scheduled task."""
    name: str
    cadence: str  # daily | weekly | monthly
    task_fn: Callable | None = None
    last_run: datetime | None = None
    next_run: datetime | None = None
    enabled: bool = True


class Scheduler:
    """
    Lightweight scheduler for portfolio operational cadences.

    No external dependencies — evaluates what should run based on current time.
    """

    def __init__(self):
        cfg = _load_config().get("scheduling", {})
        self._tasks: dict[str, ScheduledTask] = {}
        self._daily_cfg = cfg.get("daily", {})
        self._weekly_cfg = cfg.get("weekly", {})
        self._monthly_cfg = cfg.get("monthly", {})
        self._history: list[dict] = []

    def register(
        self,
        name: str,
        cadence: str,
        task_fn: Callable | None = None,
    ) -> None:
        """Register a task with a cadence."""
        self._tasks[name] = ScheduledTask(
            name=name,
            cadence=cadence,
            task_fn=task_fn,
        )

    def should_run(
        self,
        name: str,
        now: datetime | None = None,
    ) -> bool:
        """Check if a task should run at the given time."""
        task = self._tasks.get(name)
        if not task or not task.enabled:
            return False

        now = now or datetime.now(timezone.utc)

        if task.last_run is None:
            return True

        elapsed = now - task.last_run
        if task.cadence == "daily":
            return elapsed >= timedelta(hours=20)
        elif task.cadence == "weekly":
            return elapsed >= timedelta(days=6)
        elif task.cadence == "monthly":
            return elapsed >= timedelta(days=28)
        return False

    def get_due_tasks(self, now: datetime | None = None) -> list[str]:
        """Get all tasks that should run now."""
        return [name for name in self._tasks if self.should_run(name, now)]

    def mark_completed(self, name: str, now: datetime | None = None) -> None:
        """Mark a task as completed."""
        task = self._tasks.get(name)
        if task:
            task.last_run = now or datetime.now(timezone.utc)
            self._history.append({
                "task": name,
                "completed_at": task.last_run,
                "cadence": task.cadence,
            })

    def get_schedule(self) -> list[dict]:
        """Get all registered tasks and their status."""
        return [
            {
                "name": t.name,
                "cadence": t.cadence,
                "last_run": t.last_run,
                "enabled": t.enabled,
            }
            for t in self._tasks.values()
        ]

    def get_daily_tasks(self) -> list[str]:
        """Get configured daily task names."""
        return self._daily_cfg.get("tasks", [])

    def get_weekly_tasks(self) -> list[str]:
        """Get configured weekly task names."""
        return self._weekly_cfg.get("tasks", [])

    def get_monthly_tasks(self) -> list[str]:
        """Get configured monthly task names."""
        return self._monthly_cfg.get("tasks", [])

    def summary(self) -> dict:
        return {
            "total_tasks": len(self._tasks),
            "enabled": sum(1 for t in self._tasks.values() if t.enabled),
            "due": len(self.get_due_tasks()),
            "completed_today": sum(
                1 for h in self._history
                if h["completed_at"].date() == datetime.now(timezone.utc).date()
            ),
        }
