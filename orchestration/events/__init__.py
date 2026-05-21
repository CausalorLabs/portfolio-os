"""
Orchestration — Event-Driven Architecture.

Event bus for the portfolio operating system:
  - Publish/subscribe event model
  - Event routing to handlers
  - Event history with retention
  - Dependency-aware event triggering
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class Event:
    """A system event."""
    event_id: str
    timestamp: datetime
    event_type: str
    source: str
    payload: dict = field(default_factory=dict)
    status: str = "pending"  # pending | processed | failed
    parent_id: str | None = None


class EventBus:
    """
    Central event bus for the portfolio operating system.

    Supports publish/subscribe with handler registration,
    event history, and dependency-aware triggering.
    """

    def __init__(self):
        cfg = _load_config().get("events", {})
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._events: list[Event] = []
        self._max_events = cfg.get("max_events", 10000)
        self._retention_hours = cfg.get("retention_hours", 168)

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        self._handlers[event_type].append(handler)

    def publish(
        self,
        event_type: str,
        source: str,
        payload: dict | None = None,
        parent_id: str | None = None,
    ) -> Event:
        """Publish an event and invoke registered handlers."""
        event = Event(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            source=source,
            payload=payload or {},
            parent_id=parent_id,
        )

        self._events.append(event)
        self._trim()

        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
                event.status = "processed"
            except Exception as e:
                event.status = "failed"
                logger.error(f"  Event handler failed [{event_type}]: {e}")

        logger.debug(f"  Event: {event_type} from {source} ({event.status})")
        return event

    def _trim(self) -> None:
        """Trim events beyond retention/max limits."""
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)
        self._events = [e for e in self._events if e.timestamp >= cutoff]

    def recent(self, n: int = 20, event_type: str | None = None) -> list[Event]:
        """Get recent events, optionally filtered by type."""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:n]

    def pending(self) -> list[Event]:
        """Get unprocessed events."""
        return [e for e in self._events if e.status == "pending"]

    def to_dataframe(self) -> pd.DataFrame:
        """Export events as DataFrame."""
        if not self._events:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "source": e.source,
                "status": e.status,
                "parent_id": e.parent_id or "",
            }
            for e in self._events
        ])

    def summary(self) -> dict:
        from collections import Counter
        types = Counter(e.event_type for e in self._events)
        statuses = Counter(e.status for e in self._events)
        return {
            "total_events": len(self._events),
            "by_type": dict(types),
            "by_status": dict(statuses),
        }
