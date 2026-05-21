"""
Execution Engine — Portfolio State Machine.

States:
  idle → evaluating → pending_approval → executing → settling → settled → idle

Enables:
  - Event-driven workflows
  - Operational observability
  - Future broker integrations
  - Automation guardrails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/execution_engine.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# Valid state transitions
TRANSITIONS = {
    "idle": ["evaluating"],
    "evaluating": ["pending_approval", "idle"],  # can go back to idle (no trade)
    "pending_approval": ["executing", "idle"],    # can be rejected
    "executing": ["settling"],
    "settling": ["settled"],
    "settled": ["idle"],
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: datetime
    from_state: str
    to_state: str
    reason: str
    metadata: dict = field(default_factory=dict)


class PortfolioStateMachine:
    """
    Manages portfolio operational state.

    Enforces valid transitions and maintains full audit trail.
    """

    def __init__(self):
        cfg = _load_config().get("state_machine", {})
        self._state = cfg.get("default_state", "idle")
        self._auto_settle = cfg.get("auto_settle", True)
        self._settlement_days = cfg.get("settlement_days", 2)
        self._history: list[StateTransition] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == "idle"

    @property
    def is_executing(self) -> bool:
        return self._state in ("executing", "settling")

    def can_transition(self, target: str) -> bool:
        """Check if transition to target state is valid."""
        valid = TRANSITIONS.get(self._state, [])
        return target in valid

    def transition(
        self,
        target: str,
        reason: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """
        Attempt state transition.

        Returns True if successful, False if invalid.
        """
        if not self.can_transition(target):
            logger.warning(
                f"  State machine: invalid transition "
                f"{self._state} → {target}"
            )
            return False

        record = StateTransition(
            timestamp=datetime.now(),
            from_state=self._state,
            to_state=target,
            reason=reason,
            metadata=metadata or {},
        )

        self._history.append(record)
        old = self._state
        self._state = target

        logger.info(f"  State: {old} → {target} ({reason})")

        # Auto-settle
        if self._auto_settle and target == "settling":
            self.transition("settled", reason="auto_settle")
            self.transition("idle", reason="cycle_complete")

        return True

    def start_evaluation(self, reason: str = "scheduled") -> bool:
        """Begin evaluation cycle."""
        return self.transition("evaluating", reason)

    def approve_trade(self, reason: str = "utility_positive") -> bool:
        """Approve pending trade and begin execution."""
        if self._state == "evaluating":
            if not self.transition("pending_approval", reason):
                return False
        if self._state == "pending_approval":
            return self.transition("executing", reason)
        return False

    def reject_trade(self, reason: str = "insufficient_utility") -> bool:
        """Reject trade — return to idle."""
        return self.transition("idle", reason)

    def execute_and_settle(self, reason: str = "executed") -> bool:
        """Execute trade and auto-settle."""
        if self._state == "executing":
            return self.transition("settling", reason)
        return False

    def cancel(self, reason: str = "cancelled") -> bool:
        """Cancel and return to idle from any state."""
        if self._state == "idle":
            return True
        # Force transition to idle
        record = StateTransition(
            timestamp=datetime.now(),
            from_state=self._state,
            to_state="idle",
            reason=f"CANCEL: {reason}",
        )
        self._history.append(record)
        self._state = "idle"
        logger.info(f"  State: CANCELLED → idle ({reason})")
        return True

    # ── History ─────────────────────────────────────────────────────────

    def history(self) -> list[StateTransition]:
        return list(self._history)

    def summary(self) -> dict:
        """State machine summary."""
        n_cycles = sum(1 for t in self._history if t.to_state == "settled")
        n_rejections = sum(
            1 for t in self._history
            if t.from_state in ("evaluating", "pending_approval")
            and t.to_state == "idle"
        )

        return {
            "current_state": self._state,
            "total_transitions": len(self._history),
            "completed_cycles": n_cycles,
            "rejections": n_rejections,
            "last_transition": (
                self._history[-1].timestamp.isoformat()
                if self._history else None
            ),
        }
