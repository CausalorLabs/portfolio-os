"""
Orchestration — State Coordination Engine.

Global system state management:
  - Unified state snapshot (regime, NAV, model, pipeline, trust)
  - State persistence (JSON + parquet history)
  - State change detection
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class StateCoordinator:
    """
    Global system state coordinator.

    Maintains a unified view of the system state:
    regime, portfolio NAV, model version, pipeline health, trust score.
    """

    def __init__(self):
        cfg = _load_config().get("state", {})
        self._state_file = Path(cfg.get("state_file", "data/processed/system_state.json"))
        self._history_file = Path(cfg.get("history_file", "data/processed/state_history.parquet"))
        self._snapshot_on_change = cfg.get("snapshot_on_change", True)

        self._state: dict = {
            "timestamp": None,
            "regime": "unknown",
            "portfolio_nav": 0.0,
            "n_positions": 0,
            "model_version": "",
            "pipeline_status": "unknown",
            "trust_score": 0.5,
            "approval_mode": "assisted",
            "last_ingestion": None,
            "last_optimization": None,
            "last_execution": None,
        }
        self._history: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load state from disk if exists."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    saved = json.load(f)
                self._state.update(saved)
            except Exception as e:
                logger.warning(f"  Could not load state: {e}")

    def update(self, **kwargs) -> dict:
        """Update system state fields and optionally persist."""
        changed = {k: v for k, v in kwargs.items() if self._state.get(k) != v}
        if not changed:
            return self._state

        self._state.update(changed)
        self._state["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self._snapshot_on_change:
            self._snapshot()

        logger.debug(f"  State updated: {list(changed.keys())}")
        return self.get()

    def get(self) -> dict:
        """Get current system state."""
        return dict(self._state)

    def _snapshot(self) -> None:
        """Save state to disk and append to history."""
        # JSON snapshot
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

        # History
        self._history.append(dict(self._state))

    def save_history(self) -> None:
        """Persist state history to parquet."""
        if not self._history:
            return
        df = pd.DataFrame(self._history)
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._history_file, index=False)

    def get_history(self) -> pd.DataFrame:
        """Get state history as DataFrame."""
        if self._history_file.exists():
            return pd.read_parquet(self._history_file)
        if self._history:
            return pd.DataFrame(self._history)
        return pd.DataFrame()

    def summary(self) -> dict:
        return {
            "regime": self._state.get("regime"),
            "pipeline_status": self._state.get("pipeline_status"),
            "trust_score": self._state.get("trust_score"),
            "approval_mode": self._state.get("approval_mode"),
            "n_positions": self._state.get("n_positions"),
        }
