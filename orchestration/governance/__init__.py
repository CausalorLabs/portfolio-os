"""
Orchestration — Governance Layer.

Operational governance:
  - Configuration snapshots for reproducibility
  - Workflow versioning
  - Config change tracking
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/orchestration.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class GovernanceEngine:
    """
    Governance engine for configuration management and reproducibility.

    Snapshots entire config state before pipeline runs,
    enabling reproducible historical analysis.
    """

    def __init__(self):
        cfg = _load_config().get("governance", {})
        self._snapshot_dir = Path(cfg.get("snapshot_dir", "data/processed/config_snapshots"))
        self._max_snapshots = cfg.get("max_snapshots", 100)
        self._enabled = cfg.get("config_snapshots", True)
        self._snapshots: list[dict] = []

    def snapshot_configs(self, run_id: str | None = None) -> dict | None:
        """
        Snapshot all YAML configs for reproducibility.

        Returns snapshot metadata.
        """
        if not self._enabled:
            return None

        configs_dir = Path("configs")
        snapshot: dict = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        configs_content: dict = {}

        for yaml_path in sorted(configs_dir.glob("**/*.yaml")):
            try:
                key = str(yaml_path.relative_to(configs_dir))
                cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
                configs_content[key] = cfg
            except Exception:
                pass

        snapshot["configs"] = configs_content
        snapshot["hash"] = hashlib.sha256(
            json.dumps(configs_content, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # Save
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"config_snapshot_{ts}.json"
        with open(self._snapshot_dir / fname, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        self._snapshots.append({
            "file": fname,
            "hash": snapshot["hash"],
            "timestamp": snapshot["timestamp"],
            "run_id": run_id,
        })
        self._trim()

        logger.debug(f"  Config snapshot: {fname} (hash={snapshot['hash']})")
        return {"file": fname, "hash": snapshot["hash"]}

    def _trim(self) -> None:
        """Remove old snapshots beyond max limit."""
        files = sorted(self._snapshot_dir.glob("config_snapshot_*.json"))
        if len(files) > self._max_snapshots:
            for old in files[: len(files) - self._max_snapshots]:
                old.unlink()

    def get_snapshot(self, filename: str) -> dict | None:
        """Load a specific config snapshot."""
        path = self._snapshot_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def list_snapshots(self) -> list[dict]:
        """List all available snapshots."""
        if self._snapshots:
            return list(self._snapshots)
        files = sorted(self._snapshot_dir.glob("config_snapshot_*.json")) if self._snapshot_dir.exists() else []
        return [{"file": f.name} for f in files]

    def config_changed_since(self, reference_hash: str) -> bool:
        """Check if configs changed since a reference hash."""
        current = self.snapshot_configs()
        if current is None:
            return False
        return current["hash"] != reference_hash

    def summary(self) -> dict:
        snapshots = self.list_snapshots()
        return {
            "total_snapshots": len(snapshots),
            "latest": snapshots[-1] if snapshots else None,
            "enabled": self._enabled,
        }
