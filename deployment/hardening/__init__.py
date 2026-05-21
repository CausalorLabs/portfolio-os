"""
Deployment — Operational Hardening.

Sprint 8: Backup, recovery, and environment reproducibility.
  - Data backup management
  - Configuration backup
  - Environment snapshot
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/deployment.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


class HardeningEngine:
    """
    Operational hardening engine.

    Manages backups, recovery, and environment reproducibility.
    """

    def __init__(self):
        cfg = _load_config().get("hardening", {})
        backup_cfg = cfg.get("backup", {})
        self._enabled = backup_cfg.get("enabled", True)
        self._backup_dir = Path(backup_cfg.get("backup_dir", "data/backups"))
        self._max_backups = backup_cfg.get("max_backups", 30)
        self._repro_cfg = cfg.get("reproducibility", {})

    def create_backup(self, label: str = "") -> dict | None:
        """
        Create a backup of critical data files.

        Returns backup metadata.
        """
        if not self._enabled:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{ts}" + (f"_{label}" if label else "")
        backup_path = self._backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        # Critical files to back up
        critical_files = [
            "data/processed/inr_prices.parquet",
            "data/processed/portfolio_nav.parquet",
            "data/processed/target_weights.parquet",
            "data/processed/features.parquet",
            "data/processed/system_state.json",
        ]

        backed_up = []
        for file_str in critical_files:
            src = Path(file_str)
            if src.exists():
                dst = backup_path / src.name
                shutil.copy2(src, dst)
                backed_up.append(src.name)

        # Config backup
        configs_dir = Path("configs")
        if configs_dir.exists():
            config_backup = backup_path / "configs"
            config_backup.mkdir(exist_ok=True)
            for yaml_file in configs_dir.glob("*.yaml"):
                shutil.copy2(yaml_file, config_backup / yaml_file.name)

        # Metadata
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "files_backed_up": backed_up,
            "backup_path": str(backup_path),
        }
        with open(backup_path / "backup_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._trim_backups()
        logger.info(f"  Backup created: {backup_name} ({len(backed_up)} files)")
        return meta

    def _trim_backups(self) -> None:
        """Remove old backups beyond max limit."""
        if not self._backup_dir.exists():
            return
        backups = sorted(self._backup_dir.glob("backup_*"))
        if len(backups) > self._max_backups:
            for old in backups[: len(backups) - self._max_backups]:
                shutil.rmtree(old, ignore_errors=True)

    def list_backups(self) -> list[dict]:
        """List all available backups."""
        if not self._backup_dir.exists():
            return []

        result = []
        for backup_dir in sorted(self._backup_dir.glob("backup_*")):
            meta_path = backup_dir / "backup_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                result.append(meta)
            else:
                result.append({
                    "backup_path": str(backup_dir),
                    "timestamp": backup_dir.name.replace("backup_", ""),
                })
        return result

    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore from a backup.

        Copies backup files back to their original locations.
        """
        bp = Path(backup_path)
        if not bp.exists():
            logger.error(f"  Backup not found: {backup_path}")
            return False

        dest = Path("data/processed")
        dest.mkdir(parents=True, exist_ok=True)

        restored = 0
        for file in bp.glob("*.parquet"):
            shutil.copy2(file, dest / file.name)
            restored += 1
        for file in bp.glob("*.json"):
            if file.name != "backup_meta.json":
                shutil.copy2(file, dest / file.name)
                restored += 1

        logger.info(f"  Restored {restored} files from {backup_path}")
        return True

    def snapshot_environment(self) -> dict:
        """Capture environment snapshot for reproducibility."""
        import sys

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
        }

        # Requirements
        req_path = Path("requirements.txt")
        if req_path.exists():
            snapshot["requirements_hash"] = hash(req_path.read_text())

        return snapshot

    def summary(self) -> dict:
        backups = self.list_backups()
        return {
            "enabled": self._enabled,
            "total_backups": len(backups),
            "latest": backups[-1].get("timestamp") if backups else None,
            "backup_dir": str(self._backup_dir),
        }
