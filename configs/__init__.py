"""
Config loader — loads Hydra YAML configs into a flat OmegaConf dict.

Usage:
    from configs import load_config
    cfg = load_config()           # loads base.yaml
    cfg = load_config("prod")     # loads prod.yaml (inherits base)
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

_CONFIG_DIR = Path(__file__).parent / "hydra"


def load_config(env: str | None = None) -> DictConfig:
    """
    Load configuration from YAML files.

    Args:
        env: Environment name ("dev", "prod"). Loads base.yaml,
             then merges env-specific overrides on top.

    Returns:
        Merged OmegaConf DictConfig.
    """
    base_path = _CONFIG_DIR / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    cfg = OmegaConf.load(base_path)

    if env:
        env_path = _CONFIG_DIR / f"{env}.yaml"
        if env_path.exists():
            env_cfg = OmegaConf.load(env_path)
            # Remove 'defaults' key (Hydra convention, not needed for merge)
            if "defaults" in env_cfg:
                del env_cfg["defaults"]
            cfg = OmegaConf.merge(cfg, env_cfg)

    return cfg


def get_config_value(cfg: DictConfig, dotpath: str, default: object = None) -> object:
    """Get a nested config value using dot notation (e.g. 'optimization.method')."""
    try:
        return OmegaConf.select(cfg, dotpath, default=default)
    except Exception:
        return default
