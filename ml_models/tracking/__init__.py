"""
ML Alpha Engine — MLflow Tracking Integration.

Tracks:
  - Experiment runs (parameters, metrics, artifacts)
  - Model versioning
  - IC, hit ratio, stability over time
  - Feature importance snapshots
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ── MLflow wrapper ──────────────────────────────────────────────────────────


class AlphaTracker:
    """MLflow tracking wrapper for the alpha engine."""

    def __init__(self):
        self.experiment_name: str = "portfolio_alpha"
        self.tracking_uri: str | None = None
        self.run_id: str | None = None
        self._mlflow = None

    def setup(self) -> "AlphaTracker":
        """Initialize MLflow tracking."""
        try:
            import mlflow
            self._mlflow = mlflow

            cfg = _load_config().get("tracking", {})
            self.experiment_name = cfg.get("experiment_name", "portfolio_alpha")
            artifact_dir = cfg.get("artifact_dir", "data/ml_artifacts")

            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{Path(artifact_dir).resolve()}")
            mlflow.set_experiment(self.experiment_name)

            logger.info(f"MLflow tracking initialized: experiment={self.experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed — tracking disabled")
            self._mlflow = None

        return self

    def start_run(self, run_name: str | None = None) -> "AlphaTracker":
        """Start a new MLflow run."""
        if self._mlflow is None:
            return self

        self._mlflow.start_run(run_name=run_name)
        self.run_id = self._mlflow.active_run().info.run_id
        logger.info(f"MLflow run started: {run_name} ({self.run_id[:8]})")
        return self

    def log_params(self, params: dict) -> None:
        """Log hyperparameters."""
        if self._mlflow is None:
            return

        # Flatten nested dicts
        flat = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v

        try:
            self._mlflow.log_params(flat)
        except Exception as exc:
            logger.debug(f"MLflow log_params failed: {exc}")

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics (IC, hit ratio, etc.)."""
        if self._mlflow is None:
            return

        try:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                step=step,
            )
        except Exception as exc:
            logger.debug(f"MLflow log_metrics failed: {exc}")

    def log_feature_importance(self, importance: pd.DataFrame) -> None:
        """Log feature importance as artifact."""
        if self._mlflow is None or importance.empty:
            return

        try:
            path = Path("data/ml_artifacts/feature_importance.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            importance.to_csv(path, index=False)
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.debug(f"MLflow artifact logging failed: {exc}")

    def log_model(self, model, model_name: str) -> None:
        """Log a trained model."""
        if self._mlflow is None:
            return

        try:
            self._mlflow.sklearn.log_model(model, model_name)
        except Exception:
            try:
                self._mlflow.pyfunc.log_model(model_name, python_model=model)
            except Exception as exc:
                logger.debug(f"MLflow model logging failed: {exc}")

    def log_alpha_scores(self, alpha_scores: pd.DataFrame) -> None:
        """Log alpha scores as artifact."""
        if self._mlflow is None or alpha_scores.empty:
            return

        try:
            path = Path("data/ml_artifacts/alpha_scores.parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            alpha_scores.to_parquet(path, index=False)
            self._mlflow.log_artifact(str(path))
        except Exception as exc:
            logger.debug(f"MLflow alpha scores logging failed: {exc}")

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._mlflow is None:
            return

        try:
            self._mlflow.end_run()
            logger.info(f"MLflow run ended: {self.run_id[:8] if self.run_id else 'N/A'}")
        except Exception:
            pass

    def log_full_experiment(
        self,
        params: dict,
        metrics: dict,
        feature_importance: pd.DataFrame,
        alpha_scores: pd.DataFrame,
        model=None,
        run_name: str = "alpha_run",
    ) -> None:
        """Convenience: log everything for one experiment run."""
        self.start_run(run_name)
        self.log_params(params)
        self.log_metrics(metrics)
        self.log_feature_importance(feature_importance)
        self.log_alpha_scores(alpha_scores)
        if model is not None:
            self.log_model(model, "ensemble_model")
        self.end_run()


# Module-level singleton
_tracker: AlphaTracker | None = None


def get_tracker() -> AlphaTracker:
    """Get the global alpha tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = AlphaTracker().setup()
    return _tracker
