"""
ML Alpha Engine — Walk-Forward Training Framework.

Time-series aware training that prevents data leakage:
  - Expanding window splits (never random)
  - Purge gap between train/test (prevents leakage)
  - Embargo period after test (prevents leakage)
  - Walk-forward validation with IC as primary metric

Markets are temporal systems — random splits are WRONG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/ml_alpha.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


@dataclass
class WalkForwardSplit:
    """A single train/test split in the walk-forward scheme."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int = 0
    n_test: int = 0


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    fold: int
    split: WalkForwardSplit
    train_ic: float = 0.0
    test_ic: float = 0.0
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    model: object = None


# ── Walk-forward split generator ────────────────────────────────────────────


def generate_walk_forward_splits(
    dates: pd.DatetimeIndex | pd.Series,
    min_train_years: int = 3,
    test_period_years: int = 1,
    purge_days: int = 5,
    embargo_days: int = 5,
    expanding: bool = True,
) -> list[WalkForwardSplit]:
    """
    Generate time-aware train/test splits for walk-forward validation.

    Parameters
    ----------
    dates : Sorted unique dates in the dataset
    min_train_years : Minimum training window in years
    test_period_years : Test window size in years
    purge_days : Gap between train end and test start
    embargo_days : Gap after test end before next fold's train includes data
    expanding : If True, train window expands; if False, rolling window

    Returns
    -------
    List of WalkForwardSplit objects
    """
    cfg = _load_config().get("training", {}).get("walk_forward", {})
    min_train_years = cfg.get("min_train_years", min_train_years)
    test_period_years = cfg.get("test_period_years", test_period_years)
    purge_days = cfg.get("purge_days", purge_days)
    embargo_days = cfg.get("embargo_days", embargo_days)
    expanding = cfg.get("expanding_window", expanding)

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    total_days = (unique_dates[-1] - unique_dates[0]).days
    train_days = min_train_years * 365
    test_days = test_period_years * 365

    if total_days < train_days + test_days:
        logger.warning(f"Insufficient data: {total_days} days < {train_days + test_days} required")
        return []

    splits = []
    fold = 0
    start = unique_dates[0]
    train_end_cursor = start + pd.Timedelta(days=train_days)

    while True:
        train_start = start if expanding else train_end_cursor - pd.Timedelta(days=train_days)
        train_end = train_end_cursor

        test_start = train_end + pd.Timedelta(days=purge_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > unique_dates[-1]:
            break

        split = WalkForwardSplit(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        splits.append(split)

        fold += 1
        train_end_cursor = test_end + pd.Timedelta(days=embargo_days)

    logger.info(f"Generated {len(splits)} walk-forward splits "
                f"(train≥{min_train_years}yr, test={test_period_years}yr, "
                f"purge={purge_days}d, embargo={embargo_days}d)")

    return splits


# ── Walk-forward trainer ────────────────────────────────────────────────────


def walk_forward_train(
    dataset: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_factory: callable,
    splits: list[WalkForwardSplit] | None = None,
) -> list[WalkForwardResult]:
    """
    Train models using walk-forward validation.

    Parameters
    ----------
    dataset : Full dataset with date, ticker, features, and target columns
    target_col : Name of the target column (e.g. 'forward_rank_5d')
    feature_cols : List of feature column names
    model_factory : Callable that returns a new model instance
    splits : Pre-computed splits (if None, generates automatically)

    Returns
    -------
    List of WalkForwardResult objects per fold
    """
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["date"])

    if splits is None:
        splits = generate_walk_forward_splits(df["date"])

    if not splits:
        logger.error("No walk-forward splits generated")
        return []

    results = []

    for split in splits:
        logger.info(f"  Fold {split.fold}: "
                     f"train {split.train_start.date()}→{split.train_end.date()}, "
                     f"test {split.test_start.date()}→{split.test_end.date()}")

        # Split data
        train = df[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)]
        test = df[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)]

        if len(train) < 50 or len(test) < 10:
            logger.warning(f"  Fold {split.fold}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        split.n_train = len(train)
        split.n_test = len(test)

        # Prepare X, y
        valid_feats = [f for f in feature_cols if f in train.columns]
        X_train = train[valid_feats].values
        y_train = train[target_col].values
        X_test = test[valid_feats].values
        y_test = test[target_col].values

        # Drop rows with NaN in features
        train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        test_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]

        if len(X_train) < 30 or len(X_test) < 5:
            logger.warning(f"  Fold {split.fold}: too few valid rows after NaN drop")
            continue

        # Train model
        model = model_factory()

        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.error(f"  Fold {split.fold}: training failed — {exc}")
            continue

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Compute rank IC (Spearman correlation)
        from scipy.stats import spearmanr
        train_ic = spearmanr(y_train, y_pred_train)[0]
        test_ic = spearmanr(y_test, y_pred_test)[0]

        # Build test predictions DataFrame
        test_valid = test[test_mask].copy()
        test_valid = test_valid.iloc[:len(y_pred_test)]
        test_valid["prediction"] = y_pred_test
        test_valid["actual"] = y_test

        # Feature importance
        importance = _extract_importance(model, valid_feats)

        result = WalkForwardResult(
            fold=split.fold,
            split=split,
            train_ic=train_ic,
            test_ic=test_ic,
            test_predictions=test_valid[["date", "ticker", "prediction", "actual"]],
            feature_importance=importance,
            model=model,
        )
        results.append(result)

        logger.info(f"    IC: train={train_ic:.4f}, test={test_ic:.4f}")

    if results:
        avg_test_ic = np.mean([r.test_ic for r in results])
        logger.info(f"  Walk-forward complete: {len(results)} folds, avg test IC={avg_test_ic:.4f}")

    return results


def _extract_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            "feature": feature_names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Aggregate results ───────────────────────────────────────────────────────


def aggregate_walk_forward_results(results: list[WalkForwardResult]) -> dict:
    """
    Aggregate walk-forward results into a summary.

    Returns
    -------
    dict with keys: avg_train_ic, avg_test_ic, ic_stability, all_predictions,
                    avg_feature_importance
    """
    if not results:
        return {"avg_train_ic": 0, "avg_test_ic": 0, "ic_stability": 0}

    train_ics = [r.train_ic for r in results]
    test_ics = [r.test_ic for r in results]

    # Concatenate all test predictions
    all_preds = pd.concat([r.test_predictions for r in results], ignore_index=True)

    # Average feature importance across folds
    all_imp = pd.concat([r.feature_importance for r in results if not r.feature_importance.empty])
    if not all_imp.empty:
        avg_imp = all_imp.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
    else:
        avg_imp = pd.DataFrame()

    return {
        "avg_train_ic": float(np.mean(train_ics)),
        "avg_test_ic": float(np.mean(test_ics)),
        "ic_stability": float(np.std(test_ics)) if len(test_ics) > 1 else 0,
        "n_folds": len(results),
        "all_predictions": all_preds,
        "avg_feature_importance": avg_imp,
    }
