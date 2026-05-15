"""
Feature validation — leakage prevention, NaN checks, distribution stability.
"""

import numpy as np
import pandas as pd
from loguru import logger


def validate_features(store: pd.DataFrame) -> dict[str, bool]:
    """
    Run all validation checks on the feature store.
    Returns dict of check_name → passed.
    """
    results = {
        "no_missing_critical": check_missing_features(store),
        "no_infinite_values": check_infinite_values(store),
        "distributions_stable": check_distribution_stability(store),
        "no_constant_features": check_constant_features(store),
    }

    passed = sum(results.values())
    total = len(results)
    logger.info(f"Feature validation: {passed}/{total} checks passed")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        logger.info(f"  [{status}] {name}")

    return results


def check_missing_features(store: pd.DataFrame, max_nan_pct: float = 0.5) -> bool:
    """
    Check that no feature has more than max_nan_pct missing values
    (relative to its expected count per ticker).
    """
    issues = []
    for feature, group in store.groupby("feature"):
        nan_pct = group["value"].isnull().mean()
        if nan_pct > max_nan_pct:
            issues.append(f"{feature}: {nan_pct:.1%} NaN")

    if issues:
        logger.warning(f"High NaN features: {', '.join(issues[:5])}")
        return False
    return True


def check_infinite_values(store: pd.DataFrame) -> bool:
    """Check for infinite values in features."""
    inf_count = np.isinf(store["value"].dropna()).sum()
    if inf_count > 0:
        inf_features = store[np.isinf(store["value"])]["feature"].unique()
        logger.warning(f"Infinite values in: {', '.join(inf_features[:5])}")
        return False
    return True


def check_distribution_stability(
    store: pd.DataFrame,
    max_cv: float = 50.0,
) -> bool:
    """
    Check that feature distributions are not wildly unstable.
    Uses coefficient of variation (std/mean) as a proxy.
    """
    issues = []
    for feature, group in store.groupby("feature"):
        vals = group["value"].dropna()
        if len(vals) < 10:
            continue
        mean = vals.mean()
        if mean == 0:
            continue
        cv = abs(vals.std() / mean)
        if cv > max_cv:
            issues.append(f"{feature}: CV={cv:.1f}")

    if issues:
        logger.warning(f"Unstable features: {', '.join(issues[:5])}")
        return False
    return True


def check_constant_features(store: pd.DataFrame) -> bool:
    """Check for features that have zero variance (useless for modeling)."""
    issues = []
    for feature, group in store.groupby("feature"):
        vals = group["value"].dropna()
        if len(vals) > 10 and vals.std() == 0:
            issues.append(feature)

    if issues:
        logger.warning(f"Constant features: {', '.join(issues)}")
        return False
    return True


def check_lookahead_bias(
    store: pd.DataFrame,
    inr_prices: pd.DataFrame,
) -> bool:
    """
    Verify that feature dates are consistent with available price dates.
    A feature on date T should only use prices from T and earlier.

    This is a structural check — verifies that the earliest feature date
    for each ticker is not before the earliest price date.
    """
    passed = True
    for ticker in store["ticker"].unique():
        feat_min = store[store["ticker"] == ticker]["date"].min()
        price_min = inr_prices[inr_prices["ticker"] == ticker]["date"].min()

        if feat_min < price_min:
            logger.warning(
                f"Potential lookahead for {ticker}: "
                f"feature starts {feat_min.date()} but price starts {price_min.date()}"
            )
            passed = False

    if passed:
        logger.info("Lookahead check: PASSED — all features start on or after price data")
    return passed
