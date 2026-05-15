"""
Data quality validation utilities.
"""

import pandas as pd
from loguru import logger


class ValidationResult:
    """Container for a single validation check."""

    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details

    def __repr__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.name}: {self.details}"


def validate_dataframe(df: pd.DataFrame, ticker: str = "") -> list[ValidationResult]:
    """Run all validation checks on a dataframe and return results."""
    results = [
        check_not_empty(df, ticker),
        check_missing_values(df, ticker),
        check_duplicates(df, ticker),
        check_negative_prices(df, ticker),
        check_date_order(df, ticker),
    ]

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    label = f" ({ticker})" if ticker else ""
    logger.info(f"Validation{label}: {passed}/{total} checks passed")

    for r in results:
        if not r.passed:
            logger.warning(f"  {r}")

    return results


def check_not_empty(df: pd.DataFrame, ticker: str = "") -> ValidationResult:
    """Verify the dataframe is not empty."""
    if df.empty:
        return ValidationResult("not_empty", False, "DataFrame is empty")
    return ValidationResult("not_empty", True, f"{len(df)} rows")


def check_missing_values(df: pd.DataFrame, ticker: str = "") -> ValidationResult:
    """Check for missing values in key columns."""
    price_cols = [c for c in ["close", "adj_close", "nav"] if c in df.columns]
    if not price_cols:
        return ValidationResult("missing_values", True, "no price columns to check")

    missing = df[price_cols].isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        detail = ", ".join(f"{c}={v}" for c, v in missing.items() if v > 0)
        return ValidationResult("missing_values", False, f"missing: {detail}")

    return ValidationResult("missing_values", True, "no missing price values")


def check_duplicates(df: pd.DataFrame, ticker: str = "") -> ValidationResult:
    """Check for duplicate dates."""
    if "date" not in df.columns:
        return ValidationResult("duplicates", True, "no date column")

    dups = df.duplicated(subset=["date"]).sum()
    if dups > 0:
        return ValidationResult("duplicates", False, f"{dups} duplicate dates found")

    return ValidationResult("duplicates", True, "no duplicate dates")


def check_negative_prices(df: pd.DataFrame, ticker: str = "") -> ValidationResult:
    """Check for negative or zero prices."""
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close", "nav"] if c in df.columns]
    if not price_cols:
        return ValidationResult("negative_prices", True, "no price columns to check")

    for col in price_cols:
        negatives = (df[col] <= 0).sum()
        if negatives > 0:
            return ValidationResult(
                "negative_prices", False, f"{col} has {negatives} non-positive values"
            )

    return ValidationResult("negative_prices", True, "all prices positive")


def check_date_order(df: pd.DataFrame, ticker: str = "") -> ValidationResult:
    """Verify dates are in ascending order."""
    if "date" not in df.columns:
        return ValidationResult("date_order", True, "no date column")

    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        return ValidationResult("date_order", False, "dates are not in ascending order")

    return ValidationResult("date_order", True, "dates in ascending order")
