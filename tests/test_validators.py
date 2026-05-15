"""
Tests for utils/validators.py — data quality validation.
"""

import pandas as pd
import pytest

from utils.validators import (
    ValidationResult,
    validate_dataframe,
    check_not_empty,
    check_missing_values,
    check_duplicates,
    check_negative_prices,
    check_date_order,
)


class TestValidationResult:
    def test_passed_repr(self):
        r = ValidationResult("test_check", True, "ok")
        assert "✓" in repr(r)

    def test_failed_repr(self):
        r = ValidationResult("test_check", False, "bad")
        assert "✗" in repr(r)


class TestCheckNotEmpty:
    def test_empty_df_fails(self):
        result = check_not_empty(pd.DataFrame())
        assert not result.passed

    def test_non_empty_passes(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_not_empty(df)
        assert result.passed


class TestCheckMissingValues:
    def test_no_price_cols_passes(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = check_missing_values(df)
        assert result.passed

    def test_missing_close_fails(self):
        df = pd.DataFrame({"close": [1.0, None, 3.0]})
        result = check_missing_values(df)
        assert not result.passed

    def test_clean_close_passes(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        result = check_missing_values(df)
        assert result.passed


class TestCheckDuplicates:
    def test_no_date_column_passes(self):
        df = pd.DataFrame({"x": [1]})
        result = check_duplicates(df)
        assert result.passed

    def test_duplicate_dates_fail(self):
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"]})
        result = check_duplicates(df)
        assert not result.passed

    def test_unique_dates_pass(self):
        df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"]})
        result = check_duplicates(df)
        assert result.passed


class TestCheckNegativePrices:
    def test_negative_price_fails(self):
        df = pd.DataFrame({"close": [100.0, -5.0, 50.0]})
        result = check_negative_prices(df)
        assert not result.passed

    def test_positive_prices_pass(self):
        df = pd.DataFrame({"close": [100.0, 200.0]})
        result = check_negative_prices(df)
        assert result.passed


class TestCheckDateOrder:
    def test_unordered_dates_fail(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-05", "2020-01-01"])})
        result = check_date_order(df)
        assert not result.passed

    def test_ordered_dates_pass(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-05"])})
        result = check_date_order(df)
        assert result.passed


class TestValidateDataframe:
    def test_all_checks_run(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [100.0, 101.0],
        })
        results = validate_dataframe(df, "TEST")
        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_empty_df_has_failure(self):
        results = validate_dataframe(pd.DataFrame(), "EMPTY")
        assert any(not r.passed for r in results)
