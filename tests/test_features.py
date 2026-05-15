"""
Tests for features/feature_store.py and features/signal_ranker.py.
"""

import numpy as np
import pandas as pd
import pytest

from features.signal_ranker import calculate_composite_score


class TestCompositeScore:
    def test_returns_dataframe(self, feature_store):
        result = calculate_composite_score(feature_store)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, feature_store):
        result = calculate_composite_score(feature_store)
        for col in ["date", "ticker", "composite_score", "composite_rank"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rank_bounded(self, feature_store):
        result = calculate_composite_score(feature_store)
        assert result["composite_rank"].min() >= 0
        assert result["composite_rank"].max() <= 1

    def test_score_per_ticker(self, feature_store):
        result = calculate_composite_score(feature_store)
        tickers = result["ticker"].unique()
        assert len(tickers) == 4  # AAPL, SPY, RELIANCE.NS, INFY.NS

    def test_custom_weights(self, feature_store):
        weights = {
            "factor_momentum": 1.0,
            "factor_trend": 0.0,
            "factor_low_vol": 0.0,
        }
        result = calculate_composite_score(feature_store, weights=weights)
        assert not result.empty
