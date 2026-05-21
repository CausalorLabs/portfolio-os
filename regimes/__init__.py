"""
Regimes — re-exports regime detection from validation/.

MVP interface for regime intelligence. Currently wraps the POC
regime classifier; Sprint 2 will add adaptive behavior.
"""

from validation.regimes import identify_market_regimes, evaluate_regime_performance

__all__ = [
    "identify_market_regimes",
    "evaluate_regime_performance",
]
"""
Feature store — re-exports from features/ package.

This module provides the MVP interface to the feature pipeline.
Delegates to the POC feature modules under features/.
"""

from features.feature_store import build_feature_store, load_feature_store, save_feature_store
from features.signal_ranker import calculate_composite_score
from features.validators import validate_features

__all__ = [
    "build_feature_store",
    "save_feature_store",
    "load_feature_store",
    "calculate_composite_score",
    "validate_features",
]
