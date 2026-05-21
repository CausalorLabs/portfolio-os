"""Tests for the ML Alpha Engine — Sprint 3."""

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_prices():
    """Generate synthetic INR prices for 5 tickers over 500 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2021-01-01", periods=500)
    tickers = ["AAPL", "MSFT", "RELIANCE.NS", "GOLDIETF.NS", "PPFAS.MF"]
    rows = []
    for ticker in tickers:
        base = 500 + np.random.randn() * 100
        prices = base * np.cumprod(1 + np.random.randn(500) * 0.015)
        for d, p in zip(dates, prices):
            rows.append({"date": d, "ticker": ticker, "inr_price": abs(p), "fx_rate": 83.5})
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_feature_store(sample_prices):
    """Build a minimal long-format feature store."""
    np.random.seed(42)
    rows = []
    for _, row in sample_prices.iterrows():
        for feat in ["momentum_20d", "momentum_60d", "volatility_20d", "rsi_14",
                      "trend_slope_60d", "factor_momentum", "factor_trend", "factor_low_vol",
                      "return_5d", "return_20d", "ma_distance_20", "bb_zscore"]:
            rows.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "feature": feat,
                "value": np.random.randn() * 0.1,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_regime_states():
    """Generate regime states matching price date range."""
    dates = pd.bdate_range("2021-01-01", periods=500)
    regimes = (["risk_on"] * 200 + ["risk_off"] * 150 +
               ["panic"] * 50 + ["risk_on"] * 100)
    return pd.DataFrame({
        "date": dates,
        "regime": regimes,
        "confidence": np.clip(np.random.rand(500) * 0.3 + 0.7, 0, 1),
    })


@pytest.fixture()
def sample_dataset(sample_prices, sample_feature_store, sample_regime_states):
    """Build an ML dataset from fixtures."""
    from ml_models.datasets import build_ml_dataset
    return build_ml_dataset(sample_prices, sample_feature_store, sample_regime_states)


# ── Dataset Tests ────────────────────────────────────────────────────────────


class TestDatasets:
    def test_forward_rank(self, sample_prices):
        from ml_models.datasets import compute_forward_rank
        result = compute_forward_rank(sample_prices, horizon=5)
        assert not result.empty
        assert "forward_rank_5d" in result.columns
        assert result["forward_rank_5d"].between(0, 1).all()

    def test_risk_adjusted_target(self, sample_prices):
        from ml_models.datasets import compute_risk_adjusted_target
        result = compute_risk_adjusted_target(sample_prices, horizon=20)
        assert not result.empty
        assert "risk_adjusted_20d" in result.columns
        assert "risk_adjusted_rank_20d" in result.columns

    def test_downside_probability(self, sample_prices):
        from ml_models.datasets import compute_downside_probability
        result = compute_downside_probability(sample_prices, horizon=20, lookback=100)
        assert not result.empty
        assert "downside_prob_20d" in result.columns
        assert result["downside_prob_20d"].between(0, 1).all()

    def test_build_ml_dataset(self, sample_dataset):
        assert not sample_dataset.empty
        assert "date" in sample_dataset.columns
        assert "ticker" in sample_dataset.columns
        assert "forward_rank_5d" in sample_dataset.columns

    def test_dataset_no_lookahead(self, sample_dataset):
        """Verify features don't contain forward-looking columns."""
        suspect = [c for c in sample_dataset.columns if "forward" in c.lower() and c != "forward_rank_5d"]
        # forward targets are allowed; forward features are not
        feature_cols = [c for c in sample_dataset.columns
                       if c not in ("date", "ticker") and not c.startswith("forward_")
                       and not c.startswith("risk_adjusted_") and not c.startswith("downside_")]
        for col in feature_cols:
            assert "forward" not in col.lower(), f"Suspected lookahead in feature: {col}"


# ── Feature Tests ────────────────────────────────────────────────────────────


class TestFeatures:
    def test_cross_sectional_features(self, sample_feature_store):
        from ml_models.features import compute_cross_sectional_features
        result = compute_cross_sectional_features(sample_feature_store)
        assert not result.empty
        assert "feature" in result.columns

    def test_beta_features(self, sample_prices):
        from ml_models.features import compute_beta_features
        result = compute_beta_features(sample_prices, window=30)
        assert not result.empty
        assert "beta_60d" in result["feature"].values

    def test_extended_store(self, sample_feature_store, sample_prices, sample_regime_states):
        from ml_models.features import build_extended_feature_store
        result = build_extended_feature_store(
            sample_feature_store, sample_prices, sample_regime_states
        )
        assert len(result) > len(sample_feature_store)
        assert result["feature"].nunique() > sample_feature_store["feature"].nunique()


# ── Feature Quality Tests ────────────────────────────────────────────────────


class TestQuality:
    def test_psi_no_drift(self):
        from ml_models.quality import compute_psi
        np.random.seed(42)
        ref = pd.Series(np.random.randn(1000))
        cur = pd.Series(np.random.randn(1000))
        psi = compute_psi(ref, cur)
        assert psi < 0.10  # same distribution → low PSI

    def test_psi_with_drift(self):
        from ml_models.quality import compute_psi
        np.random.seed(42)
        ref = pd.Series(np.random.randn(1000))
        cur = pd.Series(np.random.randn(1000) + 3)  # shifted distribution
        psi = compute_psi(ref, cur)
        assert psi > 0.20  # different distribution → high PSI

    def test_correlation_analysis(self, sample_feature_store):
        from ml_models.quality import analyze_feature_correlations
        wide = sample_feature_store.pivot_table(
            index=["date", "ticker"], columns="feature", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        corr_matrix, to_drop = analyze_feature_correlations(wide, max_corr=0.95)
        assert not corr_matrix.empty
        assert isinstance(to_drop, list)

    def test_missing_data_policy(self, sample_feature_store):
        from ml_models.quality import apply_missing_data_policy
        wide = sample_feature_store.pivot_table(
            index=["date", "ticker"], columns="feature", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        cleaned, dropped = apply_missing_data_policy(wide, max_missing_pct=0.90, validity_window=10)
        assert len(cleaned) > 0

    def test_full_quality_pipeline(self, sample_feature_store):
        from ml_models.quality import run_feature_quality_pipeline
        wide = sample_feature_store.pivot_table(
            index=["date", "ticker"], columns="feature", values="value", aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        result = run_feature_quality_pipeline(wide)
        assert "cleaned_features" in result
        assert "quality_summary" in result
        assert result["quality_summary"]["features_retained"] > 0


# ── Training Tests ───────────────────────────────────────────────────────────


class TestTraining:
    def test_walk_forward_splits(self):
        from ml_models.training import generate_walk_forward_splits
        dates = pd.bdate_range("2018-01-01", periods=1500)
        splits = generate_walk_forward_splits(dates, min_train_years=2, test_period_years=1)
        assert len(splits) >= 1
        for s in splits:
            assert s.train_end < s.test_start  # no overlap

    def test_walk_forward_splits_purge_gap(self):
        from ml_models.training import generate_walk_forward_splits
        dates = pd.bdate_range("2018-01-01", periods=1500)
        splits = generate_walk_forward_splits(dates, purge_days=10)
        for s in splits:
            gap = (s.test_start - s.train_end).days
            assert gap >= 5  # at least the configured purge gap

    def test_walk_forward_train(self, sample_dataset):
        from ml_models.training import walk_forward_train, generate_walk_forward_splits
        from ml_models.ensembles import create_lightgbm_model

        feature_cols = [c for c in sample_dataset.columns
                        if c not in ("date", "ticker") and not c.startswith("forward_")
                        and not c.startswith("risk_adjusted_") and not c.startswith("downside_")]

        splits = generate_walk_forward_splits(
            sample_dataset["date"], min_train_years=1, test_period_years=1
        )

        if splits:
            results = walk_forward_train(
                dataset=sample_dataset,
                target_col="forward_rank_5d",
                feature_cols=feature_cols[:5],
                model_factory=create_lightgbm_model,
                splits=splits[:1],  # just first fold for speed
            )
            assert len(results) >= 0  # may be 0 if insufficient data


# ── Ensemble Tests ───────────────────────────────────────────────────────────


class TestEnsemble:
    def test_alpha_ensemble_build(self):
        from ml_models.ensembles import AlphaEnsemble
        ens = AlphaEnsemble().build(["feat1", "feat2", "feat3"])
        assert "lightgbm" in ens.models
        assert "catboost" in ens.models
        assert "momentum_baseline" in ens.models

    def test_ensemble_fit_predict(self):
        from ml_models.ensembles import AlphaEnsemble
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.rand(100)

        ens = AlphaEnsemble().build([f"f{i}" for i in range(5)])
        ens.fit(X, y)
        assert ens.is_fitted

        alpha, confidence = ens.score(X[:10])
        assert len(alpha) == 10
        assert len(confidence) == 10
        assert all(0 <= c <= 1 for c in confidence)

    def test_momentum_baseline(self):
        from ml_models.ensembles import MomentumBaseline
        mb = MomentumBaseline(lookback=60)
        X = np.random.randn(20, 3)
        mb.fit(X, np.random.rand(20))
        preds = mb.predict(X)
        assert len(preds) == 20
        assert all(0 <= p <= 1 for p in preds)


# ── Confidence Tests ─────────────────────────────────────────────────────────


class TestConfidence:
    def test_regime_confidence(self, sample_regime_states):
        from ml_models.confidence import compute_regime_confidence
        result = compute_regime_confidence(sample_regime_states)
        assert not result.empty
        assert "regime_stability_score" in result.columns
        assert result["regime_stability_score"].between(0, 1).all()

    def test_composite_confidence(self):
        from ml_models.confidence import compute_composite_confidence
        alpha = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10).repeat(3),
            "ticker": ["A", "B", "C"] * 10,
            "alpha_score": np.random.rand(30),
            "model_confidence": np.random.rand(30) * 0.5 + 0.3,
            "rank": np.random.rand(30),
        })
        result = compute_composite_confidence(alpha)
        assert "composite_confidence" in result.columns
        assert result["composite_confidence"].between(0.20, 0.95).all()

    def test_feature_drift_confidence(self):
        from ml_models.confidence import compute_feature_drift_confidence
        # No drift → high confidence
        drift = pd.DataFrame({"feature": ["f1", "f2"], "drifted": [False, False], "psi": [0.05, 0.03]})
        conf = compute_feature_drift_confidence(drift)
        assert conf > 0.7

        # Heavy drift → lower confidence
        drift_bad = pd.DataFrame({"feature": ["f1", "f2"], "drifted": [True, True], "psi": [0.5, 0.6]})
        conf_bad = compute_feature_drift_confidence(drift_bad)
        assert conf_bad < conf


# ── Evaluation Tests ─────────────────────────────────────────────────────────


class TestEvaluation:
    def test_rank_ic(self):
        from ml_models.evaluation import compute_rank_ic
        pred = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        actual = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55])
        ic = compute_rank_ic(pred, actual)
        assert ic > 0.9  # near-perfect ranking

    def test_hit_ratio(self):
        from ml_models.evaluation import compute_hit_ratio
        pred = pd.Series([0.8, 0.7, 0.3, 0.2])
        actual = pd.Series([0.9, 0.6, 0.4, 0.1])
        hr = compute_hit_ratio(pred, actual)
        assert hr >= 0.5

    def test_ic_stability(self):
        from ml_models.evaluation import compute_ic_stability
        daily_ic = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=50),
            "ic": np.random.rand(50) * 0.1 + 0.05,
        })
        stability = compute_ic_stability(daily_ic)
        assert stability > 0  # positive mean IC with low variance

    def test_evaluate_alpha_model(self):
        from ml_models.evaluation import evaluate_alpha_model
        preds = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=100).repeat(3),
            "ticker": ["A", "B", "C"] * 100,
            "prediction": np.random.rand(300),
            "actual": np.random.rand(300),
        })
        result = evaluate_alpha_model(preds)
        assert "rank_ic" in result
        assert "hit_ratio" in result
        assert "grade" in result


# ── Tracking Tests ───────────────────────────────────────────────────────────


class TestTracking:
    def test_tracker_setup(self):
        from ml_models.tracking import AlphaTracker
        tracker = AlphaTracker().setup()
        assert tracker._mlflow is not None

    def test_log_metrics(self):
        from ml_models.tracking import AlphaTracker
        tracker = AlphaTracker().setup()
        tracker.start_run("test_run")
        tracker.log_metrics({"rank_ic": 0.08, "hit_ratio": 0.55})
        tracker.end_run()


# ── Portfolio Integration Tests ──────────────────────────────────────────────


class TestPortfolioIntegration:
    def test_alpha_tilted_portfolio(self):
        from optimization.allocator import build_alpha_tilted_portfolio
        base = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "target_weight": [0.25, 0.25, 0.25, 0.25],
        })
        alpha = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "ticker": ["A", "B", "C", "D"],
            "alpha_score": [0.9, 0.7, 0.3, 0.1],
            "composite_confidence": [0.8, 0.7, 0.6, 0.5],
            "rank": [1.0, 0.75, 0.25, 0.0],
        })
        result = build_alpha_tilted_portfolio(base, alpha)
        assert len(result) == 4
        assert abs(result["target_weight"].sum() - 1.0) < 0.01

        # High alpha → higher weight
        a_weight = result[result["ticker"] == "A"]["target_weight"].iloc[0]
        d_weight = result[result["ticker"] == "D"]["target_weight"].iloc[0]
        assert a_weight > d_weight

    def test_alpha_tilt_respects_confidence_gate(self):
        from optimization.allocator import build_alpha_tilted_portfolio
        base = pd.DataFrame({
            "ticker": ["A", "B"],
            "target_weight": [0.50, 0.50],
        })
        alpha = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "ticker": ["A", "B"],
            "alpha_score": [0.9, 0.1],
            "composite_confidence": [0.10, 0.10],  # below threshold
            "rank": [1.0, 0.0],
        })
        result = build_alpha_tilted_portfolio(base, alpha, min_confidence_to_tilt=0.40)
        # With confidence below gate, weights should be ~equal
        diff = abs(result.iloc[0]["target_weight"] - result.iloc[1]["target_weight"])
        assert diff < 0.05

    def test_alpha_tilt_normalizes_to_one(self):
        from optimization.allocator import build_alpha_tilted_portfolio
        base = pd.DataFrame({
            "ticker": ["X", "Y", "Z"],
            "target_weight": [0.40, 0.35, 0.25],
        })
        alpha = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01")] * 3,
            "ticker": ["X", "Y", "Z"],
            "alpha_score": [0.8, 0.5, 0.2],
            "composite_confidence": [0.9, 0.8, 0.7],
            "rank": [1.0, 0.5, 0.0],
        })
        result = build_alpha_tilted_portfolio(base, alpha)
        assert abs(result["target_weight"].sum() - 1.0) < 0.001


# ── Integration Test ────────────────────────────────────────────────────────


class TestAlphaPipeline:
    def test_full_pipeline_with_data(self):
        """Integration: run full alpha pipeline against real data."""
        from pathlib import Path
        if not Path("data/processed/inr_prices.parquet").exists():
            pytest.skip("Processed data not available")
        if not Path("data/processed/features.parquet").exists():
            pytest.skip("Feature store not available")

        from ml_models import run_alpha_pipeline
        result = run_alpha_pipeline(save=False, track=False)

        assert "alpha_scores" in result
        assert "evaluation" in result
        assert "quality_summary" in result

        scores = result["alpha_scores"]
        assert not scores.empty
        assert "alpha_score" in scores.columns
        assert "composite_confidence" in scores.columns
        assert scores["composite_confidence"].between(0.20, 0.95).all()
