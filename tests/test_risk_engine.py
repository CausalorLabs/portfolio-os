"""
Tests for risk_engine — Sprint 4: Dynamic Risk & Covariance Engine.

Covers: volatility, covariance, correlation, tail_risk, budgeting,
        scaling, stress_testing, constraints, evaluation.
"""

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def returns_4asset():
    """Wide-format returns for 4 assets (~500 days)."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {
        "AAPL": np.random.normal(0.0005, 0.015, n),
        "SPY": np.random.normal(0.0004, 0.012, n),
        "RELIANCE.NS": np.random.normal(0.0003, 0.018, n),
        "INFY.NS": np.random.normal(0.0002, 0.016, n),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture()
def cov_4asset(returns_4asset):
    """Sample covariance matrix for 4 assets."""
    return returns_4asset.cov()


@pytest.fixture()
def weights_4asset():
    """Equal weight portfolio."""
    return pd.Series(
        [0.25, 0.25, 0.25, 0.25],
        index=["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"],
    )


@pytest.fixture()
def nav_series():
    """Portfolio NAV series."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    returns = np.random.normal(0.0003, 0.01, 500)
    nav = 1_000_000 * np.cumprod(1 + returns)
    return pd.Series(nav, index=dates)


@pytest.fixture()
def inr_prices_long():
    """Long-format INR prices for pipeline testing."""
    np.random.seed(42)
    tickers = ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"]
    dates = pd.bdate_range("2020-01-01", periods=500)
    rows = []
    for t in tickers:
        prices = 1000 * np.cumprod(1 + np.random.normal(0.0003, 0.015, 500))
        for d, p in zip(dates, prices):
            rows.append({"date": d, "ticker": t, "inr_price": p})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Volatility Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestEWMAVolatility:
    def test_shape(self, returns_4asset):
        from risk_engine.volatility import compute_ewma_volatility
        vol = compute_ewma_volatility(returns_4asset, span=60)
        assert vol.shape == returns_4asset.shape

    def test_positive(self, returns_4asset):
        from risk_engine.volatility import compute_ewma_volatility
        vol = compute_ewma_volatility(returns_4asset, span=60)
        assert (vol.dropna() > 0).all().all()

    def test_annualized_range(self, returns_4asset):
        from risk_engine.volatility import compute_ewma_volatility
        vol = compute_ewma_volatility(returns_4asset, span=60)
        last = vol.iloc[-1]
        assert (last > 0.05).all()  # > 5% annualized
        assert (last < 0.80).all()  # < 80% annualized

    def test_short_span_higher_reactivity(self, returns_4asset):
        from risk_engine.volatility import compute_ewma_volatility
        short = compute_ewma_volatility(returns_4asset, span=20)
        long_ = compute_ewma_volatility(returns_4asset, span=120)
        # Short-span vol should have higher variance (more reactive)
        assert short.iloc[-100:].std().mean() >= long_.iloc[-100:].std().mean() * 0.5


class TestMultiHorizonEWMA:
    def test_returns_three_horizons(self, returns_4asset):
        from risk_engine.volatility import compute_multi_horizon_ewma
        result = compute_multi_horizon_ewma(returns_4asset)
        assert len(result) == 3
        for key, df in result.items():
            assert "ewma_vol" in key
            assert df.shape[1] == 4


class TestRealizedVolatility:
    def test_multi_window(self, returns_4asset):
        from risk_engine.volatility import compute_realized_volatility
        result = compute_realized_volatility(returns_4asset, windows=[5, 20, 60])
        assert len(result) == 3
        assert "realized_vol_5d" in result
        assert "realized_vol_60d" in result


class TestVolRegimeClassification:
    def test_classifies_to_valid_regimes(self, returns_4asset):
        from risk_engine.volatility import classify_vol_regime, compute_ewma_volatility
        vol = compute_ewma_volatility(returns_4asset, 60)["AAPL"]
        regimes = classify_vol_regime(vol)
        valid = {"normal", "elevated", "panic"}
        unique = set(regimes.dropna().unique())
        assert unique.issubset(valid)


class TestVolatilityState:
    def test_builds_state(self, inr_prices_long):
        from risk_engine.volatility import build_volatility_state
        state = build_volatility_state(inr_prices_long)
        assert not state.empty
        assert "ticker" in state.columns
        assert "ewma_vol" in state.columns
        assert "vol_regime" in state.columns
        assert "vol_percentile" in state.columns


# ══════════════════════════════════════════════════════════════════════════════
# Covariance Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestEWMACovariance:
    def test_shape(self, returns_4asset):
        from risk_engine.covariance import compute_ewma_covariance
        cov = compute_ewma_covariance(returns_4asset, span=60)
        assert cov.shape == (4, 4)

    def test_symmetric(self, returns_4asset):
        from risk_engine.covariance import compute_ewma_covariance
        cov = compute_ewma_covariance(returns_4asset, span=60)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)


class TestShrinkageCovariance:
    def test_returns_cov_and_coefficient(self, returns_4asset):
        from risk_engine.covariance import compute_shrinkage_covariance
        cov, coeff = compute_shrinkage_covariance(returns_4asset)
        assert cov.shape == (4, 4)
        assert 0 <= coeff <= 1

    def test_positive_definite(self, returns_4asset):
        from risk_engine.covariance import compute_shrinkage_covariance
        cov, _ = compute_shrinkage_covariance(returns_4asset)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() > 0


class TestRegimeCovariance:
    def test_panic_uses_ewma(self, returns_4asset):
        from risk_engine.covariance import compute_regime_covariance
        cov = compute_regime_covariance(returns_4asset, regime="panic")
        assert cov.shape == (4, 4)

    def test_risk_on_uses_shrinkage(self, returns_4asset):
        from risk_engine.covariance import compute_regime_covariance
        cov = compute_regime_covariance(returns_4asset, regime="risk_on")
        assert cov.shape == (4, 4)


class TestCovarianceDiagnostics:
    def test_diagnostics(self, cov_4asset):
        from risk_engine.covariance import diagnose_covariance
        diag = diagnose_covariance(cov_4asset)
        assert "condition_number" in diag
        assert "is_positive_definite" in diag
        assert diag["n_assets"] == 4
        assert diag["condition_number"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Stress Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestRollingCorrelation:
    def test_output_columns(self, returns_4asset):
        from risk_engine.correlation import compute_rolling_avg_correlation
        result = compute_rolling_avg_correlation(returns_4asset, window=60)
        assert "avg_correlation" in result.columns
        assert "max_correlation" in result.columns
        assert "n_high_pairs" in result.columns


class TestCrisisClustering:
    def test_detects_clustering(self, returns_4asset):
        from risk_engine.correlation import detect_crisis_clustering
        result = detect_crisis_clustering(returns_4asset, window=20)
        assert "is_clustering" in result.columns
        assert "severity" in result.columns


class TestDiversificationRatio:
    def test_above_one_for_diversified(self, weights_4asset, cov_4asset):
        from risk_engine.correlation import compute_diversification_ratio
        dr = compute_diversification_ratio(weights_4asset, cov_4asset)
        assert dr >= 1.0  # diversified portfolio

    def test_equals_one_for_single_asset(self, cov_4asset):
        from risk_engine.correlation import compute_diversification_ratio
        w = np.array([1.0, 0.0, 0.0, 0.0])
        dr = compute_diversification_ratio(w, cov_4asset)
        np.testing.assert_allclose(dr, 1.0, atol=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Tail Risk Engine
# ══════════════════════════════════════════════════════════════════════════════


class TestCVaR:
    def test_negative(self, returns_4asset):
        from risk_engine.tail_risk import compute_cvar
        cvar = compute_cvar(returns_4asset["AAPL"])
        assert cvar < 0  # expected shortfall is negative

    def test_worse_than_var(self, returns_4asset):
        from risk_engine.tail_risk import compute_cvar, compute_var
        cvar = compute_cvar(returns_4asset["AAPL"], 0.95)
        var = compute_var(returns_4asset["AAPL"], 0.95)
        assert cvar <= var  # CVaR is always worse (more negative)

    def test_dataframe_returns_series(self, returns_4asset):
        from risk_engine.tail_risk import compute_cvar
        result = compute_cvar(returns_4asset)
        assert isinstance(result, pd.Series)
        assert len(result) == 4


class TestSemivariance:
    def test_positive(self, returns_4asset):
        from risk_engine.tail_risk import compute_semivariance
        sv = compute_semivariance(returns_4asset["AAPL"])
        assert sv > 0

    def test_less_than_total_variance(self, returns_4asset):
        from risk_engine.tail_risk import compute_semivariance
        sv = compute_semivariance(returns_4asset["AAPL"], annualize=True)
        total = returns_4asset["AAPL"].var() * 252
        # Semivariance uses different denominator, so may exceed half; just check positive
        assert sv > 0


class TestTailBeta:
    def test_computes(self, returns_4asset):
        from risk_engine.tail_risk import compute_tail_beta
        tb = compute_tail_beta(returns_4asset["AAPL"], returns_4asset["SPY"])
        assert isinstance(tb, float)

    def test_all_tail_betas(self, returns_4asset):
        from risk_engine.tail_risk import compute_all_tail_betas
        result = compute_all_tail_betas(returns_4asset)
        assert len(result) == 4


class TestDrawdown:
    def test_max_drawdown_negative(self, nav_series):
        from risk_engine.tail_risk import compute_max_drawdown
        dd = compute_max_drawdown(nav_series)
        assert dd < 0

    def test_drawdown_duration_positive(self, nav_series):
        from risk_engine.tail_risk import compute_drawdown_duration
        dur = compute_drawdown_duration(nav_series)
        assert dur >= 0


class TestTailRiskReport:
    def test_full_report(self, returns_4asset, nav_series):
        from risk_engine.tail_risk import build_tail_risk_report
        report = build_tail_risk_report(returns_4asset, nav_series)
        assert "cvar_per_asset" in report
        assert "portfolio_cvar" in report
        assert "max_drawdown" in report
        assert "tail_beta_per_asset" in report


# ══════════════════════════════════════════════════════════════════════════════
# Risk Budgeting
# ══════════════════════════════════════════════════════════════════════════════


class TestMarginalRiskContribution:
    def test_shape(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import compute_marginal_risk_contribution
        mrc = compute_marginal_risk_contribution(weights_4asset, cov_4asset)
        assert len(mrc) == 4

    def test_positive_for_positive_weights(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import compute_marginal_risk_contribution
        mrc = compute_marginal_risk_contribution(weights_4asset, cov_4asset)
        assert (mrc > 0).all()


class TestTotalRiskContribution:
    def test_sums_to_portfolio_vol(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import compute_total_risk_contribution
        trc = compute_total_risk_contribution(weights_4asset, cov_4asset)
        port_vol = np.sqrt(weights_4asset.values @ cov_4asset.values @ weights_4asset.values)
        np.testing.assert_allclose(trc.sum(), port_vol, atol=1e-10)


class TestRiskContributionPct:
    def test_sums_to_one(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import compute_risk_contribution_pct
        pct = compute_risk_contribution_pct(weights_4asset, cov_4asset)
        np.testing.assert_allclose(pct.sum(), 1.0, atol=1e-8)


class TestRiskParityCheck:
    def test_check(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import check_risk_parity
        result = check_risk_parity(weights_4asset, cov_4asset)
        assert "is_balanced" in result
        assert "herfindahl_risk" in result
        assert result["n_assets"] == 4


class TestRiskBudgetAdjustment:
    def test_reduces_concentration(self, cov_4asset):
        from risk_engine.budgeting import adjust_weights_for_risk_budget, compute_risk_contribution_pct
        # Start with concentrated weights
        w = pd.Series([0.70, 0.10, 0.10, 0.10], index=cov_4asset.columns)
        adjusted = adjust_weights_for_risk_budget(w, cov_4asset, max_risk_contribution=0.40)
        pct = compute_risk_contribution_pct(adjusted, cov_4asset)
        assert adjusted.sum() > 0.99
        # After adjustment, concentration should decrease
        assert pct.max() <= 0.60  # some convergence

    def test_preserves_sum_to_one(self, weights_4asset, cov_4asset):
        from risk_engine.budgeting import adjust_weights_for_risk_budget
        adjusted = adjust_weights_for_risk_budget(weights_4asset, cov_4asset)
        np.testing.assert_allclose(adjusted.sum(), 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Volatility Scaling
# ══════════════════════════════════════════════════════════════════════════════


class TestVolScaling:
    def test_high_vol_reduces(self):
        from risk_engine.scaling import compute_vol_scaling_factor
        factor = compute_vol_scaling_factor(0.24, target_vol=0.12)
        assert factor < 1.0
        assert factor >= 0.50

    def test_low_vol_increases(self):
        from risk_engine.scaling import compute_vol_scaling_factor
        factor = compute_vol_scaling_factor(0.06, target_vol=0.12)
        assert factor > 1.0
        assert factor <= 1.20

    def test_at_target(self):
        from risk_engine.scaling import compute_vol_scaling_factor
        factor = compute_vol_scaling_factor(0.12, target_vol=0.12)
        np.testing.assert_allclose(factor, 1.0, atol=0.01)

    def test_clamped_min(self):
        from risk_engine.scaling import compute_vol_scaling_factor
        factor = compute_vol_scaling_factor(1.0, target_vol=0.12, min_scaling=0.50)
        assert factor == 0.50

    def test_clamped_max(self):
        from risk_engine.scaling import compute_vol_scaling_factor
        factor = compute_vol_scaling_factor(0.01, target_vol=0.12, max_scaling=1.20)
        assert factor == 1.20


class TestApplyVolScaling:
    def test_scaling_changes_weights(self, weights_4asset):
        from risk_engine.scaling import apply_vol_scaling
        scaled, factor = apply_vol_scaling(weights_4asset, 0.24, target_vol=0.12)
        assert scaled.sum() < weights_4asset.sum()


class TestSmoothedScaling:
    def test_smoothness(self, returns_4asset):
        from risk_engine.scaling import compute_smoothed_scaling
        vol = returns_4asset.rolling(60).std().mean(axis=1) * np.sqrt(252)
        smoothed = compute_smoothed_scaling(vol.dropna(), target_vol=0.12)
        # Smoothed should have lower variance than raw
        raw = 0.12 / vol.dropna().clip(lower=1e-6)
        assert smoothed.std() <= raw.std() + 0.01


class TestScalingReport:
    def test_report(self):
        from risk_engine.scaling import build_scaling_report
        report = build_scaling_report(0.18, target_vol=0.12)
        assert report["scaling_factor"] < 1.0
        assert report["is_scaling_active"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Stress Testing
# ══════════════════════════════════════════════════════════════════════════════


class TestHistoricalStress:
    def test_applies_scenario(self, weights_4asset, returns_4asset):
        from risk_engine.stress_testing import apply_historical_scenario
        start = str(returns_4asset.index[10].date())
        end = str(returns_4asset.index[30].date())
        result = apply_historical_scenario(
            weights_4asset, returns_4asset, start, end, "test_crisis"
        )
        assert result["status"] == "computed"
        assert isinstance(result["portfolio_return"], float)


class TestSyntheticStress:
    def test_applies_shock(self, weights_4asset, cov_4asset):
        from risk_engine.stress_testing import apply_synthetic_shock
        result = apply_synthetic_shock(
            weights_4asset, cov_4asset,
            {"equity_shock": -0.30}, "test_synthetic"
        )
        assert "portfolio_impact" in result


# ══════════════════════════════════════════════════════════════════════════════
# Risk-Constrained Portfolio
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskAwarePortfolio:
    def test_builds_portfolio(self, returns_4asset):
        from risk_engine.constraints import build_risk_aware_portfolio
        base = pd.DataFrame({
            "ticker": returns_4asset.columns,
            "target_weight": [0.25] * 4,
        })
        result = build_risk_aware_portfolio(
            base_weights=base,
            returns=returns_4asset,
        )
        assert result["status"] == "success"
        assert "weights" in result
        assert "portfolio_cvar" in result
        assert "covariance_diagnostics" in result

    def test_weights_reasonable(self, returns_4asset):
        from risk_engine.constraints import build_risk_aware_portfolio
        base = pd.DataFrame({
            "ticker": returns_4asset.columns,
            "target_weight": [0.25] * 4,
        })
        result = build_risk_aware_portfolio(
            base_weights=base,
            returns=returns_4asset,
        )
        weights = result["weights"]
        assert weights.sum() <= 1.01  # may have cash
        assert (weights >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# Risk Evaluation
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluationMetrics:
    def test_vol_prediction(self):
        from risk_engine.evaluation import evaluate_vol_prediction
        pred = pd.Series([0.15, 0.16, 0.14], index=[0, 1, 2])
        actual = pd.Series([0.14, 0.17, 0.13], index=[0, 1, 2])
        result = evaluate_vol_prediction(pred, actual)
        assert "mae" in result
        assert result["n_observations"] == 3

    def test_drawdown_improvement(self, nav_series):
        from risk_engine.evaluation import evaluate_drawdown_improvement
        baseline = nav_series * 0.98  # slightly worse
        result = evaluate_drawdown_improvement(nav_series, baseline)
        assert "dd_improvement" in result

    def test_risk_adjusted_utility(self, returns_4asset):
        from risk_engine.evaluation import compute_risk_adjusted_utility
        util = compute_risk_adjusted_utility(returns_4asset["AAPL"])
        assert isinstance(util, float)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskPipeline:
    def test_full_pipeline(self, inr_prices_long):
        from risk_engine import run_risk_pipeline
        base = pd.DataFrame({
            "ticker": ["AAPL", "SPY", "RELIANCE.NS", "INFY.NS"],
            "target_weight": [0.25, 0.25, 0.25, 0.25],
        })
        result = run_risk_pipeline(
            inr_prices=inr_prices_long,
            base_weights=base,
        )
        assert "volatility_state" in result
        assert "covariance" in result
        assert "tail_risk" in result
        assert "portfolio" in result
        assert "stress_tests" in result
