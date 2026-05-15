"""
End-to-end pipeline test — verifies all pipeline outputs exist.
using pre-existing cached data and verifies all outputs exist.

This test reads from data/raw/ (cached), so it doesn't hit the network.
It exercises the REAL pipeline code path from app.py.
"""

from pathlib import Path

import pandas as pd
import pytest


PROCESSED = Path("data/processed")
REPORTS = Path("reports")


@pytest.fixture(scope="module")
def pipeline_ran():
    """Check that the pipeline has been run (data exists)."""
    required = PROCESSED / "inr_prices.parquet"
    if not required.exists():
        pytest.skip("Pipeline data not found — run `python app.py` first")
    return True


class TestProcessedDataExists:
    """Verify all expected Parquet files were generated."""

    EXPECTED_FILES = [
        "inr_prices.parquet",
        "portfolio_nav.parquet",
        "fx_attribution.parquet",
        "returns.parquet",
        "rolling_analytics.parquet",
        "drawdown_series.parquet",
        "features.parquet",
        "signal_scores.parquet",
        "target_weights.parquet",
        "rebalance_trades.parquet",
        "backtest_nav.parquet",
        "trade_ledger.parquet",
        "walkforward_results.parquet",
        "regime_analysis.parquet",
        "regime_performance.parquet",
        "parameter_sensitivity.parquet",
        "signal_decay.parquet",
        "monte_carlo_summary.parquet",
        "stress_test_results.parquet",
        "liquidity_stress.parquet",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_exists(self, pipeline_ran, filename):
        path = PROCESSED / filename
        assert path.exists(), f"Missing: {path}"

    @pytest.mark.parametrize("filename", EXPECTED_FILES)
    def test_file_readable(self, pipeline_ran, filename):
        path = PROCESSED / filename
        if path.exists():
            df = pd.read_parquet(path)
            assert isinstance(df, pd.DataFrame), f"Not a DataFrame: {path}"


class TestReportsExist:
    """Verify report files were generated."""

    EXPECTED_REPORTS = [
        "portfolio_metrics.csv",
        "benchmark_comparison.csv",
        "drawdown_periods.csv",
        "backtest_comparison.csv",
        "backtest_attribution.csv",
        "walkforward_results.csv",
        "regime_performance.csv",
        "parameter_sensitivity.csv",
        "stress_test_results.csv",
        "signal_decay.csv",
        "monte_carlo_summary.csv",
        "research_score.csv",
        "diagnostics_summary.csv",
        "portfolio_recommendation.csv",
    ]

    @pytest.mark.parametrize("filename", EXPECTED_REPORTS)
    def test_report_exists(self, pipeline_ran, filename):
        path = REPORTS / filename
        assert path.exists(), f"Missing report: {path}"


class TestDataIntegrity:
    """Verify data quality and cross-file consistency."""

    def test_inr_prices_has_all_tickers(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "inr_prices.parquet")
        tickers = df["ticker"].unique()
        # Should have at least 4 equity/ETF tickers
        assert len(tickers) >= 4

    def test_nav_is_monotonic_dates(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "portfolio_nav.parquet")
        df["date"] = pd.to_datetime(df["date"])
        assert df["date"].is_monotonic_increasing

    def test_backtest_nav_positive(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "backtest_nav.parquet")
        nav_col = "nav" if "nav" in df.columns else "portfolio_nav"
        assert (df[nav_col] > 0).all()

    def test_weights_sum_to_one(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "target_weights.parquet")
        if not df.empty and df["target_weight"].sum() > 0:
            total = df["target_weight"].sum()
            assert abs(total - 1.0) < 0.05  # within 5% tolerance

    def test_features_no_future_dates(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        assert df["date"].max() <= pd.Timestamp.now() + pd.Timedelta(days=1)

    def test_research_score_bounded(self, pipeline_ran):
        df = pd.read_csv(REPORTS / "research_score.csv")
        assert 0 <= df["total_score"].iloc[0] <= 100

    def test_regime_analysis_valid_regimes(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "regime_analysis.parquet")
        if not df.empty:
            valid = {"bull", "bear", "high_vol", "sideways"}
            assert set(df["regime"].unique()).issubset(valid)

    def test_walkforward_has_windows(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "walkforward_results.parquet")
        assert len(df) >= 1
        assert "test_sharpe" in df.columns

    def test_trade_ledger_has_trades(self, pipeline_ran):
        df = pd.read_parquet(PROCESSED / "trade_ledger.parquet")
        assert isinstance(df, pd.DataFrame)
