"""
Portfolio OS — Sprint 1 → 5 entry point.

Sprint 1: Downloads market data, validates, persists to parquet.
Sprint 2: FX normalization, portfolio NAV, attribution, exposure.
Sprint 3: Portfolio analytics, risk metrics, rolling diagnostics, benchmarks.
Sprint 4: Feature engineering, signal generation, feature store.
Sprint 5: Portfolio optimization, HRP, constraints, allocation engine.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ingestion.yahoo_loader import download_yahoo_data, download_batch
from ingestion.mf_loader import download_mf_data, get_scheme_name
from utils.validators import validate_dataframe

from fx.fx_loader import get_fx_series
from fx.converter import convert_prices_to_inr
from fx.attribution import calculate_fx_attribution, attribution_summary
from analytics.holdings_loader import load_holdings
from analytics.portfolio_nav import calculate_portfolio_nav, calculate_asset_contributions
from analytics.exposure import latest_exposure_snapshot, calculate_concentration_metrics
from analytics.returns import build_returns_table
from analytics.metrics import calculate_all_metrics
from analytics.drawdown import calculate_drawdown_series, calculate_drawdown_periods
from analytics.rolling import build_rolling_table
from analytics.benchmark import compare_against_benchmark, build_benchmark_nav
from analytics.charts import save_all_charts
from reports.report_generator import (
    export_metrics_csv,
    export_benchmark_comparison_csv,
    export_drawdown_periods_csv,
    generate_html_report,
)
from features.feature_store import build_feature_store, save_feature_store
from features.validators import validate_features, check_lookahead_bias
from features.signal_ranker import calculate_composite_score

from optimization.baselines import (
    equal_weight_portfolio,
    inverse_volatility_portfolio,
    risk_parity_portfolio,
)
from optimization.covariance import (
    calculate_covariance_matrix,
    calculate_shrinkage_covariance,
)
from optimization.hrp import allocate_hrp_weights
from optimization.constraints import apply_weight_caps, apply_country_constraints
from optimization.allocator import build_signal_tilted_portfolio
from optimization.turnover import calculate_turnover, calculate_weight_drift
from optimization.rebalance import should_rebalance, calculate_rebalance_trades
from optimization.reporting import generate_allocation_report


ASSET_MASTER = Path("configs/asset_master.csv")
PROCESSED_DIR = Path("data/processed")

# SBI Bluechip Fund — Direct Growth (example Indian MF)
TEST_MF_SCHEME = "119598"


def load_asset_master() -> pd.DataFrame:
    """Load the asset master CSV."""
    df = pd.read_csv(ASSET_MASTER)
    logger.info(f"Asset master loaded: {len(df)} assets")
    return df


# ── Sprint 1 ─────────────────────────────────────────────────────────────────


def run_yahoo_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download and validate all Yahoo-sourced assets."""
    yahoo_tickers = master[master["asset_type"] != "mf"]["ticker"].tolist()
    results: dict[str, pd.DataFrame] = {}

    for ticker in yahoo_tickers:
        df = download_yahoo_data(ticker)
        if not df.empty:
            validate_dataframe(df, ticker)
        results[ticker] = df

    return results


def run_mf_pipeline() -> pd.DataFrame:
    """Download and validate a test mutual fund."""
    name = get_scheme_name(TEST_MF_SCHEME)
    logger.info(f"MF scheme: {name}")

    df = download_mf_data(TEST_MF_SCHEME)
    if not df.empty:
        validate_dataframe(df, f"MF_{TEST_MF_SCHEME}")

    return df


def print_sprint1_summary(yahoo_data: dict[str, pd.DataFrame], mf_data: pd.DataFrame) -> None:
    """Print a concise summary of all downloaded data."""
    logger.info("=" * 60)
    logger.info("SPRINT 1 — DATA LAKE SUMMARY")
    logger.info("=" * 60)

    for ticker, df in yahoo_data.items():
        if df.empty:
            logger.warning(f"  {ticker:15s} — NO DATA")
        else:
            logger.info(
                f"  {ticker:15s} — {len(df):>6} rows | "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )

    if not mf_data.empty:
        logger.info(
            f"  {'MF_' + TEST_MF_SCHEME:15s} — {len(mf_data):>6} rows | "
            f"{mf_data['date'].min().date()} → {mf_data['date'].max().date()}"
        )

    raw_dir = Path("data/raw")
    parquets = list(raw_dir.glob("*.parquet"))
    logger.info(f"\nParquet files in data/raw/: {len(parquets)}")
    for p in sorted(parquets):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name:30s} {size_kb:>8.1f} KB")


# ── Sprint 2 ─────────────────────────────────────────────────────────────────


def run_sprint2(
    yahoo_data: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Execute Sprint 2 pipeline. Returns (inr_prices, nav, contributions, exposures)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 2 — FX NORMALIZATION & PORTFOLIO NAV")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. FX series
    logger.info("\n▸ Step 1 — FX Series")
    fx_series = get_fx_series()

    # 2. Holdings
    logger.info("\n▸ Step 2 — Holdings")
    holdings = load_holdings()

    # 3. FX conversion
    logger.info("\n▸ Step 3 — INR Price Conversion")
    inr_prices = convert_prices_to_inr(yahoo_data, master, fx_series)
    _save(inr_prices, "inr_prices.parquet")

    # 4. Portfolio NAV
    logger.info("\n▸ Step 4 — Portfolio NAV")
    nav = calculate_portfolio_nav(inr_prices, holdings)
    _save(nav, "portfolio_nav.parquet")

    # 5. Asset contributions
    logger.info("\n▸ Step 5 — Asset Contributions")
    contributions = calculate_asset_contributions(inr_prices, holdings)

    # 6. FX attribution
    logger.info("\n▸ Step 6 — FX Attribution")
    attr = calculate_fx_attribution(inr_prices)
    attr_summary = attribution_summary(attr)
    _save(attr, "fx_attribution.parquet")

    # 7. Exposure
    logger.info("\n▸ Step 7 — Exposure Analytics")
    exposures = latest_exposure_snapshot(contributions, master)

    # 8. Summary
    _print_sprint2_summary(nav, attr_summary)

    return inr_prices, nav, contributions, exposures


def _save(df: pd.DataFrame, filename: str) -> None:
    """Save a processed dataframe to data/processed/."""
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved → {path}  ({len(df)} rows)")


def _print_sprint2_summary(nav: pd.DataFrame, attr_summary: pd.DataFrame) -> None:
    """Final Sprint 2 summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 2 — SUMMARY")
    logger.info("=" * 60)

    if not nav.empty:
        first = nav.iloc[0]
        last = nav.iloc[-1]
        total_return = (last["portfolio_nav"] / first["portfolio_nav"] - 1) * 100
        logger.info(
            f"  Portfolio NAV: ₹{first['portfolio_nav']:,.0f} → ₹{last['portfolio_nav']:,.0f}  "
            f"({total_return:+.2f}%)"
        )
        logger.info(f"  Date range: {first['date'].date()} → {last['date'].date()}")
        logger.info(f"  Trading days: {len(nav)}")

    processed = list(PROCESSED_DIR.glob("*.parquet"))
    logger.info(f"\n  Processed files: {len(processed)}")
    for p in sorted(processed):
        size_kb = p.stat().st_size / 1024
        logger.info(f"    {p.name:30s} {size_kb:>8.1f} KB")


# ── Sprint 3 ─────────────────────────────────────────────────────────────────


def run_sprint3(
    nav: pd.DataFrame,
    inr_prices: pd.DataFrame,
    contributions: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> None:
    """Execute Sprint 3 pipeline: Analytics → Risk → Rolling → Benchmark → Charts."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 3 — PORTFOLIO ANALYTICS & RISK ENGINE")
    logger.info("=" * 60)

    # 1. Returns
    logger.info("\n▸ Step 1 — Returns Engine")
    returns_df = build_returns_table(nav)
    _save(returns_df, "returns.parquet")

    # 2. Core risk metrics
    logger.info("\n▸ Step 2 — Core Risk Metrics")
    metrics = calculate_all_metrics(nav)

    # 3. Drawdown analysis
    logger.info("\n▸ Step 3 — Drawdown Analysis")
    dd_series = calculate_drawdown_series(nav)
    dd_periods = calculate_drawdown_periods(nav)
    _save(dd_series, "drawdown_series.parquet")

    # 4. Rolling analytics
    logger.info("\n▸ Step 4 — Rolling Analytics")
    # Build benchmark returns for beta calculation
    spy_nav = build_benchmark_nav(inr_prices, "SPY")
    bench_returns = None
    if not spy_nav.empty:
        spy_aligned = spy_nav.merge(nav[["date"]], on="date", how="inner")
        spy_aligned["daily_return"] = spy_aligned["portfolio_nav"].pct_change()
        bench_returns = spy_aligned["daily_return"]

    rolling_df = build_rolling_table(nav, bench_returns)
    _save(rolling_df, "rolling_analytics.parquet")

    # 5. Concentration metrics
    logger.info("\n▸ Step 5 — Concentration Metrics")
    concentration = calculate_concentration_metrics(contributions)

    # 6. Benchmark comparison
    logger.info("\n▸ Step 6 — Benchmark Comparison")
    comparison = pd.DataFrame()
    if not spy_nav.empty:
        comparison = compare_against_benchmark(nav, spy_nav, "SPY")

    # 7. Charts
    logger.info("\n▸ Step 7 — Visualization")
    save_all_charts(
        nav=nav,
        returns_df=returns_df,
        drawdown_df=dd_series,
        rolling_df=rolling_df,
        contributions=contributions,
        exposure_snapshot=exposures,
        benchmark_nav=spy_nav if not spy_nav.empty else None,
        benchmark_name="SPY",
    )

    # 8. Reports
    logger.info("\n▸ Step 8 — Report Export")
    export_metrics_csv(metrics)
    if not comparison.empty:
        export_benchmark_comparison_csv(comparison)
    if not dd_periods.empty:
        export_drawdown_periods_csv(dd_periods)
    generate_html_report(metrics, comparison, dd_periods)

    _print_sprint3_summary(metrics, dd_periods)


def _print_sprint3_summary(metrics: dict, dd_periods: pd.DataFrame) -> None:
    """Final Sprint 3 summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 3 — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  CAGR            {metrics['cagr']:+.2%}")
    logger.info(f"  Sharpe          {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Sortino         {metrics['sortino_ratio']:.3f}")
    logger.info(f"  Max Drawdown    {metrics['max_drawdown']:+.2%}")
    logger.info(f"  Calmar          {metrics['calmar_ratio']:.3f}")

    reports = list(Path("reports").glob("*"))
    logger.info(f"\n  Reports & charts: {len(reports)} files in reports/")


# ── Sprint 4 ─────────────────────────────────────────────────────────────────


def run_sprint4(inr_prices: pd.DataFrame) -> None:
    """Execute Sprint 4 pipeline: Feature Engineering → Store → Validate → Rank."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 4 — FEATURE ENGINEERING & SIGNAL LAYER")
    logger.info("=" * 60)

    # 1. Build feature store
    logger.info("\n▸ Step 1 — Build Feature Store")
    store = build_feature_store(inr_prices)
    save_feature_store(store)

    # 2. Validate features
    logger.info("\n▸ Step 2 — Feature Validation")
    validate_features(store)
    check_lookahead_bias(store, inr_prices)

    # 3. Signal ranking
    logger.info("\n▸ Step 3 — Signal Ranking")
    scores = calculate_composite_score(store)
    _save(scores, "signal_scores.parquet")

    _print_sprint4_summary(store, scores)


def _print_sprint4_summary(store: pd.DataFrame, scores: pd.DataFrame) -> None:
    """Final Sprint 4 summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 4 — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Features computed:  {store['feature'].nunique()}")
    logger.info(f"  Tickers:            {store['ticker'].nunique()}")
    logger.info(f"  Total rows:         {len(store):,}")
    logger.info(f"  Feature store:      data/processed/features.parquet")

    size_mb = Path("data/processed/features.parquet").stat().st_size / (1024 * 1024)
    logger.info(f"  Store size:         {size_mb:.2f} MB")


# ── Sprint 5 ─────────────────────────────────────────────────────────────────


def _build_wide_returns(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot inr_prices to wide-format daily returns (columns = tickers)."""
    # Exclude FX ticker
    prices = inr_prices[~inr_prices["ticker"].str.contains("=X")].copy()
    wide = prices.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    )
    wide = wide.sort_index().ffill().dropna()
    returns = wide.pct_change().dropna()
    return returns


def run_sprint5(
    inr_prices: pd.DataFrame,
    nav: pd.DataFrame,
    contributions: pd.DataFrame,
    master: pd.DataFrame,
) -> None:
    """Execute Sprint 5: Portfolio Optimization Engine."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 5 — PORTFOLIO OPTIMIZATION ENGINE")
    logger.info("=" * 60)

    # Prepare wide-format returns
    returns = _build_wide_returns(inr_prices)
    tickers = list(returns.columns)
    logger.info(f"\n  Return matrix: {returns.shape[0]} days × {len(tickers)} assets")
    logger.info(f"  Assets: {', '.join(tickers)}")

    # ── Step 1: Baseline strategies ──────────────────────────────────────
    logger.info("\n▸ Step 1 — Baseline Strategies")
    ew = equal_weight_portfolio(tickers)
    iv = inverse_volatility_portfolio(returns, tickers)
    rp = risk_parity_portfolio(returns, tickers)

    # ── Step 2: Covariance estimation ────────────────────────────────────
    logger.info("\n▸ Step 2 — Covariance Estimation")
    cov_sample = calculate_covariance_matrix(returns, window=120)
    cov_shrink = calculate_shrinkage_covariance(returns, window=120)

    # ── Step 3: HRP allocation ───────────────────────────────────────────
    logger.info("\n▸ Step 3 — HRP Allocation")
    hrp = allocate_hrp_weights(returns, cov=cov_shrink)

    # ── Step 4: Constraints ──────────────────────────────────────────────
    logger.info("\n▸ Step 4 — Constraint Engine")
    hrp_constrained = apply_weight_caps(
        hrp, max_weight=0.40, min_weight=0.05, cash_reserve=0.0
    )
    hrp_constrained = apply_country_constraints(
        hrp_constrained, master, country_caps={"US": 0.60, "IN": 0.60}
    )

    # ── Step 5: Signal-tilted allocation ─────────────────────────────────
    logger.info("\n▸ Step 5 — Signal-Tilted Allocation")
    scores_path = PROCESSED_DIR / "signal_scores.parquet"
    if scores_path.exists():
        signal_scores = pd.read_parquet(scores_path)
        tilted = build_signal_tilted_portfolio(
            hrp_constrained, signal_scores, tilt_strength=0.20
        )
    else:
        logger.warning("No signal scores found — using constrained HRP as final")
        tilted = hrp_constrained.copy()
        tilted["strategy"] = "hrp_constrained"

    # ── Step 6: Turnover analysis ────────────────────────────────────────
    logger.info("\n▸ Step 6 — Turnover Analysis")
    # Current weights from latest contributions
    latest_date = contributions["date"].max()
    latest_contrib = contributions[contributions["date"] == latest_date][
        ["ticker", "contribution_pct"]
    ].copy()
    latest_contrib = latest_contrib.rename(
        columns={"contribution_pct": "current_weight"}
    )
    latest_contrib["current_weight"] = latest_contrib["current_weight"] / 100.0

    turnover_df = calculate_turnover(latest_contrib, tilted)

    # Weight drift simulation
    drift = calculate_weight_drift(tilted, returns, n_days=60)

    # ── Step 7: Rebalance decision ───────────────────────────────────────
    logger.info("\n▸ Step 7 — Rebalance Decision")
    rebalance = should_rebalance(
        latest_contrib, tilted, drift_threshold=0.05, method="threshold"
    )

    portfolio_value = nav.iloc[-1]["portfolio_nav"]
    if rebalance["should_rebalance"]:
        trades = calculate_rebalance_trades(
            latest_contrib, tilted, portfolio_value
        )
        _save(trades, "rebalance_trades.parquet")

    # ── Step 8: Reporting ────────────────────────────────────────────────
    logger.info("\n▸ Step 8 — Allocation Reporting")
    strategies = {
        "equal_weight": ew,
        "inverse_vol": iv,
        "risk_parity": rp,
        "hrp": hrp,
        "signal_tilted": tilted,
    }
    comparison = generate_allocation_report(
        strategies, tilted, turnover_df, rebalance
    )

    # Save final weights
    _save(tilted, "target_weights.parquet")

    _print_sprint5_summary(strategies, tilted, turnover_df, rebalance)


def _print_sprint5_summary(
    strategies: dict,
    final: pd.DataFrame,
    turnover_df: pd.DataFrame,
    rebalance: dict,
) -> None:
    """Final Sprint 5 summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPRINT 5 — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Strategies compared: {len(strategies)}")
    logger.info(f"  Final strategy:      {final['strategy'].iloc[0]}")
    logger.info(f"  Weights sum:         {final['target_weight'].sum():.4f}")
    if "total_turnover" in turnover_df.attrs:
        logger.info(f"  Turnover (1-way):    {turnover_df.attrs['total_turnover']:.2%}")
    status = "YES" if rebalance["should_rebalance"] else "NO"
    logger.info(f"  Rebalance needed:    {status} — {rebalance['reason']}")

    logger.info("\n  Target Portfolio:")
    for _, row in final.sort_values("target_weight", ascending=False).iterrows():
        logger.info(f"    {row['ticker']:15s}  {row['target_weight']:.2%}")

    outputs = [
        "data/processed/target_weights.parquet",
        "reports/strategy_comparison.csv",
        "reports/portfolio_recommendation.csv",
    ]
    logger.info(f"\n  Outputs: {len(outputs)} files")
    for o in outputs:
        logger.info(f"    {o}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Portfolio OS — Full Pipeline (Sprint 1 → 5)")
    logger.info("-" * 50)

    master = load_asset_master()

    # Sprint 1: Data ingestion
    yahoo_data = run_yahoo_pipeline(master)
    mf_data = run_mf_pipeline()
    print_sprint1_summary(yahoo_data, mf_data)

    # Sprint 2: FX normalization & portfolio engine
    inr_prices, nav, contributions, exposures = run_sprint2(yahoo_data, master)

    # Sprint 3: Analytics & risk engine
    run_sprint3(nav, inr_prices, contributions, exposures, master)

    # Sprint 4: Feature engineering & signal layer
    run_sprint4(inr_prices)

    # Sprint 5: Portfolio optimization engine
    run_sprint5(inr_prices, nav, contributions, master)

    logger.info("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
