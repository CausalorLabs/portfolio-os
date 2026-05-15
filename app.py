"""
Portfolio OS — main pipeline entry point.

Pipeline stages:
  1. Data ingestion — downloads market data, validates, persists to parquet.
  2. FX normalization — portfolio NAV, attribution, exposure.
  3. Analytics — risk metrics, rolling diagnostics, benchmarks.
  4. Feature engineering — signal generation, feature store.
  5. Optimization — HRP, constraints, allocation engine.
  6. Backtesting — friction-aware, taxes, slippage, benchmarks.
  7. Dashboard — recommendations, portfolio state.
  8. Validation — robustness & research hardening.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from ingestion.yahoo_loader import download_yahoo_data, download_batch
from ingestion.mf_loader import download_mf_data, get_scheme_name
from ingestion.fixed_income_loader import (
    generate_fixed_income_prices,
    generate_metal_proxy_prices,
    METAL_YAHOO_MAP,
)
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

from backtests.engine import run_backtest
from backtests.benchmark import run_benchmark_suite, compare_backtest_results
from backtests.attribution import calculate_performance_attribution
from backtests.reporting import generate_backtest_report

from validation.walkforward import run_walkforward_validation
from validation.regimes import identify_market_regimes, evaluate_regime_performance
from validation.robustness import run_parameter_sensitivity, evaluate_stability_surface
from validation.overfitting import detect_overfitting
from validation.signal_decay import evaluate_forward_returns, calculate_signal_decay
from validation.monte_carlo import run_monte_carlo_simulation
from validation.stress_tests import run_stress_scenarios, simulate_liquidity_stress
from validation.diagnostics import generate_diagnostics
from validation.reporting import generate_validation_report
from validation.research_score import calculate_research_score


ASSET_MASTER = Path("configs/asset_master.csv")
PROCESSED_DIR = Path("data/processed")

# Asset types that have market prices (eligible for optimization/features/backtests)
MARKET_ASSET_TYPES = {"equity", "etf", "mf", "metal"}
# Asset types with fixed rates (synthetic prices, excluded from optimization)
FIXED_ASSET_TYPES = {"fixed_income"}
# All portfolio asset types (everything except fx)
PORTFOLIO_ASSET_TYPES = MARKET_ASSET_TYPES | FIXED_ASSET_TYPES


def load_asset_master() -> pd.DataFrame:
    """Load the asset master CSV."""
    df = pd.read_csv(ASSET_MASTER)
    logger.info(f"Asset master loaded: {len(df)} assets")
    return df


# ── Data Ingestion ────────────────────────────────────────────────────────────


def run_yahoo_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download and validate all Yahoo-sourced assets (equity, etf, fx)."""
    yahoo_types = {"equity", "etf", "fx"}
    yahoo_tickers = master[master["asset_type"].isin(yahoo_types)]["ticker"].tolist()
    results: dict[str, pd.DataFrame] = {}

    for ticker in yahoo_tickers:
        df = download_yahoo_data(ticker)
        if not df.empty:
            validate_dataframe(df, ticker)
        results[ticker] = df

    return results


def run_mf_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download and validate all mutual funds from asset_master."""
    mf_rows = master[master["asset_type"] == "mf"]
    results: dict[str, pd.DataFrame] = {}

    for _, row in mf_rows.iterrows():
        ticker = row["ticker"]
        scheme_code = ticker.replace("MF_", "")
        name = get_scheme_name(scheme_code)
        logger.info(f"MF scheme: {name} ({scheme_code})")

        df = download_mf_data(scheme_code)
        if not df.empty:
            validate_dataframe(df, ticker)
            # Reshape MF NAV data to OHLCV format for pipeline compatibility
            df = df.rename(columns={"nav": "adj_close"})
            df["open"] = df["adj_close"]
            df["high"] = df["adj_close"]
            df["low"] = df["adj_close"]
            df["close"] = df["adj_close"]
            df["volume"] = 0
        results[ticker] = df

    return results


def run_fixed_income_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate synthetic daily prices for fixed-income instruments."""
    fi_rows = master[master["asset_type"] == "fixed_income"]
    results: dict[str, pd.DataFrame] = {}

    for _, row in fi_rows.iterrows():
        ticker = row["ticker"]
        rate = row.get("annual_rate", 0.0)
        if pd.isna(rate) or rate == 0:
            logger.warning(f"{ticker}: no annual_rate set, using 0% — update asset_master.csv")
            rate = 0.0
        df = generate_fixed_income_prices(ticker, rate)
        results[ticker] = df

    return results


def run_metal_pipeline(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download commodity prices for physical metals."""
    metal_rows = master[master["asset_type"] == "metal"]
    results: dict[str, pd.DataFrame] = {}

    for _, row in metal_rows.iterrows():
        ticker = row["ticker"]
        yahoo_ticker = METAL_YAHOO_MAP.get(ticker)
        if yahoo_ticker:
            df = generate_metal_proxy_prices(ticker, yahoo_ticker)
        else:
            logger.warning(f"{ticker}: no Yahoo commodity mapping — skipping")
            df = pd.DataFrame()
        results[ticker] = df

    return results


def print_ingestion_summary(all_data: dict[str, pd.DataFrame]) -> None:
    """Print a concise summary of all downloaded data."""
    logger.info("=" * 60)
    logger.info("DATA LAKE SUMMARY")
    logger.info("=" * 60)

    for ticker, df in sorted(all_data.items()):
        if df.empty:
            logger.warning(f"  {ticker:15s} — NO DATA")
        else:
            logger.info(
                f"  {ticker:15s} — {len(df):>6} rows | "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )

    raw_dir = Path("data/raw")
    parquets = list(raw_dir.glob("*.parquet"))
    logger.info(f"\nParquet files in data/raw/: {len(parquets)}")
    for p in sorted(parquets):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name:30s} {size_kb:>8.1f} KB")


# ── FX Normalization & NAV ─────────────────────────────────────────────────────


def run_fx_and_nav(
    all_data: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """FX normalization & portfolio NAV pipeline. Returns (inr_prices, nav, contributions, exposures)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FX NORMALIZATION & PORTFOLIO NAV")
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
    inr_prices = convert_prices_to_inr(all_data, master, fx_series)
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
    _print_fx_summary(nav, attr_summary)

    return inr_prices, nav, contributions, exposures


def _save(df: pd.DataFrame, filename: str) -> None:
    """Save a processed dataframe to data/processed/."""
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved → {path}  ({len(df)} rows)")


def _print_fx_summary(nav: pd.DataFrame, attr_summary: pd.DataFrame) -> None:
    """FX & NAV summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FX & NAV — SUMMARY")
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


# ── Analytics & Risk Engine ────────────────────────────────────────────────────


def run_analytics(
    nav: pd.DataFrame,
    inr_prices: pd.DataFrame,
    contributions: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> None:
    """Analytics pipeline: Returns → Risk → Rolling → Benchmark → Charts."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PORTFOLIO ANALYTICS & RISK ENGINE")
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

    _print_analytics_summary(metrics, dd_periods)


def _print_analytics_summary(metrics: dict, dd_periods: pd.DataFrame) -> None:
    """Analytics summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYTICS — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  CAGR            {metrics['cagr']:+.2%}")
    logger.info(f"  Sharpe          {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Sortino         {metrics['sortino_ratio']:.3f}")
    logger.info(f"  Max Drawdown    {metrics['max_drawdown']:+.2%}")
    logger.info(f"  Calmar          {metrics['calmar_ratio']:.3f}")

    reports = list(Path("reports").glob("*"))
    logger.info(f"\n  Reports & charts: {len(reports)} files in reports/")


# ── Feature Engineering ───────────────────────────────────────────────────────


def run_feature_engineering(inr_prices: pd.DataFrame) -> None:
    """Feature engineering pipeline: Build Store → Validate → Rank."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING & SIGNAL LAYER")
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

    _print_feature_summary(store, scores)


def _print_feature_summary(store: pd.DataFrame, scores: pd.DataFrame) -> None:
    """Feature engineering summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURES — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Features computed:  {store['feature'].nunique()}")
    logger.info(f"  Tickers:            {store['ticker'].nunique()}")
    logger.info(f"  Total rows:         {len(store):,}")
    logger.info(f"  Feature store:      data/processed/features.parquet")

    size_mb = Path("data/processed/features.parquet").stat().st_size / (1024 * 1024)
    logger.info(f"  Store size:         {size_mb:.2f} MB")


# ── Portfolio Optimization ─────────────────────────────────────────────────────


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


def run_optimization(
    inr_prices: pd.DataFrame,
    nav: pd.DataFrame,
    contributions: pd.DataFrame,
    master: pd.DataFrame,
) -> None:
    """Portfolio optimization pipeline: baselines → HRP → constraints → signal tilt → rebalance."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PORTFOLIO OPTIMIZATION ENGINE")
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
        hrp, max_weight=0.40, min_weight=0.01, cash_reserve=0.0
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

    _print_optimization_summary(strategies, tilted, turnover_df, rebalance)


def _print_optimization_summary(
    strategies: dict,
    final: pd.DataFrame,
    turnover_df: pd.DataFrame,
    rebalance: dict,
) -> None:
    """Optimization summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION — SUMMARY")
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


# ── Friction-Aware Backtesting ─────────────────────────────────────────────────


def _build_wide_prices(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """Pivot inr_prices to wide-format daily prices (columns = tickers)."""
    prices = inr_prices[~inr_prices["ticker"].str.contains("=X")].copy()
    wide = prices.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    )
    wide = wide.sort_index().ffill().dropna()
    return wide


def _build_country_map(master: pd.DataFrame) -> dict[str, str]:
    """Build ticker → country map from asset master."""
    return dict(zip(master["ticker"], master["country"]))


def _hrp_signal_strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """HRP + signal tilt strategy for backtesting."""
    from optimization.covariance import calculate_shrinkage_covariance
    from optimization.hrp import allocate_hrp_weights
    from optimization.constraints import apply_weight_caps

    ret = returns[tickers].dropna()
    if len(ret) < 60:
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}

    cov = calculate_shrinkage_covariance(ret, window=120)
    hrp_df = allocate_hrp_weights(ret, cov=cov)
    hrp_df = apply_weight_caps(hrp_df, max_weight=0.40, min_weight=0.01)
    return dict(zip(hrp_df["ticker"], hrp_df["target_weight"]))


def run_backtesting(inr_prices: pd.DataFrame, master: pd.DataFrame) -> None:
    """Friction-aware backtesting pipeline: strategy → benchmarks → attribution → reports."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FRICTION-AWARE BACKTESTING ENGINE")
    logger.info("=" * 60)

    wide_prices = _build_wide_prices(inr_prices)
    country_map = _build_country_map(master)
    initial_capital = 1_000_000.0

    # ── Step 1: Run primary strategy backtest ─────────────────────────
    logger.info("\n▸ Step 1 — Primary Strategy Backtest (HRP, quarterly)")
    primary = run_backtest(
        wide_prices=wide_prices,
        strategy_fn=_hrp_signal_strategy,
        initial_capital=initial_capital,
        frequency="quarterly",
        slippage_bps=10,
        country_map=country_map,
        warmup_days=120,
    )

    # ── Step 2: Run benchmark suite ─────────────────────────────────────
    logger.info("\n▸ Step 2 — Benchmark Suite")
    benchmarks = run_benchmark_suite(
        wide_prices=wide_prices,
        country_map=country_map,
        initial_capital=initial_capital,
        slippage_bps=10,
        warmup_days=120,
    )

    # Add primary to results for comparison
    all_results = {"hrp_optimized": primary, **benchmarks}

    # ── Step 3: Compare strategies ──────────────────────────────────────
    logger.info("\n▸ Step 3 — Strategy Comparison")
    comparison = compare_backtest_results(all_results)

    # ── Step 4: Attribution ─────────────────────────────────────────────
    logger.info("\n▸ Step 4 — Performance Attribution")
    attribution = calculate_performance_attribution(
        nav_series=primary["nav_series"],
        ledger_df=primary["ledger"].to_dataframe(),
        initial_capital=initial_capital,
    )

    # ── Step 5: Save artifacts ──────────────────────────────────────────
    logger.info("\n▸ Step 5 — Save & Report")
    primary["ledger"].save()
    _save(primary["nav_series"], "backtest_nav.parquet")

    generate_backtest_report(
        comparison=comparison,
        attribution=attribution,
        ledger_summary=primary["ledger"].summary(),
    )

    _print_backtest_summary(comparison, attribution, primary)


def _print_backtest_summary(
    comparison: pd.DataFrame,
    attribution: dict,
    primary: dict,
) -> None:
    """Backtest summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKTESTING — SUMMARY")
    logger.info("=" * 60)

    if not comparison.empty and "hrp_optimized" in comparison.index:
        row = comparison.loc["hrp_optimized"]
        logger.info(f"  HRP Optimized:")
        logger.info(f"    Net CAGR:       {row['cagr']:+.2%}")
        logger.info(f"    Sharpe:         {row['sharpe']:.3f}")
        logger.info(f"    Sortino:        {row['sortino']:.3f}")
        logger.info(f"    Max Drawdown:   {row['max_drawdown']:+.2%}")
        logger.info(f"    Total friction: ₹{row['total_friction']:,.0f}")

    if attribution:
        logger.info(f"\n  Attribution:")
        logger.info(f"    Gross CAGR:     {attribution['gross_cagr']:+.2%}")
        logger.info(f"    Net CAGR:       {attribution['net_cagr']:+.2%}")
        logger.info(f"    Friction drag:  {attribution['friction_cagr_drag']:.2%}")

    ledger_sum = primary["ledger"].summary()
    logger.info(f"\n  Execution:")
    logger.info(f"    Total trades:   {ledger_sum.get('n_trades', 0)}")
    logger.info(f"    Rebalances:     {len(primary['rebalance_log'])}")

    outputs = [
        "data/processed/backtest_nav.parquet",
        "data/processed/trade_ledger.parquet",
        "reports/backtest_comparison.csv",
        "reports/backtest_attribution.csv",
    ]
    logger.info(f"\n  Outputs: {len(outputs)} files")
    for o in outputs:
        logger.info(f"    {o}")


# ── Validation & Research Hardening ────────────────────────────────────────────


def _hrp_strategy_factory(params: dict):
    """Factory that returns a strategy_fn for given parameters."""
    window = params.get("momentum_window", 120)

    def strategy(returns: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
        from optimization.covariance import calculate_shrinkage_covariance
        from optimization.hrp import allocate_hrp_weights
        from optimization.constraints import apply_weight_caps

        ret = returns[tickers].dropna()
        if len(ret) < 60:
            n = len(tickers)
            return {t: 1.0 / n for t in tickers}

        cov = calculate_shrinkage_covariance(ret, window=window)
        hrp_df = allocate_hrp_weights(ret, cov=cov)
        hrp_df = apply_weight_caps(hrp_df, max_weight=0.40, min_weight=0.01)
        return dict(zip(hrp_df["ticker"], hrp_df["target_weight"]))

    return strategy


def run_validation(inr_prices: pd.DataFrame, master: pd.DataFrame) -> None:
    """Validation pipeline: walk-forward → regimes → sensitivity → overfitting → stress → diagnostics."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION, ROBUSTNESS & RESEARCH HARDENING")
    logger.info("=" * 60)

    wide_prices = _build_wide_prices(inr_prices)
    wide_returns = _build_wide_returns(inr_prices)
    country_map = _build_country_map(master)
    initial_capital = 1_000_000.0

    # ── 1. Walk-forward validation ───────────────────────────────────────
    logger.info("\n▸ Step 1 — Walk-Forward Validation")
    wf_results = run_walkforward_validation(
        wide_prices=wide_prices,
        strategy_fn=_hrp_signal_strategy,
        train_years=2,
        test_years=1,
        step_years=1,
        initial_capital=initial_capital,
        frequency="quarterly",
        slippage_bps=10,
        country_map=country_map,
        warmup_days=120,
    )
    if not wf_results.empty:
        _save(wf_results, "walkforward_results.parquet")

    # ── 2. Market regime analysis ────────────────────────────────────────
    logger.info("\n▸ Step 2 — Market Regime Analysis")
    benchmark_ticker = "SPY" if "SPY" in wide_prices.columns else wide_prices.columns[0]
    regimes = identify_market_regimes(wide_prices, benchmark_ticker=benchmark_ticker)
    _save(regimes, "regime_analysis.parquet")

    # Run backtest to get NAV for regime evaluation
    primary = run_backtest(
        wide_prices=wide_prices,
        strategy_fn=_hrp_signal_strategy,
        initial_capital=initial_capital,
        frequency="quarterly",
        slippage_bps=10,
        country_map=country_map,
        warmup_days=120,
    )
    nav_series = primary["nav_series"]
    regime_perf = evaluate_regime_performance(nav_series, regimes)
    if not regime_perf.empty:
        _save(regime_perf, "regime_performance.parquet")

    # ── 3. Parameter sensitivity ─────────────────────────────────────────
    logger.info("\n▸ Step 3 — Parameter Sensitivity Analysis")
    param_grid = {"momentum_window": [60, 90, 120, 180, 252]}
    sensitivity = run_parameter_sensitivity(
        wide_prices=wide_prices,
        strategy_fn_factory=_hrp_strategy_factory,
        param_grid=param_grid,
        initial_capital=initial_capital,
        frequency="quarterly",
        slippage_bps=10,
        country_map=country_map,
        warmup_days=120,
    )
    if not sensitivity.empty:
        _save(sensitivity, "parameter_sensitivity.parquet")
        stability_surface = evaluate_stability_surface(sensitivity)
        logger.info(f"  Stability score: {stability_surface.get('stability_score', 'N/A')}")

    # ── 4. Overfitting detection ─────────────────────────────────────────
    logger.info("\n▸ Step 4 — Overfitting Detection")
    overfitting_report = detect_overfitting(
        walkforward_results=wf_results,
        regime_results=regime_perf if not regime_perf.empty else None,
        sensitivity_results=sensitivity if not sensitivity.empty else None,
    )
    logger.info(f"  Assessment: {overfitting_report.get('assessment', 'N/A')}")

    # ── 5. Signal decay ──────────────────────────────────────────────────
    logger.info("\n▸ Step 5 — Signal Decay Analysis")
    signal_decay_df = pd.DataFrame()
    try:
        from features.feature_store import load_feature_store
        store = load_feature_store()
        signal_scores = calculate_composite_score(store)
        forward_returns = evaluate_forward_returns(signal_scores, wide_prices)
        signal_decay_df = calculate_signal_decay(forward_returns)
        if not signal_decay_df.empty:
            _save(signal_decay_df, "signal_decay.parquet")
    except Exception as e:
        logger.warning(f"  Signal decay skipped: {e}")

    # ── 6. Monte Carlo simulation ────────────────────────────────────────
    logger.info("\n▸ Step 6 — Monte Carlo Simulation")
    portfolio_returns = wide_returns.mean(axis=1)
    mc_result = run_monte_carlo_simulation(
        returns=portfolio_returns,
        initial_value=initial_capital,
        n_paths=1000,
        n_days=252,
        block_size=20,
        seed=42,
    )
    mc_summary = mc_result.get("summary", {})
    mc_summary_df = pd.DataFrame([mc_summary]) if mc_summary else pd.DataFrame()
    if not mc_summary_df.empty:
        _save(mc_summary_df, "monte_carlo_summary.parquet")

    # ── 7. Stress scenarios ──────────────────────────────────────────────
    logger.info("\n▸ Step 7 — Stress Testing")
    stress_results = run_stress_scenarios(
        wide_prices=wide_prices,
        strategy_fn=_hrp_signal_strategy,
        initial_capital=initial_capital,
        frequency="quarterly",
        base_slippage_bps=10,
        country_map=country_map,
        warmup_days=120,
    )
    if not stress_results.empty:
        _save(stress_results, "stress_test_results.parquet")

    # ── 8. Liquidity stress ──────────────────────────────────────────────
    logger.info("\n▸ Step 8 — Liquidity Stress")
    liquidity_results = simulate_liquidity_stress(
        wide_prices=wide_prices,
        strategy_fn=_hrp_signal_strategy,
        initial_capital=initial_capital,
        frequency="quarterly",
        country_map=country_map,
        warmup_days=120,
    )
    if not liquidity_results.empty:
        _save(liquidity_results, "liquidity_stress.parquet")

    # ── 9. Research diagnostics ──────────────────────────────────────────
    logger.info("\n▸ Step 9 — Research Diagnostics")
    diagnostics = generate_diagnostics(
        walkforward_results=wf_results,
        regime_results=regime_perf,
        sensitivity_results=sensitivity,
        stress_results=stress_results,
        signal_decay=signal_decay_df if not signal_decay_df.empty else None,
        monte_carlo_summary=mc_summary,
        overfitting_report=overfitting_report,
    )

    # ── 10. Research quality score ───────────────────────────────────────
    logger.info("\n▸ Step 10 — Research Quality Score")
    research_score = calculate_research_score(
        walkforward_results=wf_results,
        regime_results=regime_perf,
        sensitivity_results=sensitivity,
        stress_results=stress_results,
        signal_decay=signal_decay_df if not signal_decay_df.empty else None,
    )

    # ── 11. Generate validation reports ──────────────────────────────────
    logger.info("\n▸ Step 11 — Generating Validation Reports")
    generate_validation_report(
        walkforward_results=wf_results,
        regime_results=regime_perf,
        sensitivity_results=sensitivity,
        stress_results=stress_results,
        signal_decay=signal_decay_df if not signal_decay_df.empty else None,
        monte_carlo_summary=mc_summary,
        overfitting_report=overfitting_report,
        diagnostics=diagnostics,
        research_score=research_score,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Walk-forward windows:  {len(wf_results)}")
    logger.info(f"  Regime breakdown:      {len(regime_perf)} regimes")
    logger.info(f"  Param combos tested:   {len(sensitivity)}")
    logger.info(f"  Overfitting status:    {overfitting_report.get('assessment', 'N/A')}")
    logger.info(f"  Research Score:        {research_score.get('total_score', 'N/A')}/100 ({research_score.get('grade', 'N/A')})")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Portfolio OS — Full Pipeline")
    logger.info("-" * 50)

    master = load_asset_master()

    # 1. Data ingestion — all asset types
    yahoo_data = run_yahoo_pipeline(master)
    mf_data = run_mf_pipeline(master)
    fi_data = run_fixed_income_pipeline(master)
    metal_data = run_metal_pipeline(master)

    # Merge all data sources into a single dict for the pipeline
    all_data = {**yahoo_data, **mf_data, **fi_data, **metal_data}
    print_ingestion_summary(all_data)

    # 2. FX normalization & portfolio engine (all assets)
    inr_prices, nav, contributions, exposures = run_fx_and_nav(all_data, master)

    # 3. Analytics & risk engine (all assets — NAV includes everything)
    run_analytics(nav, inr_prices, contributions, exposures, master)

    # 4. Feature engineering — market assets only (fixed_income has zero volatility)
    market_tickers = master[master["asset_type"].isin(MARKET_ASSET_TYPES)]["ticker"].tolist()
    market_prices = inr_prices[inr_prices["ticker"].isin(market_tickers)]
    run_feature_engineering(market_prices)

    # 5-7: Optimization, backtesting, validation — held market assets + benchmark only
    # Exclude tickers not in holdings (e.g. SPY is benchmark-only)
    from analytics.holdings_loader import load_holdings as _load_h
    held_tickers = set(_load_h()["ticker"].tolist())
    held_market_prices = inr_prices[
        inr_prices["ticker"].isin(held_tickers & set(market_tickers))
    ]
    # Include SPY as benchmark for backtesting/validation
    bench_prices = inr_prices[inr_prices["ticker"] == "SPY"]
    opt_prices = pd.concat([held_market_prices, bench_prices]).drop_duplicates()

    # 5. Portfolio optimization engine — held market assets only
    run_optimization(held_market_prices, nav, contributions, master)

    # 6. Friction-aware backtesting — includes benchmark
    run_backtesting(opt_prices, master)

    # 7. Validation, robustness & research hardening
    run_validation(opt_prices, master)

    logger.info("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
