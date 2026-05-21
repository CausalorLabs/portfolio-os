"""
Portfolio OS — main pipeline entry point.

Pipeline stages (8 sprints):
  1.  Data ingestion — downloads market data, validates, persists to parquet.
  2.  FX normalization — portfolio NAV, attribution, exposure.
  3.  Analytics — risk metrics, rolling diagnostics, benchmarks.
  4.  Feature engineering — signal generation, feature store.
  5.  Regime Intelligence — multi-signal regime detection & behavior.
  6.  ML Alpha Engine — walk-forward ensemble, alpha scores, confidence.
  7.  Dynamic Risk Engine — volatility, covariance, tail risk, risk budgeting.
  8.  Optimization — HRP, constraints, signal-tilted allocation.
  9.  Execution Engine — utility gating, paper trading, audit.
  10. Backtesting — friction-aware, taxes, slippage, benchmarks.
  11. Attribution & Monitoring — Brinson, anomalies, alerts, audit trail.
  12. Validation — robustness & research hardening.
  13. Orchestration — state coordination, SLA, governance snapshot.
  14. Deployment — trust calibration, readiness check, hardening.
  15. Warehouse — DuckDB registration.
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

# Sprint 2: Regime Intelligence
from regimes import run_regime_pipeline, get_current_regime
from regimes.behavior import get_regime_behavior, apply_regime_constraints

# Sprint 3: ML Alpha Engine
from ml_models import run_alpha_pipeline

# Sprint 4: Dynamic Risk Engine
from risk_engine import run_risk_pipeline

# Sprint 5: Execution Engine
from execution import ExecutionEngine

# Sprint 6: Monitoring
from monitoring import MonitoringEngine

# Sprint 7: Orchestration
from orchestration import OrchestrationEngine

# Sprint 8: Deployment
from deployment import DeploymentEngine

# Warehouse
from warehouse import get_warehouse


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

    # Niyati fragility gate: tighten drift threshold when portfolio is fragile
    drift_threshold = 0.05
    try:
        from validation.niyati_stress import run_fragility_check

        metrics_now = _load_niyati_metrics()
        frag = run_fragility_check(
            current_sharpe=metrics_now["sharpe_ratio"],
            current_drawdown=metrics_now["max_drawdown"],
        )
        if frag is not None:
            collapse_risk = frag.get("collapse_risk", 0.0)
            thickness = frag.get("thickness", 1.0)
            if collapse_risk > 0.15 or thickness < 0.10:
                # Fragile: trigger rebalance sooner to keep portfolio centered
                drift_threshold = 0.03
                logger.info(
                    f"  Niyati fragility gate: collapse_risk={collapse_risk:.4f}, "
                    f"thickness={thickness:.4f} → tightening drift threshold to {drift_threshold}"
                )
            else:
                logger.info(
                    f"  Niyati fragility gate: collapse_risk={collapse_risk:.4f}, "
                    f"thickness={thickness:.4f} → normal threshold {drift_threshold}"
                )
    except Exception as exc:
        logger.warning(f"  Niyati fragility gate skipped (non-fatal): {exc}")

    rebalance = should_rebalance(
        latest_contrib, tilted, drift_threshold=drift_threshold, method="threshold"
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


# ── Regime Intelligence (Sprint 2) ─────────────────────────────────────────────


def run_regimes(inr_prices: pd.DataFrame, nav: pd.DataFrame) -> dict:
    """Sprint 2: Multi-signal regime detection pipeline."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("REGIME INTELLIGENCE ENGINE")
    logger.info("=" * 60)

    result = run_regime_pipeline(inr_prices=inr_prices, nav_series=nav, save=True)

    regime_name = result.get("current_regime", "unknown")
    quality = result.get("quality_score", {})
    behavior = result.get("behavior")

    logger.info(f"  Current regime:     {regime_name}")
    logger.info(f"  Quality score:      {quality.get('total_score', 'N/A')}/100")
    if behavior:
        logger.info(f"  Max equity weight:  {behavior.max_equity_weight:.0%}")
        logger.info(f"  Cov method:         {behavior.covariance_method}")
        logger.info(f"  Drift threshold:    {behavior.rebalance_drift_threshold}")

    return result


# ── ML Alpha Engine (Sprint 3) ────────────────────────────────────────────────


def run_ml_alpha(
    inr_prices: pd.DataFrame,
    regime_states: pd.DataFrame | None = None,
) -> dict:
    """Sprint 3: Walk-forward ML alpha generation pipeline."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ML ALPHA ENGINE")
    logger.info("=" * 60)

    from features.feature_store import load_feature_store

    try:
        feature_store = load_feature_store()
    except Exception:
        feature_store = None

    result = run_alpha_pipeline(
        inr_prices=inr_prices,
        feature_store=feature_store,
        regime_states=regime_states,
        save=True,
        track=True,
    )

    alpha_scores = result.get("alpha_scores")
    evaluation = result.get("evaluation", {})

    if alpha_scores is not None and not alpha_scores.empty:
        logger.info(f"  Alpha scores:       {len(alpha_scores)} rows")
        logger.info(f"  Tickers scored:     {alpha_scores['ticker'].nunique()}")
    logger.info(f"  Rank IC:            {evaluation.get('rank_ic', 'N/A')}")
    logger.info(f"  Grade:              {evaluation.get('grade', 'N/A')}")

    return result


# ── Dynamic Risk Engine (Sprint 4) ────────────────────────────────────────────


def run_risk_engine(
    inr_prices: pd.DataFrame,
    base_weights: pd.DataFrame,
    alpha_scores: pd.DataFrame | None = None,
    regime_behavior=None,
    master: pd.DataFrame | None = None,
    nav: pd.DataFrame | None = None,
) -> dict:
    """Sprint 4: Dynamic risk & covariance pipeline."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DYNAMIC RISK & COVARIANCE ENGINE")
    logger.info("=" * 60)

    asset_types = None
    if master is not None:
        asset_types = master.set_index("ticker")["asset_type"]

    nav_series = None
    if nav is not None and not nav.empty:
        nav_series = nav.set_index("date")["portfolio_nav"]

    result = run_risk_pipeline(
        inr_prices=inr_prices,
        base_weights=base_weights,
        alpha_scores=alpha_scores,
        regime_behavior=regime_behavior,
        asset_types=asset_types,
        nav=nav_series,
    )

    vol_state = result.get("volatility_state")
    risk_portfolio = result.get("risk_portfolio")

    if vol_state is not None and not vol_state.empty:
        latest_vol = vol_state.iloc[-1]
        logger.info(f"  Current vol regime: {latest_vol.get('vol_regime', 'N/A')}")
        logger.info(f"  EWMA vol:           {latest_vol.get('ewma_vol', 0):.4f}")

    if risk_portfolio is not None and not risk_portfolio.empty:
        logger.info(f"  Risk-aware weights: {len(risk_portfolio)} assets")

    # Persist risk engine artifacts for dashboard views
    if vol_state is not None and not vol_state.empty:
        _save(vol_state, "volatility_state.parquet")

    corr_rolling = result.get("correlation_rolling")
    if corr_rolling is not None and not corr_rolling.empty:
        _save(corr_rolling, "correlation_rolling.parquet")

    clustering = result.get("crisis_clustering")
    if clustering is not None and not clustering.empty:
        _save(clustering, "crisis_clustering.parquet")

    return result


# ── Execution Engine (Sprint 5) ───────────────────────────────────────────────


def run_execution_cycle(
    nav: pd.DataFrame,
    target_weights: pd.DataFrame,
    inr_prices: pd.DataFrame,
    master: pd.DataFrame,
    regime_name: str = "risk_on",
    regime_changed: bool = False,
    confidence: float = 0.5,
    risk_result: dict | None = None,
) -> dict:
    """Sprint 5: Utility-based execution cycle."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("UTILITY-BASED EXECUTION ENGINE")
    logger.info("=" * 60)

    from datetime import date as date_cls

    initial_capital = nav.iloc[0]["portfolio_nav"] if not nav.empty else 1_000_000.0
    engine = ExecutionEngine(initial_capital=initial_capital)

    # Build latest prices dict
    latest_date = inr_prices["date"].max()
    latest_prices_df = inr_prices[inr_prices["date"] == latest_date]
    prices = dict(zip(latest_prices_df["ticker"], latest_prices_df["inr_price"]))

    # Build target weights dict
    weights_dict = dict(zip(target_weights["ticker"], target_weights["target_weight"]))

    # Country map
    country_map = _build_country_map(master)

    # Extract risk metrics if available
    current_vol = 0.0
    target_vol = 0.15
    if risk_result:
        vol_state = risk_result.get("volatility_state")
        if vol_state is not None and not vol_state.empty:
            current_vol = float(vol_state.iloc[-1].get("ewma_vol", 0))

    result = engine.run_cycle(
        dt=date_cls.today(),
        target_weights=weights_dict,
        prices=prices,
        regime=regime_name,
        regime_changed=regime_changed,
        confidence=confidence,
        current_vol=current_vol,
        target_vol=target_vol,
        country_map=country_map,
    )

    decision = result.get("decision", "unknown")
    logger.info(f"  Execution decision: {decision}")

    # Save execution artifacts from the engine instance
    journal_df = engine.journal.to_dataframe()
    if not journal_df.empty:
        _save(journal_df, "execution_journal.parquet")
        logger.info(f"  Journal entries:    {len(journal_df)}")

    paper_df = engine.paper.to_dataframe()
    if not paper_df.empty:
        _save(paper_df, "paper_portfolio.parquet")
        logger.info(f"  Paper snapshots:    {len(paper_df)}")

    return result


# ── Monitoring (Sprint 6) ────────────────────────────────────────────────────


def run_monitoring(
    nav: pd.DataFrame,
    target_weights: pd.DataFrame,
    regime_name: str = "risk_on",
    regime_changed: bool = False,
    confidence: float = 0.5,
    risk_result: dict | None = None,
) -> dict:
    """Sprint 6: Attribution, alerts, anomaly detection, audit trail."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ATTRIBUTION & MONITORING LAYER")
    logger.info("=" * 60)

    engine = MonitoringEngine()

    # Build current weights from target
    weights = dict(zip(target_weights["ticker"], target_weights["target_weight"]))

    current_vol = 0.0
    if risk_result:
        vol_state = risk_result.get("volatility_state")
        if vol_state is not None and not vol_state.empty:
            current_vol = float(vol_state.iloc[-1].get("ewma_vol", 0))

    # Compute current drawdown
    current_dd = 0.0
    if not nav.empty:
        peak = nav["portfolio_nav"].cummax()
        dd = (nav["portfolio_nav"] - peak) / peak
        current_dd = float(dd.iloc[-1])

    # Compute latest return
    nav_return = None
    if len(nav) >= 2:
        nav_return = float(nav["portfolio_nav"].iloc[-1] / nav["portfolio_nav"].iloc[-2] - 1)

    result = engine.run_monitoring_cycle(
        weights=weights,
        target_weights=weights,
        current_vol=current_vol,
        current_drawdown=current_dd,
        current_regime=regime_name,
        regime_changed=regime_changed,
        ml_confidence=confidence,
        nav_return=nav_return,
    )

    alerts = result.get("alerts", [])
    anomalies = result.get("anomalies", {})
    if isinstance(anomalies, dict):
        n_anomalies = sum(len(v) if isinstance(v, list) else 0 for v in anomalies.values())
    elif isinstance(anomalies, list):
        n_anomalies = len(anomalies)
    else:
        n_anomalies = 0

    logger.info(f"  Alerts triggered:   {len(alerts)}")
    logger.info(f"  Anomalies found:    {n_anomalies}")

    # Save monitoring artifacts
    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame()
    if not alerts_df.empty:
        _save(alerts_df, "monitoring_alerts.parquet")

    # Save audit trail
    audit_df = engine.audit.to_dataframe()
    if not audit_df.empty:
        _save(audit_df, "audit_trail.parquet")

    # Save system health
    health_df = engine.observability.health_dataframe()
    if not health_df.empty:
        _save(health_df, "system_health.parquet")

    # Save model health
    model_df = engine.observability.model_health_dataframe()
    if not model_df.empty:
        _save(model_df, "model_health.parquet")

    return result


# ── Orchestration snapshot (Sprint 7) ─────────────────────────────────────────


def run_orchestration_snapshot() -> dict:
    """Sprint 7: Record orchestration state, SLA, governance snapshot."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ORCHESTRATION & GOVERNANCE")
    logger.info("=" * 60)

    orch = OrchestrationEngine()

    # Take governance snapshot
    orch.governance.snapshot_configs()
    logger.info("  Governance snapshot: saved")

    # Record SLA for the pipeline run
    state = orch.state.get()
    logger.info(f"  System state:       {state.get('status', 'N/A')}")

    # Check scheduling
    should_run = orch.scheduler.should_run("daily_pipeline")
    logger.info(f"  Daily pipeline due: {should_run}")

    # MLOps check
    retrain_needed = orch.mlops.check_retraining_needed(
        current_ic=0.0,
        mean_confidence=0.5,
        feature_drift_zscore=0.0,
        days_since_training=0,
    )
    logger.info(f"  Retraining needed:  {retrain_needed}")

    # Persist orchestration artifacts for dashboard
    _persist_orchestration(orch)

    return {
        "state": state,
        "should_run": should_run,
        "retrain_needed": retrain_needed,
    }


def _persist_orchestration(orch) -> None:
    """Persist orchestration artifacts for dashboard views."""
    # Events
    try:
        events_df = orch.events.to_dataframe()
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            _save(events_df, "orchestration_events.parquet")
    except Exception:
        pass

    # SLA compliance
    try:
        sla_report = orch.sla.get_compliance_report()
        if isinstance(sla_report, pd.DataFrame) and not sla_report.empty:
            _save(sla_report, "sla_records.parquet")
        elif isinstance(sla_report, dict) and sla_report:
            _save(pd.DataFrame([sla_report]), "sla_records.parquet")
        elif isinstance(sla_report, list) and sla_report:
            _save(pd.DataFrame(sla_report), "sla_records.parquet")
    except Exception:
        pass

    # System state
    try:
        state = orch.state.get()
        if isinstance(state, pd.DataFrame) and not state.empty:
            _save(state, "system_state.parquet")
        elif isinstance(state, dict) and state:
            _save(pd.DataFrame([state]), "system_state.parquet")
    except Exception:
        pass


# ── Deployment readiness (Sprint 8) ──────────────────────────────────────────


def run_deployment_check(
    alpha_evaluation: dict | None = None,
    confidence: float = 0.5,
) -> dict:
    """Sprint 8: Trust calibration & deployment readiness."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEPLOYMENT READINESS & TRUST CALIBRATION")
    logger.info("=" * 60)

    deploy = DeploymentEngine()

    # Run validation
    validation_result = deploy.validation.run_all_checks()
    n_passed = sum(1 for v in validation_result if v.passed)
    n_total = len(validation_result)
    logger.info(f"  Validation checks:  {n_passed}/{n_total} passed")

    # Trust calibration
    trust_result = deploy.trust.calibrate()
    trust_score = getattr(trust_result, "overall_trust", 0.0)
    approval_mode = getattr(trust_result, "recommended_mode", "advisory")
    logger.info(f"  Trust score:        {trust_score:.2f}")
    logger.info(f"  Approval mode:      {approval_mode}")

    # Readiness report
    rank_ic = 0.0
    grade = "C"
    if alpha_evaluation:
        rank_ic = alpha_evaluation.get("rank_ic", 0.0)
        grade = alpha_evaluation.get("grade", "C")

    readiness = deploy.run_readiness_check(
        trust_score=trust_score,
        rank_ic=rank_ic,
        grade=grade,
        confidence_mean=confidence,
    )

    deploy_ready = readiness.get("deployment_ready", False)
    logger.info(f"  Deployment ready:   {deploy_ready}")

    # Persist deployment artifacts for dashboard
    val_df = deploy.validation.to_dataframe()
    if not val_df.empty:
        _save(val_df, "validation_results.parquet")

    trust_history = deploy.trust.get_history()
    try:
        if isinstance(trust_history, pd.DataFrame) and not trust_history.empty:
            _save(trust_history, "trust_scores.parquet")
        elif isinstance(trust_history, list) and trust_history:
            trust_df = pd.DataFrame(trust_history)
            if not trust_df.empty:
                _save(trust_df, "trust_scores.parquet")
    except Exception:
        pass

    return {
        "validation": validation_result,
        "trust": trust_result,
        "readiness": readiness,
    }


# ── Warehouse registration ────────────────────────────────────────────────────


def run_warehouse_registration() -> list[str]:
    """Register all processed parquet files in DuckDB warehouse."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("WAREHOUSE REGISTRATION")
    logger.info("=" * 60)

    wh = get_warehouse()
    registered = wh.register_all()
    logger.info(f"  Tables registered:  {len(registered)}")
    wh.close()
    return registered


# ── Niyati helpers ────────────────────────────────────────────────────────────


def _load_niyati_metrics() -> dict:
    """Load current Sharpe and max drawdown from reports/portfolio_metrics.csv."""
    path = Path("reports/portfolio_metrics.csv")
    defaults = {"sharpe_ratio": 1.0, "max_drawdown": -0.15}
    if not path.exists():
        return defaults
    try:
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict() if not df.empty else {}
        return {
            "sharpe_ratio": float(row.get("sharpe_ratio", defaults["sharpe_ratio"])),
            "max_drawdown": float(row.get("max_drawdown", defaults["max_drawdown"])),
        }
    except Exception:
        return defaults


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Portfolio OS — Full Pipeline (8 Sprints)")
    logger.info("-" * 50)

    master = load_asset_master()

    # ── Sprint 1: Data Ingestion ──────────────────────────────────────
    yahoo_data = run_yahoo_pipeline(master)
    mf_data = run_mf_pipeline(master)
    fi_data = run_fixed_income_pipeline(master)
    metal_data = run_metal_pipeline(master)

    all_data = {**yahoo_data, **mf_data, **fi_data, **metal_data}
    print_ingestion_summary(all_data)

    # ── Sprint 1: FX & NAV ───────────────────────────────────────────
    inr_prices, nav, contributions, exposures = run_fx_and_nav(all_data, master)

    # ── Sprint 1: Analytics ──────────────────────────────────────────
    run_analytics(nav, inr_prices, contributions, exposures, master)

    # ── Sprint 1: Feature Engineering ────────────────────────────────
    market_tickers = master[master["asset_type"].isin(MARKET_ASSET_TYPES)]["ticker"].tolist()
    market_prices = inr_prices[inr_prices["ticker"].isin(market_tickers)]
    run_feature_engineering(market_prices)

    # ── Sprint 2: Regime Intelligence ────────────────────────────────
    regime_result = run_regimes(inr_prices, nav)
    regime_name = regime_result.get("current_regime", "risk_on")
    regime_behavior = regime_result.get("behavior")
    regime_states = regime_result.get("regimes")

    # ── Sprint 3: ML Alpha Engine ────────────────────────────────────
    alpha_result = run_ml_alpha(inr_prices, regime_states=regime_states)
    alpha_scores = alpha_result.get("alpha_scores")
    alpha_evaluation = alpha_result.get("evaluation", {})
    ml_confidence = 0.5
    if alpha_scores is not None and "model_confidence" in alpha_scores.columns:
        ml_confidence = float(alpha_scores["model_confidence"].mean())

    # ── Sprint 1: Optimization (held market assets) ──────────────────
    from analytics.holdings_loader import load_holdings as _load_h
    held_tickers = set(_load_h()["ticker"].tolist())
    held_market_prices = inr_prices[
        inr_prices["ticker"].isin(held_tickers & set(market_tickers))
    ]
    bench_prices = inr_prices[inr_prices["ticker"] == "SPY"]
    opt_prices = pd.concat([held_market_prices, bench_prices]).drop_duplicates()

    run_optimization(held_market_prices, nav, contributions, master)

    # Load optimized weights for downstream stages
    weights_path = PROCESSED_DIR / "target_weights.parquet"
    target_weights = pd.read_parquet(weights_path) if weights_path.exists() else pd.DataFrame()

    # ── Sprint 4: Dynamic Risk Engine ────────────────────────────────
    risk_result = {}
    if not target_weights.empty:
        risk_result = run_risk_engine(
            inr_prices=held_market_prices,
            base_weights=target_weights,
            alpha_scores=alpha_scores,
            regime_behavior=regime_behavior,
            master=master,
            nav=nav,
        )

        # If risk engine produced risk-aware weights, save them
        risk_portfolio = risk_result.get("risk_portfolio")
        if risk_portfolio is not None and not risk_portfolio.empty:
            _save(risk_portfolio, "risk_aware_weights.parquet")

    # ── Sprint 5: Execution Engine ───────────────────────────────────
    exec_result = {}
    if not target_weights.empty:
        # Detect regime change
        regime_changed = False
        regime_states_df = regime_result.get("regimes")
        if regime_states_df is not None and len(regime_states_df) >= 2:
            last_two = regime_states_df.tail(2)["regime"].tolist()
            regime_changed = last_two[0] != last_two[1]

        exec_result = run_execution_cycle(
            nav=nav,
            target_weights=target_weights,
            inr_prices=inr_prices,
            master=master,
            regime_name=regime_name,
            regime_changed=regime_changed,
            confidence=ml_confidence,
            risk_result=risk_result,
        )

    # ── Sprint 1: Backtesting ────────────────────────────────────────
    run_backtesting(opt_prices, master)

    # ── Sprint 6: Monitoring ─────────────────────────────────────────
    if not target_weights.empty:
        run_monitoring(
            nav=nav,
            target_weights=target_weights,
            regime_name=regime_name,
            regime_changed=regime_changed if not target_weights.empty else False,
            confidence=ml_confidence,
            risk_result=risk_result,
        )

    # ── Sprint 1: Validation ────────────────────────────────────────
    run_validation(opt_prices, master)

    # ── Niyati Structural Analysis ───────────────────────────────────
    try:
        import json

        from validation.niyati_runway import run_runway_analysis, summarize_runway
        from validation.niyati_stress import (
            run_adversarial_allocation,
            run_adversarial_survival,
            run_competition_stress,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("NIYATI STRUCTURAL ANALYSIS")
        logger.info("=" * 60)

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Runway analysis
        logger.info("\n▸ Step 9a — Runway Analysis (structural feasibility)")
        runway_result = run_runway_analysis()
        runway_summary = summarize_runway(runway_result)

        if runway_result is not None:
            (reports_dir / "niyati_runway.json").write_text(
                json.dumps(runway_result, indent=2, default=str)
            )
            logger.info(f"  Verdict:            {runway_summary['verdict']}")
            logger.info(f"  Point of no return: {runway_summary['point_of_no_return']}")
            logger.info(f"  Survival posture:   {runway_summary['survival_posture']}")
            logger.info(f"  Headline:           {runway_summary['headline']}")
        else:
            logger.warning("  Runway analysis returned no result — API may be down")

        # Adversarial allocation
        logger.info("\n▸ Step 9b — Adversarial Allocation (which class collapses first)")
        stress_result = run_adversarial_allocation()

        if stress_result is not None:
            (reports_dir / "niyati_stress.json").write_text(
                json.dumps(stress_result, indent=2, default=str)
            )
            first_collapse = stress_result.get(
                "first_collapse", stress_result.get("first_target", "unknown")
            )
            n_collapsed = stress_result.get(
                "collapse_count", stress_result.get("n_collapsed", "?")
            )
            logger.info(f"  First collapse:     {first_collapse}")
            logger.info(f"  Systems collapsed:  {n_collapsed}")
        else:
            logger.warning("  Adversarial allocation returned no result")

        # Competition stress
        logger.info("\n▸ Step 9c — Competition Stress (moderate market shock)")
        metrics = _load_niyati_metrics()
        competition_result = run_competition_stress(
            current_sharpe=metrics["sharpe_ratio"],
            current_drawdown=metrics["max_drawdown"],
            shock_level="moderate",
        )

        if competition_result is not None:
            (reports_dir / "niyati_competition.json").write_text(
                json.dumps(competition_result, indent=2, default=str)
            )
            kappa_u = competition_result.get(
                "kappa_unperturbed", competition_result.get("kappa_base", "N/A")
            )
            kappa_c = competition_result.get(
                "kappa_competitive_bound", competition_result.get("kappa_bound", "N/A")
            )
            logger.info(f"  kappa_unperturbed:  {kappa_u}")
            logger.info(f"  kappa_competitive:  {kappa_c}")
        else:
            logger.warning("  Competition stress returned no result")

        # Adversarial survival
        logger.info("\n▸ Step 9d — Adversarial Survival (crash budget verdict)")
        metrics_adv = _load_niyati_metrics()
        survival_result = run_adversarial_survival(
            current_sharpe=metrics_adv["sharpe_ratio"],
            current_drawdown=metrics_adv["max_drawdown"],
            crash_budget=0.006,
        )

        if survival_result is not None:
            (reports_dir / "niyati_survival.json").write_text(
                json.dumps(survival_result, indent=2, default=str)
            )
            sv = survival_result.get("survival_verdict", "unknown")
            sm = survival_result.get("survival_margin")
            pi_r = survival_result.get("pi_regime", "unknown")
            sm_str = f"{sm:+.4f}" if sm is not None else "N/A"
            logger.info(f"  Survival verdict:   {sv.upper()}")
            logger.info(f"  Survival margin:    {sm_str}")
            logger.info(f"  Pi regime:          {pi_r}")
        else:
            logger.warning("  Adversarial survival returned no result")

        logger.info("\n  Niyati outputs saved to reports/niyati_*.json")

    except Exception as exc:
        logger.warning(f"Niyati structural analysis skipped (non-fatal): {exc}")

    # ── Sprint 7: Orchestration snapshot ─────────────────────────────
    try:
        run_orchestration_snapshot()
    except Exception as exc:
        logger.warning(f"Orchestration snapshot skipped (non-fatal): {exc}")

    # ── Sprint 8: Deployment readiness ───────────────────────────────
    try:
        run_deployment_check(
            alpha_evaluation=alpha_evaluation,
            confidence=ml_confidence,
        )
    except Exception as exc:
        logger.warning(f"Deployment check skipped (non-fatal): {exc}")

    # ── Warehouse registration ───────────────────────────────────────
    try:
        run_warehouse_registration()
    except Exception as exc:
        logger.warning(f"Warehouse registration skipped (non-fatal): {exc}")

    logger.info("\n✓ Pipeline complete (8 sprints).")


if __name__ == "__main__":
    main()
