"""
Portfolio OS — Sprint 1 + Sprint 2 entry point.

Sprint 1: Downloads market data, validates, persists to parquet.
Sprint 2: FX normalization, portfolio NAV, attribution, exposure.
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
from analytics.exposure import latest_exposure_snapshot


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
) -> None:
    """Execute the full Sprint 2 pipeline: FX → NAV → Attribution → Exposure."""
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


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Portfolio OS — Full Pipeline (Sprint 1 + Sprint 2)")
    logger.info("-" * 50)

    master = load_asset_master()

    # Sprint 1: Data ingestion
    yahoo_data = run_yahoo_pipeline(master)
    mf_data = run_mf_pipeline()
    print_sprint1_summary(yahoo_data, mf_data)

    # Sprint 2: FX normalization & portfolio engine
    run_sprint2(yahoo_data, master)

    logger.info("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
