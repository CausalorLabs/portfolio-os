"""
Exposure analytics — country, currency, asset-class exposure tracking
and portfolio concentration metrics.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_country_exposure(
    contributions: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily country exposure as % of portfolio.

    Returns
    -------
    pd.DataFrame
        Columns: date, country, exposure_pct
    """
    merged = contributions.merge(
        asset_master[["ticker", "country"]],
        on="ticker",
        how="left",
    )
    exposure = (
        merged.groupby(["date", "country"])["contribution_pct"]
        .sum()
        .reset_index()
        .rename(columns={"contribution_pct": "exposure_pct"})
    )
    return exposure


def calculate_currency_exposure(
    contributions: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily currency exposure as % of portfolio.

    Returns
    -------
    pd.DataFrame
        Columns: date, currency, exposure_pct
    """
    merged = contributions.merge(
        asset_master[["ticker", "currency"]],
        on="ticker",
        how="left",
    )
    exposure = (
        merged.groupby(["date", "currency"])["contribution_pct"]
        .sum()
        .reset_index()
        .rename(columns={"contribution_pct": "exposure_pct"})
    )
    return exposure


def calculate_asset_class_exposure(
    contributions: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily asset-class exposure as % of portfolio.

    Returns
    -------
    pd.DataFrame
        Columns: date, asset_type, exposure_pct
    """
    merged = contributions.merge(
        asset_master[["ticker", "asset_type"]],
        on="ticker",
        how="left",
    )
    exposure = (
        merged.groupby(["date", "asset_type"])["contribution_pct"]
        .sum()
        .reset_index()
        .rename(columns={"contribution_pct": "exposure_pct"})
    )
    return exposure


def latest_exposure_snapshot(
    contributions: pd.DataFrame,
    asset_master: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Return the most recent exposure breakdown for country, currency,
    and asset class. Logs a summary.
    """
    latest_date = contributions["date"].max()
    latest = contributions[contributions["date"] == latest_date]

    country = calculate_country_exposure(latest, asset_master)
    currency = calculate_currency_exposure(latest, asset_master)
    asset_class = calculate_asset_class_exposure(latest, asset_master)

    logger.info(f"Exposure snapshot ({latest_date.date()}):")

    logger.info("  Country:")
    for _, row in country.iterrows():
        logger.info(f"    {row['country']:10s} {row['exposure_pct']:6.2f}%")

    logger.info("  Currency:")
    for _, row in currency.iterrows():
        logger.info(f"    {row['currency']:10s} {row['exposure_pct']:6.2f}%")

    logger.info("  Asset Class:")
    for _, row in asset_class.iterrows():
        logger.info(f"    {row['asset_type']:10s} {row['exposure_pct']:6.2f}%")

    return {"country": country, "currency": currency, "asset_class": asset_class}


# ── Concentration metrics ────────────────────────────────────────────────────


def calculate_concentration_metrics(contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily concentration metrics from asset contribution weights.

    Returns
    -------
    pd.DataFrame
        Columns: date, herfindahl_index, top1_pct, top3_pct,
                 effective_n (effective number of assets)
    """
    results = []

    for date, group in contributions.groupby("date"):
        weights = group["contribution_pct"].values / 100.0  # fractions

        hhi = float(np.sum(weights ** 2))
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        sorted_w = np.sort(weights)[::-1]
        top1 = float(sorted_w[0]) * 100 if len(sorted_w) >= 1 else 0.0
        top3 = float(sorted_w[:3].sum()) * 100 if len(sorted_w) >= 1 else 0.0

        results.append({
            "date": date,
            "herfindahl_index": hhi,
            "top1_pct": top1,
            "top3_pct": top3,
            "effective_n": effective_n,
        })

    df = pd.DataFrame(results)

    if not df.empty:
        latest = df.iloc[-1]
        logger.info("Concentration metrics (latest):")
        logger.info(f"  Herfindahl Index  {latest['herfindahl_index']:.4f}")
        logger.info(f"  Top 1 asset       {latest['top1_pct']:.2f}%")
        logger.info(f"  Top 3 assets      {latest['top3_pct']:.2f}%")
        logger.info(f"  Effective N       {latest['effective_n']:.2f}")

    return df
