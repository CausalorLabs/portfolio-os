"""
Exposure analytics — country, currency, and asset-class exposure tracking.
"""

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
