"""
Constraint engine — prevents portfolio stupidity.

Constraints are often MORE important than optimizer sophistication.
Enforces weight caps, country limits, and sector limits.
"""

import pandas as pd
from loguru import logger


def apply_weight_caps(
    weights: pd.DataFrame,
    max_weight: float = 0.40,
    min_weight: float = 0.01,
    cash_reserve: float = 0.0,
) -> pd.DataFrame:
    """
    Enforce per-asset weight caps and minimums.

    Excess weight from capped assets is redistributed proportionally
    to uncapped assets. Iterates until all constraints satisfied.

    Parameters
    ----------
    weights : pd.DataFrame
        Must have columns: ticker, target_weight.
    max_weight : float
        Maximum weight per asset (e.g. 0.40 = 40%).
    min_weight : float
        Minimum weight per asset (0 to remove).
    cash_reserve : float
        Reserve this fraction as cash (e.g. 0.05 = 5%).

    Returns
    -------
    pd.DataFrame
        Original df with target_weight adjusted + constraint_applied column.
    """
    df = weights.copy()
    investable = 1.0 - cash_reserve

    # Scale to investable portion
    total = df["target_weight"].sum()
    if total > 0:
        df["target_weight"] = df["target_weight"] / total * investable

    # Iterative capping (handles cascading redistributions)
    for _ in range(20):
        capped = df["target_weight"] > max_weight
        if not capped.any():
            break
        excess = (df.loc[capped, "target_weight"] - max_weight).sum()
        df.loc[capped, "target_weight"] = max_weight

        uncapped = ~capped & (df["target_weight"] > 0)
        if uncapped.any():
            uncapped_total = df.loc[uncapped, "target_weight"].sum()
            if uncapped_total > 0:
                df.loc[uncapped, "target_weight"] += (
                    df.loc[uncapped, "target_weight"] / uncapped_total * excess
                )

    # Enforce minimum weights (below min → set to 0 and redistribute)
    below_min = (df["target_weight"] < min_weight) & (df["target_weight"] > 0)
    if below_min.any():
        freed = df.loc[below_min, "target_weight"].sum()
        df.loc[below_min, "target_weight"] = 0.0
        above = df["target_weight"] > 0
        if above.any():
            above_total = df.loc[above, "target_weight"].sum()
            if above_total > 0:
                df.loc[above, "target_weight"] += (
                    df.loc[above, "target_weight"] / above_total * freed
                )

    df["constraint_applied"] = True

    changes = (df["target_weight"] - weights["target_weight"]).abs()
    modified = (changes > 1e-6).sum()
    logger.info(
        f"Weight caps applied: max={max_weight:.0%}, min={min_weight:.0%}, "
        f"cash={cash_reserve:.0%} — {modified} weights adjusted"
    )

    return df


def apply_country_constraints(
    weights: pd.DataFrame,
    asset_master: pd.DataFrame,
    country_caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Enforce country-level exposure caps.

    Parameters
    ----------
    weights : pd.DataFrame
        Must have columns: ticker, target_weight.
    asset_master : pd.DataFrame
        Must have columns: ticker, country.
    country_caps : dict
        Country → max weight (e.g. {"US": 0.50}).

    Returns
    -------
    pd.DataFrame
        Weights with country constraints enforced.
    """
    if country_caps is None:
        logger.info("Country constraints: none specified, skipping")
        return weights

    df = weights.merge(
        asset_master[["ticker", "country"]],
        on="ticker",
        how="left",
    )

    for country, cap in country_caps.items():
        mask = df["country"] == country
        country_total = df.loc[mask, "target_weight"].sum()

        if country_total > cap:
            scale = cap / country_total
            excess = country_total - cap
            df.loc[mask, "target_weight"] *= scale

            # Redistribute excess to other countries
            other = ~mask & (df["target_weight"] > 0)
            if other.any():
                other_total = df.loc[other, "target_weight"].sum()
                if other_total > 0:
                    df.loc[other, "target_weight"] += (
                        df.loc[other, "target_weight"] / other_total * excess
                    )
            logger.info(f"Country cap: {country} {country_total:.2%} → {cap:.2%}")

    if "country" in df.columns:
        df = df.drop(columns=["country"])

    return df


def apply_sector_constraints(
    weights: pd.DataFrame,
    asset_master: pd.DataFrame,
    sector_caps: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Enforce sector/asset-type level caps.

    Parameters
    ----------
    weights : pd.DataFrame
        Must have columns: ticker, target_weight.
    asset_master : pd.DataFrame
        Must have columns: ticker, asset_type.
    sector_caps : dict
        asset_type → max weight (e.g. {"equity": 0.80}).

    Returns
    -------
    pd.DataFrame
        Weights with sector constraints enforced.
    """
    if sector_caps is None:
        logger.info("Sector constraints: none specified, skipping")
        return weights

    df = weights.merge(
        asset_master[["ticker", "asset_type"]],
        on="ticker",
        how="left",
    )

    for sector, cap in sector_caps.items():
        mask = df["asset_type"] == sector
        sector_total = df.loc[mask, "target_weight"].sum()

        if sector_total > cap:
            scale = cap / sector_total
            excess = sector_total - cap
            df.loc[mask, "target_weight"] *= scale

            other = ~mask & (df["target_weight"] > 0)
            if other.any():
                other_total = df.loc[other, "target_weight"].sum()
                if other_total > 0:
                    df.loc[other, "target_weight"] += (
                        df.loc[other, "target_weight"] / other_total * excess
                    )
            logger.info(f"Sector cap: {sector} {sector_total:.2%} → {cap:.2%}")

    if "asset_type" in df.columns:
        df = df.drop(columns=["asset_type"])

    return df
