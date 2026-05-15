"""
FX attribution engine — decomposes INR returns into
local (asset) return and FX (currency) return.

For a USD-denominated asset held by an INR investor:
    combined_return ≈ local_return + fx_return + (local_return × fx_return)
"""

import pandas as pd
from loguru import logger


def calculate_fx_attribution(inr_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose each asset's INR return into local and FX components.

    Parameters
    ----------
    inr_prices : pd.DataFrame
        Output of convert_prices_to_inr. Must contain:
        date, ticker, native_price, inr_price, fx_rate, currency

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, currency,
                 local_return, fx_return, combined_return
    """
    frames: list[pd.DataFrame] = []

    for ticker, group in inr_prices.groupby("ticker"):
        df = group.sort_values("date").copy()

        df["local_return"] = df["native_price"].pct_change()
        df["combined_return"] = df["inr_price"].pct_change()

        if df["currency"].iloc[0] == "USD":
            df["fx_return"] = df["fx_rate"].pct_change()
        else:
            # INR assets have zero FX contribution
            df["fx_return"] = 0.0

        frames.append(
            df[["date", "ticker", "currency", "local_return", "fx_return", "combined_return"]]
        )

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True).dropna(subset=["local_return"])

    _log_attribution_summary(result)

    return result


def attribution_summary(attribution_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cumulative attribution per asset.

    Returns
    -------
    pd.DataFrame
        Per-ticker cumulative local_return, fx_return, combined_return.
    """
    summaries = []

    for ticker, group in attribution_df.groupby("ticker"):
        local_cum = (1 + group["local_return"]).prod() - 1
        fx_cum = (1 + group["fx_return"]).prod() - 1
        combined_cum = (1 + group["combined_return"]).prod() - 1

        summaries.append({
            "ticker": ticker,
            "currency": group["currency"].iloc[0],
            "cumulative_local_return": local_cum,
            "cumulative_fx_return": fx_cum,
            "cumulative_combined_return": combined_cum,
        })

    return pd.DataFrame(summaries)


def _log_attribution_summary(df: pd.DataFrame) -> None:
    """Log a concise attribution summary."""
    summary = attribution_summary(df)
    logger.info("FX Attribution Summary:")
    for _, row in summary.iterrows():
        logger.info(
            f"  {row['ticker']:15s} [{row['currency']}]  "
            f"local={row['cumulative_local_return']:+7.2%}  "
            f"fx={row['cumulative_fx_return']:+7.2%}  "
            f"combined={row['cumulative_combined_return']:+7.2%}"
        )
