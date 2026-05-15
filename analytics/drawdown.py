"""
Drawdown engine — peak-to-trough analysis with duration and recovery tracking.
"""

import pandas as pd
from loguru import logger


def calculate_drawdown_series(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a daily drawdown series.

    Returns
    -------
    pd.DataFrame
        Columns: date, portfolio_nav, rolling_peak, drawdown
    """
    df = nav[["date", "portfolio_nav"]].copy()
    df["rolling_peak"] = df["portfolio_nav"].cummax()
    df["drawdown"] = (df["portfolio_nav"] - df["rolling_peak"]) / df["rolling_peak"]
    return df


def calculate_drawdown_periods(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Identify distinct drawdown periods with start, trough, end,
    depth, and duration.

    Returns
    -------
    pd.DataFrame
        Columns: start_date, trough_date, end_date, depth,
                 days_to_trough, days_to_recovery, total_days
    """
    dd = calculate_drawdown_series(nav)

    # Boolean mask: are we in a drawdown?
    in_dd = dd["drawdown"] < 0

    periods: list[dict] = []
    start_idx = None

    for i in range(len(dd)):
        if in_dd.iloc[i] and start_idx is None:
            start_idx = i
        elif not in_dd.iloc[i] and start_idx is not None:
            # Drawdown just ended — record the period
            window = dd.iloc[start_idx:i + 1]
            trough_idx = window["drawdown"].idxmin()

            periods.append({
                "start_date": dd.loc[start_idx, "date"],
                "trough_date": dd.loc[trough_idx, "date"],
                "end_date": dd.loc[i, "date"],
                "depth": dd.loc[trough_idx, "drawdown"],
                "days_to_trough": (dd.loc[trough_idx, "date"] - dd.loc[start_idx, "date"]).days,
                "days_to_recovery": (dd.loc[i, "date"] - dd.loc[trough_idx, "date"]).days,
                "total_days": (dd.loc[i, "date"] - dd.loc[start_idx, "date"]).days,
            })

            start_idx = None

    # Handle ongoing drawdown (not yet recovered)
    if start_idx is not None:
        window = dd.iloc[start_idx:]
        trough_idx = window["drawdown"].idxmin()
        periods.append({
            "start_date": dd.loc[start_idx, "date"],
            "trough_date": dd.loc[trough_idx, "date"],
            "end_date": pd.NaT,
            "depth": dd.loc[trough_idx, "drawdown"],
            "days_to_trough": (dd.loc[trough_idx, "date"] - dd.loc[start_idx, "date"]).days,
            "days_to_recovery": None,
            "total_days": None,
        })

    result = pd.DataFrame(periods)

    if not result.empty:
        result = result.sort_values("depth").reset_index(drop=True)
        _log_top_drawdowns(result)

    return result


def max_drawdown_duration(nav: pd.DataFrame) -> int:
    """Return the longest drawdown duration in calendar days."""
    periods = calculate_drawdown_periods(nav)
    completed = periods.dropna(subset=["total_days"])
    if completed.empty:
        return 0
    return int(completed["total_days"].max())


def _log_top_drawdowns(periods: pd.DataFrame, top_n: int = 5) -> None:
    """Log the deepest drawdown periods."""
    n = min(top_n, len(periods))
    logger.info(f"Top {n} drawdown periods:")
    for i in range(n):
        row = periods.iloc[i]
        recovery = f"{row['days_to_recovery']}d" if pd.notna(row.get("days_to_recovery")) else "ongoing"
        logger.info(
            f"  {i+1}. {row['depth']:+.2%}  "
            f"start={row['start_date'].date()}  "
            f"trough={row['trough_date'].date()}  "
            f"recovery={recovery}"
        )
