"""
Turnover engine — measures how much the portfolio changes.

High turnover is hidden portfolio risk: it increases taxes,
slippage, and destabilizes performance.
"""

import pandas as pd
from loguru import logger


def calculate_turnover(
    current_weights: pd.DataFrame,
    target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-asset and total turnover between current and target portfolios.

    Parameters
    ----------
    current_weights : pd.DataFrame
        Columns: ticker, current_weight.
    target_weights : pd.DataFrame
        Columns: ticker, target_weight.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, current_weight, target_weight, weight_change,
                 abs_change, direction
        Plus a 'total_turnover' attribute on the df.
    """
    df = target_weights[["ticker", "target_weight"]].merge(
        current_weights[["ticker", "current_weight"]],
        on="ticker",
        how="outer",
    )
    df["current_weight"] = df["current_weight"].fillna(0.0)
    df["target_weight"] = df["target_weight"].fillna(0.0)

    df["weight_change"] = df["target_weight"] - df["current_weight"]
    df["abs_change"] = df["weight_change"].abs()
    df["direction"] = df["weight_change"].apply(
        lambda x: "BUY" if x > 0.001 else ("SELL" if x < -0.001 else "HOLD")
    )

    total_turnover = df["abs_change"].sum() / 2  # one-way turnover
    df.attrs["total_turnover"] = total_turnover

    logger.info(f"Turnover: {total_turnover:.2%} (one-way)")
    for _, row in df.sort_values("abs_change", ascending=False).iterrows():
        if row["abs_change"] > 0.001:
            logger.info(
                f"  {row['ticker']:15s}  "
                f"{row['current_weight']:.2%} → {row['target_weight']:.2%}  "
                f"({row['direction']} {row['abs_change']:.2%})"
            )

    return df


def calculate_weight_drift(
    initial_weights: pd.DataFrame,
    returns: pd.DataFrame,
    n_days: int | None = None,
) -> pd.DataFrame:
    """
    Simulate how weights drift over time due to differential returns.

    Parameters
    ----------
    initial_weights : pd.DataFrame
        Columns: ticker, target_weight.
    returns : pd.DataFrame
        Wide-format daily returns (columns = tickers).
    n_days : int, optional
        Simulate drift for last n_days. Default: all available.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, initial_weight, drifted_weight, drift
    """
    tickers = initial_weights["ticker"].tolist()
    w0 = dict(zip(initial_weights["ticker"], initial_weights["target_weight"]))

    ret = returns[tickers].dropna()
    if n_days is not None and len(ret) > n_days:
        ret = ret.iloc[-n_days:]

    if ret.empty:
        logger.warning("Weight drift: no return data")
        return initial_weights.copy()

    # Simulate value evolution
    cumulative = (1 + ret).cumprod().iloc[-1]

    # Drifted values (proportional to cumulative return)
    values = pd.Series({t: w0.get(t, 0) * cumulative.get(t, 1.0) for t in tickers})
    total = values.sum()
    drifted = values / total if total > 0 else values

    df = pd.DataFrame({
        "ticker": tickers,
        "initial_weight": [w0.get(t, 0) for t in tickers],
        "drifted_weight": [drifted.get(t, 0) for t in tickers],
    })
    df["drift"] = df["drifted_weight"] - df["initial_weight"]
    df["abs_drift"] = df["drift"].abs()
    max_drift = df["abs_drift"].max()

    logger.info(f"Weight drift ({len(ret)}D): max={max_drift:.2%}")
    for _, row in df.sort_values("abs_drift", ascending=False).iterrows():
        logger.info(
            f"  {row['ticker']:15s}  "
            f"{row['initial_weight']:.2%} → {row['drifted_weight']:.2%}  "
            f"(drift {row['drift']:+.2%})"
        )

    return df
