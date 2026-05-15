"""
Baseline portfolio strategies — naïve allocators that serve as
sanity-check benchmarks for the sophisticated optimizers.
"""

import numpy as np
import pandas as pd
from loguru import logger


def equal_weight_portfolio(tickers: list[str]) -> pd.DataFrame:
    """
    Equal weight across all assets.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, target_weight, strategy
    """
    n = len(tickers)
    w = 1.0 / n
    df = pd.DataFrame({
        "ticker": tickers,
        "target_weight": [w] * n,
        "strategy": "equal_weight",
    })
    logger.info(f"Equal weight: {n} assets × {w:.2%} each")
    return df


def inverse_volatility_portfolio(
    returns: pd.DataFrame,
    tickers: list[str] | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Weight inversely proportional to realized volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns (columns = tickers).
    tickers : list[str], optional
        Subset of tickers to allocate. Uses all columns if None.
    window : int
        Lookback for vol estimation.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, target_weight, strategy, volatility
    """
    if tickers is None:
        tickers = list(returns.columns)

    ret = returns[tickers].dropna()
    if len(ret) < window:
        logger.warning(f"Inverse vol: only {len(ret)} rows, need {window}; using all")
        vol = ret.std()
    else:
        vol = ret.iloc[-window:].std()

    # Guard against zero vol
    vol = vol.replace(0, np.nan).dropna()
    if vol.empty:
        logger.warning("All vols are zero — falling back to equal weight")
        return equal_weight_portfolio(tickers)

    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()

    df = pd.DataFrame({
        "ticker": weights.index,
        "target_weight": weights.values,
        "strategy": "inverse_volatility",
        "volatility": vol.values,
    })

    logger.info(f"Inverse volatility ({window}D): {len(df)} assets")
    for _, row in df.iterrows():
        logger.info(f"  {row['ticker']:15s}  vol={row['volatility']:.4f}  w={row['target_weight']:.2%}")

    return df


def risk_parity_portfolio(
    returns: pd.DataFrame,
    tickers: list[str] | None = None,
    window: int = 60,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Risk parity — each asset contributes equally to total portfolio risk.

    Uses iterative bisection (no optimizer dependency).

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns (columns = tickers).
    tickers : list[str], optional
        Subset of tickers.
    window : int
        Lookback for covariance estimation.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, target_weight, strategy, risk_contribution
    """
    if tickers is None:
        tickers = list(returns.columns)

    ret = returns[tickers].dropna()
    if len(ret) < window:
        cov = ret.cov().values
    else:
        cov = ret.iloc[-window:].cov().values

    n = len(tickers)
    w = np.ones(n) / n

    for _ in range(max_iter):
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-12:
            break
        # Marginal risk contribution
        mrc = cov @ w / sigma
        # Risk contribution
        rc = w * mrc
        target_rc = sigma / n
        # Adjust weights
        w_new = w * (target_rc / (rc + 1e-16))
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    # Final risk contributions
    sigma = np.sqrt(w @ cov @ w)
    mrc = cov @ w / (sigma + 1e-16)
    rc = w * mrc

    df = pd.DataFrame({
        "ticker": tickers,
        "target_weight": w,
        "strategy": "risk_parity",
        "risk_contribution": rc,
    })

    logger.info(f"Risk parity ({window}D): {n} assets")
    for _, row in df.iterrows():
        logger.info(
            f"  {row['ticker']:15s}  w={row['target_weight']:.2%}  "
            f"rc={row['risk_contribution']:.4f}"
        )

    return df
