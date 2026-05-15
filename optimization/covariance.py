"""
Covariance engine — multiple estimators for the portfolio risk structure.

Quality of covariance estimation matters MORE than return prediction
for portfolio construction.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.covariance import LedoitWolf


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """
    Sample covariance matrix (optionally windowed).

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns (columns = tickers).
    window : int, optional
        Use the last `window` days. If None, use all data.

    Returns
    -------
    pd.DataFrame
        Covariance matrix (tickers × tickers).
    """
    ret = returns.dropna()
    if window is not None and len(ret) > window:
        ret = ret.iloc[-window:]

    cov = ret.cov()
    logger.info(f"Sample covariance: {cov.shape[0]} assets, {len(ret)} observations")
    return cov


def calculate_ewma_covariance(
    returns: pd.DataFrame,
    span: int = 60,
) -> pd.DataFrame:
    """
    Exponentially-weighted covariance matrix.

    Gives more weight to recent observations — adapts faster
    to regime changes than sample covariance.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns.
    span : int
        EWMA half-life in days.

    Returns
    -------
    pd.DataFrame
        EWMA covariance matrix.
    """
    ret = returns.dropna()
    if ret.empty:
        logger.warning("EWMA covariance: no data")
        return pd.DataFrame()

    ewm = ret.ewm(span=span)
    cov = ewm.cov().iloc[-len(ret.columns):]  # last block = latest estimate

    # Extract the final cross-section
    tickers = ret.columns.tolist()
    n = len(tickers)
    last_idx = ret.index[-1]

    # ewm.cov() returns MultiIndex; grab the last date's block
    cov_full = ret.ewm(span=span).cov()
    cov_matrix = cov_full.loc[last_idx].values.reshape(n, n)

    result = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    logger.info(f"EWMA covariance (span={span}): {n} assets")
    return result


def calculate_shrinkage_covariance(
    returns: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Optimal shrinkage between sample covariance and a structured target.
    More stable than raw sample covariance, especially with few observations.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns.
    window : int, optional
        Use last `window` days if provided.

    Returns
    -------
    pd.DataFrame
        Shrinkage covariance matrix.
    """
    ret = returns.dropna()
    if window is not None and len(ret) > window:
        ret = ret.iloc[-window:]

    tickers = ret.columns.tolist()
    lw = LedoitWolf().fit(ret.values)

    cov = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
    logger.info(
        f"Ledoit-Wolf shrinkage: {len(tickers)} assets, "
        f"shrinkage={lw.shrinkage_:.4f}"
    )
    return cov
