"""
Hierarchical Risk Parity (HRP) optimizer.

Core POC optimizer — avoids the instability of mean-variance,
handles correlated assets gracefully, and produces naturally
diversified portfolios.

Pipeline:
    Returns → Correlation → Distance → Hierarchical Clustering →
    Recursive Bisection → Portfolio Weights
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix from returns."""
    corr = returns.dropna().corr()
    logger.info(f"Correlation matrix: {corr.shape[0]} assets")
    return corr


def calculate_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation to distance.

    d(i,j) = sqrt(0.5 * (1 - corr(i,j)))

    Produces a proper metric: d=0 for perfectly correlated, d=1 for
    perfectly anti-correlated.
    """
    dist = np.sqrt(0.5 * (1 - corr))
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def perform_hierarchical_clustering(dist: pd.DataFrame) -> np.ndarray:
    """
    Ward linkage clustering on the distance matrix.

    Returns
    -------
    np.ndarray
        Linkage matrix (scipy format).
    """
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="ward")
    logger.info(f"Hierarchical clustering: {dist.shape[0]} assets, ward linkage")
    return link


def _get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Compute variance of an equal-weight sub-portfolio."""
    sub_cov = cov[np.ix_(items, items)]
    n = len(items)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)


def _recursive_bisection(
    cov: np.ndarray,
    sorted_idx: list[int],
) -> np.ndarray:
    """
    Recursive bisection allocation — the heart of HRP.

    Splits the sorted asset list in half, allocates weight inversely
    proportional to cluster variance, then recurses.
    """
    n = len(sorted_idx)
    weights = np.ones(n)

    # Stack-based iteration instead of recursion for stability
    clusters = [sorted_idx]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            var_left = _get_cluster_var(cov, left)
            var_right = _get_cluster_var(cov, right)
            total_var = var_left + var_right

            if total_var < 1e-16:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / total_var  # lower var → higher weight

            # Scale weights for left and right clusters
            for i in left:
                weights[sorted_idx.index(i)] *= alpha
            for i in right:
                weights[sorted_idx.index(i)] *= (1.0 - alpha)

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    return weights


def allocate_hrp_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full HRP pipeline: correlation → distance → clustering → bisection.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily returns (columns = tickers).
    cov : pd.DataFrame, optional
        Pre-computed covariance matrix. If None, computed from returns.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, target_weight, strategy
    """
    ret = returns.dropna()
    tickers = list(ret.columns)
    n = len(tickers)

    if n == 0:
        logger.warning("HRP: no tickers")
        return pd.DataFrame(columns=["ticker", "target_weight", "strategy"])

    if n == 1:
        return pd.DataFrame({
            "ticker": tickers,
            "target_weight": [1.0],
            "strategy": "hrp",
        })

    # 1. Correlation → Distance
    corr = calculate_correlation_matrix(ret)
    dist = calculate_distance_matrix(corr)

    # 2. Hierarchical clustering
    link = perform_hierarchical_clustering(dist)

    # 3. Get quasi-diagonal ordering from clustering
    sorted_idx = list(leaves_list(link).astype(int))

    # 4. Covariance for allocation
    if cov is None:
        cov_matrix = ret.cov().values
    else:
        cov_matrix = cov.values

    # 5. Recursive bisection
    weights = _recursive_bisection(cov_matrix, sorted_idx)

    # Normalize (should already sum to ~1, but ensure)
    weights = weights / weights.sum()

    # Map back to ticker names
    weight_map = {tickers[sorted_idx[i]]: weights[i] for i in range(n)}

    df = pd.DataFrame({
        "ticker": tickers,
        "target_weight": [weight_map[t] for t in tickers],
        "strategy": "hrp",
    })

    logger.info(f"HRP allocation: {n} assets")
    for _, row in df.sort_values("target_weight", ascending=False).iterrows():
        logger.info(f"  {row['ticker']:15s}  w={row['target_weight']:.2%}")

    return df
