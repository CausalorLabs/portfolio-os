"""
Visualization layer — Plotly charts for portfolio diagnostics.

All functions return Plotly figures. Call fig.show() or save to HTML.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


# ── Performance charts ───────────────────────────────────────────────────────


def plot_nav_curve(nav: pd.DataFrame, title: str = "Portfolio NAV") -> go.Figure:
    """Line chart of portfolio NAV over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav["date"], y=nav["portfolio_nav"],
        mode="lines", name="NAV", line=dict(width=2),
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="NAV (₹)",
        template="plotly_white", height=450,
    )
    return fig


def plot_cumulative_returns(returns_df: pd.DataFrame) -> go.Figure:
    """Line chart of cumulative returns."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns_df["date"], y=returns_df["cumulative_return"] * 100,
        mode="lines", name="Cumulative Return",
        line=dict(width=2, color="#2196F3"),
    ))
    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date", yaxis_title="Return (%)",
        template="plotly_white", height=400,
    )
    return fig


def plot_portfolio_vs_benchmark(
    portfolio_nav: pd.DataFrame,
    benchmark_nav: pd.DataFrame,
    benchmark_name: str = "SPY",
) -> go.Figure:
    """Normalized NAV comparison (both rebased to 100)."""
    fig = go.Figure()

    p_norm = portfolio_nav["portfolio_nav"] / portfolio_nav["portfolio_nav"].iloc[0] * 100
    b_norm = benchmark_nav["portfolio_nav"] / benchmark_nav["portfolio_nav"].iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=portfolio_nav["date"], y=p_norm,
        mode="lines", name="Portfolio", line=dict(width=2, color="#2196F3"),
    ))
    fig.add_trace(go.Scatter(
        x=benchmark_nav["date"], y=b_norm,
        mode="lines", name=benchmark_name, line=dict(width=2, color="#FF9800"),
    ))
    fig.update_layout(
        title=f"Portfolio vs {benchmark_name} (rebased to 100)",
        xaxis_title="Date", yaxis_title="Value",
        template="plotly_white", height=450,
    )
    return fig


# ── Risk charts ──────────────────────────────────────────────────────────────


def plot_drawdown(drawdown_df: pd.DataFrame) -> go.Figure:
    """Drawdown curve (area chart, always ≤ 0)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_df["date"], y=drawdown_df["drawdown"] * 100,
        fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(width=1, color="#E53935"),
        fillcolor="rgba(229,57,53,0.25)",
    ))
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_white", height=350,
    )
    return fig


def plot_rolling_volatility(rolling_df: pd.DataFrame) -> go.Figure:
    """Rolling volatility chart."""
    fig = go.Figure()
    if "rolling_20d_vol" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["date"], y=rolling_df["rolling_20d_vol"] * 100,
            mode="lines", name="20D Vol",
            line=dict(width=1.5, color="#FF9800"),
        ))
    if "rolling_60d_vol" in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["date"], y=rolling_df["rolling_60d_vol"] * 100,
            mode="lines", name="60D Vol",
            line=dict(width=1.5, color="#2196F3"),
        ))
    fig.update_layout(
        title="Rolling Volatility",
        xaxis_title="Date", yaxis_title="Annualized Volatility (%)",
        template="plotly_white", height=350,
    )
    return fig


def plot_rolling_sharpe(rolling_df: pd.DataFrame) -> go.Figure:
    """Rolling Sharpe ratio chart."""
    fig = go.Figure()
    col = "rolling_60d_sharpe"
    if col in rolling_df.columns:
        fig.add_trace(go.Scatter(
            x=rolling_df["date"], y=rolling_df[col],
            mode="lines", name="60D Sharpe",
            line=dict(width=1.5, color="#4CAF50"),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Rolling Sharpe Ratio (60D)",
        xaxis_title="Date", yaxis_title="Sharpe",
        template="plotly_white", height=350,
    )
    return fig


# ── Exposure charts ──────────────────────────────────────────────────────────


def plot_exposure_pie(
    exposure_df: pd.DataFrame,
    label_col: str,
    title: str = "Exposure",
) -> go.Figure:
    """Pie chart for a single exposure dimension."""
    fig = go.Figure(data=[go.Pie(
        labels=exposure_df[label_col],
        values=exposure_df["exposure_pct"],
        textinfo="label+percent",
        hole=0.35,
    )])
    fig.update_layout(title=title, height=400)
    return fig


def plot_allocation_history(contributions: pd.DataFrame) -> go.Figure:
    """Stacked area chart of asset allocation over time."""
    pivot = contributions.pivot_table(
        index="date", columns="ticker",
        values="contribution_pct", aggfunc="first",
    ).fillna(0)

    fig = go.Figure()
    for ticker in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[ticker],
            mode="lines", name=ticker,
            stackgroup="one",
        ))

    fig.update_layout(
        title="Asset Allocation History",
        xaxis_title="Date", yaxis_title="Weight (%)",
        template="plotly_white", height=450,
    )
    return fig


# ── Dashboard composer ───────────────────────────────────────────────────────


def save_all_charts(
    nav: pd.DataFrame,
    returns_df: pd.DataFrame,
    drawdown_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    contributions: pd.DataFrame,
    exposure_snapshot: dict[str, pd.DataFrame],
    output_dir: str = "reports",
    benchmark_nav: pd.DataFrame | None = None,
    benchmark_name: str = "SPY",
) -> list[str]:
    """
    Generate all charts and save as HTML files in output_dir.
    Returns list of saved file paths.
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    charts = {
        "nav_curve.html": plot_nav_curve(nav),
        "cumulative_returns.html": plot_cumulative_returns(returns_df),
        "drawdown.html": plot_drawdown(drawdown_df),
        "rolling_volatility.html": plot_rolling_volatility(rolling_df),
        "rolling_sharpe.html": plot_rolling_sharpe(rolling_df),
        "allocation_history.html": plot_allocation_history(contributions),
    }

    if benchmark_nav is not None and not benchmark_nav.empty:
        charts["portfolio_vs_benchmark.html"] = plot_portfolio_vs_benchmark(
            nav, benchmark_nav, benchmark_name
        )

    if "country" in exposure_snapshot:
        charts["exposure_country.html"] = plot_exposure_pie(
            exposure_snapshot["country"], "country", "Country Exposure"
        )
    if "currency" in exposure_snapshot:
        charts["exposure_currency.html"] = plot_exposure_pie(
            exposure_snapshot["currency"], "currency", "Currency Exposure"
        )

    for filename, fig in charts.items():
        path = out / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        saved.append(str(path))

    logger.info(f"Saved {len(saved)} charts to {out}/")
    return saved
