"""
Risk Intelligence — dashboard view.

Visualizations:
  - Risk regime banner (current vol state, scaling factor)
  - Rolling volatility chart (multi-horizon EWMA)
  - Correlation heatmap (current snapshot)
  - Crisis clustering timeline
  - Risk contribution breakdown (pie + bar)
  - CVaR gauge and tail risk table
  - Stress test results table
  - Diversification ratio over time
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROCESSED = Path("data/processed")

VOL_REGIME_COLORS = {
    "normal": "#2ecc71",
    "elevated": "#f39c12",
    "panic": "#e74c3c",
}


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner=False)
def _load_risk_data() -> dict:
    """Load risk engine outputs."""
    try:
        data = {}

        vol_path = PROCESSED / "volatility_state.parquet"
        if vol_path.exists():
            data["vol_state"] = pd.read_parquet(vol_path)

        corr_path = PROCESSED / "correlation_rolling.parquet"
        if corr_path.exists():
            data["corr_rolling"] = pd.read_parquet(corr_path)

        cluster_path = PROCESSED / "crisis_clustering.parquet"
        if cluster_path.exists():
            data["clustering"] = pd.read_parquet(cluster_path)

        return data
    except Exception as exc:
        st.warning(f"Risk data loading failed: {exc}")
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def _load_returns() -> pd.DataFrame:
    """Load returns for live computation."""
    path = PROCESSED / "daily_inr_prices.parquet"
    if path.exists():
        prices = pd.read_parquet(path)
        prices["date"] = pd.to_datetime(prices["date"])
        wide = prices.pivot_table(index="date", columns="ticker", values="inr_price", aggfunc="first")
        return wide.pct_change().dropna(how="all")
    return pd.DataFrame()


# ── Chart builders ───────────────────────────────────────────────────────────


def _build_rolling_vol_chart(vol_state: pd.DataFrame) -> go.Figure:
    """Multi-ticker rolling EWMA volatility."""
    df = vol_state.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = px.line(
        df, x="date", y="ewma_vol", color="ticker",
        title="EWMA Volatility by Asset",
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Annualized Volatility",
        xaxis_title="",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def _build_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Current correlation matrix heatmap."""
    corr = returns.iloc[-60:].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Current Correlation Matrix (60d)",
        height=450,
        width=550,
    )
    return fig


def _build_rolling_corr_chart(corr_rolling: pd.DataFrame) -> go.Figure:
    """Average correlation over time with threshold line."""
    df = corr_rolling.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["avg_correlation"],
        mode="lines", name="Avg Correlation",
        line=dict(color="#3498db", width=2),
    ))
    fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                  annotation_text="Crisis Threshold")
    fig.add_hline(y=0.50, line_dash="dot", line_color="orange",
                  annotation_text="Elevated")

    fig.update_layout(
        title="Rolling Average Correlation",
        yaxis_title="Avg Pairwise Correlation",
        height=350,
    )
    return fig


def _build_risk_contribution_chart(risk_pct: dict) -> go.Figure:
    """Pie chart of risk contributions."""
    df = pd.DataFrame({
        "ticker": list(risk_pct.keys()),
        "risk_pct": list(risk_pct.values()),
    })

    fig = px.pie(
        df, names="ticker", values="risk_pct",
        title="Risk Contribution by Asset",
        hole=0.4,
    )
    fig.update_layout(height=400)
    return fig


def _build_stress_table(scenarios: list[dict]) -> pd.DataFrame:
    """Format stress test results as table."""
    if not scenarios:
        return pd.DataFrame()

    df = pd.DataFrame(scenarios)
    cols_to_show = ["scenario"]

    if "portfolio_return" in df.columns:
        cols_to_show.append("portfolio_return")
    if "portfolio_impact" in df.columns:
        cols_to_show.append("portfolio_impact")
    if "worst_day" in df.columns:
        cols_to_show.append("worst_day")
    if "status" in df.columns:
        cols_to_show.append("status")

    return df[[c for c in cols_to_show if c in df.columns]]


# ── Main render ──────────────────────────────────────────────────────────────


def render():
    """Render the Risk Intelligence dashboard view."""
    st.header("🛡️ Risk Intelligence")
    st.caption("Dynamic risk monitoring, stress testing, and risk-aware allocation")

    risk_data = _load_risk_data()
    returns = _load_returns()

    # ── Risk Regime Banner ───────────────────────────────────────────────
    if not returns.empty:
        from risk_engine.volatility import compute_ewma_volatility
        from risk_engine.scaling import compute_vol_scaling_factor

        ewma = compute_ewma_volatility(returns, span=60)
        port_vol = float(ewma.iloc[-1].mean())
        scaling = compute_vol_scaling_factor(port_vol)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vol_color = "🟢" if port_vol < 0.15 else "🟡" if port_vol < 0.25 else "🔴"
            st.metric("Portfolio Vol (EWMA)", f"{port_vol:.1%}", vol_color)
        with col2:
            st.metric("Vol Scale Factor", f"{scaling:.2f}")
        with col3:
            cash = max(0, 1 - scaling)
            st.metric("Implied Cash", f"{cash:.1%}")
        with col4:
            if not returns.empty:
                avg_corr = returns.iloc[-20:].corr().values
                mask = np.triu(np.ones(avg_corr.shape, dtype=bool), k=1)
                upper = avg_corr[mask]
                st.metric("Avg Correlation (20d)", f"{np.mean(upper):.2f}")

    st.divider()

    # ── Two-column layout ────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Volatility chart
        vol_state = risk_data.get("vol_state")
        if vol_state is not None and not vol_state.empty:
            st.plotly_chart(_build_rolling_vol_chart(vol_state), use_container_width=True)
        elif not returns.empty:
            st.subheader("Rolling Volatility")
            from risk_engine.volatility import compute_ewma_volatility
            ewma_full = compute_ewma_volatility(returns, 60)
            df_plot = ewma_full.tail(252).stack().reset_index()
            df_plot.columns = ["date", "ticker", "ewma_vol"]
            st.plotly_chart(
                _build_rolling_vol_chart(df_plot),
                use_container_width=True,
            )

        # Correlation rolling
        corr_data = risk_data.get("corr_rolling")
        if corr_data is not None and not corr_data.empty:
            st.plotly_chart(_build_rolling_corr_chart(corr_data), use_container_width=True)

    with col_right:
        # Correlation heatmap
        if not returns.empty:
            st.plotly_chart(_build_correlation_heatmap(returns), use_container_width=True)

    st.divider()

    # ── Tail Risk Section ────────────────────────────────────────────────
    st.subheader("Tail Risk")

    if not returns.empty:
        from risk_engine.tail_risk import compute_cvar, compute_semivariance

        cvar_per = compute_cvar(returns)
        semi_per = compute_semivariance(returns)

        col1, col2 = st.columns(2)
        with col1:
            if isinstance(cvar_per, pd.Series):
                tail_df = pd.DataFrame({
                    "ticker": cvar_per.index,
                    "CVaR (95%)": cvar_per.values,
                    "Semivariance": semi_per.values if isinstance(semi_per, pd.Series) else [np.nan] * len(cvar_per),
                })
                st.dataframe(tail_df.style.format({
                    "CVaR (95%)": "{:.2%}",
                    "Semivariance": "{:.4f}",
                }), hide_index=True, use_container_width=True)

        with col2:
            port_returns = returns.mean(axis=1)
            port_cvar = compute_cvar(port_returns)
            st.metric("Portfolio CVaR (95%)", f"{port_cvar:.2%}" if not np.isnan(port_cvar) else "N/A")

    st.divider()

    # ── Risk Budget Section ──────────────────────────────────────────────
    st.subheader("Risk Budget")

    if not returns.empty:
        from risk_engine.covariance import compute_regime_covariance
        from risk_engine.budgeting import compute_risk_contribution_pct

        cov = compute_regime_covariance(returns)
        n = len(returns.columns)
        eq_weights = pd.Series(np.ones(n) / n, index=returns.columns)
        risk_pct = compute_risk_contribution_pct(eq_weights, cov)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                _build_risk_contribution_chart(risk_pct.to_dict()),
                use_container_width=True,
            )
        with col2:
            budget_df = pd.DataFrame({
                "ticker": risk_pct.index,
                "weight": eq_weights.values,
                "risk_contribution": risk_pct.values,
                "ratio": risk_pct.values / eq_weights.values,
            })
            st.dataframe(budget_df.style.format({
                "weight": "{:.1%}",
                "risk_contribution": "{:.1%}",
                "ratio": "{:.2f}x",
            }), hide_index=True, use_container_width=True)

    st.divider()

    # ── Stress Test Results ──────────────────────────────────────────────
    st.subheader("Stress Testing")
    st.caption("Impact on current portfolio under historical and synthetic scenarios")

    # Placeholder for stress test results (computed by pipeline)
    st.info(
        "Run the risk engine pipeline to populate stress test results. "
        "Historical and synthetic scenarios will appear here."
    )
