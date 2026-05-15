"""
Analytics — deep risk diagnostics and correlation analysis.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_rolling_analytics,
    load_drawdown_series,
    load_inr_prices,
    load_portfolio_metrics,
)
from dashboard.utils.formatters import fmt_pct, fmt_number


def render() -> None:
    st.header("Risk Analytics")

    rolling = load_rolling_analytics()
    dd = load_drawdown_series()

    if rolling.empty:
        st.warning("No rolling analytics data. Run the pipeline first.")
        return

    # ── Metrics bar ──────────────────────────────────────────────────────
    metrics_df = load_portfolio_metrics()
    if not metrics_df.empty:
        m = metrics_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ann. Vol", fmt_pct(m.get("annualized_volatility", 0)))
        c2.metric("Skewness", fmt_number(m.get("skewness", 0)))
        c3.metric("Kurtosis", fmt_number(m.get("kurtosis", 0)))
        c4.metric("Calmar", fmt_number(m.get("calmar_ratio", 0)))
        c5.metric("Max DD", fmt_pct(m.get("max_drawdown", 0)))

    st.divider()

    # ── Rolling volatility ───────────────────────────────────────────────
    st.subheader("Rolling Volatility")
    fig_vol = go.Figure()
    if "rolling_vol_20d" in rolling.columns:
        fig_vol.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_vol_20d"] * 100,
            name="20D Vol", line=dict(color="#ff9f43", width=1.5),
        ))
    if "rolling_vol_60d" in rolling.columns:
        fig_vol.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_vol_60d"] * 100,
            name="60D Vol", line=dict(color="#ee5a24", width=1.5),
        ))
    fig_vol.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Annualized Vol %", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vol, width="stretch")

    # ── Rolling Sharpe + Drawdown ────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Rolling Sharpe (60D)")
        if "rolling_sharpe_60d" in rolling.columns:
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=rolling["date"], y=rolling["rolling_sharpe_60d"],
                mode="lines", name="Sharpe",
                line=dict(color="#6c63ff", width=1.5),
            ))
            fig_sharpe.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_sharpe.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sharpe, width="stretch")

    with right:
        st.subheader("Drawdown History")
        if not dd.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd["date"], y=dd["drawdown"] * 100,
                mode="lines", line=dict(color="#ff6b6b", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,107,107,0.12)",
            ))
            fig_dd.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="%", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, width="stretch")

    # ── Correlation heatmap ──────────────────────────────────────────────
    st.divider()
    st.subheader("Asset Correlation Matrix")

    inr = load_inr_prices()
    equity_prices = inr[~inr["ticker"].str.contains("=X")]
    wide = equity_prices.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    ).ffill().dropna()
    returns = wide.pct_change().dropna()

    if not returns.empty:
        corr = returns.corr()
        fig_corr = px.imshow(
            corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig_corr.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, width="stretch")

    # ── Rolling beta (if available) ──────────────────────────────────────
    if "rolling_beta_60d" in rolling.columns:
        st.divider()
        st.subheader("Rolling Beta vs SPY (60D)")
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_beta_60d"],
            mode="lines", line=dict(color="#00d4aa", width=1.5),
        ))
        fig_beta.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig_beta.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_beta, width="stretch")
