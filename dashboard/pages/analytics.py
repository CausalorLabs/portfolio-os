"""
Analytics — deep risk diagnostics, correlation, and concentration analysis.
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
    load_holdings,
    load_target_weights,
)
from dashboard.utils.formatters import fmt_pct, fmt_number
from dashboard.utils.exporters import export_section


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
    # Actual column names: rolling_20d_vol, rolling_60d_vol
    st.subheader("Rolling Volatility")
    fig_vol = go.Figure()
    if "rolling_20d_vol" in rolling.columns:
        fig_vol.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_20d_vol"] * 100,
            name="20D Vol", line=dict(color="#ff9f43", width=1.5),
        ))
    if "rolling_60d_vol" in rolling.columns:
        fig_vol.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_60d_vol"] * 100,
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
        if "rolling_60d_sharpe" in rolling.columns:
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Scatter(
                x=rolling["date"], y=rolling["rolling_60d_sharpe"],
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

        # ── Diversification map ──────────────────────────────────────────
        st.subheader("Diversification Map")
        st.caption("Lower correlation = better diversification benefit")
        avg_corr = corr.mean()
        fig_div = go.Figure()
        colors = ["#00d4aa" if v < 0.5 else "#ff9f43" if v < 0.75 else "#ff6b6b"
                  for v in avg_corr.values]
        fig_div.add_trace(go.Bar(
            x=avg_corr.index.tolist(),
            y=avg_corr.values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in avg_corr.values],
            textposition="outside",
        ))
        fig_div.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Avg Correlation", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_div, width="stretch")

    # ── Concentration analysis ───────────────────────────────────────────
    st.divider()
    st.subheader("Concentration Analysis")

    holdings = load_holdings()
    if not holdings.empty:
        held_tickers = holdings["ticker"].tolist()
        latest_prices = inr[inr["ticker"].isin(held_tickers)].groupby("ticker")["inr_price"].last()

        import pandas as pd
        values = []
        for _, h in holdings.iterrows():
            t = h["ticker"]
            p = latest_prices.get(t, 0)
            values.append({"ticker": t, "value": h["quantity"] * p})
        val_df = pd.DataFrame(values)
        val_df["weight"] = val_df["value"] / val_df["value"].sum()

        weights = val_df["weight"].values
        hhi = (weights ** 2).sum()
        effective_n = 1 / hhi if hhi > 0 else 0

        left, right = st.columns(2)
        with left:
            c1, c2, c3 = st.columns(3)
            c1.metric("HHI", f"{hhi:.4f}")
            c2.metric("Effective N", f"{effective_n:.1f}")
            c3.metric("Top-1 Weight", f"{weights.max():.1%}")

            # Top holdings bar
            val_df_sorted = val_df.sort_values("weight", ascending=True)
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                y=val_df_sorted["ticker"],
                x=val_df_sorted["weight"] * 100,
                orientation="h",
                marker_color="#6c63ff",
                text=[f"{w:.1f}%" for w in val_df_sorted["weight"] * 100],
                textposition="outside",
            ))
            fig_top.update_layout(
                height=250, margin=dict(l=0, r=40, t=10, b=0),
                xaxis_title="Weight %", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_top, width="stretch")

        with right:
            st.markdown("**Diversification Assessment**")
            if effective_n >= len(holdings) * 0.8:
                st.success(f"Well diversified — effective N ({effective_n:.1f}) "
                          f"close to actual N ({len(holdings)})")
            elif effective_n >= len(holdings) * 0.5:
                st.warning(f"Moderate concentration — effective N ({effective_n:.1f}) "
                          f"vs actual N ({len(holdings)})")
            else:
                st.error(f"High concentration — effective N ({effective_n:.1f}) "
                        f"much lower than actual N ({len(holdings)})")

            # Target vs current concentration comparison
            target = load_target_weights()
            if not target.empty:
                t_weights = target["target_weight"].values
                t_hhi = (t_weights ** 2).sum()
                t_eff_n = 1 / t_hhi if t_hhi > 0 else 0
                st.markdown("**Target Allocation Concentration**")
                st.metric("Target HHI", f"{t_hhi:.4f}",
                         delta=f"{t_hhi - hhi:+.4f}",
                         delta_color="inverse")
                st.metric("Target Effective N", f"{t_eff_n:.1f}",
                         delta=f"{t_eff_n - effective_n:+.1f}")

    # ── Rolling beta ─────────────────────────────────────────────────────
    if "rolling_60d_beta" in rolling.columns:
        st.divider()
        st.subheader("Rolling Beta vs SPY (60D)")
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=rolling["date"], y=rolling["rolling_60d_beta"],
            mode="lines", line=dict(color="#00d4aa", width=1.5),
        ))
        fig_beta.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig_beta.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_beta, width="stretch")

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    export_section({
        "rolling_analytics": rolling,
        "drawdown_series": dd,
    })
