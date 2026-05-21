"""
Portfolio Overview — single-screen portfolio snapshot.

KPI cards + NAV curve + allocation breakdown.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_portfolio_nav,
    load_portfolio_metrics,
    load_inr_prices,
    load_holdings,
    load_asset_master,
    load_drawdown_series,
)
from dashboard.utils.formatters import fmt_currency, fmt_pct, fmt_number


def render() -> None:
    st.header("Portfolio Overview", help="High-level snapshot of your portfolio — NAV, key risk-adjusted metrics, allocation breakdown, and recent drawdown history.")

    nav = load_portfolio_nav()
    metrics_df = load_portfolio_metrics()
    holdings = load_holdings()
    master = load_asset_master()

    if nav.empty:
        st.warning("No portfolio data. Run the pipeline first.")
        return

    # ── KPI cards ────────────────────────────────────────────────────────
    metrics = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}

    latest_nav = nav.iloc[-1]["portfolio_nav"]
    first_nav = nav.iloc[0]["portfolio_nav"]
    daily_ret = nav.iloc[-1].get("daily_return", 0.0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio NAV", fmt_currency(latest_nav), fmt_pct(daily_ret), help="Current total portfolio value in INR. Delta shows today's return.")
    c2.metric("CAGR", fmt_pct(metrics.get("cagr", 0)), help="Compound Annual Growth Rate — annualized return since inception.")
    c3.metric("Sharpe", fmt_number(metrics.get("sharpe_ratio", 0)), help="Risk-adjusted return: excess return per unit of volatility. >1.0 is good, >2.0 is excellent.")
    c4.metric("Sortino", fmt_number(metrics.get("sortino_ratio", 0)), help="Like Sharpe but only penalizes downside volatility. Higher is better.")
    c5.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown", 0)), help="Largest peak-to-trough decline in portfolio value. Measures worst-case loss.")
    c6.metric("Volatility", fmt_pct(metrics.get("annualized_volatility", 0)), help="Annualized standard deviation of returns. Lower means more stable.")

    st.divider()

    # ── NAV chart ────────────────────────────────────────────────────────
    col_chart, col_alloc = st.columns([3, 1])

    with col_chart:
        st.subheader("Portfolio NAV", help="Net Asset Value over time — the total value of all holdings in INR.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav["date"], y=nav["portfolio_nav"],
            mode="lines", name="NAV",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        ))
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="₹", xaxis_title="",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with col_alloc:
        st.subheader("Allocation", help="Current portfolio weight breakdown by asset. Each slice shows the percentage of total portfolio value.")
        # Build allocation from latest prices
        inr = load_inr_prices()
        held_tickers = holdings["ticker"].tolist()
        latest_prices = inr[inr["ticker"].isin(held_tickers)].groupby("ticker").last()

        alloc_data = []
        for _, h in holdings.iterrows():
            ticker = h["ticker"]
            qty = h["quantity"]
            price = latest_prices.loc[ticker, "inr_price"] if ticker in latest_prices.index else 0
            alloc_data.append({"ticker": ticker, "value": qty * price})

        if alloc_data:
            import pandas as pd
            alloc = pd.DataFrame(alloc_data)
            alloc["pct"] = alloc["value"] / alloc["value"].sum() * 100
            fig_pie = px.pie(
                alloc, values="pct", names="ticker",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(
                height=380, margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            fig_pie.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_pie, width="stretch")

    # ── Cumulative returns + Drawdown ────────────────────────────────────
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Cumulative Return", help="Total percentage gain/loss since the first day of the portfolio.")
        cum_ret = (nav["portfolio_nav"] / first_nav - 1) * 100
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=nav["date"], y=cum_ret,
            mode="lines", name="Cumulative %",
            line=dict(color="#6c63ff", width=2),
        ))
        fig_cum.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="%", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cum, width="stretch")

    with right:
        st.subheader("Drawdown", help="Peak-to-trough decline at each point in time. Shows how much the portfolio dropped from its previous high.")
        dd = load_drawdown_series()
        if not dd.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd["date"], y=dd["drawdown"] * 100,
                mode="lines", name="Drawdown",
                line=dict(color="#ff6b6b", width=2),
                fill="tozeroy", fillcolor="rgba(255,107,107,0.15)",
            ))
            fig_dd.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="%", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, width="stretch")

    # ── Holdings table ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Holdings", help="All current positions with quantities, asset names, country of domicile, and trading currency.")
    display = holdings.merge(master[["ticker", "asset_name", "country", "currency"]], on="ticker", how="left")
    st.dataframe(display, width="stretch", hide_index=True)
