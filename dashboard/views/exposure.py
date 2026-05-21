"""
Exposure Explorer — country, currency, and asset-class concentration.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_holdings,
    load_asset_master,
    load_inr_prices,
    load_fx_attribution,
)
from dashboard.utils.formatters import fmt_pct_plain, fmt_currency


def render() -> None:
    st.header("Exposure Explorer", help="Analyze portfolio concentration by country, currency, and asset class. Understand geographic and currency risk.")

    holdings = load_holdings()
    master = load_asset_master()
    inr = load_inr_prices()

    if holdings.empty:
        st.warning("No holdings data.")
        return

    # Merge holdings with master for country/currency/type
    # Holdings already has currency and asset_type; only pull new cols from master
    master_cols = ["ticker", "asset_name", "country"]
    if "currency" not in holdings.columns:
        master_cols.append("currency")
    if "asset_type" not in holdings.columns:
        master_cols.append("asset_type")
    merged = holdings.merge(
        master[master_cols],
        on="ticker", how="left",
    )

    # Latest prices for position values
    held_tickers = holdings["ticker"].tolist()
    latest = inr[inr["ticker"].isin(held_tickers)].groupby("ticker")["inr_price"].last()
    merged["position_value"] = merged.apply(
        lambda r: r["quantity"] * latest.get(r["ticker"], 0), axis=1
    )
    total_val = merged["position_value"].sum()
    merged["weight_pct"] = merged["position_value"] / total_val * 100

    # ── Exposure breakdowns ──────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Country", help="Portfolio weight allocation by country of domicile. Shows geographic concentration risk.")
        country = merged.groupby("country")["weight_pct"].sum().reset_index()
        fig = px.pie(
            country, values="weight_pct", names="country",
            color_discrete_sequence=["#6c63ff", "#00d4aa", "#ff9f43"],
        )
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Currency", help="Portfolio exposure by trading currency. USD positions carry INR/USD exchange rate risk.")
        currency = merged.groupby("currency")["weight_pct"].sum().reset_index()
        fig = px.pie(
            currency, values="weight_pct", names="currency",
            color_discrete_sequence=["#ee5a24", "#00d4aa"],
        )
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig, width="stretch")

    with c3:
        st.subheader("Asset Class", help="Breakdown by asset type: equities, ETFs, mutual funds, fixed income, metals. Diversification across classes reduces risk.")
        asset_class = merged.groupby("asset_type")["weight_pct"].sum().reset_index()
        fig = px.pie(
            asset_class, values="weight_pct", names="asset_type",
            color_discrete_sequence=["#ff6b6b", "#6c63ff", "#00d4aa", "#ff9f43", "#feca57", "#54a0ff"],
        )
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(fig, width="stretch")

    # ── Position detail table ────────────────────────────────────────────
    st.divider()
    st.subheader("Position Details", help="Complete list of holdings with current market value in INR and percentage weight in the portfolio.")

    display = merged[["ticker", "asset_name", "quantity", "avg_buy_price",
                       "currency", "country", "asset_type", "position_value", "weight_pct"]].copy()
    display["position_value"] = display["position_value"].apply(lambda x: f"₹{x:,.0f}")
    display["weight_pct"] = display["weight_pct"].apply(lambda x: f"{x:.1f}%")
    display.columns = ["Ticker", "Name", "Qty", "Avg Price", "Currency",
                       "Country", "Type", "Value (INR)", "Weight"]
    st.dataframe(display, width="stretch", hide_index=True)

    # ── FX attribution ───────────────────────────────────────────────────
    st.divider()
    st.subheader("FX Impact", help="For USD-denominated assets: how much of the return came from the asset itself vs. INR/USD exchange rate movement.")
    fx = load_fx_attribution()
    if not fx.empty:
        usd_tickers = master[master["currency"] == "USD"]["ticker"].tolist()
        usd_fx = fx[fx["ticker"].isin(usd_tickers)]
        if not usd_fx.empty:
            # Latest FX attribution per ticker
            latest_fx = usd_fx.groupby("ticker").last().reset_index()
            for col in ["local_return", "fx_return", "total_return"]:
                if col in latest_fx.columns:
                    latest_fx[col] = latest_fx[col].apply(lambda x: f"{x * 100:+.2f}%")

            show_cols = [c for c in ["ticker", "local_return", "fx_return", "total_return"]
                         if c in latest_fx.columns]
            st.dataframe(latest_fx[show_cols], width="stretch", hide_index=True)

    # ── Concentration metrics ────────────────────────────────────────────
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Concentration", help="Portfolio concentration metrics. HHI and Effective N measure how spread out the portfolio is across holdings.")
        weights = merged["weight_pct"].values / 100
        hhi = (weights ** 2).sum()
        effective_n = 1 / hhi if hhi > 0 else 0
        top1 = weights.max() * 100
        st.metric("HHI", f"{hhi:.4f}", help="Herfindahl-Hirschman Index: sum of squared weights. Range 0–1, lower = more diversified.")
        st.metric("Effective N", f"{effective_n:.1f}", help="Equivalent number of equal-weight positions. Closer to actual holding count = better diversified.")
        st.metric("Top-1 Weight", f"{top1:.1f}%", help="Weight of the largest single position. High values indicate concentration risk.")

    with right:
        st.subheader("Weight Distribution", help="Visual breakdown of portfolio weight for each held asset. Taller bars = larger positions.")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=merged["ticker"],
            y=merged["weight_pct"].values,
            marker_color="#00d4aa",
        ))
        fig_bar.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Weight %", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, width="stretch")
