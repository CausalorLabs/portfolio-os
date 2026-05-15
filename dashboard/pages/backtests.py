"""
Backtest Explorer — compare strategies with friction-aware metrics.

NAV comparison, drawdown overlay, friction breakdown.
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_backtest_comparison,
    load_backtest_attribution,
    load_backtest_nav,
    load_trade_ledger,
)
from dashboard.utils.formatters import fmt_currency, fmt_pct, fmt_number


def render() -> None:
    st.header("Backtest Explorer")

    comparison = load_backtest_comparison()
    attribution = load_backtest_attribution()
    bt_nav = load_backtest_nav()
    ledger = load_trade_ledger()

    if comparison.empty:
        st.warning("No backtest data. Run the pipeline first.")
        return

    # ── Metrics table ────────────────────────────────────────────────────
    st.subheader("Strategy Comparison")

    display = comparison.copy()
    display.index.name = "Strategy"

    format_cols = {
        "cagr": lambda x: fmt_pct(x),
        "volatility": lambda x: fmt_pct(x),
        "sharpe": lambda x: fmt_number(x),
        "sortino": lambda x: fmt_number(x),
        "max_drawdown": lambda x: fmt_pct(x),
        "total_friction": lambda x: fmt_currency(x),
        "final_nav": lambda x: fmt_currency(x),
    }

    formatted = display.copy()
    for col, fn in format_cols.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(fn)

    rename_map = {
        "cagr": "CAGR",
        "volatility": "Vol",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "max_drawdown": "Max DD",
        "total_friction": "Friction",
        "friction_drag_pct": "Friction %",
        "avg_turnover": "Avg Turn.",
        "n_rebalances": "Rebal.",
        "final_nav": "Final NAV",
    }
    show_cols = [c for c in rename_map if c in formatted.columns]
    formatted = formatted[show_cols].rename(columns=rename_map)
    st.dataframe(formatted, width="stretch")

    # ── NAV comparison chart ─────────────────────────────────────────────
    st.divider()
    st.subheader("Backtest NAV Curve (Primary Strategy)")

    if not bt_nav.empty and "nav" in bt_nav.columns:
        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=bt_nav["date"], y=bt_nav["nav"],
            mode="lines", name="HRP Optimized",
            line=dict(color="#00d4aa", width=2),
        ))
        fig_nav.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="₹", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_nav, width="stretch")

    # ── Friction breakdown ───────────────────────────────────────────────
    st.divider()
    st.subheader("Friction Breakdown")

    left, right = st.columns(2)

    with left:
        if not attribution.empty:
            attr = attribution.iloc[0]
            st.metric("Gross CAGR", fmt_pct(attr.get("gross_cagr", 0)))
            st.metric("Net CAGR", fmt_pct(attr.get("net_cagr", 0)))
            st.metric("Friction Drag", fmt_pct(attr.get("friction_cagr_drag", 0)))

    with right:
        if not attribution.empty:
            attr = attribution.iloc[0]
            import plotly.express as px
            import pandas as pd

            friction_data = pd.DataFrame({
                "Component": ["Slippage", "Transaction Costs", "Taxes"],
                "Amount": [
                    attr.get("slippage_drag", 0),
                    attr.get("cost_drag", 0),
                    attr.get("tax_drag", 0),
                ],
            })
            fig_fric = px.pie(
                friction_data, values="Amount", names="Component",
                color_discrete_sequence=["#ff9f43", "#6c63ff", "#ff6b6b"],
            )
            fig_fric.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_fric.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_fric, width="stretch")

    # ── Trade ledger ─────────────────────────────────────────────────────
    if not ledger.empty:
        st.divider()
        st.subheader("Trade Ledger")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", len(ledger))
        c2.metric("Total Slippage", fmt_currency(ledger["slippage_cost"].sum()))
        c3.metric("Total Costs", fmt_currency(ledger["transaction_cost"].sum()))
        c4.metric("Total Taxes", fmt_currency(ledger["tax"].sum()))

        with st.expander("View Trade Log"):
            display_ledger = ledger.copy()
            display_ledger["date"] = display_ledger["date"].dt.strftime("%Y-%m-%d")
            for col in ["market_price", "execution_price", "notional", "slippage_cost",
                        "transaction_cost", "tax", "realized_pnl"]:
                if col in display_ledger.columns:
                    display_ledger[col] = display_ledger[col].round(2)
            st.dataframe(display_ledger, width="stretch", hide_index=True)
