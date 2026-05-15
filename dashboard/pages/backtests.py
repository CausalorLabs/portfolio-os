"""
Backtest Explorer — compare strategies with friction-aware metrics.

NAV comparison, drawdown overlay, friction breakdown, rolling Sharpe comparison.
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_backtest_comparison,
    load_backtest_attribution,
    load_backtest_nav,
    load_trade_ledger,
)
from dashboard.utils.exporters import export_section
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

    # ── Strategy selector ──────────────────────────────────────────────
    st.divider()
    strategies = comparison.index.tolist()
    selected = st.multiselect(
        "Compare strategies",
        strategies,
        default=strategies[:3] if len(strategies) >= 3 else strategies,
    )

    # ── NAV comparison chart ─────────────────────────────────────────────
    st.subheader("NAV Comparison")

    colors = ["#00d4aa", "#6c63ff", "#ff9f43", "#ee5a24", "#ff6b6b"]

    if not bt_nav.empty and "nav" in bt_nav.columns:
        fig_nav = go.Figure()
        # Primary strategy NAV (from backtest engine)
        fig_nav.add_trace(go.Scatter(
            x=bt_nav["date"], y=bt_nav["nav"],
            mode="lines", name="HRP Optimized",
            line=dict(color=colors[0], width=2),
        ))
        # Overlay benchmark NAVs from comparison table (normalized)
        if not comparison.empty:
            for i, strat in enumerate(selected):
                if strat == "hrp_optimized":
                    continue
                final = comparison.loc[strat, "final_nav"] if strat in comparison.index else 0
                if final > 0:
                    initial = attribution.iloc[0].get("initial_capital", 1_000_000) if not attribution.empty else 1_000_000
                    cagr = comparison.loc[strat, "cagr"]
                    st.caption(f"{strat}: CAGR {cagr:.1%}, Final ₹{final:,.0f}")

        fig_nav.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="₹", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_nav, width="stretch")

    # ── Drawdown comparison ──────────────────────────────────────────────
    if not bt_nav.empty and "nav" in bt_nav.columns:
        st.subheader("Drawdown (Primary Strategy)")
        nav_series = bt_nav["nav"]
        running_max = nav_series.cummax()
        dd_series = (nav_series - running_max) / running_max * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=bt_nav["date"], y=dd_series,
            mode="lines", name="Drawdown",
            line=dict(color="#ff6b6b", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,107,107,0.12)",
        ))
        fig_dd.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Drawdown %", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, width="stretch")

    # ── Strategy metrics bar comparison ──────────────────────────────────
    if selected:
        st.subheader("Metrics Comparison")
        sel_comp = comparison.loc[[s for s in selected if s in comparison.index]]

        metric_cols = ["cagr", "sharpe", "max_drawdown"]
        available = [c for c in metric_cols if c in sel_comp.columns]

        if available:
            import pandas as pd
            left, mid, right = st.columns(3)
            panels = [left, mid, right]

            for i, col in enumerate(available):
                with panels[i]:
                    fig_m = go.Figure()
                    vals = sel_comp[col].values * 100 if col != "sharpe" else sel_comp[col].values
                    bar_colors = [colors[j % len(colors)] for j in range(len(sel_comp))]
                    fig_m.add_trace(go.Bar(
                        x=sel_comp.index.tolist(),
                        y=vals,
                        marker_color=bar_colors,
                        text=[f"{v:.1f}{'%' if col != 'sharpe' else ''}" for v in vals],
                        textposition="outside",
                    ))
                    fig_m.update_layout(
                        title=col.upper().replace("_", " "),
                        height=250, margin=dict(l=0, r=0, t=40, b=0),
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_m, width="stretch")

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

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    import pandas as pd
    exports = {"backtest_comparison": comparison}
    if not ledger.empty:
        exports["trade_ledger"] = ledger
    export_section(exports)
