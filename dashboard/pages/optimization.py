"""
Optimization Explorer — visualize how the optimizer allocates capital.

Current vs target weights, constraint diagnostics, signal explanations.
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_target_weights,
    load_strategy_comparison,
    load_signal_scores,
    load_holdings,
    load_inr_prices,
    load_asset_master,
)
from dashboard.utils.formatters import fmt_pct_plain


def render() -> None:
    st.header("Optimization Explorer")

    target = load_target_weights()
    strategy = load_strategy_comparison()
    master = load_asset_master()

    if target.empty:
        st.warning("No optimization data. Run the pipeline first.")
        return

    # ── Current vs Target ────────────────────────────────────────────────
    st.subheader("Current vs Target Allocation")

    # Current weights from holdings + latest prices
    holdings = load_holdings()
    inr = load_inr_prices()
    held = inr[inr["ticker"].isin(holdings["ticker"].tolist())]
    latest = held.groupby("ticker")["inr_price"].last()

    current_vals = {}
    for _, h in holdings.iterrows():
        t = h["ticker"]
        p = latest.get(t, 0)
        current_vals[t] = h["quantity"] * p
    total_val = sum(current_vals.values())
    current_w = {t: v / total_val for t, v in current_vals.items()} if total_val > 0 else {}

    tickers = target["ticker"].tolist()
    target_w = dict(zip(target["ticker"], target["target_weight"]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current",
        x=tickers,
        y=[current_w.get(t, 0) * 100 for t in tickers],
        marker_color="#6c63ff",
    ))
    fig.add_trace(go.Bar(
        name="Target",
        x=tickers,
        y=[target_w.get(t, 0) * 100 for t in tickers],
        marker_color="#00d4aa",
    ))
    fig.update_layout(
        barmode="group", height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Weight %", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # ── Weight change table ──────────────────────────────────────────────
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Weight Changes")
        import pandas as pd
        changes = pd.DataFrame({
            "Ticker": tickers,
            "Current": [f"{current_w.get(t, 0):.1%}" for t in tickers],
            "Target": [f"{target_w.get(t, 0):.1%}" for t in tickers],
            "Change": [f"{(target_w.get(t, 0) - current_w.get(t, 0)):+.1%}" for t in tickers],
            "Action": [
                "BUY" if target_w.get(t, 0) > current_w.get(t, 0) + 0.01
                else ("SELL" if target_w.get(t, 0) < current_w.get(t, 0) - 0.01 else "HOLD")
                for t in tickers
            ],
        })
        st.dataframe(changes, width="stretch", hide_index=True)

    with right:
        st.subheader("Allocation Rationale")
        scores = load_signal_scores()
        if not scores.empty:
            latest_date = scores["date"].max()
            latest_scores = scores[scores["date"] == latest_date].sort_values(
                "composite_rank", ascending=False
            )
            for _, row in latest_scores.iterrows():
                ticker = row["ticker"]
                rank = row["composite_rank"]
                tw = target_w.get(ticker, 0)
                cw = current_w.get(ticker, 0)
                delta = tw - cw

                if delta > 0.01:
                    icon = "🟢"
                    action = "Overweight"
                elif delta < -0.01:
                    icon = "🔴"
                    action = "Underweight"
                else:
                    icon = "⚪"
                    action = "Hold"

                reasons = []
                if rank > 0.7:
                    reasons.append("strong momentum & trend")
                elif rank < 0.3:
                    reasons.append("weak momentum signal")
                if "factor_low_vol" in row.index and row.get("factor_low_vol", 0.5) > 0.6:
                    reasons.append("low volatility regime")

                reason_text = ", ".join(reasons) if reasons else "balanced signal"
                st.markdown(f"{icon} **{ticker}** → {action} ({reason_text})")

    # ── Strategy comparison ──────────────────────────────────────────────
    if not strategy.empty:
        st.divider()
        st.subheader("Strategy Comparison")

        tickers_strat = strategy["ticker"].tolist()
        strat_cols = [c for c in strategy.columns if c != "ticker"]

        fig_strat = go.Figure()
        colors = ["#6c63ff", "#00d4aa", "#ff9f43", "#ee5a24", "#ff6b6b"]
        for i, col in enumerate(strat_cols):
            fig_strat.add_trace(go.Bar(
                name=col.replace("_", " ").title(),
                x=tickers_strat,
                y=strategy[col] * 100,
                marker_color=colors[i % len(colors)],
            ))
        fig_strat.update_layout(
            barmode="group", height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Weight %", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_strat, width="stretch")
