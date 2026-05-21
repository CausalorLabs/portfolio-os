"""
Recommendations — actionable portfolio suggestions with explanations.
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_portfolio_recommendation,
    load_target_weights,
    load_rebalance_trades,
    load_signal_scores,
    load_backtest_attribution,
    load_portfolio_nav,
)
from dashboard.utils.formatters import fmt_currency, fmt_pct


def render() -> None:
    st.header("Portfolio Recommendations", help="Actionable rebalance suggestions with trade-level detail, signal-driven rationale, and estimated transaction costs.")

    rec = load_portfolio_recommendation()
    target = load_target_weights()
    nav = load_portfolio_nav()

    if rec.empty:
        st.warning("No recommendations. Run the pipeline first.")
        return

    # ── Rebalance summary ────────────────────────────────────────────────
    st.subheader("Rebalance Summary", help="Key metrics for the proposed rebalance: expected turnover, number of buy/sell orders, and current portfolio value.")

    total_turnover = rec["weight_change"].abs().sum() / 2
    buys = rec[rec["action"] == "BUY"]
    sells = rec[rec["action"] == "SELL"]
    portfolio_value = nav.iloc[-1]["portfolio_nav"] if not nav.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Est. Turnover", fmt_pct(total_turnover), help="One-way portfolio turnover — what fraction of the portfolio needs to change. Lower = fewer trades needed.")
    c2.metric("Buy Orders", len(buys), help="Number of positions where the optimizer recommends increasing weight.")
    c3.metric("Sell Orders", len(sells), help="Number of positions where the optimizer recommends reducing weight.")
    c4.metric("Portfolio Value", fmt_currency(portfolio_value), help="Total current portfolio value in INR, used to estimate trade amounts.")

    # ── Trade table ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Suggested Trades", help="Detailed trade list with current vs target weights, direction (BUY/SELL), and estimated rupee value of each trade.")

    display = rec.copy()
    display["current_weight"] = display["current_weight"].apply(lambda x: f"{x:.1%}")
    display["target_weight"] = display["target_weight"].apply(lambda x: f"{x:.1%}")
    display["weight_change"] = display["weight_change"].apply(lambda x: f"{x:+.1%}")

    if portfolio_value > 0:
        display["est_trade_value"] = rec["weight_change"].apply(
            lambda x: fmt_currency(abs(x * portfolio_value))
        )
    else:
        display["est_trade_value"] = "—"

    display.columns = ["Ticker", "Current", "Target", "Change", "Action", "Est. Value"]
    st.dataframe(display, width="stretch", hide_index=True)

    # ── Weight shift visualization ───────────────────────────────────────
    st.divider()
    st.subheader("Weight Shift", help="Visual bar chart of weight changes. Green = increase (buy), red = decrease (sell). Taller bars = bigger rebalance moves.")

    fig = go.Figure()
    colors = ["#00d4aa" if x > 0 else "#ff6b6b" for x in rec["weight_change"]]
    fig.add_trace(go.Bar(
        x=rec["ticker"],
        y=rec["weight_change"] * 100,
        marker_color=colors,
        text=[f"{x:+.1f}%" for x in rec["weight_change"] * 100],
        textposition="outside",
    ))
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Weight Change %", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # ── Recommendation explanations ──────────────────────────────────────
    st.divider()
    st.subheader("Allocation Rationale", help="Signal-driven explanation for each trade: why the optimizer recommends buying or selling based on momentum, trend, and diversification.")

    scores = load_signal_scores()
    attribution = load_backtest_attribution()

    # Get latest signal scores for explanations
    latest_signals = {}
    if not scores.empty:
        latest_date = scores["date"].max()
        latest = scores[scores["date"] == latest_date]
        for _, row in latest.iterrows():
            latest_signals[row["ticker"]] = row

    for _, row in rec.iterrows():
        ticker = row["ticker"]
        action = row["action"]
        change = row["weight_change"]

        if action == "HOLD":
            continue

        signal = latest_signals.get(ticker, {})
        rank = signal.get("composite_rank", 0.5) if isinstance(signal, dict) else getattr(signal, "composite_rank", 0.5)

        reasons = []
        if action == "BUY":
            if rank > 0.6:
                reasons.append("strong composite signal (momentum + trend)")
            reasons.append("HRP cluster diversification benefit")
            if change > 0.05:
                reasons.append("currently underweight vs risk-optimal allocation")
        elif action == "SELL":
            if rank < 0.4:
                reasons.append("weak composite signal")
            reasons.append("overweight vs risk-optimal allocation")
            if change < -0.1:
                reasons.append("concentration reduction needed")

        icon = "🟢" if action == "BUY" else "🔴"
        st.markdown(f"**{icon} {action} {ticker}** (change: {change:+.1%})")
        for r in reasons:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• {r}")
        st.markdown("")

    # ── Friction estimate ────────────────────────────────────────────────
    st.divider()
    st.subheader("Estimated Rebalance Friction", help="Projected transaction costs for this rebalance: slippage (market impact), brokerage fees, and capital gains tax.")

    if not attribution.empty:
        attr = attribution.iloc[0]
        n_rebal = attr.get("n_days", 0) / 252 * 4  # rough quarterly estimate
        per_rebal_friction = attr.get("total_friction", 0) / max(n_rebal, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Est. Slippage", fmt_currency(per_rebal_friction * 0.06), help="Estimated market impact cost — the price movement caused by placing the order.")
        c2.metric("Est. Costs", fmt_currency(per_rebal_friction * 0.15), help="Estimated brokerage commissions and exchange fees.")
        c3.metric("Est. Taxes", fmt_currency(per_rebal_friction * 0.79), help="Estimated capital gains tax (short-term or long-term depending on holding period).")

        st.info(
            "💡 **Tax efficiency tip**: Taxes account for ~79% of total friction. "
            "Reducing rebalance frequency from quarterly to semi-annual could "
            "significantly improve after-tax returns."
        )
