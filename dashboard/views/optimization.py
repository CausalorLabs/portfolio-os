"""
Optimization Explorer — visualize how the optimizer allocates capital.

Current vs target weights, constraint diagnostics, signal explanations.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.loaders import (
    load_target_weights,
    load_strategy_comparison,
    load_signal_scores,
    load_holdings,
    load_inr_prices,
    load_asset_master,
    load_backtest_attribution,
)
from dashboard.utils.formatters import fmt_pct_plain, fmt_pct
from dashboard.utils.exporters import export_section


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

    # ── Constraint diagnostics ───────────────────────────────────────────
    st.divider()
    st.subheader("Constraint Diagnostics")

    max_w = st.session_state.get("max_weight", 0.40)
    min_w = st.session_state.get("min_weight", 0.05)
    us_cap = st.session_state.get("country_cap_us", 0.60)
    in_cap = st.session_state.get("country_cap_in", 0.60)

    left, right = st.columns(2)

    with left:
        st.markdown("**Weight Bounds**")
        for t in tickers:
            tw = target_w.get(t, 0)
            cw = current_w.get(t, 0)
            status = "✅" if min_w <= tw <= max_w else "⚠️"
            st.markdown(f"{status} **{t}**: target {tw:.1%} "
                       f"(bounds: {min_w:.0%}–{max_w:.0%})")

    with right:
        st.markdown("**Country Exposure Caps**")
        # Compute country exposure from target weights
        country_exposure = {}
        for t in tickers:
            row = master[master["ticker"] == t]
            if not row.empty:
                country = row.iloc[0]["country"]
                country_exposure[country] = country_exposure.get(country, 0) + target_w.get(t, 0)

        caps = {"US": us_cap, "IN": in_cap}
        for country, exp in country_exposure.items():
            cap = caps.get(country, 1.0)
            status = "✅" if exp <= cap else "⚠️ BREACHED"
            st.markdown(f"{status} **{country}**: {exp:.1%} (cap: {cap:.0%})")

        # Turnover impact
        st.markdown("")
        st.markdown("**Turnover Impact**")
        turnover = sum(abs(target_w.get(t, 0) - current_w.get(t, 0)) for t in tickers) / 2
        st.metric("Expected Turnover", f"{turnover:.1%}")

    # ── Risk contribution (marginal) ─────────────────────────────────────
    st.divider()
    st.subheader("Risk Contribution")
    st.caption("Estimated marginal risk contribution per asset (equal correlation approximation)")

    # Simple risk contribution estimate: weight × vol
    inr_eq = inr[inr["ticker"].isin(tickers)]
    wide_ret = inr_eq.pivot_table(
        index="date", columns="ticker", values="inr_price", aggfunc="first"
    ).ffill().pct_change().dropna()

    if not wide_ret.empty:
        asset_vols = wide_ret.std() * np.sqrt(252)
        target_weights = np.array([target_w.get(t, 0) for t in asset_vols.index])
        risk_contrib = target_weights * asset_vols.values
        risk_contrib_pct = risk_contrib / risk_contrib.sum() * 100

        fig_rc = go.Figure()
        fig_rc.add_trace(go.Bar(
            x=asset_vols.index.tolist(),
            y=risk_contrib_pct,
            marker_color=["#ff6b6b" if r > 30 else "#ff9f43" if r > 20 else "#00d4aa"
                         for r in risk_contrib_pct],
            text=[f"{r:.1f}%" for r in risk_contrib_pct],
            textposition="outside",
        ))
        fig_rc.update_layout(
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Risk Contribution %", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rc, width="stretch")

        # Alert if any asset dominates risk
        max_rc = risk_contrib_pct.max()
        if max_rc > 40:
            st.warning(f"⚠️ Single asset contributes {max_rc:.0f}% of total risk — "
                      "consider rebalancing for better risk diversification.")

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    import pandas as pd
    export_section({
        "target_weights": target,
        "strategy_comparison": strategy if not strategy.empty else pd.DataFrame(),
    })
