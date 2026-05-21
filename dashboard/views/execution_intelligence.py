"""
Execution Intelligence — dashboard view.

Visualizations:
  - Execution state banner (idle/evaluating/executing)
  - Drift dashboard (current vs target, thresholds)
  - Cost attribution breakdown (tax, slippage, fees)
  - Turnover analytics (monthly, budget usage)
  - Paper trading performance (NAV, benchmark, friction)
  - Decision journal (recent trade/no-trade decisions)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROCESSED = Path("data/processed")


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner=False)
def _load_paper_portfolio() -> pd.DataFrame:
    path = PROCESSED / "paper_portfolio.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def _load_execution_journal() -> pd.DataFrame:
    path = PROCESSED / "execution_journal.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# ── Chart builders ───────────────────────────────────────────────────────────


def _build_nav_chart(paper: pd.DataFrame) -> go.Figure:
    """Paper portfolio NAV over time."""
    df = paper.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["nav"],
        mode="lines", name="Paper NAV",
        line=dict(color="#2ecc71", width=2),
    ))

    fig.update_layout(
        title="Paper Portfolio NAV",
        yaxis_title="NAV (₹)",
        yaxis_tickformat=",.0f",
        height=350,
    )
    return fig


def _build_friction_chart(paper: pd.DataFrame) -> go.Figure:
    """Cumulative friction costs over time."""
    df = paper.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_costs"],
        mode="lines", name="Transaction Costs",
        line=dict(color="#e74c3c"),
        stackgroup="friction",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_taxes"],
        mode="lines", name="Taxes",
        line=dict(color="#f39c12"),
        stackgroup="friction",
    ))

    fig.update_layout(
        title="Cumulative Friction Costs",
        yaxis_title="₹",
        yaxis_tickformat=",.0f",
        height=300,
    )
    return fig


def _build_turnover_chart(paper: pd.DataFrame) -> go.Figure:
    """Turnover per period."""
    df = paper.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["turnover"] > 0]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Turnover History", height=300)
        return fig

    fig = px.bar(
        df, x="date", y="turnover",
        title="Turnover per Rebalance",
    )
    fig.add_hline(y=0.20, line_dash="dash", line_color="red",
                  annotation_text="Monthly Budget (20%)")
    fig.update_layout(
        yaxis_tickformat=".0%",
        height=300,
    )
    return fig


def _build_decision_pie(journal: pd.DataFrame) -> go.Figure:
    """Trade vs no-trade decisions."""
    if journal.empty:
        fig = go.Figure()
        fig.update_layout(title="Decision Breakdown", height=300)
        return fig

    counts = journal["action"].value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values,
        title="Decision Breakdown",
        hole=0.4,
        color_discrete_map={
            "trade": "#2ecc71",
            "no_trade": "#e74c3c",
            "harvest": "#f39c12",
        },
    )
    fig.update_layout(height=300)
    return fig


# ── Main render ──────────────────────────────────────────────────────────────


def render():
    """Render the Execution Intelligence dashboard view."""
    st.header("⚡ Execution Intelligence")
    st.caption("Utility-gated execution, paper trading, and turnover control")

    paper = _load_paper_portfolio()
    journal = _load_execution_journal()

    # ── Status Banner ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("State", "Idle", "🟢")
    with col2:
        if not paper.empty:
            last = paper.iloc[-1]
            st.metric("Paper NAV", f"₹{last['nav']:,.0f}")
        else:
            st.metric("Paper NAV", "—")
    with col3:
        if not journal.empty:
            n_trade = (journal["action"] == "trade").sum()
            n_total = len(journal)
            skip_rate = 1 - n_trade / max(n_total, 1)
            st.metric("Skip Rate", f"{skip_rate:.0%}")
        else:
            st.metric("Skip Rate", "—")
    with col4:
        if not paper.empty:
            total_friction = paper.iloc[-1]["total_costs"] + paper.iloc[-1]["total_taxes"]
            st.metric("Total Friction", f"₹{total_friction:,.0f}")
        else:
            st.metric("Total Friction", "—")

    st.divider()

    # ── Two-column layout ────────────────────────────────────────────────
    if not paper.empty:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.plotly_chart(_build_nav_chart(paper), use_container_width=True)
            st.plotly_chart(_build_friction_chart(paper), use_container_width=True)

        with col_right:
            st.plotly_chart(_build_turnover_chart(paper), use_container_width=True)
            if not journal.empty:
                st.plotly_chart(_build_decision_pie(journal), use_container_width=True)

        st.divider()

    # ── Decision Journal ─────────────────────────────────────────────────
    st.subheader("Decision Journal")

    if not journal.empty:
        display_cols = [
            c for c in ["timestamp", "action", "trigger", "expected_utility",
                        "cost_estimate", "confidence", "regime", "rationale"]
            if c in journal.columns
        ]
        st.dataframe(
            journal[display_cols].tail(20).sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Run the execution engine to populate the decision journal. "
            "Every trade AND no-trade decision will appear here with full rationale."
        )

    st.divider()

    # ── Paper Portfolio Detail ───────────────────────────────────────────
    if not paper.empty:
        st.subheader("Paper Portfolio History")
        st.dataframe(
            paper.tail(20).style.format({
                "cash": "₹{:,.0f}",
                "market_value": "₹{:,.0f}",
                "nav": "₹{:,.0f}",
                "turnover": "{:.2%}",
                "realized_pnl": "₹{:,.0f}",
                "unrealized_pnl": "₹{:,.0f}",
                "total_costs": "₹{:,.0f}",
                "total_taxes": "₹{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
