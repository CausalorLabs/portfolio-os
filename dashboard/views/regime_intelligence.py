"""
Regime Intelligence — dashboard view.

Visualizations:
  - Current regime banner with behavior parameters
  - Regime timeline (color-coded history)
  - Regime vs portfolio drawdown overlay
  - Transition heatmap
  - Regime duration distribution
  - Predictive value table (forward returns by regime)
  - Crisis alignment table
  - Quality score card
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROCESSED = Path("data/processed")

REGIME_COLORS = {
    "risk_on": "#2ecc71",
    "risk_off": "#f39c12",
    "panic": "#e74c3c",
    "high_vol": "#9b59b6",
}

REGIME_EMOJI = {
    "risk_on": "🟢",
    "risk_off": "🟡",
    "panic": "🔴",
    "high_vol": "🟣",
}


# ── Data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner=False)
def _load_regime_data() -> dict:
    """Load all regime outputs, running pipeline if needed."""
    try:
        from regimes import run_regime_pipeline
        return run_regime_pipeline(save=True)
    except Exception as exc:
        st.warning(f"Regime pipeline failed: {exc}")
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def _load_nav() -> pd.DataFrame:
    path = PROCESSED / "portfolio_nav.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


# ── Chart builders ───────────────────────────────────────────────────────────


def _build_regime_timeline(regimes: pd.DataFrame) -> go.Figure:
    """Color-coded regime timeline."""
    df = regimes.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    for regime, color in REGIME_COLORS.items():
        mask = df["regime"] == regime
        regime_df = df[mask]
        if not regime_df.empty:
            fig.add_trace(go.Scatter(
                x=regime_df["date"],
                y=[regime] * len(regime_df),
                mode="markers",
                marker=dict(color=color, size=4, symbol="square"),
                name=regime.replace("_", " ").title(),
                hovertemplate="%{x|%Y-%m-%d}<br>Confidence: %{customdata:.2f}<extra></extra>",
                customdata=regime_df["confidence"],
            ))

    fig.update_layout(
        title="Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Regime",
        height=250,
        margin=dict(t=40, b=20, l=80, r=20),
        showlegend=True,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def _build_regime_nav_overlay(regimes: pd.DataFrame, nav: pd.DataFrame) -> go.Figure | None:
    """NAV curve with regime-colored background shading."""
    if nav.empty:
        return None

    df = regimes.copy()
    df["date"] = pd.to_datetime(df["date"])
    nav = nav.copy()
    nav["date"] = pd.to_datetime(nav["date"])

    merged = df.merge(nav[["date", "portfolio_nav"]], on="date", how="inner").sort_values("date")
    if merged.empty:
        return None

    # Drawdown
    peak = merged["portfolio_nav"].cummax()
    merged["drawdown"] = (merged["portfolio_nav"] - peak) / peak

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)

    # NAV line
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["portfolio_nav"],
        mode="lines", name="NAV", line=dict(color="white", width=1.5),
    ), row=1, col=1)

    # Regime shading on NAV chart
    prev_regime = None
    start_idx = 0
    for i in range(len(merged)):
        regime = merged.iloc[i]["regime"]
        if regime != prev_regime and prev_regime is not None:
            color = REGIME_COLORS.get(prev_regime, "#666")
            fig.add_vrect(
                x0=merged.iloc[start_idx]["date"], x1=merged.iloc[i - 1]["date"],
                fillcolor=color, opacity=0.15, layer="below", line_width=0,
                row=1, col=1,
            )
            start_idx = i
        prev_regime = regime
    # Last segment
    if prev_regime:
        color = REGIME_COLORS.get(prev_regime, "#666")
        fig.add_vrect(
            x0=merged.iloc[start_idx]["date"], x1=merged.iloc[-1]["date"],
            fillcolor=color, opacity=0.15, layer="below", line_width=0,
            row=1, col=1,
        )

    # Drawdown bars
    fig.add_trace(go.Bar(
        x=merged["date"], y=merged["drawdown"],
        marker_color=[REGIME_COLORS.get(r, "#666") for r in merged["regime"]],
        name="Drawdown", opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        title="NAV with Regime Overlay",
        height=450,
        margin=dict(t=40, b=20, l=60, r=20),
        showlegend=False,
    )
    fig.update_yaxes(title_text="NAV (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)

    return fig


def _build_transition_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Transition probability heatmap."""
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[c.replace("_", " ").title() for c in matrix.columns],
        y=[r.replace("_", " ").title() for r in matrix.index],
        text=np.round(matrix.values * 100, 1),
        texttemplate="%{text:.1f}%",
        colorscale="RdYlGn_r",
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="Regime Transition Probabilities",
        height=300,
        margin=dict(t=40, b=20, l=80, r=20),
        xaxis_title="To",
        yaxis_title="From",
    )
    return fig


def _build_duration_chart(durations: pd.DataFrame) -> go.Figure:
    """Box plot of regime episode durations."""
    df = durations.copy()
    df["regime_label"] = df["regime"].str.replace("_", " ").str.title()

    fig = px.box(
        df, x="regime_label", y="duration_days",
        color="regime_label",
        color_discrete_map={k.replace("_", " ").title(): v for k, v in REGIME_COLORS.items()},
        title="Regime Episode Durations",
    )
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis_title="Regime",
        yaxis_title="Duration (days)",
        showlegend=False,
    )
    return fig


# ── Main render ──────────────────────────────────────────────────────────────


def render() -> None:
    st.header("Regime Intelligence")
    st.caption("Context-aware market regime detection — adaptive portfolio behavior")

    with st.spinner("Running regime intelligence pipeline…"):
        data = _load_regime_data()
        nav = _load_nav()

    if not data:
        st.error("Regime pipeline failed. Check logs for details.")
        return

    regimes = data.get("regimes", pd.DataFrame())
    features = data.get("features", pd.DataFrame())
    transition_matrix = data.get("transition_matrix", pd.DataFrame())
    stability = data.get("stability", {})
    durations = data.get("durations", pd.DataFrame())
    quality = data.get("quality_score", {})
    current_regime = data.get("current_regime", "unknown")
    behavior = data.get("behavior")

    # ── Current regime banner ────────────────────────────────────────────
    emoji = REGIME_EMOJI.get(current_regime, "⚪")
    label = current_regime.replace("_", " ").upper()
    confidence = regimes.iloc[-1]["confidence"] if not regimes.empty else 0

    if current_regime == "risk_on":
        st.success(f"**Current Regime: {emoji} {label}** (confidence: {confidence:.0%})")
    elif current_regime == "panic":
        st.error(f"**Current Regime: {emoji} {label}** (confidence: {confidence:.0%})")
    elif current_regime == "risk_off":
        st.warning(f"**Current Regime: {emoji} {label}** (confidence: {confidence:.0%})")
    else:
        st.info(f"**Current Regime: {emoji} {label}** (confidence: {confidence:.0%})")

    # ── Behavior parameters ──────────────────────────────────────────────
    if behavior:
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Max Equity", f"{behavior.max_equity_weight:.0%}")
        bc2.metric("Drift Threshold", f"{behavior.rebalance_drift_threshold:.0%}")
        bc3.metric("Tilt Strength", f"{behavior.tilt_strength:.2f}")
        bc4.metric("Covariance", behavior.covariance_method)

    st.divider()

    # ── Quality score ────────────────────────────────────────────────────
    if quality:
        total = quality.get("total_score", 0)
        grade = quality.get("grade", "?")
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("Quality Score", f"{total}/100 ({grade})")
        qc2.metric("Stability", f"{quality.get('stability_score', 0)}/25")
        qc3.metric("Predictive", f"{quality.get('predictive_score', 0)}/25")
        qc4.metric("Crisis Align", f"{quality.get('crisis_alignment_score', 0)}/25")
        qc5.metric("Separation", f"{quality.get('separation_score', 0)}/25")

    st.divider()

    # ── KPI row ──────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Transitions/Year", stability.get("transitions_per_year", "N/A"))
    kc2.metric("Avg Duration", f"{stability.get('avg_duration_days', 0):.0f} days")
    kc3.metric("Dominant Regime", stability.get("dominant_regime", "N/A"))
    kc4.metric("Episodes", stability.get("n_episodes", "N/A"))

    st.divider()

    # ── Timeline ─────────────────────────────────────────────────────────
    if not regimes.empty:
        st.plotly_chart(_build_regime_timeline(regimes), use_container_width=True)

    # ── NAV overlay ──────────────────────────────────────────────────────
    nav_fig = _build_regime_nav_overlay(regimes, nav)
    if nav_fig:
        st.plotly_chart(nav_fig, use_container_width=True)

    st.divider()

    # ── Transition heatmap + duration chart ──────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        if not transition_matrix.empty:
            st.plotly_chart(_build_transition_heatmap(transition_matrix), use_container_width=True)

    with col_right:
        if not durations.empty:
            st.plotly_chart(_build_duration_chart(durations), use_container_width=True)

    st.divider()

    # ── Crisis alignment table ───────────────────────────────────────────
    st.subheader("Crisis Alignment")
    from regimes.evaluation import evaluate_crisis_alignment
    crisis_df = evaluate_crisis_alignment(regimes)
    if not crisis_df.empty:
        st.dataframe(crisis_df, use_container_width=True, hide_index=True)
    else:
        st.info("No crisis data within date range.")

    # ── Regime features (expandable) ─────────────────────────────────────
    with st.expander("Regime Features (raw)", expanded=False):
        if not features.empty:
            st.dataframe(features.tail(20), use_container_width=True, hide_index=True)

    with st.expander("Transition Matrix (raw)", expanded=False):
        if not transition_matrix.empty:
            st.dataframe(transition_matrix, use_container_width=True)
