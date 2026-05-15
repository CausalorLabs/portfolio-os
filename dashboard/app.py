"""
Portfolio OS — Dashboard & Research Interface.

Launch: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.state import init_state
from dashboard.pages import overview, analytics, optimization, backtests, exposure, recommendations

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark finance theme overrides */
    .stMetric label { font-size: 0.85rem !important; color: #888 !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .stDivider { margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── State ────────────────────────────────────────────────────────────────────

init_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────

PAGES = {
    "📊 Overview": overview,
    "📈 Analytics": analytics,
    "⚖️ Optimization": optimization,
    "🧪 Backtests": backtests,
    "🌍 Exposure": exposure,
    "💡 Recommendations": recommendations,
}

with st.sidebar:
    st.title("Portfolio OS")
    st.caption("Personal Portfolio Research Engine")
    st.divider()

    page_name = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.divider()

    # ── Settings panel ───────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        st.session_state["max_weight"] = st.slider(
            "Max asset weight", 0.10, 0.60, st.session_state["max_weight"],
            step=0.05, format="%.0f%%",
        )
        st.session_state["rebalance_freq"] = st.selectbox(
            "Rebalance frequency",
            ["monthly", "quarterly"],
            index=1 if st.session_state["rebalance_freq"] == "quarterly" else 0,
        )
        st.session_state["slippage_bps"] = st.slider(
            "Slippage (bps)", 0, 50, st.session_state["slippage_bps"],
        )
        st.session_state["tilt_strength"] = st.slider(
            "Signal tilt strength", 0.0, 0.50, st.session_state["tilt_strength"],
            step=0.05,
        )

    st.divider()
    st.caption("Sprint 1→7 • POC v0.7")

# ── Render selected page ─────────────────────────────────────────────────────

PAGES[page_name].render()
