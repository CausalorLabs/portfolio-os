"""
Portfolio OS — Dashboard & Research Interface.

Launch: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.state import init_state
from dashboard.layout import apply_theme
from dashboard.components.filters import render_filters
from dashboard.views import overview, analytics, optimization, backtests, exposure, recommendations

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & State ────────────────────────────────────────────────────────────

apply_theme()
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

    # ── Research controls ────────────────────────────────────────────────
    render_filters()

    st.divider()
    st.caption("Sprint 1→8 • POC v0.8")

# ── Render selected page ─────────────────────────────────────────────────────

PAGES[page_name].render()
