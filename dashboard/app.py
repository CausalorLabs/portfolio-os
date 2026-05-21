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
from dashboard.views import overview, analytics, optimization, backtests, exposure, recommendations, structural_health, regime_intelligence, risk_intelligence, execution_intelligence, explainability

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
    "🏗️ Structural Health": structural_health,
    "🧠 Regime Intelligence": regime_intelligence,
    "🛡️ Risk Intelligence": risk_intelligence,
    "⚡ Execution Intelligence": execution_intelligence,
    "🔍 Explainability": explainability,
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
    st.caption("Portfolio OS • v1.0")

# ── Render selected page ─────────────────────────────────────────────────────

PAGES[page_name].render()
