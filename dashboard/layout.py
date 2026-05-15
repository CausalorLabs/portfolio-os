"""
Layout utilities — consistent page structure and theming.
"""

import streamlit as st

# ── CSS theme ────────────────────────────────────────────────────────────────

PORTFOLIO_OS_CSS = """
<style>
    /* Dark finance theme */
    .stMetric label { font-size: 0.85rem !important; color: #888 !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    div[data-testid="stSidebarContent"] { padding-top: 1rem; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .stDivider { margin: 1rem 0 !important; }

    /* Card-based sections */
    .stExpander { border-radius: 8px; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
"""


def apply_theme() -> None:
    """Inject Portfolio OS CSS theme."""
    st.markdown(PORTFOLIO_OS_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header."""
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def section(title: str) -> None:
    """Start a new visual section with divider + subheader."""
    st.divider()
    st.subheader(title)


def two_column_layout(ratio: tuple = (1, 1)):
    """Return two columns with given ratio."""
    return st.columns(ratio)


def three_column_layout():
    """Return three equal columns."""
    return st.columns(3)
