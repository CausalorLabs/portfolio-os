"""
Sidebar navigation component.
"""

import streamlit as st


def render_nav(pages: dict) -> str:
    """Render sidebar navigation and return selected page key."""
    with st.sidebar:
        st.title("Portfolio OS")
        st.caption("Personal Portfolio Research Engine")
        st.divider()

        page_name = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        st.divider()

    return page_name
