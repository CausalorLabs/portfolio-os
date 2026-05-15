"""
Export utilities — CSV, PNG downloads from dashboard.
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st


def csv_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "📥 Download CSV",
) -> None:
    """Render a download button for a DataFrame as CSV."""
    csv = df.to_csv(index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label=label,
        data=csv,
        file_name=f"{filename}_{ts}.csv",
        mime="text/csv",
    )


def png_download_button(
    fig,
    filename: str,
    label: str = "📥 Download PNG",
) -> None:
    """Render a download button for a Plotly figure as PNG."""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label=label,
            data=img_bytes,
            file_name=f"{filename}_{ts}.png",
            mime="image/png",
        )
    except Exception:
        st.caption("PNG export requires kaleido: `pip install kaleido`")


def export_section(exports: dict[str, pd.DataFrame]) -> None:
    """Render an export panel with download buttons for multiple datasets."""
    with st.expander("📤 Export Data", expanded=False):
        for name, df in exports.items():
            csv_download_button(df, name, label=f"📥 {name}")
