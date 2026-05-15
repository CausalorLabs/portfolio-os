"""
Reusable table components — consistent data display.
"""

import pandas as pd
import streamlit as st


def styled_dataframe(
    df: pd.DataFrame,
    hide_index: bool = True,
    height: int | None = None,
) -> None:
    """Display a dataframe with standard Portfolio OS styling."""
    kwargs = {"width": "stretch", "hide_index": hide_index}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def metric_table(
    df: pd.DataFrame,
    format_rules: dict[str, callable] | None = None,
    rename: dict[str, str] | None = None,
) -> None:
    """Display a formatted metrics table with optional formatting and renaming."""
    display = df.copy()
    if format_rules:
        for col, fn in format_rules.items():
            if col in display.columns:
                display[col] = display[col].apply(fn)
    if rename:
        show_cols = [c for c in rename if c in display.columns]
        display = display[show_cols].rename(columns=rename)
    st.dataframe(display, width="stretch", hide_index=False)
