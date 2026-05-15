"""
Reusable metric card components — consistent KPI display.
"""

import streamlit as st

from dashboard.utils.formatters import fmt_currency, fmt_pct, fmt_number


def metric_row(metrics: dict, columns: int = 6) -> None:
    """Render a row of KPI metric cards from a dict of {label: (value, delta)}."""
    cols = st.columns(columns)
    for i, (label, spec) in enumerate(metrics.items()):
        col = cols[i % columns]
        if isinstance(spec, tuple):
            col.metric(label, spec[0], spec[1] if len(spec) > 1 else None)
        else:
            col.metric(label, spec)


def portfolio_kpis(metrics: dict) -> None:
    """Render standard portfolio KPI cards from metrics dict."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Portfolio NAV", fmt_currency(metrics.get("portfolio_nav", 0)),
              fmt_pct(metrics.get("daily_return", 0)))
    c2.metric("CAGR", fmt_pct(metrics.get("cagr", 0)))
    c3.metric("Sharpe", fmt_number(metrics.get("sharpe_ratio", 0)))
    c4.metric("Sortino", fmt_number(metrics.get("sortino_ratio", 0)))
    c5.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown", 0)))
    c6.metric("Volatility", fmt_pct(metrics.get("annualized_volatility", 0)))
