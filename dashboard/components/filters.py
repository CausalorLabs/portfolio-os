"""
Interactive filter controls — allow research experimentation.

Renders in sidebar settings expander.
"""

import streamlit as st


def render_filters() -> None:
    """Render all interactive filter controls in sidebar."""
    with st.expander("⚖️ Optimization", expanded=False):
        st.session_state["rebalance_freq"] = st.selectbox(
            "Rebalance frequency",
            ["monthly", "quarterly"],
            index=1 if st.session_state.get("rebalance_freq") == "quarterly" else 0,
            key="filter_rebal_freq",
        )
        st.session_state["max_weight"] = st.slider(
            "Max asset weight", 0.10, 0.60,
            st.session_state.get("max_weight", 0.40),
            step=0.05, format="%.0f%%",
            key="filter_max_weight",
        )
        st.session_state["min_weight"] = st.slider(
            "Min asset weight", 0.00, 0.15,
            st.session_state.get("min_weight", 0.05),
            step=0.01, format="%.0f%%",
            key="filter_min_weight",
        )
        st.session_state["tilt_strength"] = st.slider(
            "Signal tilt strength", 0.0, 0.50,
            st.session_state.get("tilt_strength", 0.20),
            step=0.05,
            key="filter_tilt",
        )
        st.session_state["turnover_threshold"] = st.slider(
            "Turnover threshold", 0.01, 0.30,
            st.session_state.get("turnover_threshold", 0.10),
            step=0.01,
            key="filter_turnover",
        )

    with st.expander("📊 Risk", expanded=False):
        st.session_state["vol_target"] = st.slider(
            "Volatility target", 0.05, 0.40,
            st.session_state.get("vol_target", 0.20),
            step=0.01, format="%.0f%%",
            key="filter_vol_target",
        )
        st.session_state["country_cap_us"] = st.slider(
            "US country cap", 0.20, 0.80,
            st.session_state.get("country_cap_us", 0.60),
            step=0.05, format="%.0f%%",
            key="filter_us_cap",
        )
        st.session_state["country_cap_in"] = st.slider(
            "IN country cap", 0.20, 0.80,
            st.session_state.get("country_cap_in", 0.60),
            step=0.05, format="%.0f%%",
            key="filter_in_cap",
        )

    with st.expander("🧪 Backtests", expanded=False):
        st.session_state["slippage_bps"] = st.slider(
            "Slippage (bps)", 0, 50,
            st.session_state.get("slippage_bps", 10),
            key="filter_slippage",
        )
        st.session_state["backtest_capital"] = st.number_input(
            "Initial capital (₹)",
            min_value=100_000,
            max_value=100_000_000,
            value=st.session_state.get("backtest_capital", 1_000_000),
            step=100_000,
            key="filter_capital",
        )
        st.session_state["selected_benchmark"] = st.selectbox(
            "Benchmark",
            ["SPY", "RELIANCE.NS", "Equal Weight"],
            index=0,
            key="filter_benchmark",
        )
