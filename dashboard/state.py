"""
Session state defaults — initializes Streamlit session state with
sensible defaults for all interactive controls.
"""

import streamlit as st


def init_state() -> None:
    """Initialize session state with defaults if not already set."""
    defaults = {
        # Optimization settings
        "max_weight": 0.40,
        "min_weight": 0.05,
        "country_cap_us": 0.60,
        "country_cap_in": 0.60,
        "tilt_strength": 0.20,
        "rebalance_freq": "quarterly",
        "turnover_threshold": 0.10,
        # Risk settings
        "vol_target": 0.20,
        # Backtest settings
        "backtest_capital": 1_000_000,
        "slippage_bps": 10,
        "selected_benchmark": "SPY",
        # Strategy selection
        "selected_strategy": "hrp_optimized",
        # Date range (set from data)
        "date_range_start": None,
        "date_range_end": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
