"""
Dashboard — Personal Portfolio Command Center.

Sprint 8 command center view:
  - NAV & performance at-a-glance
  - Trust score gauge
  - Regime status
  - Recommended actions
  - Deployment readiness
  - Walk-forward comparison
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render():
    """Render the Personal Command Center dashboard."""
    st.header("🎯 Command Center", help="Personal portfolio operations hub — overview, trust calibration, validation checks, walk-forward evaluation, override control, and deployment readiness.")
    st.caption("Personal portfolio operations — trust, readiness, and control")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Trust", "Validation",
        "Walk-Forward", "Override", "Deployment",
    ])

    # ── Overview Tab ─────────────────────────────────────────────────────

    with tab1:
        st.subheader("Portfolio Status", help="At-a-glance portfolio metrics — current NAV, daily return, number of positions, and latest data date.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()

            # NAV
            nav_df = wh.read_table("portfolio_nav")
            if nav_df is not None and not nav_df.empty:
                latest_nav = nav_df.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("NAV", f"₹{latest_nav.get('nav', 0):,.0f}", help="Current portfolio Net Asset Value in INR.")

                # Daily return
                if len(nav_df) > 1:
                    prev_nav = nav_df.iloc[-2].get("nav", 1)
                    curr_nav = latest_nav.get("nav", 1)
                    daily_ret = (curr_nav / prev_nav - 1) * 100 if prev_nav else 0
                    col2.metric("Daily", f"{daily_ret:+.2f}%", help="Today's portfolio return as a percentage.")

                col3.metric("Positions", int(latest_nav.get("n_positions", 0)) if "n_positions" in nav_df.columns else "—", help="Number of active holdings in the portfolio.")
                col4.metric("Date", str(latest_nav.get("date", "—"))[:10], help="Date of the most recent portfolio valuation.")

                # NAV chart
                if "date" in nav_df.columns and "nav" in nav_df.columns:
                    fig = px.line(nav_df, x="date", y="nav", title="Portfolio NAV")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No NAV data available.")

            # System state
            state_df = wh.read_table("system_state")
            if state_df is not None and not state_df.empty:
                latest = state_df.iloc[-1]
                st.markdown("#### System State")
                c1, c2, c3 = st.columns(3)
                c1.metric("Regime", latest.get("regime", "—"), help="Current detected market regime (risk_on, risk_off, or panic).")
                c2.metric("Pipeline", latest.get("pipeline_status", "—"), help="Last pipeline execution status: completed, failed, or running.")
                c3.metric("Mode", latest.get("approval_mode", "assisted"), help="Current approval mode: advisory (no trades), assisted (human approval), or autonomous (auto-execute).")
        except Exception:
            st.info("Portfolio data not available. Run the pipeline first.")

    # ── Trust Tab ────────────────────────────────────────────────────────

    with tab2:
        st.subheader("Trust Calibration", help="Trust score measures system reliability across 5 dimensions. Higher trust enables more autonomous execution modes.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            trust_df = wh.read_table("trust_scores")
            if trust_df is not None and not trust_df.empty:
                latest = trust_df.iloc[-1]
                overall = latest.get("overall_trust", 0.5)

                # Trust gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Trust Score"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.3], "color": "red"},
                            {"range": [0.3, 0.5], "color": "orange"},
                            {"range": [0.5, 0.8], "color": "yellow"},
                            {"range": [0.8, 1.0], "color": "green"},
                        ],
                    },
                ))
                st.plotly_chart(fig, width="stretch")

                # Dimension breakdown
                st.markdown("#### Trust Dimensions")
                dims = {
                    "Model Health": latest.get("model_health", 0),
                    "Data Quality": latest.get("data_quality", 0),
                    "Regime Stability": latest.get("regime_stability", 0),
                    "Execution Reliability": latest.get("execution_reliability", 0),
                    "Operational Health": latest.get("operational_health", 0),
                }
                fig = px.bar(
                    x=list(dims.keys()), y=list(dims.values()),
                    title="Trust Dimensions", labels={"x": "", "y": "Score"},
                    color=list(dims.values()),
                    color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(fig, width="stretch")

                st.metric("Recommended Mode", latest.get("recommended_mode", "assisted"), help="System-recommended approval mode based on current trust score. Higher trust = more autonomy.")
            else:
                st.info("No trust data available yet.")
        except Exception:
            st.info("Trust data not available.")

    # ── Validation Tab ───────────────────────────────────────────────────

    with tab3:
        st.subheader("E2E Validation", help="End-to-end validation checks verifying data integrity, model outputs, and constraint compliance across the full pipeline.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            val_df = wh.read_table("validation_results")
            if val_df is not None and not val_df.empty:
                passed = val_df["passed"].sum() if "passed" in val_df.columns else 0
                total = len(val_df)
                c1, c2, c3 = st.columns(3)
                c1.metric("Checks Passed", f"{int(passed)}/{total}", help="Number of validation checks that passed out of total checks run.")
                c2.metric("Pass Rate", f"{passed/total*100:.0f}%", help="Percentage of validation checks that passed. Aim for 100%.")
                c3.metric("Critical", int((~val_df["passed"] & (val_df.get("severity", "INFO") == "CRITICAL")).sum()) if "severity" in val_df.columns else 0, help="Number of failed critical-severity checks. These must be resolved before deployment.")

                st.dataframe(val_df, width="stretch")
            else:
                st.info("No validation results yet.")
        except Exception:
            st.info("Validation data not available.")

    # ── Walk-Forward Tab ─────────────────────────────────────────────────

    with tab4:
        st.subheader("Walk-Forward Evaluation", help="Out-of-sample strategy validation — compares strategies trained on past data against their performance on unseen future data.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            wf_df = wh.read_table("walkforward_comparison")
            if wf_df is not None and not wf_df.empty:
                st.markdown("#### Strategy Comparison")
                st.dataframe(wf_df, width="stretch")

                # Bar chart of key metrics
                if "strategy" in wf_df.columns and "sharpe" in wf_df.columns:
                    fig = px.bar(
                        wf_df, x="strategy", y="sharpe",
                        title="Sharpe Ratio Comparison",
                        color="strategy",
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No walk-forward results yet.")
        except Exception:
            st.info("Walk-forward data not available.")

    # ── Override Tab ─────────────────────────────────────────────────────

    with tab5:
        st.subheader("Human Override Control", help="Configure the level of human oversight. Advisory = view-only, Assisted = approve trades, Autonomous = auto-execute.")

        st.markdown("""
        **Approval Modes:**
        - **Advisory**: Recommendations only — no execution
        - **Assisted**: Human approval required for trades
        - **Autonomous**: Fully automated (requires high trust score)
        """)

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            state_df = wh.read_table("system_state")
            if state_df is not None and not state_df.empty:
                current = state_df.iloc[-1].get("approval_mode", "assisted")
                st.metric("Current Mode", current, help="Active approval mode governing how the system handles trade recommendations.")
            else:
                st.metric("Current Mode", "assisted", help="Active approval mode governing how the system handles trade recommendations.")
        except Exception:
            st.metric("Current Mode", "assisted", help="Active approval mode governing how the system handles trade recommendations.")

    # ── Deployment Tab ───────────────────────────────────────────────────

    with tab6:
        st.subheader("Deployment Readiness", help="Stabilization report showing whether the system is ready for production. Checks model stability, data quality, and operational health.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            deploy_df = wh.read_table("stabilization_report")
            if deploy_df is not None and not deploy_df.empty:
                st.dataframe(deploy_df, width="stretch")
            else:
                st.info("No stabilization report yet. Run deployment readiness check.")
        except Exception:
            st.info("Deployment data not available.")
