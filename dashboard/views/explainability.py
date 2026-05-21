"""
Dashboard — Explainability & Attribution Intelligence View.

Visualizations:
  - Decision timeline
  - Attribution breakdown (allocation, selection, timing, currency)
  - Factor exposure decomposition
  - Risk attribution
  - Alert console
  - Confidence dashboard
  - Audit trail
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render():
    """Render the Explainability & Monitoring dashboard."""
    st.header("🔍 Explainability & Monitoring")
    st.caption("Attribution, decisions, alerts, and operational health")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Attribution", "Decisions", "Alerts",
        "System Health", "Anomalies", "Audit Trail",
    ])

    # ── Attribution Tab ──────────────────────────────────────────────────

    with tab1:
        st.subheader("Performance Attribution")

        # Attribution breakdown bar chart
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Return Decomposition")
            try:
                from warehouse import get_warehouse
                wh = get_warehouse()
                attr_df = wh.read_table("attribution_summary")
                if attr_df is not None and not attr_df.empty:
                    latest = attr_df.iloc[-1]
                    effects = {
                        "Allocation": latest.get("allocation_effect", 0),
                        "Selection": latest.get("selection_effect", 0),
                        "Timing": latest.get("timing_effect", 0),
                        "Currency": latest.get("currency_effect", 0),
                        "Interaction": latest.get("interaction_effect", 0),
                    }
                    fig = go.Figure(go.Bar(
                        x=list(effects.keys()),
                        y=list(effects.values()),
                        marker_color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#607D8B"],
                    ))
                    fig.update_layout(
                        title="Attribution Effects",
                        yaxis_title="Return Contribution",
                        yaxis_tickformat=".2%",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No attribution data available. Run the attribution pipeline first.")
            except Exception:
                st.info("Attribution data not yet computed.")

        with col2:
            st.markdown("#### Factor Exposures")
            try:
                factor_df = wh.read_table("factor_exposures")
                if factor_df is not None and not factor_df.empty:
                    fig = px.bar(
                        factor_df,
                        x="factor", y="exposure",
                        color="contribution",
                        title="Factor Exposure & Contribution",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No factor exposure data available.")
            except Exception:
                st.info("Factor data not yet computed.")

    # ── Decisions Tab ────────────────────────────────────────────────────

    with tab2:
        st.subheader("Decision Timeline")
        st.markdown("Every portfolio decision with full reasoning.")

        try:
            wh = get_warehouse()
            journal_df = wh.read_table("execution_journal")
            if journal_df is not None and not journal_df.empty:
                # Decision distribution
                col1, col2, col3 = st.columns(3)
                action_counts = journal_df["action"].value_counts()
                with col1:
                    st.metric("Total Decisions", len(journal_df))
                with col2:
                    st.metric("Trades", action_counts.get("trade", 0))
                with col3:
                    st.metric("Skipped", action_counts.get("no_trade", 0))

                # Decision pie
                fig = px.pie(
                    names=action_counts.index,
                    values=action_counts.values,
                    title="Decision Distribution",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Recent decisions table
                st.markdown("#### Recent Decisions")
                display_cols = [c for c in [
                    "timestamp", "decision_id", "action", "rationale",
                    "expected_utility", "confidence", "regime", "trigger",
                ] if c in journal_df.columns]
                st.dataframe(
                    journal_df[display_cols].tail(20).sort_index(ascending=False),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No execution journal data. Run the execution pipeline first.")
        except Exception:
            st.info("Execution journal not available.")

    # ── Alerts Tab ───────────────────────────────────────────────────────

    with tab3:
        st.subheader("Alert Console")

        try:
            wh = get_warehouse()
            alerts_df = wh.read_table("monitoring_alerts")
            if alerts_df is not None and not alerts_df.empty:
                # Alert summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Alerts", len(alerts_df))
                with col2:
                    critical = len(alerts_df[alerts_df["severity"] == "CRITICAL"])
                    st.metric("Critical", critical)
                with col3:
                    warning = len(alerts_df[alerts_df["severity"] == "WARNING"])
                    st.metric("Warning", warning)
                with col4:
                    unack = len(alerts_df[~alerts_df.get("acknowledged", False)])
                    st.metric("Unacknowledged", unack)

                # Alerts by category
                fig = px.histogram(
                    alerts_df, x="category", color="severity",
                    title="Alerts by Category",
                    color_discrete_map={
                        "CRITICAL": "#f44336",
                        "WARNING": "#ff9800",
                        "INFO": "#2196F3",
                    },
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Alert table
                st.dataframe(
                    alerts_df.tail(30).sort_index(ascending=False),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.success("No alerts. System is operating normally.")
        except Exception:
            st.info("Alert data not available.")

    # ── System Health Tab ────────────────────────────────────────────────

    with tab4:
        st.subheader("System Health")

        try:
            wh = get_warehouse()
            health_df = wh.read_table("system_health")
            if health_df is not None and not health_df.empty:
                # Status indicators
                for _, row in health_df.iterrows():
                    status = row.get("status", "unknown")
                    emoji = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(status, "⚪")
                    st.markdown(
                        f"{emoji} **{row['component']}** — "
                        f"{status} | "
                        f"Latency: {row.get('latency_ms', 0):.0f}ms | "
                        f"Errors: {row.get('error_count', 0)}"
                    )
            else:
                st.info("Health data not yet collected.")
        except Exception:
            st.info("System health data not available.")

        # Model health
        st.markdown("---")
        st.subheader("Model Health")
        try:
            model_df = wh.read_table("model_health")
            if model_df is not None and not model_df.empty:
                st.dataframe(model_df, use_container_width=True)
            else:
                st.info("Model health data not available.")
        except Exception:
            st.info("Model health not yet tracked.")

    # ── Anomalies Tab ────────────────────────────────────────────────────

    with tab5:
        st.subheader("Anomaly Detection")

        try:
            wh = get_warehouse()
            anomaly_df = wh.read_table("anomaly_log")
            if anomaly_df is not None and not anomaly_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Anomalies", len(anomaly_df))
                with col2:
                    critical = len(anomaly_df[anomaly_df["severity"] == "CRITICAL"])
                    st.metric("Critical", critical)

                fig = px.scatter(
                    anomaly_df, x="timestamp", y="zscore",
                    color="category", size=anomaly_df["zscore"].abs(),
                    hover_data=["metric", "description"],
                    title="Anomaly Timeline",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    anomaly_df.tail(20).sort_index(ascending=False),
                    use_container_width=True,
                )
            else:
                st.success("No anomalies detected.")
        except Exception:
            st.info("Anomaly data not available.")

    # ── Audit Trail Tab ──────────────────────────────────────────────────

    with tab6:
        st.subheader("Audit Trail & Traceability")

        try:
            wh = get_warehouse()
            audit_df = wh.read_table("audit_trail")
            if audit_df is not None and not audit_df.empty:
                # Pipeline flow
                fig = px.histogram(
                    audit_df, x="event_type", color="component",
                    title="Events by Type & Component",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Recent Events")
                st.dataframe(
                    audit_df.tail(30).sort_index(ascending=False),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No audit trail data yet.")
        except Exception:
            st.info("Audit data not available.")
