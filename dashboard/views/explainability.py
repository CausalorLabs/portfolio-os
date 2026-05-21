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
    st.header("🔍 Explainability & Monitoring", help="Full observability into the portfolio system — performance attribution, decision logs, alerts, system health, anomalies, and audit trail.")
    st.caption("Attribution, decisions, alerts, and operational health")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Attribution", "Decisions", "Alerts",
        "System Health", "Anomalies", "Audit Trail",
    ])

    # ── Attribution Tab ──────────────────────────────────────────────────

    with tab1:
        st.subheader("Performance Attribution", help="Decomposes portfolio return into asset-level contributions. Shows which holdings drove gains or losses.")

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
                    st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No factor exposure data available.")
            except Exception:
                st.info("Factor data not yet computed.")

    # ── Decisions Tab ────────────────────────────────────────────────────

    with tab2:
        st.subheader("Decision Timeline", help="Chronological log of every trade and no-trade decision made by the execution engine, with rationale and confidence scores.")
        st.markdown("Every portfolio decision with full reasoning.")

        try:
            wh = get_warehouse()
            journal_df = wh.read_table("execution_journal")
            if journal_df is not None and not journal_df.empty:
                # Decision distribution
                col1, col2, col3 = st.columns(3)
                action_counts = journal_df["action"].value_counts()
                with col1:
                    st.metric("Total Decisions", len(journal_df), help="Total number of trade/no-trade decisions logged by the execution engine.")
                with col2:
                    st.metric("Trades", action_counts.get("trade", 0), help="Number of times the engine decided to execute a trade.")
                with col3:
                    st.metric("Skipped", action_counts.get("no_trade", 0), help="Number of times the engine skipped trading because cost exceeded expected utility.")

                # Decision pie
                fig = px.pie(
                    names=action_counts.index,
                    values=action_counts.values,
                    title="Decision Distribution",
                    height=300,
                )
                st.plotly_chart(fig, width="stretch")

                # Recent decisions table
                st.markdown("#### Recent Decisions")
                display_cols = [c for c in [
                    "timestamp", "decision_id", "action", "rationale",
                    "expected_utility", "confidence", "regime", "trigger",
                ] if c in journal_df.columns]
                st.dataframe(
                    journal_df[display_cols].tail(20).sort_index(ascending=False),
                    width="stretch",
                    height=400,
                )
            else:
                st.info("No execution journal data. Run the execution pipeline first.")
        except Exception:
            st.info("Execution journal not available.")

    # ── Alerts Tab ───────────────────────────────────────────────────────

    with tab3:
        st.subheader("Alert Console", help="Active monitoring alerts from the pipeline. Grouped by severity (CRITICAL, WARNING, INFO) and category (data quality, risk, execution).")

        try:
            wh = get_warehouse()
            alerts_df = wh.read_table("monitoring_alerts")
            if alerts_df is not None and not alerts_df.empty:
                # Alert summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Alerts", len(alerts_df), help="Total number of monitoring alerts generated across all categories.")
                with col2:
                    critical = len(alerts_df[alerts_df["severity"] == "CRITICAL"])
                    st.metric("Critical", critical, help="Alerts requiring immediate attention — data gaps, risk limit breaches, or execution failures.")
                with col3:
                    warning = len(alerts_df[alerts_df["severity"] == "WARNING"])
                    st.metric("Warning", warning, help="Non-critical alerts that may need review — drift warnings, stale data, elevated volatility.")
                with col4:
                    unack = len(alerts_df[~alerts_df.get("acknowledged", False)])
                    st.metric("Unacknowledged", unack, help="Alerts not yet reviewed or acknowledged by the operator.")

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
                st.plotly_chart(fig, width="stretch")

                # Alert table
                st.dataframe(
                    alerts_df.tail(30).sort_index(ascending=False),
                    width="stretch",
                    height=400,
                )
            else:
                st.success("No alerts. System is operating normally.")
        except Exception:
            st.info("Alert data not available.")

    # ── System Health Tab ────────────────────────────────────────────────

    with tab4:
        st.subheader("System Health", help="Pipeline infrastructure status — CPU, memory, database connections, and component-level health indicators.")

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
        st.subheader("Model Health", help="ML model monitoring — feature drift, prediction stability, and model staleness indicators.")
        try:
            model_df = wh.read_table("model_health")
            if model_df is not None and not model_df.empty:
                st.dataframe(model_df, width="stretch")
            else:
                st.info("Model health data not available.")
        except Exception:
            st.info("Model health not yet tracked.")

    # ── Anomalies Tab ────────────────────────────────────────────────────

    with tab5:
        st.subheader("Anomaly Detection", help="Statistical anomalies detected in portfolio data — z-score outliers in returns, volumes, or risk metrics.")

        try:
            wh = get_warehouse()
            anomaly_df = wh.read_table("anomaly_log")
            if anomaly_df is not None and not anomaly_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Anomalies", len(anomaly_df), help="Count of statistical anomalies detected across all monitored metrics.")
                with col2:
                    critical = len(anomaly_df[anomaly_df["severity"] == "CRITICAL"])
                    st.metric("Critical", critical, help="High-severity anomalies (z-score > 3) that may indicate data corruption or extreme market events.")

                fig = px.scatter(
                    anomaly_df, x="timestamp", y="zscore",
                    color="category", size=anomaly_df["zscore"].abs(),
                    hover_data=["metric", "description"],
                    title="Anomaly Timeline",
                    height=350,
                )
                st.plotly_chart(fig, width="stretch")

                st.dataframe(
                    anomaly_df.tail(20).sort_index(ascending=False),
                    width="stretch",
                )
            else:
                st.success("No anomalies detected.")
        except Exception:
            st.info("Anomaly data not available.")

    # ── Audit Trail Tab ──────────────────────────────────────────────────

    with tab6:
        st.subheader("Audit Trail & Traceability", help="Immutable log of every pipeline action — data loads, model runs, trade decisions. Supports regulatory and debugging needs.")

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
                st.plotly_chart(fig, width="stretch")

                st.markdown("#### Recent Events")
                st.dataframe(
                    audit_df.tail(30).sort_index(ascending=False),
                    width="stretch",
                    height=400,
                )
            else:
                st.info("No audit trail data yet.")
        except Exception:
            st.info("Audit data not available.")
