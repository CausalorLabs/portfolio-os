"""
Dashboard — Operations & Workflow Intelligence View.

Sprint 7 dashboard visualizations:
  - Workflow pipeline status & stage timeline
  - Event stream monitor
  - Scheduling & cadence overview
  - SLA compliance
  - MLOps model health
  - System state overview
  - Retry & self-healing stats
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render():
    """Render the Operations & Workflow Intelligence dashboard."""
    st.header("🏭 Operations & Workflow Intelligence", help="Pipeline orchestration control center — monitor workflow runs, event streams, scheduling, SLA compliance, ML model health, and configuration governance.")
    st.caption("Pipeline orchestration, scheduling, SLA compliance, and MLOps")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pipeline", "Events", "Schedule",
        "SLA", "MLOps", "Governance",
    ])

    # ── Pipeline Tab ─────────────────────────────────────────────────────

    with tab1:
        st.subheader("Workflow Pipeline", help="Overview of pipeline execution runs — status, stage progression, duration, and historical run performance.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            runs_df = wh.read_table("workflow_runs")
            if runs_df is not None and not runs_df.empty:
                # Latest run summary
                latest = runs_df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", latest.get("status", "unknown"), help="Current pipeline run status: completed, running, or failed.")
                c2.metric("Stages", f"{latest.get('stages_completed', 0)}/{latest.get('stages_total', 0)}", help="Number of pipeline stages completed out of total. Each stage is a sprint module (ingestion, features, optimization, etc.).")
                c3.metric("Duration", f"{latest.get('duration_seconds', 0):.0f}s", help="Total wall-clock time for the latest pipeline run in seconds.")
                c4.metric("Run ID", latest.get("run_id", "—")[:8], help="Unique identifier for this pipeline run. First 8 characters shown.")

                # Run history
                st.markdown("#### Run History")
                fig = px.scatter(
                    runs_df.tail(50),
                    x="started_at" if "started_at" in runs_df.columns else runs_df.columns[0],
                    y="duration_seconds" if "duration_seconds" in runs_df.columns else runs_df.columns[-1],
                    color="status" if "status" in runs_df.columns else None,
                    title="Pipeline Run Duration",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No pipeline runs recorded yet.")
        except Exception:
            st.info("Pipeline data not available. Run the orchestration engine to populate.")

    # ── Events Tab ───────────────────────────────────────────────────────

    with tab2:
        st.subheader("Event Stream", help="Real-time event log from the orchestration engine — stage starts, completions, errors, and system events.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            events_df = wh.read_table("orchestration_events")
            if events_df is not None and not events_df.empty:
                # Event type distribution
                col1, col2 = st.columns(2)
                with col1:
                    type_counts = events_df["event_type"].value_counts().head(10)
                    fig = px.bar(
                        x=type_counts.index, y=type_counts.values,
                        title="Event Types", labels={"x": "Type", "y": "Count"},
                    )
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    status_counts = events_df["status"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Event Status",
                    )
                    st.plotly_chart(fig, width="stretch")

                # Recent events table
                st.markdown("#### Recent Events")
                st.dataframe(events_df.tail(20), width="stretch")
            else:
                st.info("No events recorded yet.")
        except Exception:
            st.info("Event data not available.")

    # ── Schedule Tab ─────────────────────────────────────────────────────

    with tab3:
        st.subheader("Operational Schedule", help="Planned pipeline tasks organized by frequency. Daily tasks run every trading day, weekly/monthly tasks handle deeper analysis.")

        try:
            from orchestration.scheduling import Scheduler
            scheduler = Scheduler()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Daily Tasks")
                for task in scheduler.get_daily_tasks():
                    st.markdown(f"- {task}")

            with col2:
                st.markdown("#### Weekly Tasks")
                for task in scheduler.get_weekly_tasks():
                    st.markdown(f"- {task}")

            with col3:
                st.markdown("#### Monthly Tasks")
                for task in scheduler.get_monthly_tasks():
                    st.markdown(f"- {task}")
        except Exception:
            st.info("Scheduler not available.")

    # ── SLA Tab ──────────────────────────────────────────────────────────

    with tab4:
        st.subheader("SLA Compliance", help="Service Level Agreement tracking — monitors whether pipeline components complete within their target time windows.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            sla_df = wh.read_table("sla_records")
            if sla_df is not None and not sla_df.empty:
                # Overall compliance
                met = sla_df["met"].sum() if "met" in sla_df.columns else 0
                total = len(sla_df)
                pct = (met / total * 100) if total else 100

                c1, c2, c3 = st.columns(3)
                c1.metric("Compliance", f"{pct:.1f}%", help="Overall SLA compliance rate — percentage of pipeline components that met their time targets.")
                c2.metric("SLA Met", int(met), help="Number of SLA checks that passed within the allowed time window.")
                c3.metric("Breaches", total - int(met), help="Number of SLA violations where a component exceeded its time limit. Investigate root cause for recurring breaches.")

                # Per-component
                if "component" in sla_df.columns:
                    fig = px.bar(
                        sla_df.groupby("component")["met"].mean().reset_index(),
                        x="component", y="met",
                        title="SLA Compliance by Component",
                        labels={"met": "Compliance Rate"},
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No SLA records yet.")
        except Exception:
            st.info("SLA data not available.")

    # ── MLOps Tab ────────────────────────────────────────────────────────

    with tab5:
        st.subheader("ML Operations", help="ML model health monitoring — tracks Rank IC (information coefficient), model grade, and drift over time.")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            model_df = wh.read_table("model_health")
            if model_df is not None and not model_df.empty:
                latest = model_df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", latest.get("model_id", "—"), help="Active model identifier used for signal generation.")
                c2.metric("Rank IC", f"{latest.get('rank_ic', 0):.4f}", help="Rank Information Coefficient — correlation between predicted and actual asset ranks. >0.05 = useful, >0.10 = strong.")
                c3.metric("Grade", latest.get("grade", "—"), help="Model quality grade. A/B = production-ready, C = monitor closely, D/F = consider retraining.")

                # IC over time
                if "rank_ic" in model_df.columns and len(model_df) > 1:
                    fig = px.line(
                        model_df, y="rank_ic",
                        title="Model Rank IC Over Time",
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No model health records yet.")
        except Exception:
            st.info("Model data not available.")

    # ── Governance Tab ───────────────────────────────────────────────────

    with tab6:
        st.subheader("Configuration Governance", help="Version-controlled configuration snapshots — tracks parameter changes across pipeline runs for reproducibility and audit.")

        try:
            from orchestration.governance import GovernanceEngine
            gov = GovernanceEngine()
            snapshots = gov.list_snapshots()

            if snapshots:
                st.metric("Config Snapshots", len(snapshots), help="Total number of configuration snapshots stored. Each snapshot captures the full parameter state for a pipeline run.")
                st.markdown("#### Recent Snapshots")
                st.dataframe(pd.DataFrame(snapshots).tail(20), width="stretch")
            else:
                st.info("No config snapshots yet.")
        except Exception:
            st.info("Governance data not available.")
