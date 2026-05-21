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
    st.header("🏭 Operations & Workflow Intelligence")
    st.caption("Pipeline orchestration, scheduling, SLA compliance, and MLOps")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pipeline", "Events", "Schedule",
        "SLA", "MLOps", "Governance",
    ])

    # ── Pipeline Tab ─────────────────────────────────────────────────────

    with tab1:
        st.subheader("Workflow Pipeline")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            runs_df = wh.read_table("workflow_runs")
            if runs_df is not None and not runs_df.empty:
                # Latest run summary
                latest = runs_df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", latest.get("status", "unknown"))
                c2.metric("Stages", f"{latest.get('stages_completed', 0)}/{latest.get('stages_total', 0)}")
                c3.metric("Duration", f"{latest.get('duration_seconds', 0):.0f}s")
                c4.metric("Run ID", latest.get("run_id", "—")[:8])

                # Run history
                st.markdown("#### Run History")
                fig = px.scatter(
                    runs_df.tail(50),
                    x="started_at" if "started_at" in runs_df.columns else runs_df.columns[0],
                    y="duration_seconds" if "duration_seconds" in runs_df.columns else runs_df.columns[-1],
                    color="status" if "status" in runs_df.columns else None,
                    title="Pipeline Run Duration",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No pipeline runs recorded yet.")
        except Exception:
            st.info("Pipeline data not available. Run the orchestration engine to populate.")

    # ── Events Tab ───────────────────────────────────────────────────────

    with tab2:
        st.subheader("Event Stream")

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
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    status_counts = events_df["status"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Event Status",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recent events table
                st.markdown("#### Recent Events")
                st.dataframe(events_df.tail(20), use_container_width=True)
            else:
                st.info("No events recorded yet.")
        except Exception:
            st.info("Event data not available.")

    # ── Schedule Tab ─────────────────────────────────────────────────────

    with tab3:
        st.subheader("Operational Schedule")

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
        st.subheader("SLA Compliance")

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
                c1.metric("Compliance", f"{pct:.1f}%")
                c2.metric("SLA Met", int(met))
                c3.metric("Breaches", total - int(met))

                # Per-component
                if "component" in sla_df.columns:
                    fig = px.bar(
                        sla_df.groupby("component")["met"].mean().reset_index(),
                        x="component", y="met",
                        title="SLA Compliance by Component",
                        labels={"met": "Compliance Rate"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No SLA records yet.")
        except Exception:
            st.info("SLA data not available.")

    # ── MLOps Tab ────────────────────────────────────────────────────────

    with tab5:
        st.subheader("ML Operations")

        try:
            from warehouse import get_warehouse
            wh = get_warehouse()
            model_df = wh.read_table("model_health")
            if model_df is not None and not model_df.empty:
                latest = model_df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", latest.get("model_id", "—"))
                c2.metric("Rank IC", f"{latest.get('rank_ic', 0):.4f}")
                c3.metric("Grade", latest.get("grade", "—"))

                # IC over time
                if "rank_ic" in model_df.columns and len(model_df) > 1:
                    fig = px.line(
                        model_df, y="rank_ic",
                        title="Model Rank IC Over Time",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No model health records yet.")
        except Exception:
            st.info("Model data not available.")

    # ── Governance Tab ───────────────────────────────────────────────────

    with tab6:
        st.subheader("Configuration Governance")

        try:
            from orchestration.governance import GovernanceEngine
            gov = GovernanceEngine()
            snapshots = gov.list_snapshots()

            if snapshots:
                st.metric("Config Snapshots", len(snapshots))
                st.markdown("#### Recent Snapshots")
                st.dataframe(pd.DataFrame(snapshots).tail(20), use_container_width=True)
            else:
                st.info("No config snapshots yet.")
        except Exception:
            st.info("Governance data not available.")
