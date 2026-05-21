"""
Structural Health — Niyati feasibility analysis page.

7th dashboard page showing:
  - Top row:    verdict, epsilon_star, danger_fraction, point_of_no_return
  - Middle:     future-width timeline chart (left) + phase bar chart (right)
  - Bottom left:  critical_decisions table
  - Bottom right: adversarial-allocation results bar chart
  - Banner:     survival policy / recommended posture
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path when run standalone
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Data loaders (cached) ──────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner=False)
def _load_runway() -> tuple[dict | None, dict | None]:
    """Run runway analysis and return (raw_result, summary)."""
    try:
        from validation.niyati_runway import run_runway_analysis, summarize_runway
        result = run_runway_analysis()
        summary = summarize_runway(result) if result else summarize_runway(None)
        return result, summary
    except Exception as exc:
        st.warning(f"Runway analysis unavailable: {exc}")
        return None, None


@st.cache_data(ttl=600, show_spinner=False)
def _load_adversarial() -> dict | None:
    """Run adversarial allocation analysis."""
    try:
        from validation.niyati_stress import run_adversarial_allocation
        return run_adversarial_allocation()
    except Exception as exc:
        st.warning(f"Adversarial allocation unavailable: {exc}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_survival() -> dict | None:
    """Run adversarial survival verdict (safe/critical/doomed under crash budget)."""
    try:
        from validation.niyati_stress import run_adversarial_survival
        return run_adversarial_survival(crash_budget=0.006)
    except Exception as exc:
        st.warning(f"Adversarial survival unavailable: {exc}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _load_fragility_dashboard() -> dict | None:
    """Run fragility check at current portfolio point."""
    try:
        from validation.niyati_stress import run_fragility_check
        import pandas as pd
        from pathlib import Path
        path = Path("reports/portfolio_metrics.csv")
        sharpe, dd = 1.0, -0.15
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                sharpe = float(df.iloc[0].get("sharpe_ratio", 1.0))
                dd = float(df.iloc[0].get("max_drawdown", -0.15))
        return run_fragility_check(sharpe, dd)
    except Exception as exc:
        st.warning(f"Fragility check unavailable: {exc}")
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _verdict_color(verdict: str) -> str:
    if verdict == "SURVIVES":
        return "normal"
    if verdict == "COLLAPSES":
        return "inverse"
    return "off"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.1%}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_float(val, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _build_timeline_chart(timeline: list[dict]) -> go.Figure | None:
    """Area chart of future_width (epsilon or feasibility margin) over horizon."""
    if not timeline:
        return None

    rows = []
    for step in timeline:
        t = step.get("t")
        fw = step.get("future_width")
        phase = str(step.get("phase_classification", step.get("phase", "unknown"))).lower()
        if t is not None and fw is not None:
            rows.append({"Month": int(t), "Future Width": float(fw), "Phase": phase})

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("Month")

    # Color by phase
    color_map = {"safe": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c", "danger": "#e74c3c"}
    fig = px.area(
        df,
        x="Month",
        y="Future Width",
        color="Phase",
        color_discrete_map=color_map,
        title="Structural Feasibility Width Over Horizon",
        labels={"Future Width": "ε (feasibility margin)", "Month": "Month"},
    )
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _build_phase_chart(phase_distribution: dict) -> go.Figure | None:
    """Bar chart of safe / warning / critical phase counts."""
    if not phase_distribution:
        return None

    phases = ["safe", "warning", "critical"]
    counts = [phase_distribution.get(p, 0) for p in phases]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig = go.Figure(
        go.Bar(
            x=[p.capitalize() for p in phases],
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Phase Distribution (months)",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis_title="Months",
        xaxis_title="Phase",
    )
    return fig


def _build_critical_decisions_table(result: dict) -> pd.DataFrame:
    """Extract critical_decisions from the runway result."""
    if not result:
        return pd.DataFrame()

    # critical_decisions is a dict with "ranked_transitions" list
    cd = result.get("critical_decisions", {})
    transitions = cd.get("ranked_transitions", []) if isinstance(cd, dict) else []
    if not transitions:
        return pd.DataFrame()

    rows = []
    for d in transitions:
        rows.append(
            {
                "State": d.get("state", "—"),
                "Kappa": _fmt_float(d.get("kappa"), 4),
                "Impact Ratio": _fmt_float(d.get("impact_ratio"), 3),
                "Status": str(d.get("status", "—")).replace("_", " ").capitalize(),
            }
        )
    return pd.DataFrame(rows)


def _build_adversarial_chart(adv_result: dict) -> go.Figure | None:
    """Bar chart of attack budget assigned per asset class (red if collapses)."""
    if not adv_result:
        return None

    allocations = adv_result.get("system_allocations", adv_result.get("allocations", []))
    if not allocations:
        return None

    rows = []
    for alloc in allocations:
        name = alloc.get("name", "unknown")
        budget = alloc.get("attack_budget_assigned", 0.0)
        survives = alloc.get("survives", True)
        rows.append({"Asset Class": name, "Attack Budget": float(budget), "Survives": bool(survives)})

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("Attack Budget", ascending=False)
    colors = ["#e74c3c" if not s else "#3498db" for s in df["Survives"]]

    fig = go.Figure(
        go.Bar(
            x=df["Asset Class"],
            y=df["Attack Budget"],
            marker_color=colors,
            text=[f"{'COLLAPSES' if not s else 'survives'}" for s in df["Survives"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Adversarial Budget Allocation by Asset Class",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis_title="Budget Assigned",
        xaxis_title="Asset Class",
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────


def render() -> None:
    st.header("Structural Health")
    st.caption("Niyati feasibility engine — structural runway & adversarial stress analysis")

    with st.spinner("Running structural analysis via Niyati API…"):
        raw_result, summary = _load_runway()
        adv_result = _load_adversarial()
        survival_result = _load_survival()
        fragility_result = _load_fragility_dashboard()

    if summary is None:
        st.error(
            "Niyati API is unavailable. Ensure the API key is configured and "
            "https://api.causalorlabs.com is reachable."
        )
        return

    # ── Survival verdict banner (most prominent) ─────────────────────────────
    sv = survival_result.get("survival_verdict", "unknown").lower() if survival_result else "unknown"
    sv_margin = survival_result.get("survival_margin") if survival_result else None
    pi_regime = survival_result.get("pi_regime", "") if survival_result else ""

    sv_colors = {"safe": "success", "critical": "warning", "doomed": "error"}
    sv_emoji = {"safe": "🟢", "critical": "🟡", "doomed": "🔴"}.get(sv, "⚪")
    sv_label = sv.upper()
    sv_margin_str = f"  (margin: {sv_margin:+.4f})" if sv_margin is not None else ""
    solo_omega = survival_result.get("solo_goal_omega") if survival_result else None
    stressed_omega = survival_result.get("stressed_goal_omega") if survival_result else None
    omega_str = f"  |  goal states: {solo_omega}→{stressed_omega} under crash" if (solo_omega is not None and stressed_omega is not None) else ""
    sv_text = f"**Crash Survival: {sv_emoji} {sv_label}**{sv_margin_str}{omega_str} — {pi_regime.replace('_', ' ')}"

    if sv == "safe":
        st.success(sv_text)
    elif sv == "critical":
        st.warning(sv_text)
    else:
        st.error(sv_text)

    # ── Top row: 5 KPI cards ─────────────────────────────────────────────────
    verdict = summary.get("verdict", "UNKNOWN")
    epsilon_star = summary.get("epsilon_star")
    danger_fraction = summary.get("danger_fraction")
    pnr = summary.get("point_of_no_return")
    collapse_risk = fragility_result.get("collapse_risk") if fragility_result else None
    frag_thickness = fragility_result.get("thickness") if fragility_result else None

    verdict_emoji = "✅" if verdict == "SURVIVES" else ("❌" if verdict == "COLLAPSES" else "⚠️")
    pnr_display = f"Month {pnr}" if pnr is not None else "None"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Runway Verdict",
        f"{verdict_emoji} {verdict}",
    )
    c2.metric(
        "Epsilon Star (ε*)",
        _fmt_float(epsilon_star, 4),
        help="Feasibility margin — distance to structural boundary",
    )
    c3.metric(
        "Danger Fraction",
        _fmt_pct(danger_fraction),
        help="Fraction of horizon steps classified as critical",
    )
    c4.metric(
        "Point of No Return",
        pnr_display,
        help="First month where structural collapse becomes irreversible",
    )
    frag_delta = None
    if collapse_risk is not None:
        frag_delta = "fragile" if collapse_risk > 0.15 else "stable"
    c5.metric(
        "Fragility (collapse risk)",
        _fmt_pct(collapse_risk),
        delta=frag_delta,
        delta_color="inverse",
        help="Instant structural distance to edge in market-stress direction",
    )

    st.divider()

    # ── Headline banner ──────────────────────────────────────────────────────
    headline = summary.get("headline", "")
    if headline:
        if verdict == "SURVIVES":
            st.success(headline)
        elif verdict == "COLLAPSES":
            st.error(headline)
        else:
            st.warning(headline)

    # ── Intervention deadline alert (v0.3.17+ narrative) ─────────────────────
    intervention_deadline = summary.get("intervention_deadline")
    intervention_action = summary.get("intervention_action", "")
    futures_on_best_path = summary.get("futures_on_best_path")
    root_cause = summary.get("root_cause")
    is_corridor_trap = summary.get("is_corridor_trap", False)

    if intervention_deadline is not None:
        deadline_msg = f"**Act by Month {intervention_deadline}**"
        if futures_on_best_path is not None:
            deadline_msg += f"  —  {int(futures_on_best_path)} futures preserved on recommended path"
        if root_cause:
            deadline_msg += f"  |  Root cause: `{root_cause}`"
        if intervention_action:
            deadline_msg += f"\n\n_{intervention_action}_"
        if is_corridor_trap:
            st.error(deadline_msg)
        elif verdict == "COLLAPSES":
            st.warning(deadline_msg)
        else:
            st.info(deadline_msg)

    # ── Fragility detail row ─────────────────────────────────────────────────
    if fragility_result:
        thickness = fragility_result.get("thickness")
        anisotropy = fragility_result.get("anisotropy_ratio")
        kappa = fragility_result.get("kappa")
        with st.expander("Fragility Detail (pre-rebalance gate)", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Thickness (τ*)", _fmt_float(thickness, 4),
                       help="Fraction of operating range to nearest constraint")
            fc2.metric("Kappa (κ)", _fmt_float(kappa, 4),
                       help="Instantaneous collapse rate at current point")
            fc3.metric("Anisotropy", _fmt_float(anisotropy, 2),
                       help=">10× means corridor trap — one direction is nearly locked")

    # ── Middle row: timeline chart + phase distribution ──────────────────────
    timeline = raw_result.get("timeline", []) if raw_result else []
    phase_distribution = summary.get("phase_distribution", {})

    col_left, col_right = st.columns(2)

    with col_left:
        timeline_fig = _build_timeline_chart(timeline)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Timeline data not available in API response.")

    with col_right:
        phase_fig = _build_phase_chart(phase_distribution)
        if phase_fig:
            st.plotly_chart(phase_fig, use_container_width=True)
        else:
            st.info("Phase distribution data not available.")

    st.divider()

    # ── Bottom row: critical decisions + adversarial allocation ──────────────
    col_bl, col_br = st.columns(2)

    with col_bl:
        st.subheader("Critical Decisions")
        decisions_df = _build_critical_decisions_table(raw_result)
        if not decisions_df.empty:
            st.dataframe(
                decisions_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No critical decision data in API response.")

    with col_br:
        st.subheader("Adversarial Allocation")
        if adv_result:
            adv_fig = _build_adversarial_chart(adv_result)
            if adv_fig:
                st.plotly_chart(adv_fig, use_container_width=True)
            else:
                # Show raw summary if chart data not available
                # Find first system to collapse
                allocs = adv_result.get("system_allocations", [])
                first_collapse = next(
                    (a["name"] for a in allocs if not a.get("survives", True)), "None"
                )
                n_collapsed = adv_result.get("collapsed_count", "N/A")
                pi_unified = adv_result.get("pi_unified", "N/A")
                st.metric("First Collapse", str(first_collapse))
                st.metric("Systems Collapsed", str(n_collapsed))
                st.metric("Pi Unified", _fmt_float(pi_unified, 4))
        else:
            st.warning("Adversarial allocation data unavailable.")

    st.divider()

    # ── Survival policy banner ───────────────────────────────────────────────
    posture = summary.get("survival_posture", "Maintain quarterly HRP rebalancing cadence")
    st.subheader("Recommended Survival Posture")
    st.info(f"**{posture}**")

    # ── Raw data expander ────────────────────────────────────────────────────
    with st.expander("Raw API Response", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Runway Result**")
            st.json(raw_result or {})
        with col_b:
            st.write("**Adversarial Allocation**")
            st.json(adv_result or {})
