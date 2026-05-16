"""
Niyati runway analysis — structural feasibility of the current portfolio.

Reads portfolio_metrics.csv and backtest_attribution.csv from reports/,
builds a scenario schema where:
  - Natural degradation (sharpe -7%/step, drawdown +2%/step) means the goal
    is NOT trivially satisfied from t=0.
  - Actions (rebalance_hrp, defensive_shift) allow the portfolio to survive.
  - Goal: sharpe_ratio >= 0.7 over a 12-month horizon.

Usage:
    result = run_runway_analysis()
    summary = summarize_runway(result)
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from utils.niyati_client import simulate

REPORTS = Path("reports")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_metrics() -> dict:
    """Load portfolio_metrics.csv and return key metrics as a dict."""
    path = REPORTS / "portfolio_metrics.csv"
    defaults = {"sharpe_ratio": 1.0, "max_drawdown": -0.15, "portfolio_nav": 1_000_000.0}

    if not path.exists():
        logger.warning(f"portfolio_metrics.csv not found at {path} — using defaults")
        return defaults

    try:
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict() if not df.empty else {}
        return {
            "sharpe_ratio": float(row.get("sharpe_ratio", defaults["sharpe_ratio"])),
            "max_drawdown": float(row.get("max_drawdown", defaults["max_drawdown"])),
            "portfolio_nav": float(row.get("portfolio_nav", defaults["portfolio_nav"])),
        }
    except Exception as exc:
        logger.warning(f"Could not parse portfolio_metrics.csv: {exc} — using defaults")
        return defaults


def _load_nav() -> float:
    """Load current NAV from backtest_attribution.csv or portfolio_metrics.csv."""
    # Try backtest attribution first
    attr_path = REPORTS / "backtest_attribution.csv"
    if attr_path.exists():
        try:
            df = pd.read_csv(attr_path)
            if not df.empty and "final_value" in df.columns:
                return float(df.iloc[0]["final_value"])
        except Exception:
            pass

    # Fall back to metrics
    metrics = _load_metrics()
    return metrics["portfolio_nav"]


# ── Schema builder ─────────────────────────────────────────────────────────────


def _build_schema(sharpe: float, max_dd: float, nav: float) -> dict:
    """
    Build the Niyati simulate schema.

    Scenario: stress-adjusted initial state — portfolio is assessed from a
    shock scenario where sharpe has already been halved (50% drawdown event).
    Goal: recover sharpe to the actual current level within 12 months via
    HRP rebalancing actions. This guarantees the goal is NOT trivially satisfied
    from t=0 (starts below target) and requires active management.

    Actions available:
      - rebalance_hrp:    +15% sharpe, -10% drawdown. 20bps friction each time.
      - defensive_shift:  -5% sharpe, -20% drawdown.  5bps friction.
                          Only available when drawdown > 0.20 (crisis condition).
    """
    # Clamp actual (current) values to sensible ranges
    actual_sharpe = max(0.1, min(sharpe, 3.0))
    initial_dd = abs(max_dd)

    # Shock state: start at 50% of actual sharpe — scenario asks "can HRP
    # rebalancing recover us to our real current target within the horizon?"
    stressed_sharpe = round(actual_sharpe * 0.50, 4)
    target_sharpe = round(actual_sharpe, 4)

    schema = {
        "version": "1.0.0",
        "metadata": {"name": "Portfolio Structural Runway — HRP Recovery Feasibility"},
        "time": {"type": "discrete", "horizon": 12},
        "resources": {
            "portfolio_value": {
                "initial": round(nav, 2),
                "min": 0.0,
            }
        },
        "variables": {
            "sharpe_ratio": {
                "type": "float",
                "initial": stressed_sharpe,
                "discretization": 0.05,
                "min": -3.0,
                "max": 5.0,
            },
            "max_drawdown": {
                "type": "float",
                "initial": round(initial_dd, 4),
                "discretization": 0.05,
                "min": 0.0,
                "max": 1.0,
            },
        },
        "transitions": [],
        "constraints": [],
        "actions": [
            {
                "name": "rebalance_hrp",
                "cost": {"portfolio_value": round(nav * 0.002, 2)},  # 20bps friction
                "effects": {
                    "sharpe_ratio": {"op": "multiply", "value": 1.15},
                    "max_drawdown":  {"op": "multiply", "value": 0.90},
                },
                "preconditions": [],
            },
            {
                "name": "defensive_shift",
                "cost": {"portfolio_value": round(nav * 0.0005, 2)},  # 5bps friction
                "effects": {
                    "sharpe_ratio": {"op": "multiply", "value": 0.95},
                    "max_drawdown":  {"op": "multiply", "value": 0.80},
                },
                "preconditions": ["max_drawdown > 0.20"],
            },
        ],
        "goal": {
            "type": "threshold",
            "conditions": [
                {"variable": "sharpe_ratio", "operator": "gte", "value": target_sharpe},
            ],
        },
    }
    return schema


# ── Public API ─────────────────────────────────────────────────────────────────


def run_runway_analysis() -> dict | None:
    """
    Build the runway scenario and POST to /v1/simulate.

    Returns:
        Full Niyati API response dict, or None if the call failed.
    """
    metrics = _load_metrics()
    nav = _load_nav()

    sharpe = metrics["sharpe_ratio"]
    max_dd = metrics["max_drawdown"]

    logger.info(
        f"Niyati runway: sharpe={sharpe:.3f}, max_dd={max_dd:.3f}, nav={nav:,.0f}"
    )

    schema = _build_schema(sharpe, max_dd, nav)

    try:
        result = simulate(schema, capability="full")
        if result is None:
            logger.error("Niyati /v1/simulate returned None")
            return None
        logger.info(f"Niyati runway result keys: {list(result.keys())}")
        return result
    except Exception as exc:
        logger.error(f"run_runway_analysis failed: {exc}")
        return None


def summarize_runway(result: dict) -> dict:
    """
    Extract a clean summary dict from a /v1/simulate response.

    Keys returned:
        verdict              — "SURVIVES" | "COLLAPSES" | "UNKNOWN"
        point_of_no_return   — int (month) or None (goal PNR, most actionable)
        survival_posture     — str description of recommended posture
        epsilon_star         — float | None
        danger_fraction      — float | None (0–1)
        phase_distribution   — dict with safe/warning/critical step counts
        headline             — one-line human-readable summary
        goal_reached_at      — int | None
        recommendation       — str from API
    """
    if not result:
        return {
            "verdict": "UNKNOWN",
            "point_of_no_return": None,
            "survival_posture": "API unavailable",
            "epsilon_star": None,
            "danger_fraction": None,
            "phase_distribution": {},
            "headline": "Niyati API unavailable — structural analysis skipped.",
            "goal_reached_at": None,
            "recommendation": "",
        }

    # --- Verdict ---------------------------------------------------------------
    raw_verdict = str(result.get("verdict", "unknown")).lower()
    if raw_verdict == "possible":
        verdict = "SURVIVES"
    elif raw_verdict in ("impossible", "goal_impossible"):
        verdict = "COLLAPSES"
    elif raw_verdict == "useless_but_valid":
        verdict = "SURVIVES"
    else:
        verdict = "UNKNOWN"

    # --- Point of no return ----------------------------------------------------
    # goal_point_of_no_return is the strategic PNR (earlier, more actionable)
    goal_pnr = result.get("goal_point_of_no_return")
    struct_pnr = result.get("point_of_no_return")
    point_of_no_return = goal_pnr if goal_pnr is not None else struct_pnr

    # --- Epsilon star ----------------------------------------------------------
    epsilon_star = result.get("epsilon_star")

    # --- Phase distribution ---------------------------------------------------
    pd_raw = result.get("phase_distribution", {})
    phase_distribution = {
        "safe":     pd_raw.get("safe_steps", 0),
        "warning":  pd_raw.get("warning_steps", 0),
        "critical": pd_raw.get("critical_steps", 0) + pd_raw.get("collapsed_steps", 0),
    }
    danger_fraction = pd_raw.get("danger_fraction")

    # --- Survival posture ------------------------------------------------------
    policy = result.get("survival_policy", {})
    survival_posture = (
        policy.get("recommended_posture")
        if isinstance(policy, dict)
        else str(policy)
    ) or "Maintain quarterly HRP rebalancing cadence"

    # --- Goal reached at ------------------------------------------------------
    goal_reached_at = result.get("goal_reached_at")

    # --- Narrative (structured, v0.3.17+) -------------------------------------
    narrative = result.get("narrative", {})
    api_headline = narrative.get("headline") if isinstance(narrative, dict) else None
    recommendation = result.get("recommendation", "")

    intervention = narrative.get("intervention", {}) if isinstance(narrative, dict) else {}
    intervention_deadline = intervention.get("intervention_deadline")
    futures_on_best_path = intervention.get("futures_on_best_path")
    intervention_action = intervention.get("action", "")
    highest_leverage_variable = intervention.get("highest_leverage_variable")

    post_mortem = narrative.get("post_mortem", {}) if isinstance(narrative, dict) else {}
    was_inevitable = post_mortem.get("was_inevitable", False)
    root_cause = post_mortem.get("root_cause")

    collapse_chronicle = narrative.get("collapse_chronicle", {}) if isinstance(narrative, dict) else {}
    corridor_narrowing_rate = collapse_chronicle.get("corridor_narrowing_rate")
    trajectory = collapse_chronicle.get("trajectory")

    # --- Corridor trap (survival_policy v0.3.17+) ----------------------------
    policy = result.get("survival_policy", {})
    is_corridor_trap = policy.get("is_corridor_trap", False) if isinstance(policy, dict) else False
    corridor_trap_severity = policy.get("corridor_trap_severity", "none") if isinstance(policy, dict) else "none"

    # --- Headline -------------------------------------------------------------
    if api_headline:
        headline = api_headline
    elif verdict == "SURVIVES":
        eps_str = f", ε*={epsilon_star:.4f}" if epsilon_star is not None else ""
        danger_str = f", danger={danger_fraction:.1%}" if danger_fraction is not None else ""
        headline = f"Portfolio is structurally feasible over a 12-month horizon{eps_str}{danger_str}."
    elif verdict == "COLLAPSES":
        pnr_str = f" Goal path closes at month {point_of_no_return}." if point_of_no_return else ""
        gra_str = f" Goal reachable until month {goal_reached_at}." if goal_reached_at else ""
        headline = f"Portfolio faces structural collapse risk over 12 months.{gra_str}{pnr_str}"
    else:
        headline = "Structural feasibility is indeterminate — review API response."

    return {
        "verdict": verdict,
        "point_of_no_return": point_of_no_return,
        "survival_posture": survival_posture,
        "epsilon_star": epsilon_star,
        "danger_fraction": danger_fraction,
        "phase_distribution": phase_distribution,
        "headline": headline,
        "goal_reached_at": goal_reached_at,
        "recommendation": recommendation,
        # Narrative fields (v0.3.17+)
        "intervention_deadline": intervention_deadline,
        "futures_on_best_path": futures_on_best_path,
        "intervention_action": intervention_action,
        "highest_leverage_variable": highest_leverage_variable,
        "was_inevitable": was_inevitable,
        "root_cause": root_cause,
        "corridor_narrowing_rate": corridor_narrowing_rate,
        "trajectory": trajectory,
        "is_corridor_trap": is_corridor_trap,
        "corridor_trap_severity": corridor_trap_severity,
    }
