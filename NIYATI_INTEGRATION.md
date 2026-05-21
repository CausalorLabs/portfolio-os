# Niyati × Portfolio-OS — Integration Master Document

> **Context**: Portfolio-OS was built as Phase 1 of a two-phase AI-Hybrid Portfolio system
> following the ML4T (Stefan Jansen) workflow. Phase 1 (backtesting) is complete.
> Phase 2 (live execution & paper trading) is the active build target.
> This document maps Niyati's role across both phases — what it adds to what's already
> built, and how it becomes load-bearing infrastructure in Phase 2.

---

## Project Architecture (Original Brief)

```
[ Raw Multi-Asset Data ] ──► [ Feature Engineering ] ──► [ LightGBM / LLM Sentiment ]
                                                                       │
                                                          (AI Forward-Return Views)
                                                                       │
                                                                       ▼
[ Live/Historical Holdings ] ──► [ HRP Risk Matrix ] ──► [ Black-Litterman Core ]
                                                                       │
                                                            (Optimal Weights)
                                                                       │
                                              ┌────────────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │     Phase 1: Backtester       │  ← BUILT
                              │     Phase 2: Paper Trader     │  ← TO BUILD
                              └───────────────────────────────┘
```

### Build Status

| Component | Status | Module |
|-----------|--------|--------|
| Multi-asset ingestion (Yahoo + MFAPI + Fixed Income) | ✅ Built | `ingestion/` |
| FX normalization (USD→INR) | ✅ Built | `fx/` |
| Analytics & risk metrics (CAGR, Sharpe, Drawdown) | ✅ Built | `analytics/` |
| Feature engineering (momentum, vol, trend, mean-rev) | ✅ Built | `features/` |
| HRP optimizer | ✅ Built | `optimization/hrp.py` |
| Signal-tilted allocator | ✅ Built | `optimization/allocator.py` |
| Friction-aware backtester (FIFO taxes, STT, GST) | ✅ Built | `backtests/` |
| Walk-forward / Monte Carlo / stress validation | ✅ Built | `validation/` |
| Streamlit dashboard (6 pages) | ✅ Built | `dashboard/` |
| **LightGBM ML engine** | ❌ Not built | — |
| **Black-Litterman framework** | ❌ Not built | — |
| **LLM sentiment parser** | ❌ Not built | — |
| **Live portfolio state monitor** | ❌ Not built | — |
| **Drift watcher daemon** | ❌ Not built | — |
| **Paper trading virtual wallet** | ❌ Not built | — |

---

## What Niyati Is

Niyati is the structural feasibility engine from Causalor Labs. Given a declarative model of your
system — resources, variables, actions, constraints, goal — it computes:

| Output | Meaning |
|--------|---------|
| `future_width` (Ω) | Count of goal-reaching futures that still exist at each step |
| `risk_spike` (κ) | Rate of optionality collapse — bits per step |
| `thickness` (τ*) | Geometric safety margin in the most constrained direction |
| `anisotropy` (A) | Structural brittleness — freedom in one direction, none in others |
| `point_of_no_return` | Exact step where the goal becomes structurally impossible |
| `recommended_path` | Max-bottleneck trajectory: the path that keeps the most options open |
| `critical_decisions` | Every branch point ranked by structural impact |
| `survival_policy` | Operational posture: `diversify` / `consolidate` / `hold` |
| `epsilon_star` (ε*) | Minimum perturbation that destroys goal-reachability |
| `verdict` | `possible` / `impossible` — a certificate, not a probability |

**API base**: `https://api.causalorlabs.com`

| Header | Required | Value |
|--------|----------|-------|
| `Content-Type` | Always | `application/json` |
| `X-Niyati-Version` | Always | `0.3.17` |
| `X-Niyati-Key` | `/v1/solve/*` and keyed `/v1/simulate` | your API key |

**Endpoints used in this integration**:

| Endpoint | Auth | Use |
|----------|------|-----|
| `POST /v1/simulate` | optional key | Full simulation — runway, stress, rebalance gate |
| `POST /v1/solve/fragility` | key | Fast single-point structural check — paper trade gate |
| `POST /v1/solve/multiagent` | key | BL view compatibility — cooperation deficit |
| `POST /v1/solve/adversarial` | key | Adversarial stress — survival verdict under attacker |

**Dependency** — add to `requirements.txt`:
```
requests>=2.31.0
```
No SDK required. All calls are plain HTTP POST with JSON payloads.

---

## Where Niyati Fits in the ML4T Pipeline

The original brief calls HRP + Black-Litterman "The Brakes" — the math layer that constrains
the AI's predictions before they become trades. Niyati operates one level deeper: it validates
whether the *output of The Brakes* produces a plan that is structurally survivable.

```
[ LightGBM Views ] ──► [ Black-Litterman ] ──► [ Optimal Weights ]
                                                         │
                                            ┌────────────▼────────────┐
                                            │       NIYATI            │  ← NEW LAYER
                                            │  Is this plan still     │
                                            │  structurally reachable?│
                                            │  When does it collapse? │
                                            └────────────┬────────────┘
                                                         │
                                          ┌──────────────▼──────────────┐
                                          │  Phase 1: Execute Backtest  │
                                          │  Phase 2: Execute Paper Trade│
                                          └─────────────────────────────┘
```

HRP + BL tell you: *"These are the optimal weights."*
Niyati tells you: *"Here is whether the trajectory implied by those weights stays feasible
over your planning horizon — and exactly when it stops being feasible."*

The two are not redundant. They answer different questions on different timescales.

---

## Integration Map

---

### Integration 1: Black-Litterman Views as Niyati Agents
**Pipeline stage**: Between LightGBM Views and BL core (to be built)
**Niyati endpoint**: `POST /v1/solve/multiagent` + `POST /v1/solve/adversarial`
**Phase**: Feeds into Phase 1 and Phase 2

#### The Problem

Black-Litterman requires specifying a "P matrix" — a set of views on which assets will
outperform. When those views come from LightGBM, they carry model uncertainty. If two
views conflict (e.g., LightGBM says RELIANCE outperforms while sentiment says it underperforms),
the BL blending obscures the conflict rather than surfacing it.

Niyati's multi-agent solver models each view as an agent with a budget. It computes
`optionality_loss` — how much structural optionality each view destroys when combined —
and `cooperation_deficit` — the fraction of futures lost when views act independently
rather than jointly.

#### Schema Translation

```python
# optimization/niyati_view_validator.py

from utils.niyati_client import solve_multiagent


def validate_bl_views(
    views: list,                     # [{"asset": "AAPL", "expected_return": 0.12, "confidence": 0.7}]
    base_portfolio_value_inr: float,
    current_sharpe: float,
) -> dict:
    """
    Model each BL view as an agent. Compute structural interference between views.
    Returns: optionality_loss per view, cooperation_deficit, recommended view subset.
    """
    agents = [
        {
            "id": f"view_{v['asset'].replace('.', '_')}",
            "budget": base_portfolio_value_inr * abs(v["expected_return"]) * v["confidence"],
            "role": "defender" if v["expected_return"] > 0 else "attacker",
            "attacker_strength": 1 - v["confidence"],
            "consumed_states": ["sharpe_ratio"],
        }
        for v in views
    ]

    schema = {
        "version": "1.0.0",
        "metadata": {"name": "BL View Compatibility Check"},
        "time": {"type": "discrete", "horizon": 6},
        "resources": {
            "portfolio_value": {
                "initial": base_portfolio_value_inr,
                "min": 0.0,
                "max": base_portfolio_value_inr * 2,
            }
        },
        "variables": {
            "sharpe_ratio": {
                "type": "float",
                "initial": round(current_sharpe, 2),
                "min": -2.0,
                "max": 3.0,
                "discretization": 0.1,
            }
        },
        "actions": [
            {
                "name": "blend_views",
                "cost": {"portfolio_value": 0},
                "effects": {"sharpe_ratio": {"op": "multiply", "value": 1.05}},
                "preconditions": [],
            }
        ],
        "transitions": [],
        "constraints": [],
        "goal": {
            "type": "threshold",
            "conditions": [{"variable": "sharpe_ratio", "operator": "gte", "value": 0.8}],
        },
        "theorem_inputs": {
            "agents": agents,
            "coupling_gamma": 0.10,
        },
    }

    result = solve_multiagent(schema)   # POST /v1/solve/multiagent

    multi_agent = result.get("multi_agent_analysis", {})
    cooperation_deficit = multi_agent.get("competitive", {}).get("cooperation_deficit", None)

    return {
        "verdict": result.get("verdict"),
        "cooperation_deficit": cooperation_deficit,
        "agent_reports": multi_agent.get("reports", []),
        "recommendation": (
            "views are structurally compatible — proceed to BL blend"
            if (cooperation_deficit or 1) < 0.20
            else "high cooperation deficit — prune conflicting views before BL"
        ),
    }
```

**Decision rule**: If `cooperation_deficit > 0.20`, drop the lowest-confidence conflicting
view before passing to Black-Litterman. This prevents BL from averaging incompatible views
into a noise-amplified weight.

---

### Integration 2: Portfolio Runway Monitor
**Pipeline stage**: After BL produces optimal weights; before executing backtest or paper trade
**Niyati endpoint**: `POST /v1/simulate`
**Phase**: Both

#### The Problem

The current research score (`validation/research_score.py`) grades the strategy backward —
it scores past performance. There is no forward model that asks: *given current portfolio
state, how many months of structural runway remain before the Sharpe target becomes
impossible?*

This is the core gap between what the strategy *has done* and whether it *can keep doing it*.

#### Schema Translation

```python
# validation/niyati_runway.py

import os
import pandas as pd
from utils.niyati_client import simulate


def build_runway_schema(
    portfolio_value_inr: float,
    current_sharpe: float,
    current_drawdown: float,      # absolute value, e.g. 0.18 for -18%
    monthly_friction_cost_inr: float,
    target_sharpe: float = 0.8,
    horizon_months: int = 24,
) -> dict:
    """
    Translate Portfolio-OS live state into a Niyati feasibility schema.

    Resources:
      portfolio_value  — total INR NAV, consumed by rebalancing friction
    Variables:
      sharpe_ratio     — trailing 90-day Sharpe, the strategic health gauge
      max_drawdown     — current peak-to-trough, bounds downside posture
    Actions:
      rebalance_hrp    — costs friction, improves expected Sharpe (via HRP reset)
      hold             — no cost, Sharpe drifts by market mean-reversion
      reduce_equity    — defensive allocation shift on high drawdown
    Goal:
      Sharpe >= target for the full horizon
    """
    return {
        "version": "1.0.0",
        "metadata": {"name": "Portfolio Runway — 24mo"},
        "time": {"type": "discrete", "horizon": horizon_months},
        "resources": {
            "portfolio_value": {
                "initial": portfolio_value_inr,
                "min": 0.0,
                "max": portfolio_value_inr * 3.0,
            }
        },
        "variables": {
            "sharpe_ratio": {
                "type": "float",
                "initial": round(max(-2.0, min(3.0, current_sharpe)), 1),
                "min": -2.0,
                "max": 3.0,
                "discretization": 0.1,
            },
            "max_drawdown": {
                "type": "float",
                "initial": round(min(0.95, abs(current_drawdown)), 2),
                "min": 0.0,
                "max": 1.0,
                "discretization": 0.05,
            },
        },
        "actions": [
            {
                "name": "rebalance_to_hrp",
                "cost": {"portfolio_value": monthly_friction_cost_inr},
                "effects": {
                    "sharpe_ratio": {"op": "multiply", "value": 1.08},
                    "max_drawdown": {"op": "multiply", "value": 0.95},
                },
                "preconditions": [
                    f"portfolio_value > {monthly_friction_cost_inr * 3:.0f}"
                ],
            },
            {
                "name": "hold",
                "cost": {"portfolio_value": 0.0},
                "effects": {
                    "sharpe_ratio": {"op": "multiply", "value": 0.98},
                },
                "preconditions": [],
            },
            {
                "name": "defensive_shift",
                "cost": {"portfolio_value": monthly_friction_cost_inr * 0.5},
                "effects": {
                    "sharpe_ratio": {"op": "multiply", "value": 0.93},
                    "max_drawdown": {"op": "multiply", "value": 0.78},
                },
                "preconditions": ["max_drawdown > 0.20"],
            },
        ],
        "transitions": [],
        "constraints": [],
        "goal": {
            "type": "threshold",
            "conditions": [
                {"variable": "sharpe_ratio", "operator": "gte", "value": target_sharpe}
            ],
        },
    }


def run_runway_from_pipeline_outputs() -> dict:
    """
    Read Portfolio-OS pipeline outputs and feed directly into Niyati.
    Run after app.py completes.
    """
    metrics  = pd.read_csv("reports/portfolio_metrics.csv").set_index("metric")["value"].to_dict()
    friction = pd.read_csv("reports/backtest_attribution.csv")

    portfolio_value_inr  = float(metrics.get("portfolio_value_inr", 5_000_000))
    current_sharpe       = float(metrics.get("sharpe_ratio", 0.0))
    current_drawdown     = abs(float(metrics.get("max_drawdown", 0.0)))
    friction_drag_annual = float(friction["friction_drag_pct"].mean()) / 100
    monthly_friction     = portfolio_value_inr * (friction_drag_annual / 12)

    schema = build_runway_schema(
        portfolio_value_inr=portfolio_value_inr,
        current_sharpe=current_sharpe,
        current_drawdown=current_drawdown,
        monthly_friction_cost_inr=monthly_friction,
    )

    return simulate(schema, capability="full")   # POST /v1/simulate
```

**Key outputs to act on**:

```python
result = run_runway_from_pipeline_outputs()

verdict          = result["verdict"]                        # "possible" | "impossible"
pnr_month        = result["point_of_no_return"]             # e.g. month 17
posture          = result["survival_policy"]["recommended_posture"]  # "diversify" | "hold"
critical_months  = result["critical_decisions"]["ranked_transitions"]
epsilon_star     = result["timeline"][-1].get("thickness")  # current structural margin
danger_fraction  = result["phase_distribution"]["danger_fraction"]
```

---

### Integration 3: Rebalance Signal Validation (Phase 2 Core)
**Pipeline stage**: Phase 2 — between drift watcher trigger and paper trade execution
**Niyati endpoint**: `POST /v1/simulate` + `POST /v1/solve/trajectory`
**Phase**: Phase 2 only

#### The Problem

The Phase 2 drift watcher (to be built) triggers a rebalance when any asset drifts beyond
the 5% threshold (`optimization/rebalance.py`). The current drift logic is binary: drift >
threshold → rebalance. It says nothing about *whether the rebalance itself is structurally
sound* — whether the post-rebalance portfolio enters a better or worse structural corridor.

Niyati turns this into a scored decision: before executing a paper trade, evaluate all
candidate allocations and select the one that maximizes `bottleneck_future_width`.

```python
# Phase 2 module: live/niyati_rebalance_gate.py

from typing import Dict
from utils.niyati_client import simulate


def gate_rebalance_signal(
    drift_signal: dict,                              # from optimization/rebalance.py
    current_weights: Dict[str, float],
    candidate_allocations: Dict[str, Dict[str, float]],  # strategy name → weights
    portfolio_value_inr: float,
    current_sharpe: float,
) -> dict:
    """
    Before paper trader executes a drift-triggered rebalance, score each
    candidate allocation via Niyati. Return the structurally best one.

    Args:
        drift_signal:          Output from check_rebalance_needed() in rebalance.py
        candidate_allocations: e.g. {"hrp": {...}, "hrp_signal_tilt": {...}, "equal_weight": {...}}

    Returns:
        {
            "approved": bool,
            "best_allocation": str,
            "scores": [...],
            "niyati_posture": str,
            "epsilon_star": float,
        }
    """
    if not drift_signal.get("should_rebalance"):
        return {"approved": False, "reason": "drift below threshold — no action needed"}

    scores = []
    for alloc_name, weights in candidate_allocations.items():
        turnover = sum(
            abs(weights.get(t, 0) - current_weights.get(t, 0))
            for t in set(weights) | set(current_weights)
        )
        friction_est = portfolio_value_inr * turnover * 0.009

        schema = {
            "version": "1.0.0",
            "metadata": {"name": f"Rebalance Gate: {alloc_name}"},
            "time": {"type": "discrete", "horizon": 12},
            "resources": {
                "cash": {"initial": portfolio_value_inr, "min": 0.0, "max": portfolio_value_inr * 2}
            },
            "variables": {
                "sharpe": {
                    "type": "float",
                    "initial": round(current_sharpe, 1),
                    "min": -1.0, "max": 3.0, "discretization": 0.1,
                },
                "turnover_ytd": {
                    "type": "float", "initial": 0.0,
                    "min": 0.0, "max": 4.0, "discretization": 0.1,
                },
            },
            "actions": [
                {
                    "name": f"execute_{alloc_name}",
                    "cost": {"cash": friction_est},
                    "effects": {
                        "sharpe": {"op": "multiply", "value": 1.05},
                        "turnover_ytd": {"op": "add", "value": round(turnover, 2)},
                    },
                    "preconditions": ["cash > 0", "turnover_ytd < 2.0"],
                    "valid_from": 0,
                    "valid_until": 0,
                },
                {
                    "name": "hold",
                    "cost": {"cash": 0},
                    "effects": {"sharpe": {"op": "multiply", "value": 0.99}},
                    "preconditions": [],
                },
            ],
            "transitions": [], "constraints": [],
            "goal": {
                "type": "threshold",
                "conditions": [{"variable": "sharpe", "operator": "gte", "value": 0.8}],
            },
        }

        result = simulate(schema, capability="full")   # POST /v1/simulate
        timeline = result.get("timeline", [])

        scores.append({
            "allocation": alloc_name,
            "verdict": result.get("verdict"),
            "min_thickness": min((t.get("thickness", 0) for t in timeline), default=0),
            "peak_kappa": max((t.get("risk_spike", 0) for t in timeline), default=0),
            "posture": result.get("survival_policy", {}).get("recommended_posture", "hold"),
            "point_of_no_return": result.get("point_of_no_return"),
        })

    possible = [s for s in scores if s["verdict"] == "possible"]
    if not possible:
        return {
            "approved": False,
            "reason": "all candidate allocations are structurally infeasible — hold",
            "scores": scores,
        }

    best = max(possible, key=lambda s: s["min_thickness"])
    return {
        "approved": True,
        "best_allocation": best["allocation"],
        "niyati_posture": best["posture"],
        "epsilon_star": best["min_thickness"],
        "point_of_no_return": best["point_of_no_return"],
        "scores": scores,
    }
```

---

### Integration 4: Paper Trading Guardrail (Phase 2)
**Pipeline stage**: Phase 2 — inside the paper trading virtual wallet before each simulated trade
**Niyati endpoint**: `POST /v1/solve/fragility`
**Phase**: Phase 2 only

#### The Problem

The planned paper trading virtual wallet (Phase 2.3 in the original brief) records
buy/sell executions at live market prices. Without a structural check, the wallet will
execute every signal regardless of whether the system is in a `Critical` or `Collapsed`
phase. This means the paper wallet accumulates trades in deteriorating market conditions
that a real manager would pause.

Niyati's `fragility` solver runs a single-point structural check in milliseconds —
fast enough to gate every paper trade before it is logged.

```python
# Phase 2 module: live/paper_wallet.py  (excerpt showing Niyati gate)

from dataclasses import dataclass
from datetime import datetime
from utils.niyati_client import solve_fragility


@dataclass
class PaperTrade:
    ticker: str
    direction: str       # "BUY" | "SELL"
    quantity: float
    price_inr: float
    signal_source: str   # "drift_watcher" | "lightgbm" | "manual"
    timestamp: datetime


def is_structurally_safe_to_trade(
    portfolio_value_inr: float,
    current_sharpe: float,
    trade: PaperTrade,
    fragility_threshold: float = 0.05,
) -> tuple[bool, dict]:
    """
    Single-point fragility check before logging a paper trade.
    Calls POST /v1/solve/fragility — faster than full /v1/simulate.
    Returns (is_safe, fragility_report).
    """
    trade_notional = trade.quantity * trade.price_inr
    schema = {
        "version": "1.0.0",
        "metadata": {"name": f"Trade Gate: {trade.ticker}"},
        "time": {"type": "discrete", "horizon": 3},
        "resources": {
            "portfolio_value": {
                "initial": portfolio_value_inr,
                "min": 0.0,
                "max": portfolio_value_inr * 2,
            }
        },
        "variables": {
            "sharpe": {
                "type": "float",
                "initial": round(current_sharpe, 1),
                "min": -2.0, "max": 3.0, "discretization": 0.1,
            }
        },
        "actions": [
            {
                "name": f"execute_{trade.direction}_{trade.ticker.replace('.', '_')}",
                "cost": {"portfolio_value": trade_notional * 0.01},
                "effects": {
                    "sharpe": {
                        "op": "multiply",
                        "value": 1.02 if trade.direction == "BUY" else 1.01,
                    }
                },
                "preconditions": [f"portfolio_value > {trade_notional:.0f}"],
            }
        ],
        "transitions": [], "constraints": [],
        "goal": {
            "type": "threshold",
            "conditions": [{"variable": "sharpe", "operator": "gte", "value": 0.0}],
        },
    }

    result = solve_fragility(schema)   # POST /v1/solve/fragility

    thickness = result.get("thickness", 0)
    kappa     = result.get("kappa", 99)

    # fragility endpoint returns collapse_risk, thickness, kappa directly (no timeline)
    is_safe = thickness >= fragility_threshold and kappa < 2.0

    return is_safe, {
        "thickness":    thickness,
        "kappa":        kappa,
        "collapse_risk": result.get("collapse_risk"),
    }
```

---

### Integration 5: Drift Watcher with Structural Phase Awareness (Phase 2)
**Pipeline stage**: Phase 2.2 — the cron/daemon drift monitor
**Niyati endpoint**: `POST /v1/simulate`
**Phase**: Phase 2 only

#### The Problem

The original plan's drift watcher triggers when any asset breaches 5% drift (`Δw > θ`).
This is a geometric check on weights alone. It ignores the structural question: *is the
5% drift occurring in a `Safe` phase (where rebalancing adds value) or in a `Critical`
phase (where trading adds friction without improving structural margin)?*

Niyati's `phase_classification` per step makes the drift watcher phase-aware. The alert
logic becomes:

```
IF drift > 5% AND phase == "Safe"   → rebalance (standard)
IF drift > 5% AND phase == "Warning" → rebalance with reduced turnover
IF drift > 5% AND phase == "Critical" → alert only, hold, wait for phase recovery
IF drift > 5% AND phase == "Collapsed" → defensive shift only, no HRP rebalance
```

```python
# Phase 2 module: live/drift_watcher.py

from loguru import logger
from utils.niyati_client import simulate

POSTURE_BY_PHASE = {
    "Safe":      "rebalance",
    "Warning":   "rebalance_lite",
    "Critical":  "alert_only",
    "Collapsed": "defensive_only",
}


def phase_aware_drift_check(
    drift_report: dict,       # from optimization/rebalance.py check_rebalance_needed()
    portfolio_value_inr: float,
    current_sharpe: float,
) -> dict:
    """
    Augments the standard drift check with Niyati's phase_classification.
    Uses POST /v1/simulate with a small schema (horizon 4, 1 variable) for speed.
    Returns structured action directive with phase context.
    """
    if not drift_report.get("should_rebalance"):
        return {"action": "hold", "reason": "drift within threshold"}

    schema = _build_quick_schema(portfolio_value_inr, current_sharpe)
    result = simulate(schema, capability="fast")   # POST /v1/simulate — small schema, fast

    timeline = result.get("timeline", [])
    current_phase = timeline[0].get("phase_classification", "Safe") if timeline else "Safe"
    thickness     = timeline[0].get("thickness", 0.1) if timeline else 0.1
    action        = POSTURE_BY_PHASE.get(current_phase, "hold")

    logger.info(
        f"Drift {drift_report['max_drift']:.2%} | Phase: {current_phase} "
        f"| Thickness: {thickness:.3f} | Action: {action}"
    )

    return {
        "action":           action,
        "drift":            drift_report["max_drift"],
        "drifted_assets":   drift_report.get("assets_drifted", []),
        "niyati_phase":     current_phase,
        "niyati_thickness": thickness,
        "niyati_posture":   result.get("survival_policy", {}).get("recommended_posture"),
    }


def _build_quick_schema(portfolio_value_inr: float, current_sharpe: float) -> dict:
    """Minimal schema — 1 variable, horizon 4 — keeps the API call fast for the daemon loop."""
    return {
        "version": "1.0.0",
        "metadata": {"name": "Drift Phase Check"},
        "time": {"type": "discrete", "horizon": 4},
        "resources": {
            "portfolio_value": {
                "initial": portfolio_value_inr,
                "min": 0.0,
                "max": portfolio_value_inr * 2,
            }
        },
        "variables": {
            "sharpe": {
                "type": "float",
                "initial": round(current_sharpe, 1),
                "min": -2.0, "max": 3.0, "discretization": 0.1,
            }
        },
        "actions": [
            {
                "name": "hold",
                "cost": {"portfolio_value": 0},
                "effects": {"sharpe": {"op": "multiply", "value": 0.99}},
                "preconditions": [],
            }
        ],
        "transitions": [], "constraints": [],
        "goal": {
            "type": "threshold",
            "conditions": [{"variable": "sharpe", "operator": "gte", "value": 0.5}],
        },
    }
```

---

### Integration 6: Adversarial Stress Testing
**Pipeline stage**: `validation/stress_tests.py` (Phase 1, already built)
**Niyati endpoint**: `POST /v1/solve/adversarial` + `POST /v1/solve/saddle-point`
**Phase**: Phase 1 enhancement

#### The Problem

The current `stress_tests.py` applies four hand-crafted shocks (COVID: 3% daily × 25 days,
INR depreciation: 0.2% daily × 40 days, etc.). The shock magnitudes are empirically
uncalibrated — chosen to look realistic, not derived from the portfolio's actual structural
geometry.

Niyati's `epsilon_star` IS the minimum structurally-derived shock. Using it inverts the
approach: instead of asking "what does a 30% shock do to the portfolio?", we ask "what is
the minimum shock that breaks the portfolio's structural feasibility?" The answer is
geometry-native, not historically anchored.

```python
# validation/niyati_stress.py

from utils.niyati_client import solve_adversarial

SCENARIO_ATTACKER_BUDGETS = {
    "covid_crash":       0.30,   # 30% of portfolio value as structural shock budget
    "inr_depreciation":  0.10,   # FX shock: 10%
    "tech_collapse":     0.20,   # Sector concentration: 20%
    "liquidity_crunch":  0.15,   # Liquidity: 15%
}

def run_adversarial_stress(
    portfolio_value_inr: float,
    current_sharpe: float,
) -> dict:
    results = {}

    for scenario, attacker_fraction in SCENARIO_ATTACKER_BUDGETS.items():
        attacker_budget = portfolio_value_inr * attacker_fraction
        schema = {
            "version": "1.0.0",
            "metadata": {"name": f"Stress: {scenario}"},
            "time": {"type": "discrete", "horizon": 6},
            "resources": {
                "portfolio_value": {
                    "initial": portfolio_value_inr,
                    "min": 0.0,
                    "max": portfolio_value_inr * 2,
                }
            },
            "variables": {
                "sharpe": {
                    "type": "float",
                    "initial": round(current_sharpe, 1),
                    "min": -3.0, "max": 3.0, "discretization": 0.1,
                }
            },
            "actions": [
                {
                    "name": "defensive_rebalance",
                    "cost": {"portfolio_value": portfolio_value_inr * 0.005},
                    "effects": {"sharpe": {"op": "multiply", "value": 1.08}},
                    "preconditions": ["portfolio_value > 100000"],
                },
                {
                    "name": "hold",
                    "cost": {"portfolio_value": 0},
                    "effects": {"sharpe": {"op": "multiply", "value": 0.94}},
                    "preconditions": [],
                },
            ],
            "transitions": [], "constraints": [],
            "goal": {
                "type": "threshold",
                "conditions": [{"variable": "sharpe", "operator": "gte", "value": 0.0}],
            },
            "theorem_inputs": {
                "agents": [
                    {
                        "id": "portfolio_manager",
                        "budget": portfolio_value_inr * 0.02,
                        "role": "defender",
                    },
                    {
                        "id": scenario,
                        "budget": attacker_budget,
                        "role": "attacker",
                        "attacker_strength": 0.65,
                        "consumed_states": ["sharpe"],
                    },
                ],
                "coupling_gamma": 0.12,
                "reinforcement_beta": 0.04,
            },
        }

        # POST /v1/solve/adversarial — survival verdict under rational attacker
        result = solve_adversarial(schema, adversary_budget=attacker_budget)

        results[scenario] = {
            "survival_verdict":   result.get("survival_verdict"),   # "safe"|"critical"|"doomed"
            "epsilon_star":       result.get("epsilon_star"),
            "survival_margin":    result.get("survival_margin"),    # ε* − attacker_budget
            "kappa_base":         result.get("kappa_base"),
            "kappa_under_attack": result.get("kappa_under_attack"),
            "adversarial_premium": result.get("adversarial_premium"),
            "point_of_no_return": result.get("point_of_no_return"),
            "recommended_path":   result.get("recommended_path"),
        }

    return results
```

**Calibration advantage**: `epsilon_star` is the geometry-derived minimum breaking force.
If `epsilon_star > attacker_budget`, the portfolio survives the scenario structurally.
This replaces the ad-hoc "3% shock looks like COVID" reasoning with a formal survivability
certificate.

---

### Integration 7: Tax Budget Planning (LTCG Optimizer)
**Pipeline stage**: After backtests/taxes.py produces FIFO lot data
**Niyati endpoint**: `POST /v1/simulate` + `POST /v1/solve/trajectory`
**Phase**: Phase 1 and Phase 2

```python
# backtests/niyati_tax_planner.py

from utils.niyati_client import simulate, solve_trajectory


def build_ltcg_harvest_schema(
    ltcg_exemption_inr: float = 100_000,
    realized_ytd: float = 0.0,
    ltcg_lots: list = None,    # from backtests/taxes.py: [{ticker, gain_inr, months_held}]
    horizon_months: int = 12,
) -> dict:
    """
    Model ₹1L LTCG exemption as a resource budget.
    Each eligible lot is an action that consumes budget.
    Goal: harvest ≥90% of exemption without triggering excess tax.
    Niyati returns recommended_path = optimal harvest sequence.
    """
    budget_remaining = ltcg_exemption_inr - realized_ytd
    actions = [
        {
            "name": f"harvest_{lot['ticker'].replace('.','_').replace('=','_')}",
            "cost": {"ltcg_budget": lot["gain_inr"]},
            "effects": {
                "harvested_gains": {"op": "add", "value": lot["gain_inr"]}
            },
            "preconditions": [
                f"ltcg_budget >= {lot['gain_inr']:.0f}",
                "harvested_gains < 95000",
            ],
            "valid_from": 0 if lot["months_held"] >= 12 else None,
        }
        for lot in (ltcg_lots or [])
        if lot.get("months_held", 0) >= 12 and lot.get("gain_inr", 0) > 0
    ]

    if not actions:
        actions = [{"name": "no_eligible_lots", "cost": {"ltcg_budget": 0},
                    "effects": {}, "preconditions": []}]

    return {
        "version": "1.0.0",
        "metadata": {"name": "LTCG Harvest Optimizer"},
        "time": {"type": "discrete", "horizon": horizon_months},
        "resources": {
            "ltcg_budget": {
                "initial": budget_remaining,
                "min": 0.0,
                "max": ltcg_exemption_inr,
            }
        },
        "variables": {
            "harvested_gains": {
                "type": "float",
                "initial": realized_ytd,
                "min": 0.0,
                "max": ltcg_exemption_inr * 1.5,
                "discretization": 5000.0,
            }
        },
        "actions": actions,
        "transitions": [], "constraints": [],
        "goal": {
            "type": "threshold",
            "conditions": [
                {"variable": "harvested_gains", "operator": "gte",
                 "value": ltcg_exemption_inr * 0.90}
            ],
        },
    }
```

**Usage**:
```python
schema  = build_ltcg_harvest_schema(ltcg_lots=my_lots, realized_ytd=45000)
verdict = simulate(schema)          # POST /v1/simulate — feasibility + recommended_path
path    = solve_trajectory(schema)  # POST /v1/solve/trajectory — bottleneck-optimal sequence
```

**Key outputs**:
- `recommended_path` → priority-ordered list of lots to harvest (from trajectory solver)
- `intervention_deadline` → last month to act before financial year-end (from simulate)
- `verdict` → whether full ₹1L utilization is structurally feasible

---

### Integration 8: Research Score Enhancement
**Pipeline stage**: `validation/research_score.py` (Phase 1, already built)
**Niyati endpoint**: `POST /v1/simulate`
**Phase**: Phase 1 enhancement

Two new geometry-derived components replace the heuristic thresholds in the existing
5-component research score:

| # | Component | Weight | Source |
|---|-----------|--------|--------|
| 1 | Sharpe stability | 20% | Existing |
| 2 | Drawdown consistency | 15% | Existing |
| 3 | Turnover efficiency | 15% | Existing |
| 4 | Regime robustness | 15% | **Niyati `danger_fraction`** |
| 5 | Parameter sensitivity | 10% | Existing |
| **6** | **Structural margin** | **15%** | **Niyati `epsilon_star` normalized** |
| **7** | **Corridor trap risk** | **10%** | **Niyati `anisotropy_ratio`** |

```python
# validation/niyati_research_score.py

def extract_niyati_score_components(niyati_result: dict) -> dict:
    """
    Two new scoring components from a runway Niyati result.
    Plug directly into research_score.py composite calculation.
    """
    timeline = niyati_result.get("timeline", [])
    phase_dist = niyati_result.get("phase_distribution", {})

    # Component 6: Structural Margin
    tau_vals = [t.get("thickness", 0) for t in timeline if t.get("thickness") is not None]
    mean_tau = sum(tau_vals) / len(tau_vals) if tau_vals else 0
    structural_margin_score = min(100.0, mean_tau * 400)   # τ = 0.25 → score = 100

    # Component 7: Corridor Trap Risk (lower anisotropy = safer)
    aniso_vals = [t.get("anisotropy_ratio", 1.0) for t in timeline]
    mean_aniso = sum(aniso_vals) / len(aniso_vals) if aniso_vals else 1.0
    is_trapped = any(t.get("is_corridor_trap", False) for t in timeline)
    corridor_score = max(0.0, 100 - (mean_aniso - 1) * 20)
    if is_trapped:
        corridor_score = max(0.0, corridor_score - 30)

    # Component 4 replacement: regime robustness from Niyati phases
    danger_fraction = phase_dist.get("danger_fraction", 0.5)
    regime_score = max(0.0, 100 * (1 - danger_fraction))

    return {
        "structural_margin_score":  round(structural_margin_score, 1),
        "corridor_trap_score":      round(corridor_score, 1),
        "regime_robustness_niyati": round(regime_score, 1),
        "mean_tau":         round(mean_tau, 4),
        "mean_anisotropy":  round(mean_aniso, 4),
        "is_corridor_trapped": is_trapped,
        "danger_fraction":  round(danger_fraction, 3),
    }
```

---

## Phase 2 Architecture with Niyati Embedded

```
                  ┌─────────────────────────────────────────────────────┐
                  │              PHASE 2 LIVE SYSTEM                    │
                  └─────────────────────────────────────────────────────┘

  [Market Close]
       │
       ▼
  ┌──────────────────┐      ┌──────────────────────────────────────────┐
  │ Live Holdings     │      │        DRIFT WATCHER DAEMON             │
  │ (Broker/CSV/API) │─────►│  calc drift for every asset              │
  └──────────────────┘      │  Δw = |current_weight - target_weight|   │
                            └──────────────┬───────────────────────────┘
                                           │  drift > 5% triggered
                                           ▼
                            ┌──────────────────────────────────────────┐
                            │    NIYATI PHASE CHECK  (Integration 5)   │
                            │    POST /v1/simulate  capability="fast"  │
                            │    Returns: Safe / Warning / Critical     │
                            └──────────────┬───────────────────────────┘
                                           │
                       ┌───────────────────┼──────────────────────────┐
                       │                   │                          │
                    "Safe"             "Warning"              "Critical/Collapsed"
                       │                   │                          │
                       ▼                   ▼                          ▼
              Generate signals     Generate signals            Alert only.
              (LightGBM + BL)      (HRP only, no tilt)         Hold portfolio.
                       │                   │
                       └────────┬──────────┘
                                ▼
                   ┌────────────────────────────┐
                   │  BL VIEW VALIDATOR          │
                   │  (Integration 1)            │
                   │  Niyati multi-agent:        │
                   │  cooperation_deficit < 0.20?│
                   └────────────┬───────────────┘
                                │  views compatible
                                ▼
                   ┌────────────────────────────┐
                   │  REBALANCE GATE             │
                   │  (Integration 3)            │
                   │  Score all candidate allocs │
                   │  Select max(min_thickness)  │
                   └────────────┬───────────────┘
                                │  best allocation selected
                                ▼
                   ┌────────────────────────────┐
                   │  PAPER TRADE GATE           │
                   │  (Integration 4)            │
                   │  Per-trade fragility check  │
                   │  thickness >= 0.05?         │
                   └────────────┬───────────────┘
                                │  trade approved
                                ▼
                   ┌────────────────────────────┐
                   │  PAPER WALLET               │
                   │  Log trade + Niyati state   │
                   │  sqlite / JSON ledger       │
                   └────────────────────────────┘
```

---

## Schema Design Reference

When translating Portfolio-OS concepts into Niyati schemas:

| Portfolio-OS concept | Niyati field | Convention |
|----------------------|-------------|-----------|
| Portfolio INR NAV | `resources.portfolio_value.initial` | Current NAV from `portfolio_nav.parquet` |
| Rebalance friction | `actions[].cost.portfolio_value` | From `backtests/costs.py` output |
| Sharpe ratio | `variables.sharpe_ratio` | `discretization: 0.1` |
| Max drawdown | `variables.max_drawdown` | `discretization: 0.05` |
| Asset weight | `variables.w_<ticker>` | `discretization: 0.01` |
| Target Sharpe floor | `goal.conditions[].value` | e.g. 0.8 |
| Planning horizon | `time.horizon` | Months for runway, quarters for multi-year |
| Drift threshold | `actions[].preconditions` | `"turnover_ytd < 2.0"` |
| Adversarial shock | `theorem_inputs.agents[role=attacker]` | `budget = shock_magnitude_inr` |
| Sub-portfolio (INR/USD) | `theorem_inputs.composition_subsystems` | `thickness` and `kappa` per sub-system |
| BL views | `theorem_inputs.agents` | `role: defender` for bullish, `attacker` for bearish |

### Discretization Guidelines

| Variable | Discretization | Notes |
|----------|----------------|-------|
| Sharpe ratio | 0.1 | Fine enough for strategy decisions |
| Max drawdown | 0.05 | 5% bands |
| Asset weight | 0.01 | 1% increments |
| LTCG harvested | 5000 | INR 5K bands |
| IC (signal quality) | 0.01 | 1% granularity |
| Horizon | ≤ 24 months | Beyond 24 → use Hybrid mode |

---

## Shared Client Wrapper

All Niyati calls go through one thin module. No SDK — pure `requests`.

```python
# utils/niyati_client.py

import os
import requests
from loguru import logger

_BASE   = "https://api.causalorlabs.com"
_VER    = "0.3.17"
_TIMEOUT = 12   # seconds; Niyati's own SDK uses ~3.5s + retries, 12s is safe margin


def _headers() -> dict:
    h = {"Content-Type": "application/json", "X-Niyati-Version": _VER}
    key = os.getenv("NIYATI_API_KEY", "")
    if key:
        h["X-Niyati-Key"] = key
    else:
        logger.warning("NIYATI_API_KEY not set — anonymous tier (60 req/min, simulate only)")
    return h


def _post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{_BASE}{path}", json=payload, headers=_headers(), timeout=_TIMEOUT)
    if not resp.ok:
        logger.error(f"Niyati {path} → HTTP {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()


# ── Primary surfaces ────────────────────────────────────────────────────────

def simulate(schema: dict, capability: str = "full", budget: float = None) -> dict:
    """
    POST /v1/simulate
    Full scenario evaluation. Use for runway, stress testing, rebalance scoring.
    capability: "full" | "fast"
    """
    payload = {"schema": schema, "capability": capability}
    if budget is not None:
        payload["budget"] = budget
    return _post("/v1/simulate", payload)


def solve_fragility(schema: dict) -> dict:
    """
    POST /v1/solve/fragility
    Single-point collapse risk — epsilon_star, thickness, kappa.
    Faster than full simulate. Use for per-trade gate in paper wallet.
    """
    return _post("/v1/solve/fragility", {"schema": schema})


def solve_multiagent(schema: dict) -> dict:
    """
    POST /v1/solve/multiagent
    Interference footprint across agents. Use for BL view compatibility check.
    """
    return _post("/v1/solve/multiagent", {"schema": schema})


def solve_adversarial(schema: dict, adversary_budget: float) -> dict:
    """
    POST /v1/solve/adversarial
    Two-player survival verdict under a rational attacker.
    Use for adversarial stress scenarios.
    """
    return _post("/v1/solve/adversarial", {"schema": schema, "adversary_budget": adversary_budget})


def solve_trajectory(schema: dict) -> dict:
    """
    POST /v1/solve/trajectory
    Max-bottleneck recommended path. Use for LTCG harvest sequencing.
    """
    return _post("/v1/solve/trajectory", {"schema": schema})
```

Add to `.env`:
```
NIYATI_API_KEY=your_key_here
```

Add to `requirements.txt`:
```
requests>=2.31.0
```

---

## Implementation Priority

| Priority | Integration | Effort | When |
|----------|-------------|--------|------|
| **P0** | Portfolio Runway Monitor | 1 day | Now — consumes existing CSV outputs |
| **P0** | Adversarial Stress Testing | 1 day | Now — replaces uncalibrated stress_tests.py |
| **P0** | Research Score Enhancement | 1 day | Now — 7-component score with geometry-derived components |
| **P1** | Drift Watcher Phase Awareness | 2 days | Phase 2 build — first Phase 2 module |
| **P1** | Paper Trade Gate | 1 day | Phase 2 build — inside paper wallet |
| **P1** | Rebalance Gate | 2 days | Phase 2 build — between drift trigger and execution |
| **P2** | BL View Validator | 2 days | When Black-Litterman is built |
| **P2** | LTCG Harvest Optimizer | 2 days | Year-end, high seasonal value |
| **P3** | Dashboard Structural Health page | 2 days | After P0/P1 integrations produce data |

---

## What Niyati Does Not Replace

| Component | Keep | Reason |
|-----------|------|--------|
| `backtests/engine.py` | Yes | Historical simulation — Niyati is forward-looking |
| `analytics/metrics.py` | Yes | Backward-looking risk metrics are complementary |
| `optimization/hrp.py` | Yes | Allocation algorithm — Niyati validates the trajectory, not the weights |
| `features/signal_ranker.py` | Yes | Alpha signal engineering is a separate concern |
| `backtests/taxes.py` | Yes | FIFO lot tracking — Niyati uses its output as schema input |
| LightGBM (to build) | Yes | Niyati consumes its views, not replaces them |
| Black-Litterman (to build) | Yes | Niyati validates BL output, not replaces BL |

---

## Summary

Portfolio-OS is a backward-looking research engine: it measures what the strategy *has done*
and optimizes weights based on historical structure. The original brief calls HRP + BL "The
Brakes" — the math layer that constrains the AI's predictions.

Niyati operates one level above: it validates whether the plan implied by The Brakes is
structurally survivable over the forward horizon. It answers the question none of the existing
modules can answer:

> *Given the current portfolio state, how many months of structural runway remain —
> and exactly when does the goal become impossible?*

In Phase 1, Niyati enhances validation with geometry-derived stress tests, a 7-component
research score, and a forward-looking runway model that replaces heuristic thresholds.

In Phase 2, Niyati becomes load-bearing infrastructure: the phase-aware drift watcher, the
per-trade fragility gate, and the rebalance scoring layer all depend on Niyati's structural
classifications to decide when to act, what to execute, and when to hold.

---

## Phase 1 Handoff Checklist

> **Handoff date**: 2026-05-16
> **Handing off from**: Niyati integration engineer (Causalor Labs)
> **Handing off to**: Portfolio-OS engineer
>
> This checklist captures the exact state of the Niyati integration at Phase 1 close.
> Every ✅ item is live, tested against the production API, and committed to the repo.
> Every 🔵 item is specified and ready to build in Phase 2.

---

### Integration 1 — Portfolio Runway Monitor (`/v1/simulate`)

**Status: ✅ Phase 1 Complete**

| Task | Status | File |
|------|--------|------|
| Schema builder `_build_schema()` — stressed initial state, correct format | ✅ | `validation/niyati_runway.py:71` |
| Live API tested — schema format validated against production | ✅ | — |
| `run_runway_analysis()` — POSTs to `/v1/simulate`, returns full result | ✅ | `validation/niyati_runway.py:156` |
| `summarize_runway()` — extracts verdict, ε*, danger_fraction, PNR, posture | ✅ | `validation/niyati_runway.py:187` |
| Dashboard Structural Health page — timeline chart, phase distribution | ✅ | `dashboard/views/structural_health.py` |
| Integrated into `app.py` Step 9 — saves `reports/niyati_runway.json` | ✅ | `app.py` |

**Live result (2026-05-16, re-verified)**: verdict=COLLAPSES (goal_impossible), ε*=0.017, danger_fraction=23%, 10 safe + 3 warning months. Goal reachable until month 5. Point of no return: month 12.

API v0.3.17 now returns a fully structured `narrative` object:
- **pre_mortem**: 18 initial futures, 53% capacity utilization, binding variable = `max_drawdown`
- **collapse_chronicle**: trajectory=gradual_decline, corridor_narrowing_rate=19.6x/month, first warning at month 12
- **post_mortem**: goal_lost_at=12, was_inevitable=**false** (collapse was preventable), recovery_window_existed=true
- **intervention**: deadline=month 11, futures_on_best_path=13, action="Relax `max_drawdown` to maximize futures preserved"

Survival policy: regime=corridor, is_corridor_trap=false, corridor_trap_severity=none, optimal_threshold=0.95.

**What this gives portfolio-OS**: The first forward-looking certificate in the system. The new `intervention.intervention_deadline=11` is the most actionable field — it tells the portfolio-OS engineer exactly when they must act (month 11, not month 12). The `was_inevitable=false` is reassuring: collapse is preventable if HRP rebalancing executes on the recommended path. The `futures_on_best_path=13` confirms there are 13 viable recovery trajectories still open if action is taken now.

**Phase 2 extension**: 🔵 Add multi-scenario runs (bull/bear/base) and refresh on each pipeline execution.

---

### Integration 2 — Adversarial Stress Testing (`/v1/solve/adversarial-allocation` + `/v1/solve/saddle-point`)

**Status: ✅ Phase 1 Complete**

| Task | Status | File |
|------|--------|------|
| `run_adversarial_allocation()` — which asset class collapses first | ✅ | `validation/niyati_stress.py` |
| `run_saddle_point()` — sustained stress, rounds to collapse | ✅ | `validation/niyati_stress.py` |
| Dashboard adversarial allocation bar chart | ✅ | `dashboard/views/structural_health.py:202` |
| Integrated into `app.py` Step 9 — saves `reports/niyati_stress.json` | ✅ | `app.py` |
| Live API tested — both endpoints validated against production | ✅ | — |

**Live result (2026-05-16)**:
- *Adversarial allocation*: Metal collapses first (0.275 attack budget assigned), pi_unified=0.40, 5 of 6 asset classes survive under single-round optimal attack.
- *Saddle-point*: regime=inevitable_collapse, rounds_to_collapse=8, attacker_optimal_target=Metal, pi_per_round=0.20.

**What this gives portfolio-OS**: Replaces the heuristic stress scenarios in `validation/stress_tests.py` with game-theoretic analysis. The adversarial-allocation result identifies Metal as the weakest link — the portfolio breaks first there under optimal adversarial pressure. The saddle-point result (8 rounds to collapse at current budgets) gives a monthly clock: if sustained market stress persists for 8 months, the Metal position destroys portfolio structure. This is actionable for position sizing — Metal's weight should be reviewed at each rebalance against its tau_star margin.

**Phase 2 extension**: 🔵 Feed saddle-point rounds_to_collapse into the drift watcher's phase classification. 🔵 Add collapse timeline chart to dashboard.

---

### Integration 3 — Adversarial Survival Verdict (`/v1/solve/adversarial`)

**Status: ✅ Phase 1 Complete**

| Task | Status | File |
|------|--------|------|
| `solve_adversarial()` client function | ✅ | `utils/niyati_client.py:151` |
| `_build_portfolio_state_graph()` — discrete 2D grid (sharpe × drawdown) | ✅ | `validation/niyati_stress.py` |
| `run_adversarial_survival()` — survival verdict under crash budget ξ=0.006 | ✅ | `validation/niyati_stress.py` |
| Survival verdict banner in Structural Health dashboard | ✅ | `dashboard/views/structural_health.py:263` |
| Integrated into `app.py` Step 9d — saves `reports/niyati_survival.json` | ✅ | `app.py` |
| Live API tested — state graph format validated against production | ✅ | — |

**Live result (2026-05-16, re-verified)**: survival_verdict=critical, survival_margin≈0, ε*=0.006, pi_regime=inevitable_collapse, budget_fraction_at_risk=0.60.

API v0.3.17 now returns richer adversarial fields:
- **solo_optionality**: 6 reachable states without adversary; **solo_goal_omega**: 4 goal states reachable solo
- **stressed_optionality**: 3 reachable states under crash; **stressed_goal_omega**: 1 goal state reachable under crash
- **kappa_base**: 0.693 (natural collapse rate); **kappa_under_attack**: 1.693 (collapse rate under crash — 2.4× faster)
- **adversarial_premium**: 1.0 (the crash doubles the collapse rate exactly)
- **recommended_path**: explicit state IDs [s0640_d0194 → s0760_d0172 → s0880_d0150]

**What this gives portfolio-OS**: The `stressed_goal_omega=1` is the sharpest number from this endpoint — under a crash scenario, only 1 goal state remains reachable (down from 4 in the solo case). The portfolio has 75% of its goal space destroyed by a single crash event. The recommended_path gives the explicit trajectory the HRP defender should follow to reach that 1 remaining goal state.

**Phase 2 extension**: 🔵 Run continuously as a daemon and alert if verdict flips from critical → doomed between scheduled rebalances.

---

### Integration 4 — Fragility Check / Pre-Rebalance Gate (`/v1/solve/fragility`)

**Status: ✅ Phase 1 Complete**

| Task | Status | File |
|------|--------|------|
| `solve_fragility()` client function | ✅ | `utils/niyati_client.py:192` |
| `run_fragility_check()` — instant polytope check at current portfolio point | ✅ | `validation/niyati_stress.py` |
| Fragility gate in `app.py` Step 7 — tightens drift threshold if fragile | ✅ | `app.py` |
| Fragility detail expander in Structural Health dashboard | ✅ | `dashboard/views/structural_health.py:336` |
| 5th KPI card: Fragility (collapse_risk) | ✅ | `dashboard/views/structural_health.py:315` |
| Live API tested — polytope format validated against production | ✅ | — |

**Live result (2026-05-16, re-verified)**: collapse_risk=0.074 (7.4%), thickness=0.168, kappa=0.012, anisotropy_ratio=13.33.

*Δ from prior run*: collapse_risk improved 8.7%→7.4%, thickness improved 0.134→0.168, anisotropy improved 16.08→13.33. The API geometry recalibration produced slightly more conservative (safer) fragility readings.

**What this gives portfolio-OS**: The fastest structural check in the system — runs in milliseconds before any rebalance decision. The anisotropy of 16× is the critical finding: the portfolio has near-normal freedom in one direction but is almost locked in the perpendicular. This means a rebalance that moves in the "wrong" direction (even within drift bounds) can push the portfolio into a corner from which recovery is difficult. The fragility gate in `app.py` acts on this: when `collapse_risk > 0.15` or `thickness < 0.10`, the drift threshold tightens from 5% → 3%, requiring more convincing drift before triggering a trade.

**Phase 2 extension**: 🔵 Run per-trade as the paper wallet gate — block any trade that raises collapse_risk above 0.15.

---

### Integration 5 — Competition Check (`/v1/solve/competition`)

**Status: ✅ Phase 1 Complete (advisory only)**

| Task | Status | File |
|------|--------|------|
| `solve_competition()` client function | ✅ | `utils/niyati_client.py:58` |
| `run_competition_check()` — kappa under moderate/severe adversarial pressure | ✅ | `validation/niyati_stress.py` |
| Integrated into `app.py` Step 9 — saves to stress JSON | ✅ | `app.py` |

**Live result (2026-05-16, re-verified)**: kappa_unperturbed=0.0527, kappa_competitive_bound=3.531, adversarial_premium=3.479, sequential_stress_index=67. New fields: tau_star=0.158, adversarial_pressure=3.479.

*Δ from prior run*: kappa values recalibrated ~20% lower (0.066→0.053 unperturbed, 4.41→3.53 competitive bound). The adversarial multiplier is now 67× instead of 67× — SSI unchanged. The API added `tau_star` (structural thickness at this competition point) and renamed `adversarial_premium` → also exposed as `adversarial_pressure`.

**What this gives portfolio-OS**: Quantifies the *multiplier effect* of adversarial pressure on collapse speed. Unperturbed, the portfolio loses optionality at κ=0.066 per step — slow degradation. Under competition, κ jumps to 4.41: the collapse rate is 67× faster than baseline. The sequential_stress_index=67 is a single summary number for dashboards and alerts. This replaces guesswork about "how bad is a bad month" with a geometry-derived multiplier.

**Phase 2 extension**: 🔵 Add to research score as the 7th component (adversarial_premium as a penalty factor).

---

### Integration 6 — Dashboard: Structural Health Page

**Status: ✅ Phase 1 Complete**

| Task | Status | File |
|------|--------|------|
| Page created as 7th dashboard page | ✅ | `dashboard/views/structural_health.py` |
| Survival verdict banner (🟢/🟡/🔴) | ✅ | `:263` |
| 5 KPI cards: verdict, ε*, danger_fraction, PNR, fragility | ✅ | `:292` |
| Fragility detail expander (thickness, kappa, anisotropy) | ✅ | `:336` |
| Future-width timeline chart | ✅ | `:355` |
| Phase distribution bar chart | ✅ | `:364` |
| Critical decisions table | ✅ | `:374` |
| Adversarial allocation bar chart (red = collapses) | ✅ | `:387` |
| Recommended survival posture banner | ✅ | `:409` |
| Raw API response expander for debugging | ✅ | `:415` |
| Registered in `app.py` navigation | ✅ | `app.py` |

---

### Integration 7 — Research Score Enhancement

**Status: 🔵 Phase 2 — Specified, Not Built**

The integration doc (Section 8) specifies a 7-component research score where 3 components are Niyati-derived. The current `features/signal_ranker.py` uses only backward-looking signals.

| Component | Source | Status |
|-----------|--------|--------|
| Momentum score | `features/` | ✅ Exists |
| Volatility regime | `analytics/` | ✅ Exists |
| Drawdown severity | `analytics/` | ✅ Exists |
| **Structural thickness (τ*)** | Niyati fragility | 🔵 Phase 2 |
| **Optionality score (Ω)** | Niyati simulate | 🔵 Phase 2 |
| **Collapse speed (κ)** | Niyati fragility | 🔵 Phase 2 |
| **Adversarial premium** | Niyati competition | 🔵 Phase 2 |

**Blocked on**: LightGBM feature pipeline (not yet built). Niyati components ready to plug in once the feature store exists.

---

### Integration 8 — Drift Watcher Phase Awareness

**Status: 🔵 Phase 2 — Specified, Not Built**

The drift watcher daemon does not yet exist. When built, it should:
- Query Niyati phase (safe / warning / critical) every 6 hours
- Tighten drift threshold to 3% in warning phase, 1% in critical phase
- Log phase transitions as structured events
- Trigger an emergency rebalance if phase = critical AND survival_verdict = doomed

**Depends on**: Live portfolio state monitor (not yet built), daemon infrastructure.

---

### Integration 9 — Paper Trade Gate (Per-Trade Fragility)

**Status: 🔵 Phase 2 — Specified, Not Built**

The paper trading virtual wallet does not yet exist. When built, Niyati should gate every proposed trade:
1. Compute proposed portfolio point after trade (adjusted sharpe, drawdown)
2. POST to `/v1/solve/fragility` with proposed point
3. Block trade if collapse_risk increases by >0.05 or crosses 0.15 threshold
4. Log fragility delta for every executed trade

**Depends on**: Paper trading virtual wallet (not yet built), live position tracking.

---

### Integration 10 — LTCG Harvest Optimizer + BL Validator

**Status: 🔵 Phase 2 / Phase 3 — Specified, Not Built**

| Integration | Endpoint | Depends On | Phase |
|-------------|----------|------------|-------|
| LTCG Harvest Sequencing | `/v1/solve/trajectory` | Tax-lot tracking (built) | Phase 2 year-end |
| Black-Litterman View Validator | `/v1/solve/multiagent` | BL framework (not built) | Phase 3 |

---

## Endpoint Reference: What's Live vs. Deferred

| Endpoint | Status | Used In |
|----------|--------|---------|
| `POST /v1/simulate` | ✅ Live | `validation/niyati_runway.py` — now extracts `narrative.intervention` (deadline, futures_on_best_path, root_cause) |
| `POST /v1/solve/adversarial-allocation` | ✅ Live | `validation/niyati_stress.py` |
| `POST /v1/solve/saddle-point` | ✅ Live | `validation/niyati_stress.py` — now includes per-round `system_states` |
| `POST /v1/solve/competition` | ✅ Live | `validation/niyati_stress.py` — now returns `tau_star`, `adversarial_pressure` |
| `POST /v1/solve/adversarial` | ✅ Live | `validation/niyati_stress.py` — now returns `solo_goal_omega`, `stressed_goal_omega`, `kappa_under_attack`, `recommended_path` |
| `POST /v1/solve/fragility` | ✅ Live | `validation/niyati_stress.py`, `app.py` — geometry recalibrated in v0.3.17 |
| `POST /v1/solve/trajectory` | 🔵 Phase 2 | — (LTCG harvest) |
| `POST /v1/solve/reachability` | 🔵 Phase 2 | — (drift watcher) |
| `POST /v1/solve/nash` | 🔵 Phase 3 | — (multi-manager) |
| `POST /v1/solve/multiagent` | 🔵 Phase 3 | — (BL validator) |
| `POST /v1/solve/pareto` | 🔵 Phase 3 | — (multi-objective) |

---

## Key Schema Lessons (for portfolio-OS engineer)

These are bugs that were found and fixed during integration — do not revert them.

| Issue | Wrong | Correct |
|-------|-------|---------|
| Variable type field | `{"initial": 1.0}` | `{"type": "float", "initial": 1.0}` |
| Action effects format | `[{"variable": "x", "operation": "multiply", "value": 1.1}]` | `{"x": {"op": "multiply", "value": 1.1}}` |
| Action cost format | `{"resource": "portfolio_value", "amount": 2000}` | `{"portfolio_value": 2000}` |
| Preconditions format | `[{"variable": "x", "operator": "gt", "value": 0.2}]` | `["x > 0.20"]` |
| Transitions in simulate | Any transition with effects | `"transitions": []` — effects in transitions not supported |
| Top-level fields | Missing | Requires `version`, `metadata`, `time` at root |
| State graph endpoints | Pass schema dict | Pass `{state_space: {states, transitions}, start, goals, ...}` |
| Polytope endpoints | Pass schema dict | Pass `{constraints: [{normal, offset}], point, direction}` |

---

## Files Modified in Phase 1

| File | Change |
|------|--------|
| `utils/niyati_client.py` | Added `solve_adversarial()`, `solve_fragility()` client functions |
| `validation/niyati_runway.py` | New file — runway analysis via `/v1/simulate` |
| `validation/niyati_stress.py` | New file — adversarial allocation, saddle-point, competition, survival, fragility |
| `dashboard/views/structural_health.py` | New file — 7th dashboard page with all Niyati outputs |
| `app.py` | Added Step 9 (Niyati analysis), Step 7 fragility gate, Step 9d survival verdict |
