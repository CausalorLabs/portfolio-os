# Niyati Output Report

**Date:** 2026-05-17
**API version:** 0.3.17
**Base URL:** `https://api.causalorlabs.com`
**Portfolio data:** `reports/portfolio_metrics.csv` + `reports/backtest_attribution.csv`
**Endpoints hit:** 6 (all Phase 1 integrations)

All results are live from the production API against current portfolio-os CSV data.
Trust level on `/v1/simulate`: **High** (exact BFS, Discrete mode — not probabilistic).

---

## Endpoint 1 — `/v1/simulate` — Structural Runway

### Raw output

| Field | Value |
|---|---|
| Verdict | `goal_impossible` |
| Trust level | `High` (exact BFS, Discrete mode) |
| ε* (epsilon star) | 0.0170 |
| Danger fraction | 23% (3 of 13 steps in warning) |
| Safe steps | 10 |
| Warning steps | 3 |
| Critical steps | 0 |
| Goal reached at | Month 5 |
| Goal PNR | Month 12 |
| Intervention deadline | **Month 11** |
| Futures on best path | 13 |
| Root cause | `max_drawdown` |
| Was inevitable | **false** — collapse is preventable |
| Trajectory | `gradual_decline`, narrowing at 19.6×/month |
| Survival regime | `corridor` — stay centered, preserve thickness |
| Corridor trap | None (`severity: none`) |

### Narrative (API-generated)

> Goal unreachable from T=12: 1 futures remain but the goal path is closed (`max_drawdown` binding).

**Pre-mortem:** 18 futures available at start, operating at 53% of theoretical capacity. Primary constraint variable: `max_drawdown`.

**Collapse chronicle:** Gradual decline throughout horizon driven by `max_drawdown`. First warning at month 12. Corridor narrowing rate: 19.6×/month.

**Post-mortem:** Goal futures exhausted at T=12. System remains operational (1 future) but goal path is permanently closed. Root cause: `max_drawdown`. Collapse was **not inevitable** — recovery window existed.

**Intervention:** Act by month 11. Follow the recommended path immediately. Relax `max_drawdown` to maximize futures preserved. 13 futures remain on the best path.

### What it means for portfolio-OS

This is the only forward-looking certificate in the system. Every other validation module (walk-forward, Monte Carlo, `stress_tests.py`) is backward-looking — it tells you what happened. The runway verdict tells you whether the current portfolio *can* recover to target Sharpe within 12 months under active HRP management.

`was_inevitable: false` is the most important field in this response. The portfolio is not structurally doomed — it is on a gradual decline that HRP rebalancing can reverse if executed before month 11. The 13 futures on the recommended path are the viable recovery trajectories still open right now.

**Files:** `validation/niyati_runway.py`, `app.py` Step 9, `dashboard/views/structural_health.py`

---

## Endpoint 2 — `/v1/solve/adversarial-allocation` — Which Asset Class Breaks First

### Raw output

| System | τ* | κ base | Budget assigned | κ with attack | Survives |
|---|---|---|---|---|---|
| **Metal** | 0.55 | 0.55 | **0.275** | 1.050 | **No** |
| IN_equity | 0.60 | 0.50 | 0.025 | 0.542 | Yes |
| US_equity | 0.65 | 0.45 | 0.000 | 0.450 | Yes |
| ETF | 0.70 | 0.35 | 0.000 | 0.350 | Yes |
| MF | 0.72 | 0.30 | 0.000 | 0.300 | Yes |
| Fixed Income | 0.85 | 0.15 | 0.000 | 0.150 | Yes |

| Summary field | Value |
|---|---|
| Collapsed count | 1 of 6 |
| pi_unified | 0.40 |
| Total κ post-attack | 2.842 |
| Attacker budget used | 0.30 (100%) |

### What it means for portfolio-OS

A rational attacker with 30% structural budget concentrates 91.7% of it on Metal. It takes 27.5% to push Metal's κ above its τ* threshold (1.050 > 0.550 = collapsed). The remaining 2.5% nudges IN_equity but cannot collapse it. Fixed income, ETF, and MF are structurally robust enough that attacking them is inefficient — the attacker ignores them entirely.

This replaces uniform shock scenarios in `validation/stress_tests.py` with solved game theory. Metal is the named structural weak link. Its τ*=0.55 and κ_base=0.55 are the worst in the portfolio. Position sizing decisions at each rebalance should weight this finding — Metal carries disproportionate structural risk relative to its allocation weight.

**Files:** `validation/niyati_stress.py` (`run_adversarial_allocation`), `app.py` Step 9, `dashboard/views/structural_health.py` (adversarial bar chart)

---

## Endpoint 3 — `/v1/solve/saddle-point` — Sustained Stress Timeline

### Raw output

| Field | Value |
|---|---|
| Regime | `inevitable_collapse` |
| Survival value V(x) | 0.0 |
| Rounds to collapse | **8** |
| π per round | 0.20 |
| Attacker optimal target | Metal |
| Defender optimal target | Metal |

### What it means for portfolio-OS

Both players converge on Metal every round — attacker to destroy it, defender to protect it. Net pressure per round is Π=0.20 (attacker α=0.08 + coupling γ=0.15 − defender β=0.03). At that rate ε* drains to zero in 8 rounds. The defender cannot win at current budget ratios.

Translates the adversarial allocation result into a clock. "8 months of sustained adversarial pressure" is the structural planning horizon. If macro conditions enter a prolonged stress regime, round 1 starts at entry. The defender needs either a larger reinforcement budget per round (β > α + γ) or reduced Metal exposure to shift the regime from `inevitable_collapse` to `critical_boundary`.

**Files:** `validation/niyati_stress.py` (`run_saddle_point`), `app.py` Step 9

---

## Endpoint 4 — `/v1/solve/adversarial` — Crash Survival Verdict

### Raw output

| Field | Value |
|---|---|
| Survival verdict | **critical** |
| Survival margin | ≈ 0 (2.6×10⁻¹⁸) |
| ε* | 0.006 |
| Solo optionality | 6 states |
| Solo goal omega | 4 goal states reachable |
| Stressed optionality | 3 states |
| Stressed goal omega | **1** goal state reachable under crash |
| κ base | 0.693 |
| κ under attack | 1.693 (2.4× faster) |
| Adversarial premium | 1.0 |
| pi regime | `inevitable_collapse` |
| Budget fraction at risk | 60% |
| Recommended path | `s0640_d0194 → s0760_d0172 → s0880_d0150` |

### What it means for portfolio-OS

Against a crash of ξ=0.006 (60bps — one bad day), the portfolio sits exactly on the survival boundary. The crash destroys 75% of the goal space: 4 goal states reachable solo → 1 reachable under crash. κ doubles under attack (0.693 → 1.693), meaning the portfolio loses structural options 2.4× faster during a crash than naturally.

One goal state remains reachable, and the API gives the exact path to reach it: target a Sharpe corridor of 0.64 → 0.76 → 0.88 while keeping drawdown within the d0194 → d0172 → d0150 band.

This endpoint is the emergency gate in `app.py`. `critical` holds — no drift override yet. If the next adverse event tips verdict to `doomed`, the rebalance executes regardless of observed drift magnitude.

**Files:** `validation/niyati_stress.py` (`run_adversarial_survival`), `app.py` Step 9d, `dashboard/views/structural_health.py` (survival banner)

---

## Endpoint 5 — `/v1/solve/fragility` — Pre-Rebalance Gate

### Raw output

| Field | Value |
|---|---|
| Collapse risk | **7.4%** |
| Thickness τ* | 0.168 |
| κ (instantaneous) | 0.012 |
| Anisotropy ratio | **13.3×** |

### What it means for portfolio-OS

The portfolio is 7.4% of the way to structural collapse in the market stress direction (Sharpe dropping, drawdown rising). Thickness of 0.168 means a 16.8% buffer before the nearest constraint is hit. Anisotropy of 13.3× means the portfolio has 13× more maneuver room in the Sharpe axis than the drawdown axis — wide in one direction, tight in the other.

Current readings are below both gate thresholds (`collapse_risk < 0.15`, `thickness > 0.10`), so drift threshold stays at 5% and rebalancing proceeds normally. The anisotropy reading is the key operational signal: rebalance moves that increase drawdown even slightly are structurally far more dangerous than moves that affect Sharpe. The engine identifies which axis is locked.

**Gate logic in `app.py`:** if `collapse_risk > 0.15` OR `thickness < 0.10` → drift threshold tightens from 5% → 3%.

**Files:** `validation/niyati_stress.py` (`run_fragility_check`), `app.py` Step 7, `dashboard/views/structural_health.py` (fragility KPI + expander)

---

## Endpoint 6 — `/v1/solve/competition` — Adversarial Pressure Multiplier

### Raw output

| Field | Value |
|---|---|
| κ unperturbed | 0.053 |
| κ competitive bound | 3.531 |
| Adversarial premium | 3.479 |
| τ* at operating point | 0.158 |
| Adversarial pressure α | 3.479 |
| Sequential stress index | **67×** |

### What it means for portfolio-OS

Without adversarial pressure, the portfolio loses optionality at κ=0.053 per step — slow baseline drift. Under moderate shock (attacker strength 0.55), κ jumps to 3.531. The collapse rate is **67× faster** in a competitive market environment than in calm conditions. τ*=0.158 at the current operating point confirms the fragility reading — thin margin from this directional angle.

A "bad month" is not uniformly bad — it is structurally 67× worse when market dynamics are adversarial. This is the multiplier that benchmark stress scenarios cannot capture. The SSI=67 is the single number for executive or risk-committee reporting.

Phase 2: feed SSI as the 7th signal component in the research score, penalising high-SSI regimes before position sizing decisions are finalized.

**Files:** `validation/niyati_stress.py` (`run_competition_stress`), `app.py` Step 9

---

## Current Portfolio Structural Status — Summary

| Signal | Value | Threshold | Status |
|---|---|---|---|
| Runway verdict | COLLAPSES (`goal_impossible`) | — | Act before month 11 |
| Epsilon star ε* | 0.017 | > 0.010 = healthy | Marginal |
| Danger fraction | 23% | < 15% = safe | Elevated |
| Survival verdict | CRITICAL | SAFE = green | On the boundary |
| Stressed goal states | 1 of 4 survive crash | ≥ 2 = acceptable | Thin |
| Collapse risk | 7.4% | < 15% = gate open | Gate open ✓ |
| Thickness τ* | 0.168 | > 0.10 = gate open | Gate open ✓ |
| Anisotropy | 13.3× | < 10× = no trap | Watch drawdown axis |
| Rounds to collapse | 8 | > 12 = safe horizon | Below safe horizon |
| Adversarial multiplier | 67× | < 20× = low stress | High |
| Root constraint | `max_drawdown` | — | Primary lever to relax |

### Bottom line

The portfolio is structurally alive but thin. The fragility gate is open — rebalancing proceeds at normal 5% drift threshold. But the adversarial picture is consistent across all four stress endpoints: Metal is the named weak link, 8 months is the structural clock under sustained pressure, and the survival margin under a single crash event is essentially zero.

The most actionable output from this run: `max_drawdown` is the binding variable across every endpoint that exposes root cause. Relaxing it — via defensive_shift actions that reduce drawdown exposure — is the highest-leverage intervention available. The recommended path (s0640→s0760→s0880) is still open. The intervention deadline is month 11. The collapse was not inevitable.
