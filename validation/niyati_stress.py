"""
Niyati stress analysis — adversarial allocation, competition, saddle-point,
adversarial survival verdict, and fragility gate.

Functions:
  run_adversarial_allocation() — which asset class collapses first under attack
  run_competition_stress()     — market shock vs. HRP defender
  run_saddle_point()           — sustained multi-round stress to collapse
  run_adversarial_survival()   — survive/doomed verdict under explicit crash budget
  run_fragility_check()        — instant polytope fragility at current portfolio point
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from utils.niyati_client import (
    solve_adversarial,
    solve_adversarial_allocation,
    solve_competition,
    solve_fragility,
    solve_saddle_point,
)

PROCESSED = Path("data/processed")

# ── Asset class definitions ────────────────────────────────────────────────────

# Tickers per asset class (must match asset_master.csv asset_type groupings)
ASSET_CLASS_TICKERS: dict[str, list[str]] = {
    "IN_equity": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BAJFINANCE.NS"],
    "US_equity": ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
    "ETF": ["NIFTYBEES.NS", "JUNIORBEES.NS", "GOLDBEES.NS", "BANKBEES.NS"],
    "MF": [],  # populated dynamically from processed data
    "fixed_income": [],
    "metal": ["GC=F", "SI=F", "GOLD"],
}

# Fallback tau_star / kappa_base per class when parquet data is unavailable
# tau_star: 0–1, higher = more robust (1 - fragility)
# kappa_base: 0–1, higher = faster collapse
_DEFAULTS: dict[str, dict] = {
    "IN_equity":   {"tau_star": 0.60, "kappa_base": 0.50},
    "US_equity":   {"tau_star": 0.65, "kappa_base": 0.45},
    "ETF":         {"tau_star": 0.70, "kappa_base": 0.35},
    "MF":          {"tau_star": 0.72, "kappa_base": 0.30},
    "fixed_income": {"tau_star": 0.85, "kappa_base": 0.15},
    "metal":       {"tau_star": 0.55, "kappa_base": 0.55},
}


# ── Parquet loaders ────────────────────────────────────────────────────────────


def _load_inr_prices() -> pd.DataFrame:
    path = PROCESSED / "inr_prices.parquet"
    if not path.exists():
        logger.warning(f"inr_prices.parquet not found at {path}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:
        logger.warning(f"Could not load inr_prices.parquet: {exc}")
        return pd.DataFrame()


def _load_asset_master() -> pd.DataFrame:
    path = Path("configs/asset_master.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ── System builder ─────────────────────────────────────────────────────────────


def _build_systems() -> list[dict]:
    """
    Build 6 system descriptors from processed parquet data.

    For each asset class:
      tau_star  = 1 - abs(worst Sharpe in class, normalised 0–1 across all classes)
      kappa_base = avg volatility in class, normalised 0–1 across all classes

    Falls back to hardcoded defaults if data is unavailable.
    """
    prices_df = _load_inr_prices()
    master_df = _load_asset_master()

    if prices_df.empty or master_df.empty:
        logger.warning("Parquet data unavailable — using default system parameters")
        return [
            {"name": cls, **params}
            for cls, params in _DEFAULTS.items()
        ]

    # Map tickers to asset_type
    ticker_type: dict[str, str] = {}
    if not master_df.empty and "ticker" in master_df.columns and "asset_type" in master_df.columns:
        ticker_type = dict(zip(master_df["ticker"], master_df["asset_type"]))

    # Normalise asset_type → our 6 classes
    type_to_class = {
        "equity": None,  # split by country below
        "etf": "ETF",
        "mf": "MF",
        "fixed_income": "fixed_income",
        "metal": "metal",
    }

    # Country map for equity split
    ticker_country: dict[str, str] = {}
    if "ticker" in master_df.columns and "country" in master_df.columns:
        ticker_country = dict(zip(master_df["ticker"], master_df["country"]))

    # Calculate per-ticker daily returns and volatility
    if "inr_price" not in prices_df.columns or "ticker" not in prices_df.columns:
        logger.warning("inr_prices.parquet missing expected columns — using defaults")
        return [{"name": cls, **params} for cls, params in _DEFAULTS.items()]

    prices_wide = (
        prices_df.pivot_table(index="date", columns="ticker", values="inr_price", aggfunc="first")
        .sort_index()
        .ffill()
    )
    returns_wide = prices_wide.pct_change().dropna()

    if returns_wide.empty or len(returns_wide) < 20:
        logger.warning("Insufficient return history — using default system parameters")
        return [{"name": cls, **params} for cls, params in _DEFAULTS.items()]

    # Per-class Sharpe and volatility
    class_sharpes: dict[str, list[float]] = {c: [] for c in _DEFAULTS}
    class_vols: dict[str, list[float]] = {c: [] for c in _DEFAULTS}

    for ticker in returns_wide.columns:
        ret = returns_wide[ticker].dropna()
        if len(ret) < 20:
            continue
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * (252 ** 0.5)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

        asset_type = ticker_type.get(ticker, "")
        country = ticker_country.get(ticker, "IN")

        if asset_type == "equity":
            cls = "US_equity" if str(country).upper() in ("US", "USA") else "IN_equity"
        elif asset_type in type_to_class and type_to_class[asset_type]:
            cls = type_to_class[asset_type]
        else:
            continue

        class_sharpes[cls].append(sharpe)
        class_vols[cls].append(ann_vol)

    # Normalise to [0, 1]
    all_worst_sharpes = []
    all_avg_vols = []
    class_stats: dict[str, dict] = {}

    for cls in _DEFAULTS:
        sharpes = class_sharpes[cls]
        vols = class_vols[cls]
        worst_sharpe = min(sharpes) if sharpes else 0.0
        avg_vol = sum(vols) / len(vols) if vols else _DEFAULTS[cls]["kappa_base"]
        class_stats[cls] = {"worst_sharpe": worst_sharpe, "avg_vol": avg_vol}
        all_worst_sharpes.append(worst_sharpe)
        all_avg_vols.append(avg_vol)

    sharpe_min = min(all_worst_sharpes)
    sharpe_max = max(all_worst_sharpes)
    vol_min = min(all_avg_vols)
    vol_max = max(all_avg_vols)

    systems = []
    for cls in _DEFAULTS:
        ws = class_stats[cls]["worst_sharpe"]
        av = class_stats[cls]["avg_vol"]

        # tau_star: 1 - normalised fragility (higher sharpe → more robust)
        if sharpe_max > sharpe_min:
            norm_sharpe = (ws - sharpe_min) / (sharpe_max - sharpe_min)
        else:
            norm_sharpe = 0.5
        tau_star = round(max(0.05, min(0.95, 1.0 - abs(norm_sharpe - 1.0))), 4)

        # kappa_base: normalised volatility (higher vol → faster collapse)
        if vol_max > vol_min:
            kappa_base = round((av - vol_min) / (vol_max - vol_min), 4)
        else:
            kappa_base = _DEFAULTS[cls]["kappa_base"]
        kappa_base = max(0.05, min(0.95, kappa_base))

        systems.append({"name": cls, "tau_star": tau_star, "kappa_base": kappa_base})
        logger.debug(f"  {cls}: tau_star={tau_star}, kappa_base={kappa_base}")

    return systems


# ── A) Adversarial allocation ──────────────────────────────────────────────────


def run_adversarial_allocation(api_key: str | None = None) -> dict | None:
    """
    Determine which asset class the adversary hits first and which collapse.

    Parameters match the research spec:
      attacker_budget   = 0.30  (30% of structural budget — market crash)
      coupling_gamma    = 0.15  (moderate IN/US contagion)
      reinforcement_beta = 0.05 (quarterly HRP defensive rate)

    Returns:
        Parsed API response or None.  Key fields to look for:
          first_collapse, collapse_count, pi_unified, allocations (per system).
    """
    systems = _build_systems()
    logger.info(
        f"Niyati adversarial-allocation: {len(systems)} systems, budget=0.30"
    )

    try:
        result = solve_adversarial_allocation(
            systems=systems,
            attacker_budget=0.30,
            coupling_gamma=0.15,
            reinforcement_beta=0.05,
        )
        if result is None:
            logger.error("Niyati adversarial-allocation returned None")
            return None

        # Log key output
        allocs = result.get("system_allocations", [])
        first = next((a["name"] for a in allocs if not a.get("survives", True)), "none")
        n_collapse = result.get("collapsed_count", "?")
        pi = result.get("pi_unified")
        logger.info(
            f"Niyati adversarial-allocation: first_collapse={first}, "
            f"n_collapsed={n_collapse}, pi_unified={pi}"
        )
        return result
    except Exception as exc:
        logger.error(f"run_adversarial_allocation failed: {exc}")
        return None


# ── B) Competition stress ──────────────────────────────────────────────────────


def run_competition_stress(
    current_sharpe: float,
    current_drawdown: float,
    shock_level: str = "moderate",
) -> dict | None:
    """
    Run a market shock vs. HRP defender competition analysis.

    Polytope:   sharpe in [-2, 3], drawdown in [0, 0.5]
    Defender:   rebalancing can improve sharpe +0.15, reduce drawdown -0.05
    Attacker:   mild=0.30, moderate=0.55, severe=0.80

    Returns:
        Parsed API response or None.  Key fields:
          kappa_unperturbed, kappa_competitive_bound,
          adversarial_premium, sequential_stress_index.
    """
    strength_map = {"mild": 0.30, "moderate": 0.55, "severe": 0.80}
    attacker_strength = strength_map.get(shock_level, 0.55)

    # 4-constraint polytope: sharpe in [-2, 3], drawdown in [0, 0.5]
    constraints = [
        {"normal": [1.0, 0.0], "offset": 3.0},   # sharpe <= 3
        {"normal": [-1.0, 0.0], "offset": 2.0},  # sharpe >= -2
        {"normal": [0.0, 1.0], "offset": 0.5},   # drawdown <= 0.5
        {"normal": [0.0, -1.0], "offset": 0.0},  # drawdown >= 0
    ]

    defender_position = [round(float(current_sharpe), 4), round(float(abs(current_drawdown)), 4)]
    defender_action = [0.15, -0.05]  # +0.15 sharpe, -0.05 drawdown

    logger.info(
        f"Niyati competition stress: shock={shock_level} ({attacker_strength}), "
        f"position={defender_position}"
    )

    try:
        result = solve_competition(
            constraints=constraints,
            defender_position=defender_position,
            defender_action=defender_action,
            attacker_strength=attacker_strength,
        )
        if result is None:
            logger.error("Niyati /v1/solve/competition returned None")
            return None

        kappa_u = result.get("kappa_unperturbed", result.get("kappa_base", None))
        kappa_c = result.get("kappa_competitive_bound", result.get("kappa_bound", None))
        premium = result.get("adversarial_premium", result.get("premium", None))
        ssi = result.get("sequential_stress_index", result.get("stress_index", None))
        logger.info(
            f"Niyati competition: kappa_u={kappa_u}, kappa_c={kappa_c}, "
            f"premium={premium}, ssi={ssi}"
        )
        return result
    except Exception as exc:
        logger.error(f"run_competition_stress failed: {exc}")
        return None


# ── C) Saddle-point ───────────────────────────────────────────────────────────


def run_saddle_point(
    asset_class_systems: list[dict] | None = None,
    horizon: int = 12,
) -> dict | None:
    """
    Sustained multi-round stress: monthly attacker vs. quarterly defender.

    Parameters:
      attacker_budget_per_round = 0.08  (monthly market pressure)
      defender_budget_per_round = 0.03  (monthly HRP rebalancing capacity)
      coupling_gamma            = 0.15
      horizon                   = 12 months

    Args:
        asset_class_systems: Pre-built systems list (uses _build_systems() if None).
        horizon:             Number of rounds.

    Returns:
        Parsed API response or None.  Key fields:
          regime, rounds_to_collapse, system_states,
          attacker_optimal, defender_optimal.
    """
    systems = asset_class_systems if asset_class_systems is not None else _build_systems()
    logger.info(
        f"Niyati saddle-point: {len(systems)} systems, horizon={horizon}, "
        f"atk=0.08, def=0.03"
    )

    try:
        result = solve_saddle_point(
            systems=systems,
            attacker_budget_per_round=0.08,
            defender_budget_per_round=0.03,
            coupling_gamma=0.15,
            horizon=horizon,
        )
        if result is None:
            logger.error("Niyati /v1/solve/saddle-point returned None")
            return None

        regime = result.get("regime", result.get("equilibrium_regime", "unknown"))
        rtc = result.get("rounds_to_collapse", result.get("collapse_round", None))
        logger.info(f"Niyati saddle-point: regime={regime}, rounds_to_collapse={rtc}")
        return result
    except Exception as exc:
        logger.error(f"run_saddle_point failed: {exc}")
        return None


# ── D) Portfolio state-graph builder ─────────────────────────────────────────


def _build_portfolio_state_graph(
    current_sharpe: float,
    current_drawdown: float,
) -> tuple[dict, str, list[str]]:
    """
    Build a discrete 2-D portfolio state graph for state-graph endpoints.

    State space: (sharpe_bin × drawdown_bin) grid centred on current metrics.
    Transitions: rebalance_hrp action (sharpe +15%, drawdown -10%, cost 0.002).

    Returns:
        state_space   — {"states": {...}, "transitions": [...]}
        start         — state ID of the 40%-shocked portfolio (stress scenario)
        goals         — list of state IDs representing ≥85% recovery to current
    """
    raw_sharpe = max(0.1, min(current_sharpe, 3.0))
    raw_dd = abs(current_drawdown)

    # Build 8 sharpe bins and 4 drawdown bins spanning ±60% of current values
    def _bins(center: float, n: int, step: float) -> list[float]:
        half = (n - 1) / 2
        return [round(center + (i - half) * step, 3) for i in range(n)]

    sharpe_step = max(0.05, round(raw_sharpe * 0.12, 3))
    dd_step = max(0.02, round(raw_dd * 0.15, 3))
    sharpe_bins = sorted(set(max(0.05, v) for v in _bins(raw_sharpe, 9, sharpe_step)))
    dd_bins = sorted(set(max(0.01, v) for v in _bins(raw_dd, 5, dd_step)))

    def snap(val: float, bins: list[float]) -> float:
        return min(bins, key=lambda b: abs(b - val))

    def sid(s: float, d: float) -> str:
        return f"s{int(round(s * 1000)):04d}_d{int(round(d * 1000)):04d}"

    states: dict = {}
    for s in sharpe_bins:
        for d in dd_bins:
            nid = sid(s, d)
            states[nid] = {"id": nid, "attributes": {"sharpe": s, "drawdown": d}, "valid_from": 0}

    transitions: list[dict] = []
    for s in sharpe_bins:
        for d in dd_bins:
            from_id = sid(s, d)
            new_s = snap(s * 1.15, sharpe_bins)
            new_d = snap(d * 0.90, dd_bins)
            to_id = sid(new_s, new_d)
            if from_id != to_id:
                transitions.append({"from": from_id, "to": to_id, "cost": 0.002, "valid_from": 0})

    # Start: 60% of current sharpe (simulated market shock)
    start_s = snap(raw_sharpe * 0.60, sharpe_bins)
    start_d = snap(raw_dd * 1.30, dd_bins)
    start = sid(start_s, start_d)

    # Goal: all states with sharpe ≥ 85% of current and drawdown ≤ 110% of current
    goal_sharpe_floor = raw_sharpe * 0.85
    goal_dd_ceiling = raw_dd * 1.10
    goals = [
        sid(s, d)
        for s in sharpe_bins
        for d in dd_bins
        if s >= goal_sharpe_floor and d <= goal_dd_ceiling
    ]
    if not goals:
        # Fallback: nearest state to current metrics
        goals = [sid(snap(raw_sharpe, sharpe_bins), snap(raw_dd, dd_bins))]

    state_space = {"states": states, "transitions": transitions}
    logger.debug(
        f"State graph: {len(states)} states, {len(transitions)} transitions, "
        f"start={start}, goals={len(goals)}"
    )
    return state_space, start, goals


# ── E) Adversarial survival ───────────────────────────────────────────────────


def run_adversarial_survival(
    current_sharpe: float | None = None,
    current_drawdown: float | None = None,
    crash_budget: float = 0.006,
) -> dict | None:
    """
    Survival verdict under an explicit adversary crash budget.

    Scenario:
      - Portfolio is modelled as a discrete Sharpe × Drawdown state graph.
      - Defender (HRP) has budget 0.010 (5 rebalances at 20bps each).
      - Adversary (market crash) has budget `crash_budget` (default 0.006 = 3 shocks).
      - Verdict: "safe" (ε* > adversary budget), "critical" (boundary), or "doomed".
      - survival_margin = ε* − crash_budget: negative means structurally compromised.

    Args:
        current_sharpe:   Current portfolio Sharpe ratio. Loads from reports/ if None.
        current_drawdown: Current max drawdown (negative fraction). Loads if None.
        crash_budget:     Adversary budget in transition-cost units (default 0.006).

    Returns:
        API response with: survival_verdict, survival_margin, epsilon_star,
        solo_optionality, stressed_optionality, pi_regime, budget_fraction_at_risk.
    """
    # Load metrics if not provided
    if current_sharpe is None or current_drawdown is None:
        path = Path("reports") / "portfolio_metrics.csv"
        if path.exists():
            try:
                import pandas as _pd
                df = _pd.read_csv(path)
                if not df.empty:
                    row = df.iloc[0]
                    current_sharpe = current_sharpe or float(row.get("sharpe_ratio", 1.0))
                    current_drawdown = current_drawdown or float(row.get("max_drawdown", -0.15))
            except Exception:
                pass
        current_sharpe = current_sharpe or 1.0
        current_drawdown = current_drawdown or -0.15

    defender_budget = 0.010  # 5 rebalances at 20bps each

    state_space, start, goals = _build_portfolio_state_graph(current_sharpe, current_drawdown)

    logger.info(
        f"Niyati adversarial survival: sharpe={current_sharpe:.3f}, "
        f"dd={current_drawdown:.3f}, crash_budget={crash_budget}"
    )

    try:
        result = solve_adversarial(
            state_space=state_space,
            start=start,
            goals=goals,
            agent_budget=defender_budget,
            adversary_budget=crash_budget,
        )
        if result is None:
            logger.error("Niyati /v1/solve/adversarial returned None")
            return None

        verdict = result.get("survival_verdict", "unknown")
        margin = result.get("survival_margin")
        pi_regime = result.get("pi_regime", "unknown")
        logger.info(
            f"Niyati adversarial: verdict={verdict}, margin={margin}, pi_regime={pi_regime}"
        )
        return result
    except Exception as exc:
        logger.error(f"run_adversarial_survival failed: {exc}")
        return None


# ── F) Fragility check (pre-rebalance gate) ───────────────────────────────────


def run_fragility_check(
    current_sharpe: float,
    current_drawdown: float,
) -> dict | None:
    """
    Instant polytope fragility check at the current portfolio point.

    Maps the 2-D (sharpe, drawdown) space to a polytope and evaluates
    how close the current state is to the structural boundary in the
    direction of a market stress (sharpe drops, drawdown rises).

    Returns collapse_risk, thickness, kappa, anisotropy_ratio.
    High collapse_risk (>0.15) or thin thickness (<0.10) → conservative rebalance.

    Args:
        current_sharpe:   Current Sharpe ratio.
        current_drawdown: Current max drawdown (negative fraction).

    Returns:
        API response dict, or None on failure.
    """
    sharpe = max(0.05, float(current_sharpe))
    drawdown = abs(float(current_drawdown))

    # Polytope: [sharpe, drawdown] box with comfortable bounds
    constraints = [
        {"normal": [1.0, 0.0],  "offset":  3.0},   # sharpe ≤ 3.0
        {"normal": [-1.0, 0.0], "offset":  0.2},   # sharpe ≥ -0.2  (0.2 below 0)
        {"normal": [0.0, 1.0],  "offset":  0.60},  # drawdown ≤ 60%
        {"normal": [0.0, -1.0], "offset":  0.0},   # drawdown ≥ 0
    ]

    point = [round(sharpe, 4), round(drawdown, 4)]
    # Direction: market stress vector — sharpe erodes, drawdown worsens
    direction = [-0.10, 0.05]

    logger.info(f"Niyati fragility check: point={point}")

    try:
        result = solve_fragility(
            constraints=constraints,
            point=point,
            direction=direction,
        )
        if result is None:
            logger.error("Niyati /v1/solve/fragility returned None")
            return None

        cr = result.get("collapse_risk")
        th = result.get("thickness")
        kappa = result.get("kappa")
        ar = result.get("anisotropy_ratio")
        logger.info(
            f"Niyati fragility: collapse_risk={cr:.4f}, thickness={th:.4f}, "
            f"kappa={kappa:.4f}, anisotropy={ar:.2f}"
        )
        return result
    except Exception as exc:
        logger.error(f"run_fragility_check failed: {exc}")
        return None
