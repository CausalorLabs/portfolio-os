"""
Niyati API client — pure requests-based, no SDK.

Base URL:  https://api.causalorlabs.com
Version:   0.3.17
"""

import os

import requests
from loguru import logger

BASE_URL = "https://api.causalorlabs.com"
NIYATI_KEY = os.getenv("NIYATI_API_KEY", "niyati_int_0149_0149_0149")
NIYATI_VERSION = "0.3.17"
TIMEOUT = 30

_HEADERS = {
    "Content-Type": "application/json",
    "X-Niyati-Version": NIYATI_VERSION,
    "X-Niyati-Key": NIYATI_KEY,
}


def _post(path: str, payload: dict) -> dict | None:
    """POST to the Niyati API. Returns parsed JSON or None on error."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, headers=_HEADERS, timeout=TIMEOUT)
        if not resp.ok:
            logger.error(
                f"Niyati API error {resp.status_code} at {path}: {resp.text[:400]}"
            )
            resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error(f"Niyati request failed for {path}: {exc}")
        return None


def simulate(schema: dict, capability: str = "full") -> dict | None:
    """
    POST /v1/simulate — full scenario simulation.

    Args:
        schema:     Niyati scenario schema dict (resources, variables, transitions,
                    actions, goal, horizon, …).
        capability: API capability tier (default "full").

    Returns:
        Parsed API response dict, or None on failure.
    """
    payload = {"schema": schema, "capability": capability}
    logger.debug(f"Niyati /v1/simulate — capability={capability}")
    return _post("/v1/simulate", payload)


def solve_competition(
    constraints: list[dict],
    defender_position: list[float],
    defender_action: list[float],
    attacker_strength: float,
) -> dict | None:
    """
    POST /v1/solve/competition — market shock vs defender.

    Args:
        constraints:        Polytope as list of {"normal": [...], "offset": float}.
        defender_position:  Current position vector.
        defender_action:    Defender's response action vector.
        attacker_strength:  Attacker budget scalar (0–1).

    Returns:
        Parsed API response dict, or None on failure.
    """
    payload = {
        "constraints": constraints,
        "defender_position": defender_position,
        "defender_action": defender_action,
        "attacker_strength": attacker_strength,
    }
    logger.debug(f"Niyati /v1/solve/competition — attacker_strength={attacker_strength}")
    return _post("/v1/solve/competition", payload)


def solve_adversarial_allocation(
    systems: list[dict],
    attacker_budget: float,
    coupling_gamma: float,
    reinforcement_beta: float,
) -> dict | None:
    """
    POST /v1/solve/adversarial-allocation — which asset class gets hit first.

    Args:
        systems:            List of {"name": str, "tau_star": float, "kappa_base": float}.
        attacker_budget:    Total budget the attacker can deploy (0–1).
        coupling_gamma:     Contagion coupling factor between systems (0–1).
        reinforcement_beta: Defender's reinforcement rate per round (0–1).

    Returns:
        Parsed API response dict, or None on failure.
    """
    payload = {
        "systems": systems,
        "attacker_budget": attacker_budget,
        "coupling_gamma": coupling_gamma,
        "reinforcement_beta": reinforcement_beta,
    }
    logger.debug(
        f"Niyati /v1/solve/adversarial-allocation — {len(systems)} systems, "
        f"budget={attacker_budget}"
    )
    return _post("/v1/solve/adversarial-allocation", payload)


def solve_saddle_point(
    systems: list[dict],
    attacker_budget_per_round: float,
    defender_budget_per_round: float,
    coupling_gamma: float,
    horizon: int,
) -> dict | None:
    """
    POST /v1/solve/saddle-point — sustained stress, rounds to collapse.

    Args:
        systems:                   List of {"name": str, "tau_star": float, "kappa_base": float}.
        attacker_budget_per_round: Monthly attacker budget (0–1).
        defender_budget_per_round: Monthly defender budget (0–1).
        coupling_gamma:            Contagion coupling factor (0–1).
        horizon:                   Number of rounds.

    Returns:
        Parsed API response dict, or None on failure.
    """
    payload = {
        "systems": systems,
        "attacker_budget_per_round": attacker_budget_per_round,
        "defender_budget_per_round": defender_budget_per_round,
        "coupling_gamma": coupling_gamma,
        "horizon": horizon,
    }
    logger.debug(
        f"Niyati /v1/solve/saddle-point — horizon={horizon}, "
        f"atk={attacker_budget_per_round}, def={defender_budget_per_round}"
    )
    return _post("/v1/solve/saddle-point", payload)


def solve_adversarial(
    state_space: dict,
    start: str,
    goals: list[str],
    agent_budget: float,
    adversary_budget: float,
    reinforcement_budget: float = 0.0,
) -> dict | None:
    """
    POST /v1/solve/adversarial — survival verdict under explicit adversary budget ξ.

    Args:
        state_space:          {"states": {id: {...}}, "transitions": [{from,to,cost}]}
        start:                Start state ID (defender's starting state).
        goals:                List of goal state IDs.
        agent_budget:         Defender's total friction budget.
        adversary_budget:     ξ — attacker's perturbation budget.
        reinforcement_budget: β — additional defender reinforcement (default 0.0).

    Returns:
        Dict with: survival_verdict, survival_margin, epsilon_star,
                   solo/stressed optionality, pi_regime, recommended_path. None on failure.
    """
    payload = {
        "state_space": state_space,
        "agent": {"start": start, "budget": agent_budget},
        "adversary": {"budget": adversary_budget},
        "goals": goals,
        "reinforcement_budget": reinforcement_budget,
    }
    logger.debug(
        f"Niyati /v1/solve/adversarial — start={start}, "
        f"adv_budget={adversary_budget}"
    )
    return _post("/v1/solve/adversarial", payload)


def solve_fragility(
    constraints: list[dict],
    point: list[float],
    direction: list[float],
) -> dict | None:
    """
    POST /v1/solve/fragility — single-point structural check.

    Args:
        constraints: Polytope as list of {"normal": [...], "offset": float}.
        point:       Current state vector to check.
        direction:   Direction of perturbation.

    Returns:
        Parsed API response dict, or None on failure.
    """
    payload = {
        "constraints": constraints,
        "point": point,
        "direction": direction,
    }
    logger.debug(f"Niyati /v1/solve/fragility — point={point}")
    return _post("/v1/solve/fragility", payload)
