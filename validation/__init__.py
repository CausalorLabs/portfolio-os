"""
Validation — research hardening, robustness testing, and overfitting detection.

Public API:
    from validation.walkforward import run_walkforward_validation
    from validation.regimes import identify_market_regimes, evaluate_regime_performance
    from validation.robustness import run_parameter_sensitivity, evaluate_stability_surface
    from validation.overfitting import detect_overfitting, calculate_strategy_stability
    from validation.signal_decay import calculate_signal_decay, evaluate_forward_returns
    from validation.monte_carlo import run_monte_carlo_simulation
    from validation.stress_tests import run_stress_scenarios, simulate_liquidity_stress
    from validation.diagnostics import generate_diagnostics
    from validation.reporting import generate_validation_report
    from validation.research_score import calculate_research_score
"""

from validation.walkforward import run_walkforward_validation
from validation.regimes import identify_market_regimes, evaluate_regime_performance
from validation.robustness import run_parameter_sensitivity, evaluate_stability_surface
from validation.overfitting import detect_overfitting, calculate_strategy_stability
from validation.signal_decay import calculate_signal_decay, evaluate_forward_returns
from validation.monte_carlo import run_monte_carlo_simulation
from validation.stress_tests import run_stress_scenarios, simulate_liquidity_stress
from validation.diagnostics import generate_diagnostics
from validation.reporting import generate_validation_report
from validation.research_score import calculate_research_score
