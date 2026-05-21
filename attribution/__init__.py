"""
Attribution — re-exports from backtests/ and fx/ packages.

MVP interface for performance attribution.
"""

from backtests.attribution import calculate_performance_attribution, calculate_allocation_attribution
from fx.attribution import calculate_fx_attribution, attribution_summary

__all__ = [
    "calculate_performance_attribution",
    "calculate_allocation_attribution",
    "calculate_fx_attribution",
    "attribution_summary",
]
