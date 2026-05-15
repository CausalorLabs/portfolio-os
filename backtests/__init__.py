"""
backtests — Friction-aware portfolio backtesting engine.
"""

from backtests.engine import run_backtest
from backtests.portfolio_state import PortfolioState
from backtests.benchmark import run_benchmark_suite, compare_backtest_results
from backtests.attribution import calculate_performance_attribution
from backtests.reporting import generate_backtest_report
