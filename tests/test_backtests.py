"""
Tests for backtests/engine.py — friction-aware backtest engine.
"""

import numpy as np
import pandas as pd
import pytest

from backtests.engine import run_backtest
from backtests.portfolio_state import PortfolioState, TaxLot
from backtests.costs import calculate_transaction_costs
from backtests.taxes import calculate_capital_gains_tax


class TestRunBacktest:
    def test_returns_dict(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        assert isinstance(result, dict)

    def test_required_keys(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        for key in ["nav_series", "ledger", "portfolio_state", "rebalance_log"]:
            assert key in result, f"Missing key: {key}"

    def test_nav_series_is_dataframe(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        nav = result["nav_series"]
        assert isinstance(nav, pd.DataFrame)
        assert "nav" in nav.columns or "portfolio_nav" in nav.columns

    def test_nav_starts_near_initial_capital(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        nav = result["nav_series"]
        nav_col = "nav" if "nav" in nav.columns else "portfolio_nav"
        first_nav = nav[nav_col].iloc[0]
        assert abs(first_nav - 1_000_000) < 50_000  # within 5%

    def test_rebalance_log_not_empty(self, wide_prices, equal_weight_strategy, country_map):
        result = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=10,
            country_map=country_map,
            warmup_days=60,
        )
        assert len(result["rebalance_log"]) > 0

    def test_higher_slippage_lower_nav(self, wide_prices, equal_weight_strategy, country_map):
        result_low = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=5,
            country_map=country_map,
            warmup_days=60,
        )
        result_high = run_backtest(
            wide_prices=wide_prices,
            strategy_fn=equal_weight_strategy,
            initial_capital=1_000_000,
            frequency="quarterly",
            slippage_bps=100,
            country_map=country_map,
            warmup_days=60,
        )
        nav_col = "nav"
        nav_low = result_low["nav_series"][nav_col].iloc[-1]
        nav_high = result_high["nav_series"][nav_col].iloc[-1]
        assert nav_low > nav_high


class TestPortfolioState:
    def test_initial_state(self):
        ps = PortfolioState(cash=1_000_000)
        assert ps.cash == 1_000_000
        assert len(ps.holdings) == 0

    def test_buy_updates_holdings(self):
        ps = PortfolioState(cash=1_000_000)
        ps.buy("AAPL", 10, 150.0, pd.Timestamp("2020-06-01"))
        assert "AAPL" in ps.holdings
        assert ps.holdings["AAPL"] == 10

    def test_buy_reduces_cash(self):
        ps = PortfolioState(cash=1_000_000)
        ps.buy("AAPL", 10, 150.0, pd.Timestamp("2020-06-01"))
        assert ps.cash < 1_000_000

    def test_nav_calculation(self):
        ps = PortfolioState(cash=500_000)
        ps.buy("AAPL", 10, 100.0, pd.Timestamp("2020-01-01"))
        nav = ps.nav({"AAPL": 150.0})
        expected = ps.cash + 10 * 150.0
        assert abs(nav - expected) < 1.0


class TestTransactionCosts:
    def test_india_buy(self):
        cost = calculate_transaction_costs("RELIANCE.NS", 10, 2500.0, "BUY", "IN")
        assert isinstance(cost, dict)
        assert cost["total_cost"] > 0

    def test_india_sell_has_stt(self):
        cost = calculate_transaction_costs("RELIANCE.NS", 10, 2500.0, "SELL", "IN")
        assert cost["total_cost"] > 0

    def test_us_buy(self):
        cost = calculate_transaction_costs("AAPL", 10, 15000.0, "BUY", "US")
        assert isinstance(cost, dict)


class TestCapitalGainsTax:
    def test_short_term_india(self):
        result = calculate_capital_gains_tax(
            sell_price=200.0,
            cost_price=100.0,
            quantity=10,
            purchase_date=pd.Timestamp("2024-06-01"),
            sell_date=pd.Timestamp("2024-12-01"),
            country="IN",
        )
        assert result["tax"] > 0
        assert result["is_ltcg"] is False

    def test_long_term_india(self):
        result = calculate_capital_gains_tax(
            sell_price=200.0,
            cost_price=100.0,
            quantity=10,
            purchase_date=pd.Timestamp("2023-01-01"),
            sell_date=pd.Timestamp("2024-06-01"),
            country="IN",
        )
        assert result["is_ltcg"] is True

    def test_no_gain_no_tax(self):
        result = calculate_capital_gains_tax(
            sell_price=100.0,
            cost_price=100.0,
            quantity=10,
            purchase_date=pd.Timestamp("2023-01-01"),
            sell_date=pd.Timestamp("2024-06-01"),
            country="IN",
        )
        assert result["tax"] == 0.0

    def test_loss_no_tax(self):
        result = calculate_capital_gains_tax(
            sell_price=50.0,
            cost_price=100.0,
            quantity=10,
            purchase_date=pd.Timestamp("2023-01-01"),
            sell_date=pd.Timestamp("2024-06-01"),
            country="IN",
        )
        assert result["tax"] == 0.0
