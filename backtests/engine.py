"""
Core backtesting engine — time-based portfolio simulation with
friction-aware execution.

For each rebalance date:
    1. Compute target weights (from optimizer)
    2. Generate trade orders
    3. Simulate execution (slippage)
    4. Apply transaction costs
    5. Apply taxes
    6. Update portfolio state
    7. Record metrics

Supports monthly and quarterly rebalance frequencies.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from backtests.portfolio_state import PortfolioState
from backtests.rebalance import calculate_target_positions, generate_trade_orders
from backtests.execution import simulate_execution
from backtests.costs import calculate_transaction_costs
from backtests.taxes import calculate_tax_on_lots
from backtests.ledger import TradeLedger


def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """
    Select rebalance dates from the full date index.

    Monthly  → last trading day of each month.
    Quarterly → last trading day of each quarter.
    """
    s = pd.Series(range(len(dates)), index=dates)
    if frequency == "monthly":
        grouped = s.resample("ME").last()
    elif frequency == "quarterly":
        grouped = s.resample("QE").last()
    else:
        grouped = s.resample("ME").last()

    rebal_dates = dates[grouped.dropna().astype(int).values]
    return list(rebal_dates)


def _prices_on_date(
    wide_prices: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Get price dict for a date from wide-format prices."""
    if date not in wide_prices.index:
        # Find nearest prior date
        prior = wide_prices.index[wide_prices.index <= date]
        if prior.empty:
            return {}
        date = prior[-1]
    return wide_prices.loc[date].to_dict()


def run_backtest(
    wide_prices: pd.DataFrame,
    strategy_fn,
    initial_capital: float = 1_000_000.0,
    frequency: str = "quarterly",
    slippage_bps: float = 10,
    country_map: dict[str, str] | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    warmup_days: int = 120,
) -> dict:
    """
    Run a full friction-aware backtest.

    Parameters
    ----------
    wide_prices : pd.DataFrame
        Wide-format daily prices in INR (columns = tickers, index = dates).
    strategy_fn : callable
        Function(returns_up_to_date, tickers) → dict[str, float] (weights).
    initial_capital : float
        Starting cash in INR.
    frequency : str
        "monthly" or "quarterly".
    slippage_bps : float
        Slippage in basis points.
    country_map : dict
        ticker → country code.
    start_date : pd.Timestamp, optional
        Backtest start. Default: after warmup period.
    end_date : pd.Timestamp, optional
        Backtest end. Default: last available date.
    warmup_days : int
        Days of data before first rebalance (for vol/signal estimation).

    Returns
    -------
    dict
        Keys: nav_series (pd.DataFrame), ledger (TradeLedger),
              portfolio_state (PortfolioState), rebalance_log (list),
              daily_states (list)
    """
    if country_map is None:
        country_map = {}

    dates = wide_prices.index
    tickers = list(wide_prices.columns)

    if start_date is None:
        start_date = dates[warmup_days] if len(dates) > warmup_days else dates[0]
    if end_date is None:
        end_date = dates[-1]

    mask = (dates >= start_date) & (dates <= end_date)
    sim_dates = dates[mask]

    rebal_dates = _get_rebalance_dates(sim_dates, frequency)
    rebal_set = set(rebal_dates)

    logger.info(
        f"Backtest: {start_date.date()} → {end_date.date()}, "
        f"{frequency}, {len(rebal_dates)} rebalances, "
        f"{len(tickers)} assets, ₹{initial_capital:,.0f} capital"
    )

    # Initialize
    state = PortfolioState(cash=initial_capital)
    ledger = TradeLedger()
    daily_states: list[dict] = []
    rebalance_log: list[dict] = []

    for date in sim_dates:
        prices = _prices_on_date(wide_prices, date)
        if not prices:
            continue

        # Record daily state
        snap = state.snapshot(date, prices)
        daily_states.append(snap)

        # Rebalance on designated dates
        if date in rebal_set:
            _step_rebalance(
                state=state,
                ledger=ledger,
                rebalance_log=rebalance_log,
                date=date,
                prices=prices,
                wide_prices=wide_prices,
                strategy_fn=strategy_fn,
                tickers=tickers,
                country_map=country_map,
                slippage_bps=slippage_bps,
                warmup_days=warmup_days,
            )

    # Build NAV series
    nav_df = pd.DataFrame(daily_states)

    logger.info(
        f"Backtest complete: {len(daily_states)} days, "
        f"{len(rebalance_log)} rebalances"
    )

    return {
        "nav_series": nav_df,
        "ledger": ledger,
        "portfolio_state": state,
        "rebalance_log": rebalance_log,
        "daily_states": daily_states,
    }


def _step_rebalance(
    state: PortfolioState,
    ledger: TradeLedger,
    rebalance_log: list[dict],
    date: pd.Timestamp,
    prices: dict[str, float],
    wide_prices: pd.DataFrame,
    strategy_fn,
    tickers: list[str],
    country_map: dict[str, str],
    slippage_bps: float,
    warmup_days: int,
) -> None:
    """Execute a single rebalance step."""
    # 1. Compute returns up to this date
    hist = wide_prices.loc[:date]
    returns = hist.pct_change().dropna()

    if len(returns) < warmup_days:
        return

    # 2. Get target weights from strategy
    target_weights = strategy_fn(returns, tickers)
    if not target_weights:
        return

    # 3. Calculate target positions
    current_nav = state.nav(prices)
    if current_nav <= 0:
        return

    target_pos = calculate_target_positions(target_weights, current_nav, prices)

    # 4. Generate trade orders
    orders = generate_trade_orders(
        state.holdings, target_pos, prices, country_map
    )

    if not orders:
        rebalance_log.append({"date": date, "n_trades": 0, "turnover": 0.0})
        return

    # 5. Simulate execution
    executed = simulate_execution(orders, prices, slippage_bps)

    # 6. Apply costs, taxes, and update state
    total_turnover = 0.0
    for trade in executed:
        ticker = trade["ticker"]
        action = trade["action"]
        qty = trade["quantity"]
        exec_price = trade["execution_price"]
        country = trade["country"]

        # Transaction costs
        cost_detail = calculate_transaction_costs(
            ticker, qty, exec_price, action, country
        )
        txn_cost = cost_detail["total_cost"]

        # Tax (sells only)
        tax = 0.0
        realized = 0.0
        if action == "SELL":
            lots = state.tax_lots.get(ticker, [])
            tax, _ = calculate_tax_on_lots(lots, exec_price, qty, date, country)
            realized = state.sell(ticker, qty, exec_price, date, txn_cost, tax)
        else:
            state.buy(ticker, qty, exec_price, date, txn_cost)

        # Record in ledger
        ledger.record(
            date=date,
            ticker=ticker,
            action=action,
            quantity=qty,
            market_price=trade["market_price"],
            execution_price=exec_price,
            slippage_cost=trade["slippage_cost"],
            transaction_cost=txn_cost,
            tax=tax,
            realized_pnl=realized,
            country=country,
        )

        total_turnover += trade["notional"]

    turnover_pct = total_turnover / current_nav if current_nav > 0 else 0.0

    rebalance_log.append({
        "date": date,
        "n_trades": len(executed),
        "turnover": turnover_pct,
        "nav_before": current_nav,
        "nav_after": state.nav(prices),
    })
