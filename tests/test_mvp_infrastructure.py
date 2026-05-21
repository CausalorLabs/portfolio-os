"""Tests for MVP infrastructure — configs, contracts, warehouse, API, monitoring."""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── Config tests ─────────────────────────────────────────────────────────────


class TestConfig:
    def test_load_base_config(self):
        from configs import load_config

        cfg = load_config()
        assert cfg.system.name == "portfolio-os"
        assert cfg.system.version == "1.0.0-mvp"
        assert cfg.optimization.method == "hrp"

    def test_load_dev_config(self):
        from configs import load_config

        cfg = load_config("dev")
        assert cfg.system.env == "dev"
        assert cfg.system.log_level == "DEBUG"

    def test_load_prod_config(self):
        from configs import load_config

        cfg = load_config("prod")
        assert cfg.system.env == "prod"
        assert cfg.system.log_level == "WARNING"
        assert cfg.validation.monte_carlo.n_paths == 5000

    def test_config_values(self):
        from configs import load_config

        cfg = load_config()
        assert cfg.analytics.trading_days == 252
        assert cfg.backtesting.initial_capital == 1_000_000.0
        assert cfg.rebalance.drift_threshold == 0.05
        assert cfg.features.composite_weights.momentum == 0.40

    def test_missing_env_returns_base(self):
        from configs import load_config

        cfg = load_config("nonexistent")
        assert cfg.system.name == "portfolio-os"


# ── Contract tests ───────────────────────────────────────────────────────────


class TestContracts:
    def test_holdings_snapshot(self):
        from contracts import AssetType, Country, Currency, HoldingsSnapshot

        h = HoldingsSnapshot(
            date=date(2024, 1, 1),
            ticker="AAPL",
            quantity=100,
            avg_cost=150.0,
            currency=Currency.USD,
            asset_class=AssetType.EQUITY,
            country=Country.US,
        )
        assert h.ticker == "AAPL"
        assert h.quantity == 100

    def test_holdings_negative_quantity_rejected(self):
        from contracts import AssetType, Country, Currency, HoldingsSnapshot

        with pytest.raises(Exception):
            HoldingsSnapshot(
                date=date(2024, 1, 1),
                ticker="AAPL",
                quantity=-10,
                avg_cost=150.0,
                currency=Currency.USD,
                asset_class=AssetType.EQUITY,
                country=Country.US,
            )

    def test_price_record(self):
        from contracts import Currency, PriceRecord

        p = PriceRecord(
            date=date(2024, 1, 1),
            ticker="AAPL",
            close=195.50,
            currency=Currency.USD,
        )
        assert p.close == 195.50

    def test_fx_rate_positive(self):
        from contracts import FXRate

        with pytest.raises(Exception):
            FXRate(date=date(2024, 1, 1), pair="USDINR", rate=-83.5)

    def test_portfolio_metrics(self):
        from contracts import PortfolioMetrics

        m = PortfolioMetrics(
            cagr=0.15,
            sharpe_ratio=1.8,
            max_drawdown=-0.12,
            annualized_volatility=0.18,
        )
        assert m.sharpe_ratio == 1.8

    def test_allocation_weight_bounds(self):
        from contracts import AllocationWeight

        with pytest.raises(Exception):
            AllocationWeight(ticker="AAPL", weight=1.5, method="hrp")

    def test_signal_score_bounds(self):
        from contracts import SignalScore

        s = SignalScore(
            date=date(2024, 1, 1),
            ticker="AAPL",
            composite_score=0.85,
        )
        assert 0 <= s.composite_score <= 1

    def test_execution_context_validation(self):
        from contracts import ExecutionContext

        with pytest.raises(Exception):
            ExecutionContext(
                execution_id="short",
                pipeline="test",
                started_at=datetime.now(timezone.utc),
                environment="dev",
            )

    def test_rebalance_decision(self):
        from contracts import RebalanceDecision, RebalanceMethod

        d = RebalanceDecision(
            should_rebalance=True,
            method=RebalanceMethod.THRESHOLD,
            max_drift=0.08,
            threshold=0.05,
            reason="Drift exceeds threshold",
        )
        assert d.should_rebalance is True


# ── Warehouse tests ──────────────────────────────────────────────────────────


class TestWarehouse:
    def test_available_tables(self):
        from warehouse import Warehouse

        wh = Warehouse(db_path=":memory:")
        tables = wh.available_tables()
        assert "inr_prices" in tables
        assert "portfolio_nav" in tables
        wh.close()

    def test_write_and_read(self, tmp_path):
        from warehouse import Warehouse

        wh = Warehouse(db_path=":memory:", data_dir=tmp_path)
        df = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"], "inr_price": [16000.0], "fx_rate": [83.5]})
        wh.write_parquet(df, "inr_prices")
        result = wh.read_table("inr_prices")
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"
        wh.close()

    def test_query(self, tmp_path):
        from warehouse import Warehouse

        wh = Warehouse(db_path=":memory:", data_dir=tmp_path)
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "ticker": ["AAPL", "MSFT"], "inr_price": [16000.0, 30000.0], "fx_rate": [83.5, 83.6]})
        wh.write_parquet(df, "inr_prices")
        result = wh.query("SELECT ticker, inr_price FROM inr_prices WHERE ticker = 'AAPL'")
        assert len(result) == 1
        wh.close()

    def test_unknown_table_raises(self):
        from warehouse import Warehouse

        wh = Warehouse(db_path=":memory:")
        with pytest.raises(ValueError, match="Unknown table"):
            wh.read_table("nonexistent_table")
        wh.close()


# ── Monitoring tests ─────────────────────────────────────────────────────────


class TestMonitoring:
    def test_check_drift_alerts(self):
        from monitoring.alerts import check_drift

        current = {"AAPL": 0.30, "MSFT": 0.20, "GOOGL": 0.50}
        target = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.50}
        alerts = check_drift(current, target, threshold=0.04)
        assert len(alerts) == 2  # AAPL drifted 5%, MSFT drifted 5%
        assert all(a.drift > 0.04 for a in alerts)

    def test_check_drift_no_alerts(self):
        from monitoring.alerts import check_drift

        current = {"AAPL": 0.25, "MSFT": 0.25}
        target = {"AAPL": 0.25, "MSFT": 0.25}
        alerts = check_drift(current, target, threshold=0.05)
        assert len(alerts) == 0

    def test_check_drawdown(self):
        from monitoring.alerts import check_drawdown

        nav = pd.Series([100, 105, 103, 95, 85, 80])
        alert = check_drawdown(nav, threshold=-0.15)
        assert alert is not None
        assert alert.current_drawdown < -0.15

    def test_check_drawdown_no_breach(self):
        from monitoring.alerts import check_drawdown

        nav = pd.Series([100, 105, 103, 100])
        alert = check_drawdown(nav, threshold=-0.15)
        assert alert is None

    def test_check_concentration(self):
        from monitoring.alerts import check_concentration

        weights = {"AAPL": 0.60, "MSFT": 0.40}
        alert = check_concentration(weights, hhi_threshold=0.25)
        assert alert is not None  # HHI = 0.36 + 0.16 = 0.52

    def test_pipeline_context(self):
        from monitoring import pipeline_context

        with pipeline_context("test_pipeline") as exec_id:
            assert len(exec_id) == 12

    def test_step_timer(self):
        from monitoring import step_timer

        with step_timer("test_step"):
            pass  # should not raise


# ── API tests ────────────────────────────────────────────────────────────────


class TestAPI:
    @pytest.fixture()
    def client(self):
        from api import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0-mvp"

    def test_portfolio_current(self, client):
        r = client.get("/portfolio/current")
        # Either 200 (data exists) or 503 (no data)
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "nav" in data
            assert "metrics" in data
            assert "holdings" in data

    def test_portfolio_history(self, client):
        r = client.get("/portfolio/history?limit=10")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, list)
            assert len(data) <= 10

    def test_rebalance_proposed(self, client):
        r = client.get("/rebalance/proposed")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "decision" in data
            assert "trades" in data

    def test_regime_current(self, client):
        r = client.get("/regime/current")
        assert r.status_code in (200, 503)
