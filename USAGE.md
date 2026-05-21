# Usage Guide — Portfolio OS

## Prerequisites

- **Python 3.13+** (tested on 3.13.3 via Homebrew on macOS)
- **Internet connection** for initial market data download
- **Docker** (optional) for containerized deployment

---

## Installation

```bash
# Clone the repository
git clone <repo-url> portfolio-os
cd portfolio-os

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

The main pipeline ingests data, computes analytics, detects regimes, generates ML alpha, optimizes allocations, backtests with friction, executes with utility gating, and runs full validation:

```bash
python app.py
```

This executes the full pipeline sequentially:

1. **Downloads market data** from Yahoo Finance and MFAPI (first run fetches ~5 years of history)
2. **Normalizes all prices** to INR base currency
3. **Computes analytics** — returns, risk metrics, drawdowns, rolling stats, benchmark comparison
4. **Engineers features** — momentum, volatility, trend, mean reversion, composite signals
5. **Detects regimes** — multi-signal regime classification (risk_on/risk_off/crisis/recovery)
6. **Generates ML alpha** — walk-forward LightGBM + CatBoost ensemble with confidence scoring
7. **Estimates dynamic risk** — regime-aware covariance, vol scaling, tail risk (CVaR)
8. **Optimizes portfolio** — HRP allocation, signal tilt, weight constraints, rebalance trades
9. **Evaluates execution utility** — friction vs alpha analysis, tax-loss harvesting, turnover budgeting
10. **Backtests** — runs the strategy with realistic taxes, slippage, and transaction costs
11. **Generates attribution** — Brinson decomposition, factor analysis, decision explanations
12. **Validates** — walk-forward, regime analysis, Monte Carlo, stress tests, overfitting detection
13. **Persists to warehouse** — registers 44 tables in DuckDB

**Runtime:** ~2–3 minutes on first run (data download), ~1 minute on subsequent runs (data cached locally).

**Output:** All computed data is persisted to `data/processed/` as Parquet files. Reports go to `reports/`. DuckDB warehouse is available for SQL queries.

---

## Running the Dashboard

The dashboard is a **Streamlit** application with 13 views that reads pre-computed data from `data/processed/` and `reports/`. You must run the pipeline at least once before launching the dashboard.

### How to launch

```bash
streamlit run dashboard/app.py
```

This opens the dashboard at **http://localhost:8501** in your browser.

### Dashboard Pages

| Page | What it shows |
|------|---------------|
| **📊 Overview** | KPI cards (NAV, CAGR, Sharpe, Sortino, Max DD), NAV curve, allocation pie |
| **📈 Analytics** | Rolling Sharpe/volatility, drawdown analysis, return distributions, risk metrics |
| **⚖️ Optimization** | Target weights, HRP dendrogram, weight comparison, constraints |
| **🧪 Backtests** | NAV curves, friction breakdown, trade ledger, strategy comparison |
| **🌍 Exposure** | Country/currency/asset-type allocation, concentration metrics |
| **💡 Recommendations** | Rebalance trades, signal scores, current vs target weights |
| **🏗️ Structural Health** | Research Quality Score, validation results, walk-forward OOS |
| **🔄 Regime Intelligence** | Regime detection, transition matrix, regime-specific performance |
| **⚠️ Risk Intelligence** | Risk decomposition, covariance, stress testing, tail risk |
| **⚡ Execution Intelligence** | Utility analysis, paper trading, turnover, execution state |
| **🔍 Explainability** | Attribution, factor exposures, decision narratives, alerts |
| **🛠️ Operations** | Pipeline events, SLA compliance, MLOps status, system health |
| **🎯 Command Center** | Trust scores, deployment readiness, human override layer |

---

## Running the API

The REST API serves portfolio data and actions via FastAPI:

```bash
uvicorn api.main:app --reload --port 8000
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/portfolio` | Current portfolio state & NAV |
| `POST` | `/rebalance` | Trigger rebalance computation |
| `GET` | `/regime` | Current regime classification |

API docs available at **http://localhost:8000/docs** (Swagger UI).

---

## Docker Deployment

Run the full system in containers:

```bash
cd infra
docker-compose up --build
```

This starts:
- **API** on port `8000`
- **Dashboard** on port `8501`

---

## Configuration

### Asset Universe

Edit [`configs/asset_master.csv`](configs/asset_master.csv) to add or remove assets:

```csv
ticker,asset_name,asset_type,country,currency,exchange,annual_rate
AAPL,Apple Inc,equity,US,USD,NASDAQ,
RELIANCE.NS,Reliance Industries,equity,IN,INR,NSE,
VOO,Vanguard S&P 500 ETF,etf,US,USD,NYSE,
MF_119598,SBI Bluechip Fund - Direct Growth,mf,IN,INR,AMFI,
EPF,Employee Provident Fund,fixed_income,IN,INR,GOV,8.25
PPF,Public Provident Fund,fixed_income,IN,INR,GOV,7.10
FD_SBI,SBI Fixed Deposit,fixed_income,IN,INR,BANK,7.10
SGB,Sovereign Gold Bond,fixed_income,IN,INR,GOV,2.50
GOLD_PHYS,Physical Gold,metal,IN,INR,PHYSICAL,
SILVER_PHYS,Physical Silver,metal,IN,INR,PHYSICAL,
USDINR=X,USD/INR FX Rate,fx,GLOBAL,INR,FX,
```

**Supported asset types:**

| Type | Source | Description |
|------|--------|-------------|
| `equity` | Yahoo Finance | Individual stocks (NSE/NASDAQ/NYSE) |
| `etf` | Yahoo Finance | Exchange-traded funds |
| `mf` | MFAPI (AMFI) | Indian mutual funds — ticker must be `MF_<scheme_code>` |
| `fixed_income` | Synthetic | EPF, PPF, FD, NPS, SGB — set `annual_rate` column |
| `metal` | Yahoo (commodity) | Physical gold/silver — uses GC=F/SI=F as price proxy |
| `fx` | Yahoo Finance | Currency pairs — exactly one USD/INR row required |

**Rules:**
- `ticker` must be a valid Yahoo Finance symbol, `MF_<AMFI code>`, or a custom label for fixed-income
- `country` must be `US`, `IN`, or `GLOBAL`
- `currency` determines whether FX conversion is applied
- `annual_rate` is required for `fixed_income` (percentage, e.g. 8.25 for 8.25%)
- There must be exactly one `fx` row for the USD/INR pair
- Fixed-income and metal assets are included in total NAV but excluded from optimization/backtesting

### Holdings

Edit [`data/holdings/current_holdings.csv`](data/holdings/current_holdings.csv) to reflect your actual positions:

```csv
ticker,quantity,avg_buy_price,currency,asset_type
AAPL,10,180.00,USD,equity
RELIANCE.NS,15,2500.00,INR,equity
VOO,5,420.00,USD,etf
MF_119598,1000,85.50,INR,mf
EPF,1,500000.00,INR,fixed_income
PPF,1,300000.00,INR,fixed_income
FD_SBI,1,200000.00,INR,fixed_income
SGB,10,6500.00,INR,fixed_income
GOLD_PHYS,50,7500.00,INR,metal
SILVER_PHYS,1000,95.00,INR,metal
```

**Notes:**
- For **mutual funds**, `quantity` is the number of units, `avg_buy_price` is the average NAV at purchase
- For **fixed-income** (EPF/PPF/FD/NPS), use `quantity=1` and `avg_buy_price` as the total principal amount
- For **SGB**, `quantity` is the number of bond units, `avg_buy_price` is the issue price per unit
- For **physical metals**, `quantity` is grams, `avg_buy_price` is the cost per gram (INR)

### Environment Variables

Create a `.env` file in the project root if needed (currently optional):

```env
# No required keys — all data sources are public APIs
```

---

## Running Tests

```bash
# Run all 560 tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_validators.py

# Run tests by sprint module
pytest tests/test_metrics.py              # Sprint 1 — analytics
pytest tests/test_features.py             # Sprint 1 — features
pytest tests/test_optimization.py         # Sprint 1 — optimization
pytest tests/test_backtests.py            # Sprint 1 — backtesting
pytest tests/test_returns_drawdown.py     # Sprint 1 — returns/drawdown
pytest tests/test_validation.py           # Sprint 1 — validation
pytest tests/test_integration.py          # Sprint 2 — regime intelligence
pytest tests/test_e2e.py                  # Sprint 3-8 — ML, risk, execution, monitoring, orchestration, deployment

# Run tests matching a keyword
pytest -k "regime"
pytest -k "ml_alpha"
pytest -k "risk_engine"
pytest -k "execution"
pytest -k "monitoring"
pytest -k "orchestration"
pytest -k "deployment"

# Run with coverage report
pytest --cov=. --cov-report=term-missing
```

---

## Common Workflows

### Fresh start (new machine)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py                      # Full pipeline (~3 min)
streamlit run dashboard/app.py     # Explore results
```

### Daily portfolio check

```bash
source venv/bin/activate
python app.py                      # Re-run with latest data
streamlit run dashboard/app.py     # Check recommendations
```

### After editing asset_master.csv or holdings

```bash
python app.py                      # Re-run to pick up changes
```

### Orchestration mode (event-driven)

```bash
# The orchestration engine manages the full pipeline lifecycle:
# - Event bus publishes stage completion events
# - Dependency graph resolves execution order
# - Retry engine handles transient failures (exponential backoff + circuit breaker)
# - SLA tracker monitors pipeline latency
python app.py                      # Orchestration runs automatically
```

### ML model retraining (MLOps)

```bash
# MLOps triggers retraining when:
# - Regime shift detected
# - Feature drift exceeds threshold
# - Scheduled cadence (weekly)
# - Manual trigger via API
python app.py                      # Retraining happens automatically if triggered
```

### Trust calibration & deployment readiness

```bash
# Check trust scores in Command Center dashboard view
# Trust dimensions: data_quality, model_stability, execution_reliability,
#                   risk_compliance, operational_health
streamlit run dashboard/app.py     # Navigate to Command Center
```

### Failure simulation (chaos testing)

```bash
# Failure simulation tests system resilience:
# - Component failure injection
# - Data corruption scenarios
# - Latency injection
# - Recovery verification
python app.py                      # Failure sim runs as part of deployment validation
```

### Research iteration

```bash
python app.py                      # Pipeline with latest signals
# Check reports/research_score.csv for quality score
# Check reports/overfitting_flags.csv for overfitting warnings
# Check reports/walkforward_results.csv for OOS performance
streamlit run dashboard/app.py     # Visual inspection
```

### Docker deployment

```bash
cd infra
docker-compose up --build          # Start API + Dashboard
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Swagger: http://localhost:8000/docs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate the venv: `source venv/bin/activate` |
| `No portfolio data` in dashboard | Run `python app.py` first to generate data |
| Yahoo Finance download fails | Check internet connection; yfinance may rate-limit — retry after a few minutes |
| `streamlit: command not found` | Install: `pip install streamlit` |
| Dashboard shows stale data | Re-run `python app.py` to refresh |
| Port 8501 in use | Kill the old Streamlit process or use `streamlit run dashboard/app.py --server.port 8502` |
| Port 8000 in use | Kill the old uvicorn process or use `--port 8001` |
| Parquet read errors | Delete `data/processed/*.parquet` and re-run `python app.py` |
| DuckDB lock errors | Ensure only one process accesses the warehouse at a time |
| MLflow tracking errors | Check `mlruns/` directory permissions |
| Docker build fails | Ensure Docker Desktop is running; check `infra/` Dockerfiles |

---

## Data Flow

```
Yahoo Finance / MFAPI
        │
        ▼
   data/raw/*.parquet              ← Raw market data (cached)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  Pipeline Stages                       │
│                                                       │
│  Ingestion → FX → Analytics → Features → Regimes     │
│       → ML Alpha → Risk Engine → Optimization         │
│       → Execution → Backtesting → Attribution         │
│       → Validation → Warehouse                        │
└───────────────────────────────────────────────────────┘
        │
        ├──▶ data/processed/*.parquet  ← Computed results (44 tables)
        ├──▶ reports/*.csv             ← Summary reports
        ├──▶ reports/*.html            ← Interactive charts
        ├──▶ warehouse (DuckDB)        ← SQL-queryable warehouse
        └──▶ dashboard/                ← Streamlit reads processed data
```

The pipeline is **idempotent** — running it again overwrites computed outputs with fresh results. Raw data is cached and only re-downloaded if the cache is missing.

### Orchestration Lifecycle

```
Event Bus (pub/sub)
        │
        ▼
Dependency Graph (DAG)  ←── Scheduling (daily/weekly/monthly)
        │
        ├──▶ Stage execution with retry & circuit breaker
        ├──▶ SLA monitoring (latency tracking)
        ├──▶ MLOps (retraining triggers, shadow deployment)
        └──▶ Governance (config snapshots, versioning)
```
