# Usage Guide — Portfolio OS

## Prerequisites

- **Python 3.13+** (tested on 3.13.3 via Homebrew on macOS)
- **Internet connection** for initial market data download

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

The main pipeline ingests data, computes analytics, optimizes allocations, backtests with friction, and runs validation — all in one command:

```bash
python app.py
```

This executes Sprints 1 through 8 sequentially:

1. **Downloads market data** from Yahoo Finance and MFAPI (first run fetches ~5 years of history)
2. **Normalizes all prices** to INR base currency
3. **Computes analytics** — returns, risk metrics, drawdowns, rolling stats, benchmark comparison
4. **Engineers features** — momentum, volatility, trend, mean reversion, composite signals
5. **Optimizes portfolio** — HRP allocation, signal tilt, weight constraints, rebalance trades
6. **Backtests** — runs the strategy with realistic taxes, slippage, and transaction costs
7. **Generates reports** — CSV summaries and HTML charts saved to `reports/`
8. **Validates** — walk-forward, regime analysis, Monte Carlo, stress tests, overfitting detection

**Runtime:** ~2–3 minutes on first run (data download), ~1 minute on subsequent runs (data cached locally).

**Output:** All computed data is persisted to `data/processed/` as Parquet files. Reports go to `reports/`.

---

## Running the Dashboard

The dashboard is a **Streamlit** application that reads pre-computed data from `data/processed/` and `reports/`. You must run the pipeline at least once before launching the dashboard.

### When to run the dashboard

- **After running the pipeline** (`python app.py`) to explore results interactively
- Anytime you want to inspect portfolio state, analytics, backtest results, or rebalance recommendations
- The dashboard reads from saved Parquet files, so it works offline after the initial pipeline run

### How to launch

```bash
# From the project root
streamlit run dashboard/app.py
```

This opens the dashboard at **http://localhost:8501** in your browser.

### Dashboard Pages

| Page | What it shows |
|------|---------------|
| **📊 Overview** | KPI cards (NAV, CAGR, Sharpe, Sortino, Max DD, Volatility), NAV curve chart, allocation pie chart |
| **📈 Analytics** | Rolling Sharpe and volatility charts, drawdown analysis, return distributions, detailed risk metrics |
| **⚖️ Optimization** | Target weights table, HRP dendrogram, weight comparison across strategies, constraint visualization |
| **🧪 Backtests** | Backtest NAV curves, friction breakdown (taxes vs costs vs slippage), trade ledger, strategy comparison |
| **🌍 Exposure** | Country allocation, currency breakdown, asset-type distribution, concentration metrics |
| **💡 Recommendations** | Actionable rebalance trades with estimated values, signal scores, current vs target weights |

### Stopping the dashboard

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## Configuration

### Asset Universe

Edit [`configs/asset_master.csv`](configs/asset_master.csv) to add or remove assets:

```csv
ticker,asset_name,asset_type,country,currency,exchange
AAPL,Apple Inc,equity,US,USD,NASDAQ
SPY,SPDR S&P 500 ETF Trust,etf,US,USD,NYSE
RELIANCE.NS,Reliance Industries,equity,IN,INR,NSE
INFY.NS,Infosys Ltd,equity,IN,INR,NSE
USDINR=X,USD/INR FX Rate,fx,GLOBAL,INR,FX
```

**Rules:**
- `ticker` must be a valid Yahoo Finance symbol (or a `=X` FX pair)
- `country` must be `US`, `IN`, or `GLOBAL` (used for tax and cost calculations)
- `currency` determines whether FX conversion is applied
- There must be exactly one `fx` row for the USD/INR pair

### Holdings

Edit [`data/holdings/current_holdings.csv`](data/holdings/current_holdings.csv) to reflect your actual positions:

```csv
ticker,quantity,avg_buy_price,currency,asset_type
AAPL,10,180.00,USD,equity
SPY,5,420.00,USD,etf
RELIANCE.NS,15,2500.00,INR,equity
INFY.NS,25,1400.00,INR,equity
```

### Environment Variables

Create a `.env` file in the project root if needed (currently optional):

```env
# No required keys — all data sources are public APIs
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_validators.py

# Run tests matching a keyword
pytest -k "metrics"

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

### Research iteration

```bash
python app.py                      # Pipeline with latest signals
# Check reports/research_score.csv for quality score
# Check reports/overfitting_flags.csv for overfitting warnings
# Check reports/walkforward_results.csv for OOS performance
streamlit run dashboard/app.py     # Visual inspection
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
| Parquet read errors | Delete `data/processed/*.parquet` and re-run `python app.py` |

---

## Data Flow

```
Yahoo Finance / MFAPI
        │
        ▼
   data/raw/*.parquet          ← Raw market data (cached)
        │
        ▼
   data/processed/*.parquet    ← Computed results (pipeline output)
        │
        ├──▶ reports/*.csv     ← Summary reports
        ├──▶ reports/*.html    ← Interactive charts
        └──▶ dashboard/        ← Streamlit reads processed data
```

The pipeline is **idempotent** — running it again overwrites computed outputs with fresh results. Raw data is cached and only re-downloaded if the cache is missing.
