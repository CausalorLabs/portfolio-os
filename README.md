# Portfolio OS — Personal Portfolio Research Engine

A modular quantitative portfolio research system that ingests multi-currency market data, normalizes it to a base currency, engineers alpha signals, optimizes allocations, backtests with realistic friction (taxes, slippage, transaction costs), validates out-of-sample, and surfaces everything through an interactive dashboard.

---

## Thesis

> **Can friction-aware portfolio optimization outperform passive allocation on a risk-adjusted basis?**

Most retail portfolio tools ignore the real costs of trading — taxes, slippage, FX spreads, and transaction fees. Portfolio OS models these explicitly and measures whether the net-of-friction alpha justifies active management.

---

## Supported Assets

| Type | Examples | Source |
|------|----------|--------|
| US equities | AAPL | Yahoo Finance |
| US ETFs | SPY | Yahoo Finance |
| Indian equities (NSE) | RELIANCE.NS, INFY.NS | Yahoo Finance |
| Indian mutual funds | SBI Bluechip Direct Growth | MFAPI |
| FX rates | USD/INR | Yahoo Finance |

Assets are declared in [`configs/asset_master.csv`](configs/asset_master.csv). Current holdings live in [`data/holdings/current_holdings.csv`](data/holdings/current_holdings.csv).

---

## Architecture

Portfolio OS is built as a modular pipeline, each stage adding a layer of capability. The entry point [`app.py`](app.py) orchestrates all stages sequentially.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ingestion   │────▶│  FX & NAV    │────▶│  Analytics   │────▶│  Features    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  Validation  │◀────│  Backtesting │◀────│ Optimization │◀───────────┘
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────────────┐
                     │  Dashboard   │
                     └──────────────┘
```

### Module Breakdown

| # | Module | Purpose |
|--------|--------|---------|
| **1** | `ingestion/` | Download and validate market data from Yahoo Finance and MFAPI. Persist raw data as Parquet files. |
| **2** | `fx/` | Normalize all prices to INR base currency. Calculate portfolio NAV, FX attribution, and exposure breakdowns. |
| **3** | `analytics/` | Compute risk metrics (CAGR, Sharpe, Sortino, max drawdown, Calmar), rolling analytics, drawdown periods, and benchmark comparisons. Generate HTML charts and CSV reports. |
| **4** | `features/` | Engineer alpha signals — momentum, volatility, trend, mean reversion, factor composites. Build a feature store with composite scoring and signal ranking. |
| **5** | `optimization/` | Portfolio construction — Hierarchical Risk Parity (HRP), equal weight, inverse volatility, risk parity baselines. Weight constraints, signal-tilted allocation, turnover management, and rebalance scheduling. |
| **6** | `backtests/` | Friction-aware backtesting engine with realistic transaction costs (Indian STT, GST, stamp duty; US SEC fees), capital gains taxes (STCG/LTCG with lot tracking), slippage modeling, and benchmark comparison suite. |
| **7** | `dashboard/` | Streamlit-based interactive research interface with 6 pages: Overview, Analytics, Optimization, Backtests, Exposure, and Recommendations. |
| **8** | `validation/` | Research hardening — walk-forward validation, market regime analysis, parameter sensitivity, overfitting detection, signal decay, Monte Carlo simulation, stress testing, and a composite Research Quality Score. |

---

## Project Structure

```
portfolio-os/
├── app.py                  # Main pipeline entry point
│
├── ingestion/              # Data loaders
│   ├── yahoo_loader.py     #   Yahoo Finance downloader
│   └── mf_loader.py        #   Indian mutual fund loader (MFAPI)
│
├── fx/                     # Currency normalization
│   ├── fx_loader.py        #   FX rate fetcher (USD/INR)
│   ├── converter.py        #   Multi-currency → INR converter
│   └── attribution.py      #   FX vs local-return attribution
│
├── analytics/              # Portfolio analytics & risk
│   ├── metrics.py          #   Core metrics (CAGR, Sharpe, Sortino, etc.)
│   ├── portfolio_nav.py    #   Portfolio NAV calculation
│   ├── returns.py          #   Daily, log, cumulative, rolling returns
│   ├── drawdown.py         #   Drawdown series & period analysis
│   ├── rolling.py          #   Rolling analytics (vol, Sharpe, beta)
│   ├── benchmark.py        #   Benchmark comparison engine
│   ├── exposure.py         #   Country/currency/asset-type exposure
│   ├── holdings_loader.py  #   Holdings CSV reader
│   └── charts.py           #   Plotly HTML chart generation
│
├── features/               # Alpha signal engineering
│   ├── feature_store.py    #   Build/save/load feature store
│   ├── signal_ranker.py    #   Composite score & ranking
│   ├── momentum.py         #   Momentum features (5d–252d)
│   ├── volatility.py       #   Realized volatility features
│   ├── trend.py            #   SMA ratios, trend strength
│   ├── mean_reversion.py   #   Z-scores, Bollinger signals
│   ├── returns.py          #   Return-based features
│   ├── factor_features.py  #   Multi-factor composites
│   └── validators.py       #   Feature validation & lookahead checks
│
├── optimization/           # Portfolio construction
│   ├── hrp.py              #   Hierarchical Risk Parity
│   ├── allocator.py        #   Signal-tilted portfolio builder
│   ├── baselines.py        #   Equal weight, inverse vol, risk parity
│   ├── constraints.py      #   Weight caps, country limits
│   ├── covariance.py       #   Covariance estimation (shrinkage)
│   ├── turnover.py         #   Turnover calculation & drift
│   ├── rebalance.py        #   Rebalance scheduling & trade generation
│   └── reporting.py        #   Allocation reports
│
├── backtests/              # Friction-aware backtesting
│   ├── engine.py           #   Core backtest engine
│   ├── portfolio_state.py  #   Portfolio state & tax lot tracking
│   ├── ledger.py           #   Trade ledger
│   ├── costs.py            #   Transaction cost models (IN/US)
│   ├── taxes.py            #   Capital gains tax engine (STCG/LTCG)
│   ├── execution.py        #   Order execution with slippage
│   ├── rebalance.py        #   Backtest rebalance logic
│   ├── benchmark.py        #   Benchmark strategy suite
│   ├── attribution.py      #   Performance attribution
│   └── reporting.py        #   Backtest reports
│
├── validation/             # Research hardening
│   ├── walkforward.py      #   Walk-forward train/test validation
│   ├── regimes.py          #   Market regime detection & eval
│   ├── robustness.py       #   Parameter sensitivity analysis
│   ├── overfitting.py      #   Overfitting detection
│   ├── signal_decay.py     #   Signal IC decay analysis
│   ├── monte_carlo.py      #   Bootstrap Monte Carlo simulation
│   ├── stress_tests.py     #   Stress scenarios & liquidity stress
│   ├── diagnostics.py      #   Research health diagnostics
│   ├── research_score.py   #   Composite Research Quality Score
│   └── reporting.py        #   Validation report generator
│
├── dashboard/              # Streamlit research interface
│   ├── app.py              #   Dashboard entry point
│   ├── layout.py           #   Theme & styling
│   ├── state.py            #   Session state management
│   ├── views/              #   6 dashboard views
│   │   ├── overview.py     #     KPI cards, NAV curve, allocation
│   │   ├── analytics.py    #     Risk metrics, rolling charts, drawdowns
│   │   ├── optimization.py #     Weight targets, HRP tree, constraints
│   │   ├── backtests.py    #     NAV comparison, trade log, friction
│   │   ├── exposure.py     #     Country, currency, asset-type breakdown
│   │   └── recommendations.py #  Rebalance trades, signal scores
│   ├── components/         #   Reusable UI components
│   │   ├── charts.py       #     Chart wrappers
│   │   ├── filters.py      #     Sidebar filters
│   │   ├── metrics.py      #     KPI metric cards
│   │   ├── nav.py          #     Navigation helpers
│   │   └── tables.py       #     Data table renderers
│   └── utils/              #   Dashboard utilities
│       ├── loaders.py      #     Data loaders (Parquet → DataFrame)
│       ├── formatters.py   #     Number/currency formatters
│       └── exporters.py    #     Data export helpers
│
├── reports/                # Generated outputs
│   └── report_generator.py #   CSV & HTML report builder
│
├── configs/
│   └── asset_master.csv    # Asset universe declaration
│
├── data/
│   ├── raw/                # Raw market data (Parquet)
│   ├── processed/          # Computed results (Parquet)
│   ├── holdings/           # Portfolio holdings (CSV)
│   ├── cache/              # Temporary cache
│   └── exports/            # User exports
│
├── tests/                  # Test suite
├── notebooks/              # Research notebooks
├── utils/
│   └── validators.py       # Data quality checks
│
├── conftest.py             # Pytest shared fixtures
├── requirements.txt
├── .env                    # API keys (gitignored)
└── .gitignore
```

---

## Key Outputs

### Data Files (`data/processed/`)

| File | Description |
|------|-------------|
| `inr_prices.parquet` | All asset prices normalized to INR |
| `portfolio_nav.parquet` | Daily portfolio NAV series |
| `fx_attribution.parquet` | FX vs local-return breakdown |
| `returns.parquet` | Daily, log, cumulative, rolling returns |
| `rolling_analytics.parquet` | Rolling Sharpe, volatility, beta |
| `drawdown_series.parquet` | Daily drawdown series |
| `features.parquet` | Full feature store (momentum, vol, trend) |
| `signal_scores.parquet` | Composite signal scores & ranks |
| `target_weights.parquet` | Optimized portfolio weights |
| `rebalance_trades.parquet` | Suggested rebalance trades |
| `backtest_nav.parquet` | Backtest NAV series with friction |
| `trade_ledger.parquet` | Complete trade history |
| `walkforward_results.parquet` | Walk-forward OOS performance |
| `regime_analysis.parquet` | Market regime classifications |
| `regime_performance.parquet` | Per-regime strategy metrics |
| `parameter_sensitivity.parquet` | Parameter grid search results |
| `signal_decay.parquet` | Signal IC at multiple horizons |
| `monte_carlo_summary.parquet` | Monte Carlo simulation stats |
| `stress_test_results.parquet` | Stress scenario impacts |
| `liquidity_stress.parquet` | Slippage sensitivity analysis |

### Reports (`reports/`)

| File | Description |
|------|-------------|
| `portfolio_metrics.csv` | Core risk metrics summary |
| `benchmark_comparison.csv` | Portfolio vs benchmark performance |
| `drawdown_periods.csv` | Drawdown period detail |
| `strategy_comparison.csv` | Optimization strategy comparison |
| `backtest_comparison.csv` | Backtest strategy comparison |
| `backtest_attribution.csv` | Gross/net CAGR, friction drag |
| `walkforward_results.csv` | Walk-forward train/test Sharpe |
| `regime_performance.csv` | Performance by market regime |
| `parameter_sensitivity.csv` | Sharpe across parameter grid |
| `stress_test_results.csv` | Stress scenario impact analysis |
| `signal_decay.csv` | Signal IC decay across horizons |
| `monte_carlo_summary.csv` | MC return/drawdown distribution |
| `research_score.csv` | Composite research quality score |
| `diagnostics_summary.csv` | Research health diagnostics |
| `overfitting_flags.csv` | Overfitting detection flags |
| `portfolio_recommendation.csv` | Actionable rebalance trades |
| `portfolio_report.html` | Full HTML portfolio report |
| `*.html` | Interactive Plotly charts |

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.13 | Core runtime |
| Data | Pandas, NumPy | Data manipulation |
| Storage | Parquet (PyArrow) | Columnar persistence |
| Market Data | yfinance, mftool | Data ingestion |
| Statistics | SciPy, scikit-learn, statsmodels | Statistical analysis |
| Optimization | Custom HRP | Portfolio construction |
| Visualization | Plotly | Interactive charts |
| Dashboard | Streamlit | Research interface |
| Logging | Loguru | Structured logging |
| Config | python-dotenv | Environment management |

---

## Recent Results

| Metric | Value |
|--------|-------|
| Portfolio CAGR | +25.55% |
| Sharpe Ratio | 1.074 |
| Sortino Ratio | 1.358 |
| Max Drawdown | -30.00% |
| Backtest Net CAGR (HRP) | +19.01% |
| Backtest Sharpe | 0.791 |
| Friction Drag | 0.88% |
| Research Quality Score | 70.8/100 (Grade B) |
| Overfitting Assessment | ACCEPTABLE |
| Monte Carlo Prob of Loss | 12.0% |
| MC CVaR (5th percentile) | -19.13% |

---

## Development History

| Commit | Description |
|--------|-------------|
| `2e1e434` | Project bootstrap — structure, loaders, validators |
| `9531ca9` | FX normalization, portfolio NAV, attribution, exposure |
| `66d5ab2` | Analytics & risk engine — returns, metrics, drawdown, rolling, benchmark, charts, reports |
| `d51e4a8` | Feature engineering — returns, momentum, volatility, trend, mean reversion, factors, signal ranker |
| `eea4c3e` | Portfolio optimization — HRP, baselines, constraints, signal-tilt, turnover, rebalance |
| `3ca05b1` | Friction-aware backtesting — taxes, slippage, costs, benchmarks, attribution |
| `2a9ff07` | Streamlit dashboard — overview, analytics, optimization, backtests, exposure, recommendations |
| `d360945` | Complete dashboard architecture per spec |
| `6ecf628` | Validation, robustness & research hardening |

---

## License

Private research project. Not intended for distribution.
