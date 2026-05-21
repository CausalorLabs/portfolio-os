# Portfolio OS — Personal Portfolio Operating System

A production-grade quantitative portfolio operating system that automates the full investment lifecycle: data ingestion, regime detection, ML-driven alpha generation, dynamic risk management, utility-based execution, performance attribution, and continuous operations — all governed by trust-calibrated automation.

---

## Thesis

> **Can a fully automated, friction-aware portfolio system with ML-driven signals, regime awareness, and trust-calibrated execution outperform passive allocation on a risk-adjusted basis — while remaining personally deployable?**

Most retail portfolio tools ignore the real costs of trading — taxes, slippage, FX spreads, and transaction fees — and operate without awareness of market regimes or model health. Portfolio OS models these explicitly, detects regime shifts, generates ML alpha, manages risk dynamically, and gates execution through utility analysis and trust scoring.

---

## Supported Assets

| Type | Examples | Source |
|------|----------|--------|
| US equities | AAPL | Yahoo Finance |
| US ETFs | SPY, VOO | Yahoo Finance |
| Indian equities (NSE) | RELIANCE.NS, INFY.NS | Yahoo Finance |
| Indian mutual funds | SBI Bluechip Direct Growth | MFAPI |
| Fixed income | EPF, PPF, FD, NPS, SGB | Synthetic (annual_rate) |
| Commodities | Physical Gold, Silver | Yahoo (GC=F, SI=F proxy) |
| FX rates | USD/INR | Yahoo Finance |

Assets are declared in [`configs/asset_master.csv`](configs/asset_master.csv).

---

## Architecture

Portfolio OS is built as an 8-sprint modular system. The `OrchestrationEngine` manages the daily lifecycle with event-driven architecture, dependency-aware execution, retry logic, and SLA compliance.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Ingestion  │─▶│   Features  │─▶│   Regimes   │─▶│  ML Alpha   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                                                           │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│ Attribution │◀─│  Execution  │◀─│Optimization │◀────────┘
└─────────────┘  └─────────────┘  └─────────────┘
       │                                 ▲
       ▼                                 │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Monitoring  │  │Orchestration│  │ Risk Engine │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│  Dashboard  │  │ Deployment  │
│ (13 views)  │  │(Trust/Ops)  │
└─────────────┘  └─────────────┘
```

### Sprint Architecture

| Sprint | Focus | Key Capabilities |
|--------|-------|------------------|
| **1** | Infrastructure & Core Pipeline | Ingestion, FX, analytics, features, optimization, backtesting, validation, warehouse, API, dashboard |
| **2** | Regime Intelligence | Multi-signal regime detection (risk_on/risk_off/crisis/recovery), regime-aware behavior, transition analysis |
| **3** | ML Alpha Engine | Walk-forward LightGBM + CatBoost ensemble, SHAP explainability, feature quality pipeline, MLflow tracking |
| **4** | Dynamic Risk & Covariance | EWMA volatility, regime-aware covariance, tail risk (CVaR), risk budgeting, vol scaling, stress testing |
| **5** | Utility-Based Execution | Utility engine (friction gating), tax-loss harvesting, slippage modeling, paper trading, state machine |
| **6** | Attribution & Monitoring | Brinson attribution, factor decomposition, alert engine, anomaly detection, audit trail, notifications |
| **7** | Orchestration & Automation | Event bus, DAG dependencies, retry/self-healing, scheduling, MLOps, SLA tracking, governance |
| **8** | Hardening & Deployment | E2E validation, chaos testing, trust calibration, walk-forward evaluation, human override layer |

---

## Module Breakdown

| # | Module | Purpose |
|---|--------|---------|
| 1 | `ingestion/` | Yahoo Finance, MFAPI, fixed income, metals — Parquet persistence |
| 2 | `fx/` | Multi-currency → INR normalization, FX attribution |
| 3 | `analytics/` | Risk metrics (CAGR, Sharpe, Sortino, Calmar), drawdowns, rolling stats |
| 4 | `features/` | Momentum, volatility, trend, mean reversion, factor composites, signal ranking |
| 5 | `optimization/` | HRP, risk parity, inverse vol, signal-tilted allocation, constraints |
| 6 | `backtests/` | Friction-aware engine: Indian STT/GST + US SEC fees, STCG/LTCG taxes, slippage |
| 7 | `validation/` | Walk-forward, Monte Carlo, overfitting detection, stress testing, Research Score |
| 8 | `regimes/` | Multi-signal regime detection, regime behavior, features, transitions |
| 9 | `ml_models/` | Walk-forward ensemble (LightGBM + CatBoost), SHAP, MLflow, feature quality |
| 10 | `risk_engine/` | 8-step pipeline: volatility, covariance, correlation, tail risk, budgeting, vol scaling, stress tests |
| 11 | `execution/` | Utility engine, rebalancing, tax harvesting, slippage, paper trading, state machine |
| 12 | `monitoring/` | Attribution, explainability, alerts (5 categories), anomaly detection, notifications, audit trail |
| 13 | `orchestration/` | Event bus, dependency graph, retry engine, scheduler, MLOps, state, SLA, governance |
| 14 | `deployment/` | Validation framework, failure simulation, trust calibration, walk-forward, security, hardening |
| 15 | `warehouse/` | DuckDB query layer over 44 Parquet tables |
| 16 | `api/` | FastAPI REST: health, portfolio, rebalance, regime |
| 17 | `dashboard/` | Streamlit with 13 views + 5 reusable component modules |
| 18 | `contracts/` | 50+ Pydantic v2 data models |
| 19 | `configs/` | Hydra YAML configs (8 config files) |
| 20 | `infra/` | Docker (API + Dashboard + warehouse) |

---

## Project Structure

```
portfolio-os/
├── app.py                      # Main pipeline entry point
├── configs/
│   ├── asset_master.csv        # Asset universe
│   ├── orchestration.yaml      # Sprint 7: workflow, events, scheduling, MLOps
│   ├── deployment.yaml         # Sprint 8: validation, trust, override
│   └── hydra/                  # Hydra config hierarchy
│       ├── base.yaml
│       ├── execution_engine.yaml
│       ├── risk_engine.yaml
│       ├── ml_alpha.yaml
│       └── monitoring.yaml
│
├── ingestion/                  # Data loaders (yahoo, mf, fixed income)
├── fx/                         # Currency normalization & attribution
├── analytics/                  # Portfolio analytics & risk metrics
├── features/                   # Alpha signal engineering
├── optimization/               # Portfolio construction (HRP, constraints)
├── backtests/                  # Friction-aware backtesting
├── validation/                 # Research hardening (walk-forward, Monte Carlo)
│
├── regimes/                    # Regime Intelligence Engine
│   ├── detectors/              #   Multi-signal regime detection
│   ├── behavior/               #   Regime-specific behavior params
│   ├── features/               #   Regime-aware features
│   ├── evaluation/             #   Regime model evaluation
│   └── transitions/            #   Transition detection & analysis
│
├── ml_models/                  # ML Alpha Engine
│   ├── training/               #   LightGBM + CatBoost walk-forward
│   ├── inference/              #   Alpha score generation
│   ├── evaluation/             #   Rank IC, hit rate grading
│   ├── explainability/         #   SHAP explanations
│   ├── ensembles/              #   Model stacking
│   ├── features/               #   Feature importance analysis
│   ├── quality/                #   Feature drift & quality checks
│   ├── confidence/             #   Regime-aware confidence scoring
│   └── tracking/               #   MLflow experiment tracking
│
├── risk_engine/                # Dynamic Risk & Covariance Engine
│   ├── volatility/             #   EWMA, realized, regime-aware vol
│   ├── covariance/             #   LedoitWolf, shrinkage, regime-aware
│   ├── correlation/            #   Rolling correlation, crisis clustering
│   ├── tail_risk/              #   CVaR, semivariance, tail beta
│   ├── budgeting/              #   Risk contribution per asset
│   ├── scaling/                #   Vol targeting / scaling
│   ├── stress_testing/         #   Historical + synthetic scenarios
│   ├── constraints/            #   Risk-based constraints
│   └── evaluation/             #   Risk model evaluation
│
├── execution/                  # Utility-Based Execution Engine
│   ├── utility_engine/         #   Friction vs alpha utility analysis
│   ├── rebalancing/            #   Regime-adaptive rebalance triggers
│   ├── tax_engine/             #   STCG/LTCG, tax-loss harvesting
│   ├── slippage/               #   Market impact modeling
│   ├── simulation/             #   Execution simulation
│   ├── paper_trading/          #   Virtual portfolio management
│   ├── audit/                  #   Execution journal & audit
│   ├── turnover/               #   Turnover budgeting
│   └── state_machine/          #   Portfolio lifecycle states
│
├── monitoring/                 # Attribution & Monitoring Layer
│   ├── attribution/            #   Brinson-Hood-Beebower + factor attribution
│   ├── explainability/         #   Decision explanations + trade narratives
│   ├── alerts/                 #   5-category alert engine
│   ├── notifications/          #   Telegram, Slack, Email channels
│   ├── observability/          #   Component health tracking
│   ├── anomaly_detection/      #   Z-score anomaly detection
│   └── audit/                  #   Lineage tracing & audit trail
│
├── orchestration/              # Orchestration & Automation Engine
│   ├── events/                 #   Pub/sub event bus
│   ├── dependencies/           #   DAG dependency graph
│   ├── retries/                #   Exponential backoff + circuit breaker
│   ├── scheduling/             #   Daily/weekly/monthly cadences
│   ├── mlops/                  #   Retraining triggers, shadow deployment
│   ├── state/                  #   Global system state coordination
│   ├── sla/                    #   Pipeline SLA compliance
│   └── governance/             #   Config snapshots, versioning
│
├── deployment/                 # MVP Hardening & Deployment
│   ├── validation/             #   E2E integrity checks
│   ├── failure_sim/            #   Chaos testing (failure simulation)
│   ├── trust/                  #   5-dimension trust calibration
│   ├── walkforward/            #   Long-horizon survivability
│   ├── security/               #   Rate limiting, auth, CORS
│   ├── hardening/              #   Backup/restore, reproducibility
│   └── report/                 #   MVP stabilization assessment
│
├── warehouse/                  # DuckDB query layer (44 tables)
├── api/                        # FastAPI REST API
│   └── routers/                #   health, portfolio, rebalance, regime
├── contracts/                  # Pydantic v2 data models (50+)
├── dashboard/                  # Streamlit research interface (13 views)
│   ├── views/
│   │   ├── overview.py         #   KPIs, NAV curve, allocation
│   │   ├── analytics.py        #   Risk metrics, rolling, drawdowns
│   │   ├── optimization.py     #   Weights, HRP, constraints
│   │   ├── backtests.py        #   NAV comparison, friction, trade log
│   │   ├── exposure.py         #   Country/currency/type breakdown
│   │   ├── recommendations.py  #   Rebalance trades, signals
│   │   ├── structural_health.py#   Research quality, validation
│   │   ├── regime_intelligence.py # Regime detection, transitions
│   │   ├── risk_intelligence.py#   Risk decomposition, stress tests
│   │   ├── execution_intelligence.py # Utility, paper trading
│   │   ├── explainability.py   #   Attribution, decisions, alerts
│   │   ├── operations.py       #   Pipeline, events, SLA, MLOps
│   │   └── command_center.py   #   Trust, override, deployment
│   ├── components/             #   Charts, filters, metrics, tables
│   └── utils/                  #   Loaders, formatters, exporters
│
├── infra/                      # Infrastructure
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
│
├── reports/                    # Generated outputs (HTML + CSV)
├── data/
│   ├── raw/                    # Raw market data (Parquet)
│   ├── processed/              # Pipeline outputs (Parquet)
│   ├── holdings/               # Portfolio holdings (CSV)
│   ├── backups/                # Operational backups
│   └── cache/                  # Temporary cache
│
├── tests/                      # 560 tests (all passing)
├── notebooks/                  # Research notebooks
└── requirements.txt
```

---

## Key Outputs

### Data Files (`data/processed/`)

| File | Description |
|------|-------------|
| `inr_prices.parquet` | All prices normalized to INR |
| `portfolio_nav.parquet` | Daily portfolio NAV |
| `features.parquet` | Full feature store |
| `alpha_scores.parquet` | ML alpha predictions + confidence |
| `regime_states.parquet` | Regime classifications |
| `target_weights.parquet` | Optimized portfolio weights |
| `rebalance_trades.parquet` | Proposed trades |
| `backtest_nav.parquet` | Backtest NAV with friction |
| `trade_ledger.parquet` | Complete trade history |
| `system_state.json` | Current system state |

### Warehouse (44 DuckDB-registered tables)

Includes: `inr_prices`, `portfolio_nav`, `features`, `alpha_scores`, `regime_states`, `target_weights`, `backtest_nav`, `trade_ledger`, `volatility_state`, `risk_budget`, `paper_portfolio`, `execution_journal`, `attribution_summary`, `factor_exposures`, `monitoring_alerts`, `anomaly_log`, `audit_trail`, `workflow_runs`, `trust_scores`, `validation_results`, and more.

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.13 | Core runtime |
| Data | Pandas, NumPy, Polars | Data manipulation |
| Storage | Parquet, DuckDB | Columnar persistence + warehouse |
| Market Data | yfinance, mftool | Data ingestion |
| ML | LightGBM, CatBoost, SHAP | Alpha generation & explainability |
| Tracking | MLflow | Experiment tracking |
| Risk | PyPortfolioOpt, scikit-learn (LedoitWolf) | Portfolio optimization & covariance |
| Statistics | SciPy, statsmodels | Statistical analysis |
| API | FastAPI, uvicorn | REST API |
| Dashboard | Streamlit, Plotly | 13-view research interface |
| Config | Hydra, OmegaConf, Pydantic v2 | Config + data contracts |
| Logging | Loguru | Structured logging |
| Infra | Docker, docker-compose | Containerization |

---

## Test Coverage

| Sprint | Module | Tests |
|--------|--------|-------|
| 1 | Core pipeline (ingestion, features, optimization, analytics, backtests) | ~150 |
| 2 | Regime Intelligence | 23 |
| 3 | ML Alpha Engine | 32 |
| 4 | Dynamic Risk & Covariance | 52 |
| 5 | Utility-Based Execution | 54 |
| 6 | Attribution & Monitoring | 75 |
| 7 | Orchestration & Automation | 57 |
| 8 | Hardening & Deployment | 56 |
| | **Total** | **560 passing** |

---

## Development History

| Commit | Sprint | Description |
|--------|--------|-------------|
| `2e1e434` | 1 | Project bootstrap — loaders, validators |
| `9531ca9` | 1 | FX normalization, NAV, attribution, exposure |
| `66d5ab2` | 1 | Analytics — returns, metrics, drawdown, rolling, benchmark |
| `d51e4a8` | 1 | Feature engineering — momentum, vol, trend, mean reversion |
| `eea4c3e` | 1 | Optimization — HRP, baselines, constraints, signal-tilt |
| `3ca05b1` | 1 | Backtesting — taxes, slippage, costs, benchmarks |
| `2a9ff07` | 1 | Dashboard — 6 views |
| `6ecf628` | 1 | Validation — walk-forward, Monte Carlo, stress tests |
| `d360945` | 1 | Infrastructure — warehouse, API, Docker, configs, contracts |
| `d14dfac` | 2 | Regime Intelligence Engine (23 tests) |
| `bf73ba3` | 3 | ML Alpha Engine (32 tests) |
| `d4b8ee8` | 4 | Dynamic Risk & Covariance Engine (52 tests) |
| `32bb5e3` | 5 | Utility-Based Execution Engine (54 tests) |
| `81628a0` | 6 | Attribution, Explainability & Monitoring (75 tests) |
| `326187c` | 7+8 | Orchestration + Hardening & Deployment (113 tests) |

---

## License

Private research project. Not intended for distribution.
