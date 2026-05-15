# Portfolio OS — Personal Portfolio Research Engine

## Objective
Validate whether friction-aware portfolio optimization can outperform passive allocation on a risk-adjusted basis.

## Supported Assets
- US equities & ETFs
- Indian equities (NSE)
- Indian mutual funds
- USD/INR FX

## Project Structure
```
portfolio-os/
├── data/           # raw, processed, cache, exports
├── ingestion/      # data loaders (yahoo, mf)
├── fx/             # FX normalization engine
├── analytics/      # portfolio analytics
├── features/       # feature engineering
├── optimization/   # HRP, risk parity
├── backtests/      # friction-aware backtesting
├── dashboard/      # Streamlit UI
├── configs/        # asset master, parameters
├── notebooks/      # research notebooks
├── utils/          # validators, helpers
├── tests/          # test suite
└── reports/        # output reports
```

## Quick Start
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Tech Stack
| Layer | Tool |
|-------|------|
| Language | Python 3.11 |
| Storage | DuckDB + Parquet |
| Optimization | PyPortfolioOpt |
| Backtesting | vectorbt |
| Visualization | Plotly |
| UI | Streamlit |
