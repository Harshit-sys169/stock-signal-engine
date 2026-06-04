# Getting Started

## Prerequisites

Python 3.11+, `make` command, and a free FRED API key from [FRED Economic Data](https://fredaccount.stlouisfed.org/apikey).

On Windows, install GNU Make via [this link](https://gnuwin32.sourceforge.io/packages/make.htm) or use WSL.

## Setup

```bash
git clone https://github.com/Harshit-sys169/stock-signal-engine.git
cd stock-signal-engine

make setup
```

This creates a virtual environment and installs all dependencies. Add your FRED API key to the `.env` file after setup.

## First Run

```bash
make download      # Fetch 5 years of market data
make validate      # Validate data quality
make features      # Build feature engineering pipeline
make train         # Train model with walk-forward validation
make backtest      # Run backtest and generate report
```

View the HTML report in `outputs/backtest_report.html`.

Or run everything at once with `make all`.

## Daily Signal Generation

After NSE market close, run:

```bash
make daily
```

Outputs latest signals to `outputs/signals/signals_{YYYY-MM-DD}.csv`.

## Dashboard

```bash
make dashboard
```

Opens at `http://localhost:8501`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| make not found | Install GNU Make or use WSL |
| FRED_API_KEY error | Get key from [FRED](https://fredaccount.stlouisfed.org/apikey) and add to `.env` |
| yfinance timeout | Retry the command |
| Jupyter kernel error | Run `make setup` again |

## Next Steps

- Read [Architecture](ARCHITECTURE.md) for system design details
- Check [sample_output/](sample_output/) for example signal files
- See [Development](DEVELOPMENT.md) for extending the system
