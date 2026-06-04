# Quick Start Guide

Get the NSE-Alpha signal system running in 5 minutes.

## Prerequisites

- Python 3.11+
- `make` command (Windows users: install via [GNU Make for Windows](https://gnuwin32.sourceforge.io/packages/make.htm) or use WSL)
- Free FRED API key from [FRED Economic Data](https://fredaccount.stlouisfed.org/apikey)

## Installation

```bash
# Clone the repository
git clone https://github.com/Harshit-sys169/stock-signal-engine.git
cd stock-signal-engine

# Run setup (creates virtual environment and installs dependencies)
make setup

# Add your FRED API key to .env
# Edit the .env file and add:
# FRED_API_KEY=your_key_here
```

## First Run

```bash
# Download all required market data
make download

# Validate data quality
make validate

# Build feature engineering pipeline
make features

# Train the model with walk-forward validation
make train

# Run backtest and generate report
make backtest

# View the HTML backtest report
open outputs/backtest_report.html
```

Or run the full pipeline at once:

```bash
make all
```

## Daily Signal Generation

After the market closes (3:30 PM IST), run:

```bash
make daily
```

This outputs the latest Buy/Hold/Sell signals to `outputs/signals/signals_{date}.csv`.

## View the Dashboard

```bash
make dashboard
```

Opens Streamlit dashboard at `http://localhost:8501`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `make: command not found` | Install GNU Make or use WSL on Windows |
| `FRED_API_KEY` error | Get free key from [FRED](https://fredaccount.stlouisfed.org/apikey) and add to `.env` |
| `yfinance` download timeout | Internet connection issue; retry the command |
| Jupyter kernel error | Run `make setup` again to ensure venv is active |

## Next Steps

- Review the [Architecture Guide](ARCHITECTURE.md) to understand the system design
- Check [sample outputs](sample_output/) for example signals
- See [Development Guide](DEVELOPMENT.md) for contributing or extending the system

---

**Time estimate:** ~10 minutes for first setup + 5-10 minutes for initial data download
