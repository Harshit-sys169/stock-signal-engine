# NSE-Alpha v1 — Makefile
# Requires: Python 3.11+, make
#
# Usage:
#   make setup        — create venv and install dependencies
#   make download     — download all market data
#   make validate     — validate downloaded data quality
#   make features     — build feature panel
#   make train        — train model with walk-forward CV
#   make backtest     — run backtest and generate report
#   make report       — generate HTML report only
#   make daily        — run daily signal generation (Phase 2)
#   make dashboard    — launch Streamlit dashboard (Phase 3)
#   make notebook     — launch Jupyter notebook
#   make clean        — remove cached data and outputs
#   make all          — full pipeline: download → features → train → backtest

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
JUPYTER := .venv/bin/jupyter
STREAMLIT := .venv/bin/streamlit

.PHONY: setup download validate features train backtest report daily dashboard notebook clean all

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cp -n .env.example .env || true
	@echo ""
	@echo "Setup complete. Edit .env and add your FRED API key."
	@echo "Then run: make all"

download:
	$(PYTHON) -c "from data.downloader import download_all; download_all(force=True)"

validate:
	$(PYTHON) -c "\
from data.downloader import download_all; \
from data.validate import run_validation; \
datasets = download_all(); \
run_validation(datasets)"

features:
	$(PYTHON) -c "\
from data.downloader import download_all; \
from features.pipeline import build_feature_panel; \
datasets = download_all(); \
build_feature_panel(datasets, save=True)"

train:
	$(PYTHON) -c "\
from features.pipeline import load_feature_panel; \
from models.train import run_training_pipeline; \
panel = load_feature_panel(); \
run_training_pipeline(panel)"

backtest:
	$(PYTHON) -c "\
from features.pipeline import load_feature_panel; \
from models.train import walk_forward_train; \
from data.downloader import download_all; \
from backtest.engine import run_backtest; \
from backtest.report import generate_report; \
import pandas as pd; \
panel = load_feature_panel(); \
oos_df, _ = walk_forward_train(panel); \
datasets = download_all(); \
eq, trades = run_backtest(oos_df, datasets['ohlcv']); \
eq.to_parquet('outputs/equity_curve.parquet'); \
trades.to_parquet('outputs/trade_log.parquet') if len(trades) > 0 else None; \
bm = datasets['index']['nifty50']; \
path = generate_report(eq, trades, oos_df, bm); \
print(f'Report: {path}')"

report:
	$(PYTHON) backtest/report.py

daily:
	$(PYTHON) scripts/run_daily.py

dashboard:
	$(STREAMLIT) run dashboard/app.py

notebook:
	$(JUPYTER) notebook notebooks/01_phase1.ipynb

clean:
	rm -f data/raw/*.parquet data/cache/*.parquet
	rm -f outputs/feature_panel.parquet
	rm -f outputs/equity_curve.parquet
	rm -f outputs/trade_log.parquet
	rm -f outputs/wf_predictions.parquet
	rm -f outputs/feature_importance.csv
	rm -f outputs/signals/*.csv
	rm -f models/saved/*.txt models/saved/*.pkl
	@echo "Cache cleared."

all: download validate features train backtest
	@echo ""
	@echo "Full pipeline complete."
	@echo "Open outputs/backtest_report.html to view results."
	@echo "Run 'make dashboard' to launch the Streamlit dashboard."
