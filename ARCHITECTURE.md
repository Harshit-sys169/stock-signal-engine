# System Architecture

## Overview

NSE-Alpha is a **daily signal system** that predicts 5-day forward returns for NSE India equities using machine learning. The pipeline follows three phases:

```
PHASE 1 (Research)     → PHASE 2 (Daily Production)    → PHASE 3 (Dashboard)
   Notebooks              Signal Generation               Web Interface
   Feature Design         Automated Inference             Risk Monitoring
   Model Training         CSV Output (50+ stocks)         Performance Analytics
```

## Data Flow

```
                          ┌─ Technical Indicators (RSI, MACD, etc.)
Market Data (OHLCV) ──┤
                          ├─ Macro Features (VIX, USD/INR, Repo Rate)
                          └─ Sector Indices
                                    ↓
                          Feature Engineering Pipeline
                                    ↓
                          Normalized Feature Matrix
                                    ↓
                          LightGBM Model (Walk-Forward Trained)
                                    ↓
                    Prediction + Confidence Score
                                    ↓
                    Risk Filters & Position Sizing
                                    ↓
        Output: Buy / Sell / Hold Signals (Confidence 0-1)
```

## Core Modules

### 1. **data/** — Data Acquisition & Validation

| Module | Responsibility |
|--------|-----------------|
| `universe.py` | Manages Nifty 50 + Next 50 ticker lists with sector mapping |
| `downloader.py` | Fetches OHLCV, VIX, sector indices, macro data from yfinance/FRED |
| `validate.py` | Data quality checks: missing values, outliers, stale data |

**Key Design Decision:** All data is cached locally in `data/cache/` to enable offline model training and fast feature rebuilds.

---

### 2. **features/** — Feature Engineering Pipeline

| Module | Purpose | Example Features |
|--------|---------|-------------------|
| `technical.py` | Momentum & volatility indicators | RSI(14), MACD, Bollinger Bands, ATR, OBV, momentum ratios |
| `macro.py` | Macro economic context | India VIX, sector returns, USD/INR, LIBOR rate |
| `pipeline.py` | Orchestrator for full feature matrix | Combines all features, handles missing values, normalizes |

**Key Design Decision:** 50+ features are engineered from ~5 raw time series, keeping the feature space manageable while capturing market dynamics.

---

### 3. **models/** — ML Training & Inference

| Module | Responsibility |
|--------|-----------------|
| `train.py` | LightGBM model training with walk-forward validation |
| `predict.py` | Daily inference: loads trained model, generates signals |

**Key Architecture:**
- **Walk-Forward Validation:** 6-month test windows prevent data leakage and ensure realistic OOS performance
- **Labels:** Binary classification of 5-day forward returns (up/down) with confidence calibration
- **Hyperparameters:** Stored in `config/settings.py` for reproducibility and easy tuning

---

### 4. **backtest/** — Historical Validation

| Module | Responsibility |
|--------|-----------------|
| `engine.py` | Trade simulator: entry/exit logic, position sizing, risk limits |
| `metrics.py` | Performance calculation: Sharpe, drawdown, win rate, CAGR |
| `report.py` | HTML report generation with equity curve and monthly returns |

**Key Features:**
- Transaction costs & slippage modelled at 0.15% per trade
- Risk limits: max position size, portfolio-level drawdown constraints
- Walk-forward evaluation ensures all metrics are out-of-sample

---

### 5. **config/** — Single Source of Truth

`settings.py` contains:
- Universe definitions (Nifty 50/Next 50)
- Feature parameters (indicator periods, thresholds)
- Model hyperparameters (learning rate, max depth, regularization)
- Risk management rules (max drawdown, position limits)
- API credentials (loaded from `.env`)

**Design Benefit:** Change any system behavior in one place; entire pipeline updates.

---

## Execution Workflows

### Workflow 1: Training Pipeline (One-time / Retraining)

```
1. make download      → Fetch 5 years of market data
2. make validate      → Quality checks
3. make features      → Build 50+ features
4. make train         → LightGBM with walk-forward CV
5. make backtest      → Validate on historical data
6. make report        → Generate HTML backtest report
```

### Workflow 2: Daily Signal Generation

```
Time: 3:30 PM IST (after NSE close)
1. Fetch latest OHLCV data
2. Compute technical + macro features
3. Load trained model from disk
4. Run inference on 100 stocks
5. Apply risk filters & confidence thresholds
6. Output signals to CSV: outputs/signals/signals_{YYYY-MM-DD}.csv
```

### Workflow 3: Dashboard Monitoring

```
Dashboard (Streamlit) reads:
  - Latest signals from signals/ directory
  - Backtest report (HTML)
  - Equity curve (CSV)
  - Performance metrics (live calculation)
```

---

## Key Design Decisions

### Why LightGBM?

- **Fast Training:** Handles high-dimensional feature space efficiently
- **Walk-Forward Ready:** Retrains in minutes, enabling frequent model updates
- **Interpretable:** Feature importance is human-readable
- **Robust:** Less prone to overfitting than deep neural networks on small datasets

### Why Walk-Forward Validation?

- **Realistic Performance:** All metrics are out-of-sample; no data leakage
- **Time-Aware:** Respects temporal order of data; no look-ahead bias
- **Production-Ready:** Metrics match live trading performance better

### Why Confidence Scores?

- Each prediction includes a probability estimate (0-1)
- Signals below 0.60 confidence are flagged as non-actionable
- Enables risk-conscious position sizing: high confidence → larger position

### Universe: Nifty 50 + Next 50

- **Liquidity:** Most liquid large-cap stocks in India
- **Tradeable:** Low bid-ask spreads, sufficient volume
- **Homogeneous:** Similar market dynamics, easier feature engineering
- **Realistic Size:** 100 stocks = manageable for daily processing

---

## Extensibility

### Add New Features

1. Implement in `features/technical.py` or `features/macro.py`
2. Update `features/pipeline.py` to include in the feature matrix
3. Retrain model with `make train`

### Change Model Algorithm

1. Edit `models/train.py` (replace LightGBM with XGBoost, CatBoost, etc.)
2. Keep the same `predict_signals()` interface
3. `predict.py` continues to work unchanged

### Add New Data Sources

1. Add download logic to `data/downloader.py`
2. Add validation checks to `data/validate.py`
3. Reference in `features/pipeline.py`

---

## Testing

Run validation before deploying:

```bash
make validate      # Data quality checks
make backtest      # Historical performance
```

---

## Performance Constraints

| Constraint | Current | Limit |
|-----------|---------|-------|
| Feature compute time | ~2 sec | < 30 sec |
| Model inference | ~1 sec | < 5 sec |
| Total pipeline (after data download) | ~5 min | < 30 min |
| Memory usage | ~500 MB | < 2 GB |

---

## File Size Estimates

| Directory | Size | Notes |
|-----------|------|-------|
| `data/cache/` | ~500 MB | 5 years OHLCV + macro data |
| `models/saved/` | ~50 MB | Trained model artifacts |
| `outputs/signals/` | ~1 MB | Historical signal CSVs |

---

## Deployment Checklist

- [ ] `.env` file configured with FRED_API_KEY
- [ ] `data/cache/` populated with `make download`
- [ ] Model trained with `make train`
- [ ] Backtest passed with acceptable metrics
- [ ] Daily signal generation runs without errors
- [ ] Dashboard loads and displays latest signals

---

**For contributing or questions, see [Development Guide](DEVELOPMENT.md).**
