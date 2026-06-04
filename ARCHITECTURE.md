# System Architecture

This document covers how the system works, why it's structured this way, and how to extend it.

## Overview

The project predicts 5-day forward returns for 100 NSE stocks using machine learning. It's built in three phases:

Phase 1 (research) - notebook-based exploration and model development
Phase 2 (production) - automated daily signal generation
Phase 3 (dashboard) - web interface for monitoring signals and performance

## Data Pipeline

Market data flows through the system in this order:

Raw market data (OHLCV) -> Technical indicators -> Macro features -> Feature engineering -> LightGBM model -> Predictions -> Signal output

## Modules

### config/

Single configuration file (`settings.py`) that controls:

- Universe definitions (Nifty 50, Next 50)
- Feature parameters (RSI periods, MACD settings, etc.)
- Model hyperparameters
- Risk management rules
- API credentials

Changing a parameter here updates the entire pipeline consistently.

### data/

Three modules handle data:

`universe.py` - Maintains lists of Nifty 50 and Next 50 stocks with sector mapping.

`downloader.py` - Fetches OHLCV data from yfinance, macro data from FRED API. Data is cached locally in `data/cache/` to avoid repeated downloads.

`validate.py` - Quality checks: missing values, outliers, data staleness. Runs before training to catch issues early.

### features/

Feature engineering is split into two modules:

`technical.py` - Momentum and volatility indicators (RSI, MACD, Bollinger Bands, ATR, OBV, momentum ratios).

`macro.py` - Macro economic context (India VIX, sector returns, USD/INR, LIBOR rate).

`pipeline.py` - Combines all features, handles missing values, normalizes output.

The feature matrix has 50+ indicators engineered from about 5 raw time series.

### models/

Two modules manage model training and inference:

`train.py` - LightGBM model trained with walk-forward validation. 6-month test windows ensure no data leakage.

`predict.py` - Loads trained model and generates daily signals. Outputs CSV with ticker, signal direction, confidence score, expected return, entry price.

Walk-forward validation means each model is trained on data up to a point, then tested on the next 6 months. This prevents look-ahead bias and gives realistic performance metrics.

### backtest/

Three modules for historical validation:

`engine.py` - Trade simulator. Applies entry/exit rules, position sizing, risk limits.

`metrics.py` - Calculates performance: Sharpe ratio, max drawdown, win rate, CAGR, profit factor.

`report.py` - Generates HTML backtest report with equity curve and monthly returns.

All backtest metrics are out-of-sample. Transaction costs (0.15% per trade) and slippage are included.

## Why These Choices

### LightGBM

Chosen because:
- Trains fast on high-dimensional data
- Can be retrained daily if needed
- Feature importance is interpretable
- Robust to overfitting on small datasets

### Walk-Forward Validation

Walk-forward validation prevents overfitting and gives realistic metrics. Each test period uses a model trained only on prior data, matching live trading behavior.

### Confidence Scores

Each prediction includes a probability (0-1). Signals below 0.60 confidence are marked non-actionable. This enables risk-aware position sizing.

### Universe: Nifty 50 + Next 50

These 100 stocks are liquid enough to trade realistically. Feature engineering is simpler with homogeneous assets compared to mixing large-cap and micro-cap stocks.

## Adding Features

To add a new technical indicator:

1. Implement in `features/technical.py`
2. Update `features/pipeline.py` to include it
3. Retrain with `make train`

To add a new data source:

1. Add download logic to `data/downloader.py`
2. Add validation checks to `data/validate.py`
3. Reference in `features/pipeline.py`

## Changing Models

Want to try XGBoost or CatBoost instead? The interface is:

```python
def train_model():
    # Return trained model
    return model

def predict_signals(model, features):
    # Return predictions with confidence
    return predictions
```

As long as these functions work the same way, `predict.py` continues to work unchanged.

## Performance Constraints

Current system runs in:

- Feature computation: ~2 seconds
- Model inference: ~1 second
- Total pipeline (after data download): ~5 minutes
- Memory usage: ~500 MB

Max constraints:

- Feature compute: 30 seconds
- Inference: 5 seconds
- Total: 30 minutes
- Memory: 2 GB

## File Sizes

Data cache: ~500 MB (5 years of OHLCV + macro data)
Trained model: ~50 MB
Historical signals: ~1 MB
