"""
models/predict.py
─────────────────
Loads the saved LightGBM model and generates today's signals
for the full universe.

Output per ticker:
  - signal        : Buy / Hold / Sell
  - confidence    : probability of predicted class
  - prob_buy      : probability of Buy
  - prob_hold     : probability of Hold
  - prob_sell     : probability of Sell
  - expected_5d_return : weighted expected return (buy_prob * buy_avg - sell_prob * sell_avg)
  - entry_price   : today's adjusted close
  - signal_date   : today's date
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from config.settings import (
    MODELS_DIR, OUTPUTS_DIR,
    CLASS_LABELS, SIGNAL_COLUMN,
    CONFIDENCE_COL, RETURN_PRED_COL,
    MIN_CONFIDENCE,
    BUY_THRESHOLD, SELL_THRESHOLD,
)
from models.train import get_feature_columns, NON_FEATURE_COLS

logger = logging.getLogger(__name__)


def load_model() -> "lgb.Booster":
    """Load the saved final LightGBM model from disk."""
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed — pip install lightgbm")

    model_path = MODELS_DIR / "lgbm_final.txt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run models/train.py first to train and save the model."
        )
    model = lgb.Booster(model_file=str(model_path))
    logger.info(f"Loaded model from {model_path}")
    return model


def predict_latest(
    panel: pd.DataFrame,
    model: "lgb.Booster" | None = None,
    signal_date: date | None = None,
) -> pd.DataFrame:
    """
    Generate signals for the most recent available date in the panel.

    Parameters
    ----------
    panel      : pd.DataFrame — feature panel (from features/pipeline.py)
    model      : lgb.Booster — if None, loads from disk
    signal_date: date — if None, uses the latest date in panel

    Returns
    -------
    pd.DataFrame
        One row per ticker, sorted by confidence descending.
        Columns: ticker, signal, signal_label, confidence,
                 prob_buy, prob_hold, prob_sell,
                 expected_5d_return, entry_price, signal_date
    """
    if model is None:
        model = load_model()

    feature_cols = get_feature_columns(panel)

    # Use latest date in panel
    if signal_date is None:
        signal_date = panel.index.max()

    today_data = panel[panel.index == signal_date]

    if today_data.empty:
        # Fall back to last available date
        signal_date = panel.index.max()
        today_data  = panel[panel.index == signal_date]
        logger.warning(f"Requested date unavailable — using {signal_date}")

    X = today_data[feature_cols].fillna(0)

    proba = model.predict(X)   # shape: (n_tickers, 3)
    preds = proba.argmax(axis=1)

    results = today_data[["ticker", "adj_close"]].copy()
    results["signal"]            = preds
    results["signal_label"]      = [CLASS_LABELS[p] for p in preds]
    results["confidence"]        = proba.max(axis=1)
    results["prob_sell"]         = proba[:, 0]
    results["prob_hold"]         = proba[:, 1]
    results["prob_buy"]          = proba[:, 2]
    results["entry_price"]       = results["adj_close"]
    results["signal_date"]       = signal_date

    # Expected 5-day return: rough expectation using average thresholds
    # Buy expected: mean of (BUY_THRESHOLD, 2*BUY_THRESHOLD)
    # Sell expected: mean of (SELL_THRESHOLD, 2*SELL_THRESHOLD)
    buy_exp  =  BUY_THRESHOLD  * 1.5
    sell_exp =  SELL_THRESHOLD * 1.5   # negative
    results[RETURN_PRED_COL] = (
        results["prob_buy"]  * buy_exp
        + results["prob_hold"] * 0.0
        + results["prob_sell"] * sell_exp
    )

    # Apply minimum confidence filter — flag low-confidence signals
    results["actionable"] = results["confidence"] >= MIN_CONFIDENCE

    results = results.drop(columns=["adj_close"]).sort_values(
        "confidence", ascending=False
    ).reset_index(drop=True)

    return results


def generate_daily_signals(
    panel: pd.DataFrame,
    model: "lgb.Booster" | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate and optionally save today's signals.

    This is the function called by run_daily.py after market close.
    """
    signals = predict_latest(panel, model)

    actionable = signals[signals["actionable"]]
    buy_signals  = actionable[actionable["signal"] == 2]
    sell_signals = actionable[actionable["signal"] == 0]

    logger.info(f"Signal date: {signals['signal_date'].iloc[0]}")
    logger.info(f"Total tickers: {len(signals)}")
    logger.info(f"Actionable Buy  (conf ≥ {MIN_CONFIDENCE}): {len(buy_signals)}")
    logger.info(f"Actionable Sell (conf ≥ {MIN_CONFIDENCE}): {len(sell_signals)}")

    if save:
        today_str = str(signals["signal_date"].iloc[0])[:10]
        out_path  = OUTPUTS_DIR / f"signals_{today_str}.csv"
        signals.to_csv(out_path, index=False)
        logger.info(f"Signals saved → {out_path}")

    return signals


if __name__ == "__main__":
    from features.pipeline import load_feature_panel
    panel = load_feature_panel()
    sigs = generate_daily_signals(panel)
    print(sigs[sigs["actionable"]].head(10).to_string())
