"""
features/pipeline.py
────────────────────
Orchestrates the full feature-engineering pipeline.

Steps:
  1. Build macro features from index, VIX, and FRED data.
  2. For each ticker: build technical features → merge macro → label.
  3. Stack all tickers into a single panel DataFrame.
  4. Save to parquet for use by model training.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import (
    OUTPUTS_DIR, FORWARD_DAYS,
    BUY_THRESHOLD, SELL_THRESHOLD,
    MAX_MISSING_PCT,
)
from features.technical import build_technical_features
from features.macro import build_macro_features, merge_macro_into_ticker

logger = logging.getLogger(__name__)


# ── Label Generation ───────────────────────────────────────────────────────

def build_labels(adj_close: pd.Series) -> pd.DataFrame:
    """
    Build prediction target labels from adjusted close prices.

    Label = 5-day forward return
    Signal = Buy(2) / Hold(1) / Sell(0)

    Note: forward-shifted, so final FORWARD_DAYS rows will be NaN.
    """
    fwd_return = adj_close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    signal = pd.Series(1, index=fwd_return.index, name="signal")  # default Hold
    signal[fwd_return >  BUY_THRESHOLD]  = 2  # Buy
    signal[fwd_return <  SELL_THRESHOLD] = 0  # Sell

    return pd.DataFrame({
        "fwd_return_5d": fwd_return,
        "signal":        signal.astype("Int64"),
    })


# ── Per-Ticker Feature Build ───────────────────────────────────────────────

def build_ticker_features(
    ticker: str,
    price_df: pd.DataFrame,
    macro_features: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Build complete feature + label DataFrame for one ticker.

    Parameters
    ----------
    ticker        : str, NSE symbol without .NS
    price_df      : DataFrame with OHLCV columns
    macro_features: DataFrame from build_macro_features()

    Returns
    -------
    pd.DataFrame or None (if ticker has insufficient data)
    """
    try:
        # Technical features
        tech = build_technical_features(price_df)

        # Labels
        labels = build_labels(tech["adj_close"])

        # Merge macro
        full = merge_macro_into_ticker(tech, macro_features, ticker)

        # Attach labels
        full = pd.concat([full, labels], axis=1)

        # Add ticker identifier
        full["ticker"] = ticker

        # Drop rows where label is unknown (last FORWARD_DAYS rows)
        full = full.dropna(subset=["signal"])

        # Drop rows with excessive NaN in features (early lookback rows)
        feature_cols = [c for c in full.columns
                        if c not in ("signal", "fwd_return_5d", "ticker")]
        row_nan_pct = full[feature_cols].isna().mean(axis=1)
        full = full[row_nan_pct < 0.30]   # drop rows where >30% features are NaN

        if len(full) < 252:  # need at least 1 year of data
            logger.warning(f"{ticker}: only {len(full)} valid rows — skipping")
            return None

        return full

    except Exception as e:
        logger.error(f"{ticker}: feature build failed — {e}")
        return None


# ── Full Pipeline ──────────────────────────────────────────────────────────

def build_feature_panel(
    datasets: dict,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build the full feature panel for all tickers.

    Parameters
    ----------
    datasets : dict
        Output of data.downloader.download_all()
        Keys: 'ohlcv', 'index', 'vix', 'macro'
    save : bool
        If True, saves panel to parquet.

    Returns
    -------
    pd.DataFrame
        Multi-stock panel with columns:
          [all features] + fwd_return_5d + signal + ticker
        Index: DatetimeIndex (date of feature observation)
    """
    logger.info("Building macro features...")
    macro_features = build_macro_features(
        index_df = datasets["index"],
        vix_df   = datasets["vix"],
        macro_df = datasets["macro"],
    )

    ohlcv = datasets["ohlcv"]
    logger.info(f"Building features for {len(ohlcv)} tickers...")

    ticker_frames = []
    failed = []

    for ticker, price_df in tqdm(ohlcv.items(), desc="Feature engineering"):
        result = build_ticker_features(ticker, price_df, macro_features)
        if result is not None:
            ticker_frames.append(result)
        else:
            failed.append(ticker)

    if not ticker_frames:
        raise RuntimeError("No ticker features were successfully built.")

    panel = pd.concat(ticker_frames, axis=0)
    panel = panel.sort_values(["ticker", panel.index.name or "Date"])

    logger.info(f"Panel shape: {panel.shape}")
    logger.info(f"Tickers: {panel['ticker'].nunique()} / {len(ohlcv)}")
    logger.info(f"Date range: {panel.index.min()} → {panel.index.max()}")

    if failed:
        logger.warning(f"Failed tickers ({len(failed)}): {failed}")

    # Signal distribution
    dist = panel["signal"].value_counts().sort_index()
    labels = {0: "Sell", 1: "Hold", 2: "Buy"}
    for k, count in dist.items():
        pct = count / len(panel) * 100
        logger.info(f"  {labels.get(k, k)}: {count:,}  ({pct:.1f}%)")

    if save:
        out_path = OUTPUTS_DIR.parent / "feature_panel.parquet"
        panel.to_parquet(out_path)
        logger.info(f"Saved panel → {out_path}")

    return panel


def load_feature_panel() -> pd.DataFrame:
    """Load a previously built feature panel from disk."""
    path = OUTPUTS_DIR.parent / "feature_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature panel not found at {path}. "
            "Run build_feature_panel() first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    from data.downloader import download_all
    datasets = download_all()
    panel = build_feature_panel(datasets)
    print(panel.head(3))
