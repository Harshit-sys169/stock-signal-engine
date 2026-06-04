"""
features/technical.py
─────────────────────
Computes all technical indicator features for a single ticker DataFrame.
Uses pandas-ta for indicator calculation.

All features are computed strictly from data available at time t
to prevent any lookahead bias.
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    import warnings
    warnings.warn("pandas-ta not installed — using manual fallback implementations")

from config.settings import (
    RSI_WINDOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_WINDOW, BB_STD, ATR_WINDOW,
    SMA_WINDOWS, EMA_WINDOWS, MOMENTUM_WINDOWS,
    VOLUME_MA_WINDOW, PRICE_COLUMN,
)


# ── Individual Feature Builders ────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    if _HAS_TA:
        return ta.rsi(close, length=window)
    # Manual RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"RSI_{window}")


def _macd(close: pd.Series) -> pd.DataFrame:
    if _HAS_TA:
        result = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if result is not None:
            result.columns = ["macd", "macd_hist", "macd_signal"]
            return result
    # Manual MACD
    ema_fast   = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":        macd_line,
        "macd_signal": signal_line,
        "macd_hist":   histogram,
    })


def _bollinger(close: pd.Series) -> pd.DataFrame:
    if _HAS_TA:
        result = ta.bbands(close, length=BB_WINDOW, std=BB_STD)
        if result is not None:
            # Rename to standard names
            cols = result.columns.tolist()
            result.columns = ["bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent"]
            return result[["bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent"]]
    # Manual Bollinger
    mid = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std()
    upper = mid + BB_STD * std
    lower = mid - BB_STD * std
    bw = (upper - lower) / mid.replace(0, np.nan)
    pct = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_lower":     lower,
        "bb_mid":       mid,
        "bb_upper":     upper,
        "bb_bandwidth": bw,
        "bb_percent":   pct,
    })


def _atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if _HAS_TA:
        result = ta.atr(high, low, close, length=ATR_WINDOW)
        if result is not None:
            return result.rename("atr")
    # Manual ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_WINDOW).mean().rename("atr")


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    if _HAS_TA:
        result = ta.obv(close, volume)
        if result is not None:
            return result.rename("obv")
    # Manual OBV
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")


def _sma_features(close: pd.Series) -> pd.DataFrame:
    frames = {}
    for w in SMA_WINDOWS:
        sma = close.rolling(w).mean()
        frames[f"sma_{w}"]          = sma
        frames[f"price_to_sma_{w}"] = close / sma.replace(0, np.nan) - 1
    return pd.DataFrame(frames)


def _ema_features(close: pd.Series) -> pd.DataFrame:
    frames = {}
    for w in EMA_WINDOWS:
        ema = close.ewm(span=w, adjust=False).mean()
        frames[f"ema_{w}"]          = ema
        frames[f"price_to_ema_{w}"] = close / ema.replace(0, np.nan) - 1
    return pd.DataFrame(frames)


def _momentum_features(close: pd.Series) -> pd.DataFrame:
    frames = {}
    for w in MOMENTUM_WINDOWS:
        frames[f"return_{w}d"] = close.pct_change(w)
    return pd.DataFrame(frames)


def _volume_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    vol_ma      = volume.rolling(VOLUME_MA_WINDOW).mean()
    vol_ratio   = volume / vol_ma.replace(0, np.nan)
    log_volume  = np.log1p(volume)
    price_range = (close - close.shift(1)).abs()
    return pd.DataFrame({
        "volume_ratio":   vol_ratio,
        "log_volume":     log_volume,
        "volume_ma":      vol_ma,
        "price_range_1d": price_range,
    })


def _candle_features(open_: pd.Series, high: pd.Series,
                     low: pd.Series, close: pd.Series) -> pd.DataFrame:
    body        = (close - open_).abs()
    total_range = (high - low).replace(0, np.nan)
    return pd.DataFrame({
        "hl_spread":      (high - low) / close.replace(0, np.nan),
        "oc_spread":      (close - open_) / open_.replace(0, np.nan),
        "body_ratio":     body / total_range,
        "upper_shadow":   (high - close.clip(lower=open_)) / total_range,
        "lower_shadow":   (close.clip(upper=open_) - low) / total_range,
    })


# ── Master Feature Builder ─────────────────────────────────────────────────

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all technical features for a single ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Adj Close, Volume.
        Index must be DatetimeIndex sorted ascending.

    Returns
    -------
    pd.DataFrame
        Same DatetimeIndex.
        All feature columns — NaN rows at the start (due to lookback windows)
        are left as-is; caller handles dropna.
    """
    required = {"Open", "High", "Low", "Close", PRICE_COLUMN, "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    close  = df[PRICE_COLUMN].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    open_  = df["Open"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)

    # Normalise ATR by price for comparability across stocks
    atr_raw = _atr(high, low, close)
    atr_pct = (atr_raw / close.replace(0, np.nan)).rename("atr_pct")

    parts = [
        _rsi(close).rename("rsi").to_frame(),
        _macd(close),
        _bollinger(close),
        atr_raw.to_frame(),
        atr_pct.to_frame(),
        _obv(close, volume).to_frame(),
        _sma_features(close),
        _ema_features(close),
        _momentum_features(close),
        _volume_features(close, volume),
        _candle_features(open_, high, low, close),
    ]

    features = pd.concat(parts, axis=1)
    features.index = df.index

    # Keep raw price for label construction later
    features["adj_close"] = close
    features["volume_raw"] = volume

    return features


if __name__ == "__main__":
    # Quick smoke test
    import yfinance as yf
    raw = yf.download("TCS.NS", start="2022-01-01", end="2024-01-01",
                      auto_adjust=False, progress=False)
    feats = build_technical_features(raw)
    print(feats.shape)
    print(feats.tail(3).to_string())
