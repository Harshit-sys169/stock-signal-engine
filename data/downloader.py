"""
data/downloader.py
──────────────────
Downloads and caches all raw data required by NSE-Alpha v1.

Sources:
  - yfinance      : OHLCV + adjusted close for all 100 NSE tickers
  - yfinance      : Nifty 50 index, sector indices, India VIX (fallback)
  - FRED API      : USD/INR exchange rate, India repo rate
  - NSE direct    : India VIX (primary, falls back to yfinance ^INDIAVIX)

All data is saved to DATA_DIR as parquet files for fast re-loading.
Re-download is skipped if cached file exists and is fresh (< 1 day old).
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    DATA_DIR, HISTORY_START, HISTORY_END,
    INDEX_TICKER, SECTOR_INDICES,
    ENABLE_MACRO_FRED, FRED_API_KEY, FRED_SERIES,
    INDIA_VIX_FALLBACK_TICKER,
    MAX_MISSING_PCT, GAP_FILL_LIMIT,
    PRICE_COLUMN,
)
from data.universe import get_universe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")


# ── Helpers ────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _is_fresh(path: Path, max_age_hours: int = 20) -> bool:
    """Return True if the file exists and was modified within max_age_hours."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def _save(df: pd.DataFrame, name: str) -> None:
    path = _cache_path(name)
    df.to_parquet(path)
    logger.info(f"Saved  {name}  →  {path}  ({len(df)} rows)")


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(name))


# ── OHLCV Download ─────────────────────────────────────────────────────────

def download_ohlcv(
    tickers: list[str] | None = None,
    start: str = HISTORY_START,
    end: str = HISTORY_END,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for all tickers in the universe.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are clean NSE symbols (no .NS suffix).
        Each DataFrame has columns: Open, High, Low, Close, Adj Close, Volume.
        Index is DatetimeIndex.
    """
    cache_name = "ohlcv_raw"
    if not force and _is_fresh(_cache_path(cache_name)):
        logger.info("Loading OHLCV from cache")
        panel = _load(cache_name)
        return _panel_to_dict(panel)

    if tickers is None:
        tickers = get_universe()

    logger.info(f"Downloading OHLCV for {len(tickers)} tickers  [{start} → {end}]")

    # yfinance bulk download
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        threads=True,
        progress=True,
    )

    # Flatten MultiIndex columns → long format stored as wide parquet
    if isinstance(raw.columns, pd.MultiIndex):
        # Shape: (dates, (field, ticker))
        panel = raw.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        # Single ticker edge case
        panel = raw.reset_index()
        panel["Ticker"] = tickers[0]

    panel["Ticker"] = panel["Ticker"].str.replace(".NS", "", regex=False)
    _save(panel, cache_name)
    return _panel_to_dict(panel)


def _panel_to_dict(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert flat panel back to per-ticker DataFrames."""
    result = {}
    for ticker, grp in panel.groupby("Ticker"):
        df = grp.set_index("Date").drop(columns=["Ticker"], errors="ignore")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        result[ticker] = df
    return result


# ── Data Quality ───────────────────────────────────────────────────────────

def clean_ohlcv(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Quality filter:
      - Remove tickers with > MAX_MISSING_PCT missing adjusted close days
      - Forward-fill gaps up to GAP_FILL_LIMIT consecutive days
      - Drop rows where Adj Close <= 0
    """
    cleaned = {}
    dropped = []
    total_days = None

    for ticker, df in data.items():
        if PRICE_COLUMN not in df.columns:
            dropped.append((ticker, "no Adj Close column"))
            continue

        if total_days is None:
            total_days = len(df)

        missing = df[PRICE_COLUMN].isna().sum()
        missing_pct = missing / max(len(df), 1)

        if missing_pct > MAX_MISSING_PCT:
            dropped.append((ticker, f"{missing_pct:.1%} missing"))
            continue

        # Forward-fill small gaps
        df = df.copy()
        df[PRICE_COLUMN] = (
            df[PRICE_COLUMN]
            .ffill(limit=GAP_FILL_LIMIT)
        )

        # Drop invalid prices
        df = df[df[PRICE_COLUMN] > 0]
        cleaned[ticker] = df

    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers:")
        for t, reason in dropped:
            logger.warning(f"  {t}: {reason}")

    logger.info(f"Clean universe: {len(cleaned)} tickers")
    return cleaned


# ── Index & Sector Indices ─────────────────────────────────────────────────

def download_index_data(
    force: bool = False,
) -> pd.DataFrame:
    """
    Download Nifty 50 index + all sector indices.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns = [nifty50, bank, it, fmcg, auto, pharma]
        Values are adjusted close prices.
    """
    cache_name = "index_data"
    if not force and _is_fresh(_cache_path(cache_name)):
        logger.info("Loading index data from cache")
        return _load(cache_name)

    tickers = {
        "nifty50": INDEX_TICKER,
        **SECTOR_INDICES,
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(
                ticker,
                start=HISTORY_START,
                end=HISTORY_END,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data for {ticker} ({name})")
                continue
            frames[name] = df["Close"].rename(name)
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")

    if not frames:
        raise RuntimeError("Could not download any index data")

    combined = pd.concat(frames.values(), axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index().ffill(limit=5)

    _save(combined, cache_name)
    return combined


# ── India VIX ─────────────────────────────────────────────────────────────

def download_vix(force: bool = False) -> pd.DataFrame:
    """
    Download India VIX.
    Tries NSE direct API first; falls back to yfinance ^INDIAVIX.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, single column 'vix'.
    """
    cache_name = "india_vix"
    if not force and _is_fresh(_cache_path(cache_name)):
        logger.info("Loading VIX from cache")
        return _load(cache_name)

    vix = None

    # Attempt 1: NSE direct (may require browser-like headers)
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        session = requests.Session()
        # Prime cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        start_dt = datetime.strptime(HISTORY_START, "%Y-%m-%d")
        end_dt   = datetime.strptime(HISTORY_END,   "%Y-%m-%d")

        all_rows = []
        cursor = start_dt
        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=365), end_dt)
            url = (
                f"https://www.nseindia.com/api/historical/vixhistory"
                f"?from={cursor.strftime('%d-%m-%Y')}"
                f"&to={chunk_end.strftime('%d-%m-%Y')}"
            )
            resp = session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            all_rows.extend(data)
            cursor = chunk_end + timedelta(days=1)
            time.sleep(0.5)   # polite delay

        if all_rows:
            vix_df = pd.DataFrame(all_rows)
            vix_df["Date"] = pd.to_datetime(
                vix_df["EOD_TIMESTAMP"], format="%d-%b-%Y"
            )
            vix_df = vix_df.set_index("Date")[["EOD_VIX_CLOSE"]].rename(
                columns={"EOD_VIX_CLOSE": "vix"}
            )
            vix = vix_df.sort_index()
            logger.info("VIX: downloaded from NSE direct")

    except Exception as e:
        logger.warning(f"NSE VIX direct failed ({e}), trying yfinance fallback")

    # Attempt 2: yfinance fallback
    if vix is None or vix.empty:
        try:
            raw = yf.download(
                INDIA_VIX_FALLBACK_TICKER,
                start=HISTORY_START,
                end=HISTORY_END,
                auto_adjust=True,
                progress=False,
            )
            if not raw.empty:
                vix = raw[["Close"]].rename(columns={"Close": "vix"})
                vix.index = pd.to_datetime(vix.index)
                vix = vix.sort_index()
                logger.info("VIX: downloaded from yfinance fallback")
        except Exception as e2:
            logger.warning(f"yfinance VIX fallback also failed: {e2}")

    if vix is None or vix.empty:
        logger.warning("VIX unavailable — will be skipped in feature engineering")
        vix = pd.DataFrame(columns=["vix"])

    _save(vix, cache_name)
    return vix


# ── FRED Macro Data ────────────────────────────────────────────────────────

def download_macro_fred(force: bool = False) -> pd.DataFrame:
    """
    Download macro series from FRED:
      - USD/INR exchange rate (daily)
      - India lending rate / repo rate proxy (monthly → ffill to daily)

    Returns
    -------
    pd.DataFrame
        DatetimeIndex aligned to trading days.
        Columns: usd_inr, repo_rate.
    """
    if not ENABLE_MACRO_FRED:
        logger.info("FRED macro feeds disabled (ENABLE_MACRO_FRED=False)")
        return pd.DataFrame()

    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping macro download")
        return pd.DataFrame()

    cache_name = "macro_fred"
    if not force and _is_fresh(_cache_path(cache_name)):
        logger.info("Loading FRED macro from cache")
        return _load(cache_name)

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
    except ImportError:
        logger.warning("fredapi not installed — pip install fredapi")
        return pd.DataFrame()

    frames = {}
    for col_name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=HISTORY_START,
                observation_end=HISTORY_END,
            )
            s.name = col_name
            frames[col_name] = s
            logger.info(f"FRED: downloaded {series_id} ({col_name})")
        except Exception as e:
            logger.warning(f"FRED: failed to get {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1)
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    # Forward-fill monthly series to daily; limit 31 days
    macro = macro.ffill(limit=31)

    _save(macro, cache_name)
    return macro


# ── Master Download ────────────────────────────────────────────────────────

def download_all(force: bool = False) -> dict:
    """
    Run all downloads in sequence and return a dict of all datasets.

    Returns
    -------
    dict with keys:
        'ohlcv'   : dict[str, pd.DataFrame]  — per-ticker price data
        'index'   : pd.DataFrame             — Nifty + sector indices
        'vix'     : pd.DataFrame             — India VIX
        'macro'   : pd.DataFrame             — FRED macro series
    """
    logger.info("=" * 55)
    logger.info("NSE-Alpha v1 — Data Download")
    logger.info("=" * 55)

    raw_ohlcv   = download_ohlcv(force=force)
    clean       = clean_ohlcv(raw_ohlcv)
    index_data  = download_index_data(force=force)
    vix_data    = download_vix(force=force)
    macro_data  = download_macro_fred(force=force)

    logger.info("-" * 55)
    logger.info(f"OHLCV  : {len(clean)} tickers")
    logger.info(f"Index  : {index_data.shape}")
    logger.info(f"VIX    : {len(vix_data)} rows")
    logger.info(f"Macro  : {macro_data.shape if not macro_data.empty else 'skipped'}")
    logger.info("=" * 55)

    return {
        "ohlcv":  clean,
        "index":  index_data,
        "vix":    vix_data,
        "macro":  macro_data,
    }


if __name__ == "__main__":
    datasets = download_all()
    print(f"\nSample ticker (TCS):")
    print(datasets["ohlcv"].get("TCS", pd.DataFrame()).tail(3))
