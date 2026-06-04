"""
features/macro.py
─────────────────
Builds macro / market-level features and merges them into a
per-ticker feature DataFrame.

Features built:
  - India VIX level + 5-day change
  - Nifty 50 rolling 5d and 20d return
  - Sector index return matching the stock's sector
  - USD/INR 5-day change
  - RBI repo rate level
"""

import numpy as np
import pandas as pd

from config.settings import (
    VIX_CHANGE_WINDOW,
    SECTOR_RETURN_WINDOW,
    MACRO_CHANGE_WINDOW,
    EXCHANGE_SUFFIX,
)
from data.universe import get_sector


def build_macro_features(
    index_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build market-wide feature DataFrame aligned to trading dates.

    Parameters
    ----------
    index_df : pd.DataFrame
        Columns: nifty50, bank, it, fmcg, auto, pharma (prices).
    vix_df   : pd.DataFrame
        Column: vix.
    macro_df : pd.DataFrame
        Columns: usd_inr, repo_rate (may be empty if FRED disabled).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex = union of all inputs, forward-filled.
        All macro feature columns.
    """
    frames = {}

    # ── Nifty 50 features ──────────────────────────────────────────────
    if "nifty50" in index_df.columns:
        nifty = index_df["nifty50"]
        frames["nifty_return_5d"]  = nifty.pct_change(5)
        frames["nifty_return_20d"] = nifty.pct_change(20)
        frames["nifty_level"]      = nifty

    # ── Sector index features ──────────────────────────────────────────
    for sector in ["bank", "it", "fmcg", "auto", "pharma"]:
        if sector in index_df.columns:
            s = index_df[sector]
            frames[f"sector_{sector}_return_{SECTOR_RETURN_WINDOW}d"] = (
                s.pct_change(SECTOR_RETURN_WINDOW)
            )

    # ── India VIX ──────────────────────────────────────────────────────
    if not vix_df.empty and "vix" in vix_df.columns:
        vix = vix_df["vix"]
        frames["vix_level"]      = vix
        frames["vix_change_5d"]  = vix.pct_change(VIX_CHANGE_WINDOW)
        frames["vix_high_flag"]  = (vix > vix.rolling(252).mean() * 1.2).astype(int)

    # ── FRED macro ─────────────────────────────────────────────────────
    if not macro_df.empty:
        if "usd_inr" in macro_df.columns:
            inr = macro_df["usd_inr"]
            frames["usd_inr"]             = inr
            frames["usd_inr_change_5d"]   = inr.pct_change(MACRO_CHANGE_WINDOW)

        if "repo_rate" in macro_df.columns:
            repo = macro_df["repo_rate"]
            frames["repo_rate"]           = repo
            frames["repo_rate_change_60d"]= repo.diff(60)

    if not frames:
        return pd.DataFrame()

    macro_features = pd.DataFrame(frames)
    macro_features.index = pd.to_datetime(macro_features.index)
    macro_features = macro_features.sort_index()

    # Forward-fill macro to daily (handles weekends/holidays)
    macro_features = macro_features.ffill(limit=5)

    return macro_features


def merge_macro_into_ticker(
    ticker_features: pd.DataFrame,
    macro_features: pd.DataFrame,
    ticker_symbol: str,
) -> pd.DataFrame:
    """
    Merge macro features into a per-ticker feature DataFrame.
    Also adds the relevant sector index return for this stock's sector.

    Parameters
    ----------
    ticker_features : pd.DataFrame
        Output of build_technical_features(), DatetimeIndex.
    macro_features  : pd.DataFrame
        Output of build_macro_features(), DatetimeIndex.
    ticker_symbol   : str
        NSE symbol without .NS suffix, e.g. 'RELIANCE'.

    Returns
    -------
    pd.DataFrame
        ticker_features with macro columns appended.
        Index aligned — only dates present in ticker_features are kept.
    """
    if macro_features.empty:
        return ticker_features

    # Align macro to ticker dates via reindex + ffill
    aligned_macro = (
        macro_features
        .reindex(ticker_features.index.union(macro_features.index))
        .ffill(limit=5)
        .reindex(ticker_features.index)
    )

    # Add sector-specific return as a focused feature
    sector = get_sector(ticker_symbol)
    sector_col = f"sector_{sector}_return_{SECTOR_RETURN_WINDOW}d"
    if sector_col in aligned_macro.columns:
        aligned_macro["own_sector_return"] = aligned_macro[sector_col]
    else:
        aligned_macro["own_sector_return"] = np.nan

    return pd.concat([ticker_features, aligned_macro], axis=1)


if __name__ == "__main__":
    # Smoke test using dummy data
    dates = pd.bdate_range("2023-01-01", "2024-01-01")
    idx = pd.DataFrame({
        "nifty50": np.random.randn(len(dates)).cumsum() + 18000,
        "bank":    np.random.randn(len(dates)).cumsum() + 42000,
        "it":      np.random.randn(len(dates)).cumsum() + 30000,
        "fmcg":    np.random.randn(len(dates)).cumsum() + 50000,
        "auto":    np.random.randn(len(dates)).cumsum() + 12000,
        "pharma":  np.random.randn(len(dates)).cumsum() + 14000,
    }, index=dates)
    vix = pd.DataFrame({"vix": np.random.uniform(10, 30, len(dates))}, index=dates)
    macro = pd.DataFrame()
    mf = build_macro_features(idx, vix, macro)
    print(mf.shape)
    print(mf.tail(3))
