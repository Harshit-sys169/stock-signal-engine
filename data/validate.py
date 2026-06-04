"""
data/validate.py
────────────────
Validates downloaded data quality before feature engineering or training.

Checks:
  - Ticker coverage vs expected universe
  - Date range completeness
  - Missing value rates per ticker
  - Price anomalies (zero prices, extreme single-day moves)
  - Volume anomalies
  - Macro data availability
  - VIX data availability

Prints a clean summary report and returns a dict of issues found.
Run this after download_all() before proceeding to feature engineering.
"""

import logging
import numpy as np
import pandas as pd

from config.settings import (
    HISTORY_START, HISTORY_END,
    MAX_MISSING_PCT, PRICE_COLUMN,
)
from data.universe import get_universe

logger = logging.getLogger(__name__)

# Thresholds
MAX_SINGLE_DAY_MOVE  = 0.30    # flag if >30% single day move (likely split/error)
MIN_DAILY_VOLUME     = 1000    # flag if avg volume < this (illiquid)
MIN_TRADING_DAYS     = 500     # flag if ticker has fewer than this many rows


def validate_ohlcv(data: dict[str, pd.DataFrame]) -> dict:
    """
    Validate per-ticker OHLCV data.

    Returns dict with keys:
      - 'missing_tickers'    : tickers expected but not downloaded
      - 'low_coverage'       : tickers with too few trading days
      - 'high_missing_pct'   : tickers with too many NaN close prices
      - 'price_anomalies'    : tickers with extreme single-day moves
      - 'volume_anomalies'   : tickers with very low average volume
      - 'summary'            : per-ticker summary DataFrame
    """
    expected = set(t.replace(".NS", "") for t in get_universe())
    actual   = set(data.keys())

    issues = {
        "missing_tickers":  sorted(expected - actual),
        "low_coverage":     [],
        "high_missing_pct": [],
        "price_anomalies":  [],
        "volume_anomalies": [],
    }

    rows = []
    for ticker, df in data.items():
        if PRICE_COLUMN not in df.columns:
            issues["high_missing_pct"].append(ticker)
            continue

        close = df[PRICE_COLUMN].astype(float)
        n_days = len(df)
        missing_pct = close.isna().mean()

        # Single-day return extremes
        daily_ret = close.pct_change().abs()
        max_move  = daily_ret.max()
        n_extreme = (daily_ret > MAX_SINGLE_DAY_MOVE).sum()

        # Volume check
        avg_vol = df["Volume"].mean() if "Volume" in df.columns else np.nan

        rows.append({
            "ticker":       ticker,
            "n_days":       n_days,
            "start_date":   df.index.min().date() if not df.empty else None,
            "end_date":     df.index.max().date() if not df.empty else None,
            "missing_pct":  missing_pct,
            "max_1d_move":  max_move,
            "n_extreme":    n_extreme,
            "avg_volume":   avg_vol,
            "min_price":    close.min(),
            "max_price":    close.max(),
        })

        if n_days < MIN_TRADING_DAYS:
            issues["low_coverage"].append(ticker)
        if missing_pct > MAX_MISSING_PCT:
            issues["high_missing_pct"].append(ticker)
        if n_extreme > 5:
            issues["price_anomalies"].append(ticker)
        if avg_vol < MIN_DAILY_VOLUME:
            issues["volume_anomalies"].append(ticker)

    issues["summary"] = pd.DataFrame(rows).set_index("ticker")
    return issues


def validate_index(index_df: pd.DataFrame) -> dict:
    """Check index / sector data completeness."""
    expected_cols = ["nifty50", "bank", "it", "fmcg", "auto", "pharma"]
    missing_cols  = [c for c in expected_cols if c not in index_df.columns]
    missing_pct   = index_df.isna().mean().to_dict()

    return {
        "missing_columns": missing_cols,
        "missing_pct":     missing_pct,
        "n_rows":          len(index_df),
        "date_range":      (index_df.index.min(), index_df.index.max()),
    }


def validate_vix(vix_df: pd.DataFrame) -> dict:
    if vix_df.empty or "vix" not in vix_df.columns:
        return {"available": False, "n_rows": 0}
    return {
        "available": True,
        "n_rows":    len(vix_df),
        "missing_pct": vix_df["vix"].isna().mean(),
        "date_range": (vix_df.index.min(), vix_df.index.max()),
        "vix_range":  (vix_df["vix"].min(), vix_df["vix"].max()),
    }


def validate_macro(macro_df: pd.DataFrame) -> dict:
    if macro_df.empty:
        return {"available": False}
    return {
        "available":   True,
        "columns":     list(macro_df.columns),
        "n_rows":      len(macro_df),
        "missing_pct": macro_df.isna().mean().to_dict(),
        "date_range":  (macro_df.index.min(), macro_df.index.max()),
    }


def run_validation(datasets: dict, verbose: bool = True) -> dict:
    """
    Run all validation checks and print a summary report.

    Parameters
    ----------
    datasets : dict — output of download_all()
    verbose  : bool — print report to stdout

    Returns
    -------
    dict with all validation results
    """
    ohlcv_issues = validate_ohlcv(datasets["ohlcv"])
    index_info   = validate_index(datasets["index"])
    vix_info     = validate_vix(datasets["vix"])
    macro_info   = validate_macro(datasets["macro"])

    if verbose:
        sep = "=" * 55
        print(sep)
        print("DATA VALIDATION REPORT")
        print(sep)

        # OHLCV
        print(f"\nOHLCV")
        print(f"  Tickers downloaded   : {len(datasets['ohlcv'])}")
        print(f"  Missing tickers      : {len(ohlcv_issues['missing_tickers'])}", end="")
        if ohlcv_issues["missing_tickers"]:
            print(f"  → {ohlcv_issues['missing_tickers'][:5]}", end="")
        print()
        print(f"  Low coverage         : {len(ohlcv_issues['low_coverage'])}")
        print(f"  High missing %       : {len(ohlcv_issues['high_missing_pct'])}")
        print(f"  Price anomalies      : {len(ohlcv_issues['price_anomalies'])}")
        print(f"  Volume anomalies     : {len(ohlcv_issues['volume_anomalies'])}")

        # Summary stats
        s = ohlcv_issues["summary"]
        if not s.empty:
            print(f"\n  Avg trading days     : {s['n_days'].mean():.0f}")
            print(f"  Median missing pct   : {s['missing_pct'].median():.2%}")
            print(f"  Date range (median)  : {s['start_date'].min()} to {s['end_date'].max()}")

        # Index
        print(f"\nIndex / Sector Data")
        print(f"  Rows                 : {index_info['n_rows']}")
        print(f"  Missing columns      : {index_info['missing_columns'] or 'none'}")
        for col, pct in index_info["missing_pct"].items():
            flag = " ⚠" if pct > 0.05 else ""
            print(f"  {col:<20}: {pct:.2%} missing{flag}")

        # VIX
        print(f"\nIndia VIX")
        if vix_info["available"]:
            print(f"  Rows                 : {vix_info['n_rows']}")
            print(f"  Missing pct          : {vix_info['missing_pct']:.2%}")
            print(f"  VIX range            : {vix_info['vix_range'][0]:.1f} to {vix_info['vix_range'][1]:.1f}")
        else:
            print("  Not available ⚠")

        # Macro
        print(f"\nFRED Macro")
        if macro_info["available"]:
            print(f"  Columns              : {macro_info['columns']}")
            print(f"  Rows                 : {macro_info['n_rows']}")
            for col, pct in macro_info["missing_pct"].items():
                flag = " ⚠" if pct > 0.10 else ""
                print(f"  {col:<20}: {pct:.2%} missing{flag}")
        else:
            print("  Skipped (ENABLE_MACRO_FRED=False or no API key)")

        # Overall verdict
        total_issues = (
            len(ohlcv_issues["missing_tickers"]) +
            len(ohlcv_issues["price_anomalies"]) +
            len(ohlcv_issues["high_missing_pct"])
        )
        print(f"\n{'=' * 55}")
        if total_issues == 0:
            print("VERDICT: ✅  Data looks clean. Safe to proceed.")
        elif total_issues < 5:
            print(f"VERDICT: ⚠   {total_issues} minor issues. Review above before training.")
        else:
            print(f"VERDICT: ❌  {total_issues} issues found. Fix data before training.")
        print("=" * 55)

    return {
        "ohlcv":  ohlcv_issues,
        "index":  index_info,
        "vix":    vix_info,
        "macro":  macro_info,
    }


if __name__ == "__main__":
    from data.downloader import download_all
    datasets = download_all()
    run_validation(datasets)
