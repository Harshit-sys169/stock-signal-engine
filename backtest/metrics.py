"""
backtest/metrics.py
───────────────────
Computes all trading performance metrics from backtest output.

Metrics computed:
  - CAGR
  - Sharpe ratio (annualised)
  - Sortino ratio
  - Maximum drawdown
  - Win rate
  - Profit factor
  - Average win / average loss
  - Trade count by exit reason
  - Benchmark comparison vs Nifty 50 (buy-and-hold)
"""

import numpy as np
import pandas as pd

from config.settings import (
    TARGET_SHARPE, TARGET_WIN_RATE,
    TARGET_PROFIT_FACTOR, TARGET_CAGR_ALPHA,
    INITIAL_CAPITAL,
)

TRADING_DAYS_PER_YEAR = 252


def cagr(equity: pd.Series, initial: float | None = None) -> float:
    """Compound Annual Growth Rate."""
    if initial is None:
        initial = equity.iloc[0]
    final = equity.iloc[-1]
    n_years = len(equity) / TRADING_DAYS_PER_YEAR
    if n_years <= 0 or initial <= 0:
        return 0.0
    return (final / initial) ** (1 / n_years) - 1


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """
    Annualised Sharpe ratio.
    risk_free_rate default = 6.5% (approximate India 1yr T-bill rate).
    """
    excess = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Annualised Sortino ratio (uses downside deviation)."""
    excess = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a negative fraction)."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def win_rate(trade_log: pd.DataFrame) -> float:
    if trade_log.empty:
        return 0.0
    return (trade_log["pnl"] > 0).mean()


def profit_factor(trade_log: pd.DataFrame) -> float:
    if trade_log.empty:
        return 0.0
    wins  = trade_log["pnl"][trade_log["pnl"] > 0].sum()
    losses = trade_log["pnl"][trade_log["pnl"] < 0].abs().sum()
    return wins / losses if losses > 0 else np.inf


def compute_all_metrics(
    equity_df:   pd.DataFrame,
    trade_log:   pd.DataFrame,
    benchmark:   pd.Series | None = None,
    initial_cap: float = INITIAL_CAPITAL,
) -> dict:
    """
    Compute full set of trading performance metrics.

    Parameters
    ----------
    equity_df  : DataFrame with column 'value' (daily portfolio value)
    trade_log  : DataFrame with trade records from backtest engine
    benchmark  : Nifty 50 price series (optional) for alpha calculation
    initial_cap: Initial portfolio capital

    Returns
    -------
    dict of all metrics
    """
    equity = equity_df["value"]
    daily_ret = equity.pct_change().dropna()

    metrics = {}

    # ── Return metrics ─────────────────────────────────────────────
    metrics["cagr"]              = cagr(equity, initial_cap)
    metrics["total_return"]      = (equity.iloc[-1] - initial_cap) / initial_cap
    metrics["sharpe_ratio"]      = sharpe_ratio(daily_ret)
    metrics["sortino_ratio"]     = sortino_ratio(daily_ret)
    metrics["max_drawdown"]      = max_drawdown(equity)
    metrics["avg_daily_return"]  = daily_ret.mean()
    metrics["volatility_annual"] = daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # ── Trade metrics ──────────────────────────────────────────────
    if not trade_log.empty:
        metrics["total_trades"]     = len(trade_log)
        metrics["win_rate"]         = win_rate(trade_log)
        metrics["profit_factor"]    = profit_factor(trade_log)
        metrics["avg_win_pct"]      = (
            trade_log.loc[trade_log["pnl"] > 0, "return_pct"].mean()
        )
        metrics["avg_loss_pct"]     = (
            trade_log.loc[trade_log["pnl"] < 0, "return_pct"].mean()
        )
        metrics["avg_hold_days"]    = trade_log["hold_days"].mean()
        metrics["total_pnl"]        = trade_log["pnl"].sum()
        # Exit reason breakdown
        reason_counts = trade_log["exit_reason"].value_counts().to_dict()
        metrics["exits_stop_loss"]    = reason_counts.get("stop_loss",    0)
        metrics["exits_take_profit"]  = reason_counts.get("take_profit",  0)
        metrics["exits_hold_expired"] = reason_counts.get("hold_expired", 0)
    else:
        metrics.update({
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "avg_hold_days": 0,
            "total_pnl": 0, "exits_stop_loss": 0,
            "exits_take_profit": 0, "exits_hold_expired": 0,
        })

    # ── Benchmark comparison ───────────────────────────────────────
    if benchmark is not None:
        bm_aligned = benchmark.reindex(equity.index).ffill()
        bm_ret     = bm_aligned.pct_change().dropna()
        bm_cagr    = cagr(bm_aligned)
        metrics["benchmark_cagr"]   = bm_cagr
        metrics["alpha"]            = metrics["cagr"] - bm_cagr
        metrics["benchmark_sharpe"] = sharpe_ratio(bm_ret)
        metrics["benchmark_mdd"]    = max_drawdown(bm_aligned)
        # Beta
        aligned = pd.concat([daily_ret, bm_ret], axis=1).dropna()
        if len(aligned) > 10:
            cov   = aligned.cov().iloc[0, 1]
            b_var = aligned.iloc[:, 1].var()
            metrics["beta"] = cov / b_var if b_var > 0 else np.nan
        else:
            metrics["beta"] = np.nan

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print metrics with pass/fail against targets."""
    targets = {
        "sharpe_ratio":  (TARGET_SHARPE,        "≥", True),
        "win_rate":      (TARGET_WIN_RATE,       "≥", True),
        "profit_factor": (TARGET_PROFIT_FACTOR,  "≥", True),
        "max_drawdown":  (-0.15,                 "≥", True),   # drawdown is negative
        "alpha":         (TARGET_CAGR_ALPHA,     "≥", False),  # optional
    }

    print("\n" + "=" * 55)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("=" * 55)

    formatters = {
        "cagr":              lambda v: f"{v:.2%}",
        "total_return":      lambda v: f"{v:.2%}",
        "sharpe_ratio":      lambda v: f"{v:.3f}",
        "sortino_ratio":     lambda v: f"{v:.3f}",
        "max_drawdown":      lambda v: f"{v:.2%}",
        "volatility_annual": lambda v: f"{v:.2%}",
        "win_rate":          lambda v: f"{v:.2%}",
        "profit_factor":     lambda v: f"{v:.3f}",
        "avg_win_pct":       lambda v: f"{v:.2%}",
        "avg_loss_pct":      lambda v: f"{v:.2%}",
        "avg_hold_days":     lambda v: f"{v:.1f}d",
        "total_pnl":         lambda v: f"₹{v:,.0f}",
        "benchmark_cagr":    lambda v: f"{v:.2%}",
        "alpha":             lambda v: f"{v:.2%}",
        "beta":              lambda v: f"{v:.3f}",
    }

    for key, value in metrics.items():
        fmt = formatters.get(key, lambda v: str(v))
        try:
            val_str = fmt(value)
        except Exception:
            val_str = str(value)

        flag = ""
        if key in targets:
            threshold, op, required = targets[key]
            passed = value >= threshold
            flag = " ✅" if passed else " ❌"

        print(f"  {key:<26} {val_str}{flag}")

    print("=" * 55)
