"""
backtest/report.py
──────────────────
Generates a self-contained HTML performance report from backtest results.

Output: outputs/backtest_report.html
Open in any browser. No server required.

Includes:
  - Key metrics scorecard with pass/fail against targets
  - Equity curve vs Nifty 50 benchmark (interactive Plotly)
  - Drawdown chart
  - Monthly returns heatmap
  - Trade log table
  - Exit reason breakdown
  - Sector P&L breakdown
  - Walk-forward fold performance
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config.settings import (
    OUTPUTS_DIR, MAX_DRAWDOWN_PCT,
    TARGET_SHARPE, TARGET_WIN_RATE,
    TARGET_PROFIT_FACTOR, TARGET_CAGR_ALPHA,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, HOLD_PERIOD_DAYS,
    MIN_CONFIDENCE, TRANSACTION_COST, SLIPPAGE,
)
from backtest.metrics import compute_all_metrics, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

REPORT_PATH = OUTPUTS_DIR.parent / "backtest_report.html"
INITIAL_CAPITAL = 1_000_000


def _metric_card(label: str, value: str, passed: bool | None = None) -> str:
    if passed is True:
        border = "#26a641"
        badge  = '<span style="color:#26a641;font-size:18px">✓</span>'
    elif passed is False:
        border = "#da3633"
        badge  = '<span style="color:#da3633;font-size:18px">✗</span>'
    else:
        border = "#444"
        badge  = ""

    return f"""
    <div style="background:#161b22;border:1px solid {border};border-radius:8px;
                padding:16px 20px;min-width:160px;text-align:center;">
      <div style="color:#8b949e;font-size:12px;margin-bottom:6px">{label}</div>
      <div style="color:#e6edf3;font-size:24px;font-weight:600">{value} {badge}</div>
    </div>"""


def _monthly_returns_heatmap(equity: pd.Series) -> go.Figure:
    daily_ret = equity.pct_change().dropna()
    monthly   = (1 + daily_ret).resample("ME").prod() - 1

    df = monthly.reset_index()
    df.columns = ["date", "return"]
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")

    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=month_order)

    fig = px.imshow(
        pivot * 100,
        color_continuous_scale=["#da3633", "#0d1117", "#26a641"],
        color_continuous_midpoint=0,
        text_auto=".1f",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont_size=10)
    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="",
    )
    return fig


def generate_report(
    equity_df:   pd.DataFrame,
    trade_log:   pd.DataFrame,
    oos_df:      pd.DataFrame | None = None,
    benchmark:   pd.Series | None   = None,
) -> Path:
    """
    Generate full HTML backtest report.

    Parameters
    ----------
    equity_df  : DataFrame with 'value', 'drawdown' columns
    trade_log  : DataFrame from backtest engine
    oos_df     : Walk-forward OOS predictions (optional)
    benchmark  : Nifty 50 price series (optional)

    Returns
    -------
    Path to generated HTML report
    """
    equity = equity_df["value"]
    metrics = compute_all_metrics(equity_df, trade_log, benchmark, INITIAL_CAPITAL)

    # ── Equity curve chart ─────────────────────────────────────────
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="NSE-Alpha v1", line=dict(color="#26a641", width=2),
        fill="tozeroy", fillcolor="rgba(38,166,65,0.08)",
    ))
    if benchmark is not None:
        bm = benchmark.reindex(equity.index).ffill()
        bm_scaled = bm / bm.iloc[0] * INITIAL_CAPITAL
        fig_equity.add_trace(go.Scatter(
            x=bm_scaled.index, y=bm_scaled.values,
            name="Nifty 50 (B&H)", line=dict(color="#6e7681", width=1.5, dash="dash"),
        ))
    fig_equity.update_layout(
        title="Portfolio Equity Curve", template="plotly_dark",
        height=380, yaxis_title="Value (INR)",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    # ── Drawdown chart ─────────────────────────────────────────────
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["drawdown"] * 100,
        fill="tozeroy", fillcolor="rgba(218,54,51,0.4)",
        line=dict(color="#da3633", width=1),
        name="Drawdown",
    ))
    fig_dd.add_hline(
        y=MAX_DRAWDOWN_PCT * 100, line_dash="dash",
        line_color="orange", annotation_text="Max allowed",
    )
    fig_dd.update_layout(
        title="Drawdown (%)", template="plotly_dark",
        height=220, yaxis_title="%",
        margin=dict(l=40, r=20, t=40, b=30),
    )

    # ── Monthly returns heatmap ────────────────────────────────────
    fig_monthly = _monthly_returns_heatmap(equity)

    # ── Trade analysis charts ──────────────────────────────────────
    if not trade_log.empty:
        fig_returns = px.histogram(
            trade_log, x=trade_log["return_pct"] * 100,
            nbins=40, color_discrete_sequence=["steelblue"],
            title="Trade Return Distribution (%)",
            labels={"x": "Return (%)"},
        )
        fig_returns.add_vline(x=0, line_color="white", line_dash="dash")
        fig_returns.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=40, r=20, t=40, b=30),
        )

        exit_counts = trade_log["exit_reason"].value_counts().reset_index()
        exit_counts.columns = ["reason", "count"]
        fig_exits = px.pie(
            exit_counts, names="reason", values="count",
            title="Exit Reasons",
            color_discrete_sequence=["#da3633", "#26a641", "#6e7681"],
        )
        fig_exits.update_layout(
            template="plotly_dark", height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        sector_pnl = trade_log.groupby("sector")["pnl"].sum().sort_values().reset_index()
        sector_pnl["color"] = sector_pnl["pnl"].apply(
            lambda x: "#26a641" if x >= 0 else "#da3633"
        )
        fig_sector = go.Figure(go.Bar(
            x=sector_pnl["pnl"], y=sector_pnl["sector"],
            orientation="h",
            marker_color=sector_pnl["color"],
        ))
        fig_sector.update_layout(
            title="P&L by Sector (INR)", template="plotly_dark",
            height=320, margin=dict(l=80, r=20, t=40, b=30),
        )
    else:
        fig_returns = go.Figure()
        fig_exits   = go.Figure()
        fig_sector  = go.Figure()

    # ── Walk-forward fold metrics ──────────────────────────────────
    fold_html = ""
    if oos_df is not None and "fold" in oos_df.columns:
        from sklearn.metrics import accuracy_score, f1_score
        fold_rows = []
        for fold_num, grp in oos_df.groupby("fold"):
            acc = accuracy_score(grp["signal"].astype(int), grp["pred_signal"].astype(int))
            f1  = f1_score(grp["signal"].astype(int), grp["pred_signal"].astype(int),
                           average="macro", zero_division=0)
            fold_rows.append({
                "Fold": int(fold_num),
                "Start": grp.index.min().date(),
                "End":   grp.index.max().date(),
                "Rows":  f"{len(grp):,}",
                "Accuracy": f"{acc:.3f}",
                "Macro-F1": f"{f1:.3f}",
            })
        fold_df   = pd.DataFrame(fold_rows)
        fold_html = fold_df.to_html(index=False, border=0,
            classes="table", escape=False)

    # ── Metric scorecards ──────────────────────────────────────────
    def pct(v): return f"{v:.2%}"
    def flt(v): return f"{v:.3f}"
    def inr(v): return f"₹{v:,.0f}"

    cards_html = "".join([
        _metric_card("CAGR",            pct(metrics.get("cagr", 0))),
        _metric_card("Sharpe Ratio",     flt(metrics.get("sharpe_ratio", 0)),
                     metrics.get("sharpe_ratio", 0) >= TARGET_SHARPE),
        _metric_card("Max Drawdown",     pct(metrics.get("max_drawdown", 0)),
                     metrics.get("max_drawdown", 0) >= MAX_DRAWDOWN_PCT),
        _metric_card("Win Rate",         pct(metrics.get("win_rate", 0)),
                     metrics.get("win_rate", 0) >= TARGET_WIN_RATE),
        _metric_card("Profit Factor",    flt(metrics.get("profit_factor", 0)),
                     metrics.get("profit_factor", 0) >= TARGET_PROFIT_FACTOR),
        _metric_card("Total Trades",     str(int(metrics.get("total_trades", 0)))),
        _metric_card("Total P&L",        inr(metrics.get("total_pnl", 0))),
        _metric_card("Alpha vs Nifty",   pct(metrics.get("alpha", 0)) if "alpha" in metrics else "N/A",
                     metrics.get("alpha", 0) >= TARGET_CAGR_ALPHA if "alpha" in metrics else None),
    ])

    # ── Trade log table ────────────────────────────────────────────
    trade_table_html = ""
    if not trade_log.empty:
        tl = trade_log.copy().sort_values("entry_date", ascending=False).head(50)
        tl["return_pct"] = (tl["return_pct"] * 100).round(2).astype(str) + "%"
        tl["pnl"]        = tl["pnl"].apply(lambda x: f"₹{x:+,.0f}")
        tl["entry_price"]= tl["entry_price"].apply(lambda x: f"₹{x:,.2f}")
        tl["exit_price"] = tl["exit_price"].apply(lambda x: f"₹{x:,.2f}")
        display_cols     = ["ticker", "sector", "entry_date", "exit_date",
                            "entry_price", "exit_price", "return_pct", "pnl",
                            "hold_days", "exit_reason"]
        trade_table_html = tl[display_cols].to_html(
            index=False, border=0, classes="table", escape=False
        )

    # ── Assemble HTML ──────────────────────────────────────────────
    config_rows = [
        ("Market",           "NSE India"),
        ("Universe",         "Nifty 50 + Nifty Next 50 (100 stocks)"),
        ("Signal",           "Buy / Hold / Sell"),
        ("Forward window",   f"{HOLD_PERIOD_DAYS} trading days"),
        ("Stop-loss",        f"{STOP_LOSS_PCT:.1%}"),
        ("Take-profit",      f"{TAKE_PROFIT_PCT:.1%}"),
        ("Min confidence",   f"{MIN_CONFIDENCE:.0%}"),
        ("Transaction cost", f"{TRANSACTION_COST:.2%} + {SLIPPAGE:.2%} slippage"),
        ("Initial capital",  f"₹{INITIAL_CAPITAL:,.0f}"),
        ("Generated",        datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    config_html = "".join(
        f"<tr><td style='color:#8b949e;padding:4px 12px 4px 0'>{k}</td>"
        f"<td style='color:#e6edf3;padding:4px 0'>{v}</td></tr>"
        for k, v in config_rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NSE-Alpha v1 — Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif;
          font-size: 14px; padding: 32px; }}
  h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 600; color: #8b949e; margin: 32px 0 14px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .table th {{ background: #161b22; color: #8b949e; padding: 8px 12px;
               text-align: left; border-bottom: 1px solid #30363d; }}
  .table td {{ padding: 7px 12px; border-bottom: 1px solid #21262d; }}
  .table tr:hover td {{ background: #161b22; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
            padding: 8px; margin-bottom: 20px; }}
  .config-table {{ font-size: 13px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 28px; }}
</style>
</head>
<body>
<h1>NSE-Alpha v1 — Backtest Report</h1>
<p class="subtitle">Walk-forward backtest · NSE India · Daily signals · Large-cap universe</p>

<h2>Performance Scorecard</h2>
<div class="cards">{cards_html}</div>

<h2>Equity Curve</h2>
<div class="chart" id="equity"></div>

<h2>Drawdown</h2>
<div class="chart" id="drawdown"></div>

<h2>Monthly Returns</h2>
<div class="chart" id="monthly"></div>

<div class="two-col">
  <div>
    <h2>Trade Return Distribution</h2>
    <div class="chart" id="returns"></div>
  </div>
  <div>
    <h2>Exit Reasons</h2>
    <div class="chart" id="exits"></div>
  </div>
</div>

<h2>P&amp;L by Sector</h2>
<div class="chart" id="sector"></div>

{"<h2>Walk-Forward Fold Results</h2><div style='overflow-x:auto'>" + fold_html + "</div>" if fold_html else ""}

<h2>Recent Trades (last 50)</h2>
<div style="overflow-x:auto">
{trade_table_html or "<p style='color:#8b949e'>No trades available.</p>"}
</div>

<h2>System Configuration</h2>
<table class="config-table">{config_html}</table>

<script>
  var eq   = {fig_equity.to_json()};
  var dd   = {fig_dd.to_json()};
  var mo   = {fig_monthly.to_json()};
  var ret  = {fig_returns.to_json()};
  var ex   = {fig_exits.to_json()};
  var sec  = {fig_sector.to_json()};
  var cfg  = {{responsive: true, displayModeBar: false}};
  Plotly.newPlot('equity',    eq.data,   eq.layout,   cfg);
  Plotly.newPlot('drawdown',  dd.data,   dd.layout,   cfg);
  Plotly.newPlot('monthly',   mo.data,   mo.layout,   cfg);
  Plotly.newPlot('returns',   ret.data,  ret.layout,  cfg);
  Plotly.newPlot('exits',     ex.data,   ex.layout,   cfg);
  Plotly.newPlot('sector',    sec.data,  sec.layout,  cfg);
</script>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    logger.info(f"Report saved → {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    equity_path = OUTPUTS_DIR.parent / "equity_curve.parquet"
    trade_path  = OUTPUTS_DIR.parent / "trade_log.parquet"
    oos_path    = OUTPUTS_DIR.parent / "wf_predictions.parquet"

    if not equity_path.exists():
        print("Run backtest/engine.py first to generate equity_curve.parquet")
    else:
        equity_df = pd.read_parquet(equity_path)
        trade_log = pd.read_parquet(trade_path) if trade_path.exists() else pd.DataFrame()
        oos_df    = pd.read_parquet(oos_path)   if oos_path.exists()   else None

        path = generate_report(equity_df, trade_log, oos_df)
        print(f"Report: {path}")
