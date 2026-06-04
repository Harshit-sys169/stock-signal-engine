"""
dashboard/app.py
────────────────
Phase 3 — Streamlit dashboard for NSE-Alpha v1.

Shows:
  - Today's Buy/Sell/Hold signals with confidence scores
  - Portfolio state (open positions, P&L)
  - Equity curve from backtest
  - Feature importance chart

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import OUTPUTS_DIR, CLASS_LABELS, MIN_CONFIDENCE

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE-Alpha v1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data Loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_signals() -> pd.DataFrame | None:
    files = sorted(glob.glob(str(OUTPUTS_DIR / "signals_*.csv")))
    if not files:
        return None
    return pd.read_csv(files[-1])


@st.cache_data(ttl=300)
def load_all_signals() -> pd.DataFrame:
    files = sorted(glob.glob(str(OUTPUTS_DIR / "signals_*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(ttl=3600)
def load_equity_curve() -> pd.DataFrame | None:
    path = OUTPUTS_DIR.parent / "equity_curve.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_feature_importance() -> pd.DataFrame | None:
    path = OUTPUTS_DIR.parent / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=3600)
def load_trade_log() -> pd.DataFrame | None:
    path = OUTPUTS_DIR.parent / "trade_log.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("NSE-Alpha v1")
    st.caption("Advisory Signal System — NSE India")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Today's Signals", "Portfolio", "Backtest", "Model"],
    )
    st.divider()
    min_conf = st.slider(
        "Min Confidence", 0.50, 0.95, MIN_CONFIDENCE, 0.05,
        help="Filter signals by minimum model confidence"
    )
    show_hold = st.checkbox("Show Hold signals", value=False)


# ── Page: Today's Signals ──────────────────────────────────────────────────

if page == "Today's Signals":
    st.header("📊 Today's Signals")

    signals = load_latest_signals()
    if signals is None:
        st.warning(
            "No signal files found. "
            "Run `python scripts/run_daily.py` first to generate signals."
        )
        st.stop()

    signal_date = signals["signal_date"].iloc[0] if "signal_date" in signals.columns else "N/A"
    st.caption(f"Signal date: **{signal_date}**")

    # Apply filters
    filtered = signals[signals["confidence"] >= min_conf].copy()
    if not show_hold:
        filtered = filtered[filtered["signal"] != 1]

    # KPI row
    buy_count  = (filtered["signal"] == 2).sum()
    sell_count = (filtered["signal"] == 0).sum()
    hold_count = (signals["signal"] == 1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buy Signals",  buy_count,  delta=None)
    c2.metric("Sell Signals", sell_count, delta=None)
    c3.metric("Hold",         hold_count, delta=None)
    c4.metric("Total Universe", len(signals))

    st.divider()

    # Signal table
    display_cols = ["ticker", "signal_label", "confidence",
                    "expected_5d_return", "entry_price",
                    "prob_buy", "prob_hold", "prob_sell"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def highlight_signal(row):
        if row.get("signal_label") == "Buy":
            return ["background-color: #1a3a1a"] * len(row)
        elif row.get("signal_label") == "Sell":
            return ["background-color: #3a1a1a"] * len(row)
        return [""] * len(row)

    styled = (
        filtered[display_cols]
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
        .style
        .apply(highlight_signal, axis=1)
        .format({
            "confidence":        "{:.1%}",
            "expected_5d_return":"{:+.2%}",
            "entry_price":       "₹{:,.2f}",
            "prob_buy":          "{:.1%}",
            "prob_hold":         "{:.1%}",
            "prob_sell":         "{:.1%}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # Confidence distribution
    st.subheader("Confidence Distribution")
    fig = px.histogram(
        signals, x="confidence", color="signal_label",
        nbins=30, barmode="overlay", opacity=0.7,
        color_discrete_map={"Buy": "#26a641", "Sell": "#da3633", "Hold": "#6e7681"},
        labels={"confidence": "Confidence Score", "signal_label": "Signal"},
    )
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Portfolio ────────────────────────────────────────────────────────

elif page == "Portfolio":
    st.header("💼 Portfolio")

    all_sigs = load_all_signals()
    if all_sigs.empty:
        st.info("No signal history available yet.")
        st.stop()

    trades = load_trade_log()
    if trades is not None and not trades.empty:
        st.subheader("Trade History")
        st.dataframe(
            trades.sort_values("entry_date", ascending=False)
            .head(50)
            .style.format({
                "return_pct":  "{:+.2%}",
                "pnl":         "₹{:+,.0f}",
                "entry_price": "₹{:,.2f}",
                "exit_price":  "₹{:,.2f}",
            }, na_rep="—"),
            use_container_width=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades",  len(trades))
        col2.metric("Win Rate",      f"{(trades['pnl'] > 0).mean():.1%}")
        col3.metric("Total P&L",     f"₹{trades['pnl'].sum():+,.0f}")
        wins  = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        losses= trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
        pf = wins / losses if losses > 0 else float("inf")
        col4.metric("Profit Factor", f"{pf:.2f}")
    else:
        st.info("Run the backtest engine to populate trade history.")


# ── Page: Backtest ─────────────────────────────────────────────────────────

elif page == "Backtest":
    st.header("📉 Backtest Results")

    equity = load_equity_curve()
    if equity is None:
        st.info(
            "No equity curve found. "
            "Run `backtest/engine.py` to generate backtest results."
        )
        st.stop()

    # Equity curve chart
    st.subheader("Portfolio Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity["value"],
        name="NSE-Alpha v1", line=dict(color="#26a641", width=2),
    ))
    fig.update_layout(
        template="plotly_dark", height=400,
        yaxis_title="Portfolio Value (₹)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    if "drawdown" in equity.columns:
        st.subheader("Drawdown")
        fig2 = px.area(
            equity, x=equity.index, y="drawdown",
            color_discrete_sequence=["#da3633"],
            labels={"drawdown": "Drawdown"},
        )
        fig2.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig2, use_container_width=True)


# ── Page: Model ────────────────────────────────────────────────────────────

elif page == "Model":
    st.header("🧠 Model Insights")

    fi = load_feature_importance()
    if fi is None:
        st.info("Feature importance not found. Train the model first.")
        st.stop()

    st.subheader("Top 30 Feature Importances (Gain)")
    top_fi = fi.head(30)
    fig = px.bar(
        top_fi.sort_values("importance"),
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Greens",
    )
    fig.update_layout(template="plotly_dark", height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
