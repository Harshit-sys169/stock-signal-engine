"""
backtest/engine.py
──────────────────
Walk-forward backtest engine.

Simulates the advisory system as if it were paper-traded:
  - Each day: check OOS signals from walk-forward predictions
  - Apply entry conditions (confidence >= MIN_CONFIDENCE)
  - Track open positions (stop-loss, take-profit, hold period)
  - Apply all risk constraints (max positions, sector caps, correlation filter)
  - Log every trade for metrics calculation

Transaction costs and slippage are applied on both entry and exit.
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from config.settings import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, HOLD_PERIOD_DAYS,
    POSITION_SIZE_PCT, MAX_OPEN_POSITIONS, SHORT_SELLING,
    TRANSACTION_COST, SLIPPAGE,
    MIN_CONFIDENCE, MAX_DRAWDOWN_PCT,
    MIN_CASH_BUFFER_PCT, MAX_SECTOR_EXPOSURE,
    CORR_FILTER_THRESHOLD,
    CLASS_LABELS,
)
from data.universe import get_sector

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 1_000_000  # 10 lakh INR


@dataclass
class Position:
    ticker:       str
    entry_date:   pd.Timestamp
    entry_price:  float
    shares:       float
    capital:      float
    sector:       str
    hold_days:    int = 0
    exit_date:    pd.Timestamp | None = None
    exit_price:   float | None = None
    pnl:          float = 0.0
    exit_reason:  str = ""


@dataclass
class Portfolio:
    cash:          float = INITIAL_CAPITAL
    total_capital: float = INITIAL_CAPITAL
    open_positions: list = field(default_factory=list)
    closed_trades:  list = field(default_factory=list)
    equity_curve:   list = field(default_factory=list)
    peak_value:     float = INITIAL_CAPITAL
    halted:         bool = False

    def portfolio_value(self, price_map: dict[str, float]) -> float:
        pos_value = sum(
            p.shares * price_map.get(p.ticker, p.entry_price)
            for p in self.open_positions
        )
        return self.cash + pos_value

    def sector_exposure(self, sector: str, price_map: dict[str, float]) -> float:
        total = self.portfolio_value(price_map)
        if total <= 0:
            return 0.0
        sector_value = sum(
            p.shares * price_map.get(p.ticker, p.entry_price)
            for p in self.open_positions
            if p.sector == sector
        )
        return sector_value / total


def _apply_cost(price: float, is_buy: bool) -> float:
    """Apply transaction cost + slippage to a trade price."""
    total_cost = TRANSACTION_COST + SLIPPAGE
    if is_buy:
        return price * (1 + total_cost)
    return price * (1 - total_cost)


def run_backtest(
    oos_predictions: pd.DataFrame,
    price_data:      dict[str, pd.DataFrame],
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full walk-forward backtest.

    Parameters
    ----------
    oos_predictions : pd.DataFrame
        Output from models/train.py walk_forward_train().
        Must have columns: ticker, pred_signal, confidence, index=Date.
    price_data : dict[str, pd.DataFrame]
        Per-ticker OHLCV from downloader.
    initial_capital : float

    Returns
    -------
    (equity_curve_df, trade_log_df)
    """
    portfolio = Portfolio(cash=initial_capital, total_capital=initial_capital,
                          peak_value=initial_capital)

    all_dates = sorted(oos_predictions.index.unique())

    for current_date in all_dates:
        if portfolio.halted:
            break

        # Build price map for today
        price_map = {}
        for ticker, df in price_data.items():
            day_data = df[df.index == current_date]
            if not day_data.empty and "Adj Close" in day_data.columns:
                price_map[ticker] = float(day_data["Adj Close"].iloc[0])

        # ── 1. Update open positions ───────────────────────────────
        still_open = []
        for pos in portfolio.open_positions:
            current_price = price_map.get(pos.ticker, pos.entry_price)
            pos.hold_days += 1
            ret = (current_price - pos.entry_price) / pos.entry_price

            # Check exit conditions
            exit_reason = None
            if ret <= STOP_LOSS_PCT:
                exit_reason = "stop_loss"
            elif ret >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            elif pos.hold_days >= HOLD_PERIOD_DAYS:
                exit_reason = "hold_expired"

            if exit_reason:
                exit_price = _apply_cost(current_price, is_buy=False)
                pos.exit_date   = current_date
                pos.exit_price  = exit_price
                pos.exit_reason = exit_reason
                pos.pnl = (exit_price - pos.entry_price) * pos.shares
                portfolio.cash += pos.shares * exit_price
                portfolio.closed_trades.append(pos)
            else:
                still_open.append(pos)

        portfolio.open_positions = still_open

        # ── 2. Check drawdown halt ─────────────────────────────────
        port_value = portfolio.portfolio_value(price_map)
        if port_value > portfolio.peak_value:
            portfolio.peak_value = port_value
        drawdown = (port_value - portfolio.peak_value) / portfolio.peak_value
        if drawdown <= MAX_DRAWDOWN_PCT:
            logger.warning(f"HALT: drawdown {drawdown:.1%} at {current_date}")
            portfolio.halted = True
            portfolio.equity_curve.append({
                "date": current_date, "value": port_value, "drawdown": drawdown
            })
            break

        # ── 3. Process new Buy signals ─────────────────────────────
        if not portfolio.halted:
            day_signals = oos_predictions[oos_predictions.index == current_date]
            buy_signals = day_signals[
                (day_signals["pred_signal"] == 2) &
                (day_signals["confidence"] >= MIN_CONFIDENCE)
            ].sort_values("confidence", ascending=False)

            for _, sig in buy_signals.iterrows():
                ticker = sig["ticker"]
                if len(portfolio.open_positions) >= MAX_OPEN_POSITIONS:
                    break
                if ticker not in price_map:
                    continue
                # Skip if already holding this ticker
                if any(p.ticker == ticker for p in portfolio.open_positions):
                    continue

                sector = get_sector(ticker)
                # Sector cap check
                if portfolio.sector_exposure(sector, price_map) >= MAX_SECTOR_EXPOSURE:
                    continue
                # Cash buffer check
                min_cash = portfolio.total_capital * MIN_CASH_BUFFER_PCT
                if portfolio.cash <= min_cash:
                    break
                # Correlation filter (skip if similar position exists)
                # Simplified: check if same sector already has 3+ positions
                sector_count = sum(
                    1 for p in portfolio.open_positions if p.sector == sector
                )
                if sector_count >= 3:
                    continue

                # Size position
                position_capital = min(
                    portfolio.total_capital * POSITION_SIZE_PCT,
                    portfolio.cash - min_cash,
                )
                if position_capital <= 0:
                    continue

                entry_price = _apply_cost(price_map[ticker], is_buy=True)
                shares = position_capital / entry_price

                pos = Position(
                    ticker=ticker,
                    entry_date=current_date,
                    entry_price=entry_price,
                    shares=shares,
                    capital=position_capital,
                    sector=sector,
                )
                portfolio.cash -= shares * entry_price
                portfolio.open_positions.append(pos)

        # ── 4. Record equity curve ─────────────────────────────────
        port_value = portfolio.portfolio_value(price_map)
        portfolio.total_capital = port_value  # track for position sizing
        portfolio.equity_curve.append({
            "date":     current_date,
            "value":    port_value,
            "cash":     portfolio.cash,
            "n_open":   len(portfolio.open_positions),
            "drawdown": (port_value - portfolio.peak_value) / portfolio.peak_value,
        })

    # Build output DataFrames
    equity_df = pd.DataFrame(portfolio.equity_curve).set_index("date")
    equity_df.index = pd.to_datetime(equity_df.index)

    if portfolio.closed_trades:
        trade_log = pd.DataFrame([
            {
                "ticker":       t.ticker,
                "sector":       t.sector,
                "entry_date":   t.entry_date,
                "exit_date":    t.exit_date,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "shares":       t.shares,
                "capital":      t.capital,
                "pnl":          t.pnl,
                "return_pct":   t.pnl / t.capital if t.capital else 0,
                "hold_days":    t.hold_days,
                "exit_reason":  t.exit_reason,
            }
            for t in portfolio.closed_trades
        ])
    else:
        trade_log = pd.DataFrame()

    logger.info(f"Backtest complete: {len(portfolio.closed_trades)} trades, "
                f"final value = ₹{equity_df['value'].iloc[-1]:,.0f}")

    return equity_df, trade_log


if __name__ == "__main__":
    from features.pipeline import load_feature_panel
    from models.train import walk_forward_train
    from data.downloader import download_all

    panel = load_feature_panel()
    oos_df, _ = walk_forward_train(panel)
    datasets = download_all()

    equity, trades = run_backtest(oos_df, datasets["ohlcv"])
    print(equity.tail(5))
    print(f"\nTotal trades: {len(trades)}")
