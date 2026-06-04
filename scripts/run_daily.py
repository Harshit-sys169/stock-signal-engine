"""
scripts/run_daily.py
────────────────────
Phase 2 — Daily batch runner.

Run this script after NSE market close (3:30 PM IST) every trading day.
It fetches the latest data, builds features, generates signals,
and saves them to outputs/signals/signals_YYYY-MM-DD.csv.

Usage:
    python scripts/run_daily.py

Schedule with cron (runs at 4 PM IST = 10:30 UTC):
    30 10 * * 1-5 /path/to/venv/bin/python /path/to/scripts/run_daily.py

Or with APScheduler (see bottom of this file).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUTS_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "daily.log",
                            mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def is_market_day() -> bool:
    """Return True if today is a weekday (rough NSE holiday check)."""
    return datetime.now().weekday() < 5  # Mon–Fri


def run() -> None:
    if not is_market_day():
        logger.info("Not a trading day — skipping")
        return

    logger.info("=" * 55)
    logger.info(f"NSE-Alpha v1 — Daily Run  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    logger.info("=" * 55)

    # ── Step 1: Download latest data ──────────────────────────────
    logger.info("Step 1/4: Downloading latest market data...")
    from data.downloader import download_all
    datasets = download_all(force=True)   # always refresh on daily run

    # ── Step 2: Build features for latest date ────────────────────
    logger.info("Step 2/4: Building features...")
    from features.pipeline import build_feature_panel
    panel = build_feature_panel(datasets, save=True)

    # ── Step 3: Generate signals ──────────────────────────────────
    logger.info("Step 3/4: Generating signals...")
    from models.predict import generate_daily_signals
    signals = generate_daily_signals(panel, save=True)

    # ── Step 4: Summary ───────────────────────────────────────────
    logger.info("Step 4/4: Summary")
    actionable = signals[signals["actionable"]]
    buy  = actionable[actionable["signal"] == 2]
    sell = actionable[actionable["signal"] == 0]

    logger.info(f"Signal date : {signals['signal_date'].iloc[0]}")
    logger.info(f"Buy signals : {len(buy)}")
    logger.info(f"Sell signals: {len(sell)}")
    logger.info("Top Buy signals:")
    if not buy.empty:
        top = buy[["ticker", "confidence", "expected_5d_return", "entry_price"]].head(5)
        for _, row in top.iterrows():
            logger.info(
                f"  {row['ticker']:<15} conf={row['confidence']:.2f}  "
                f"exp_ret={row['expected_5d_return']:+.2%}  "
                f"price=₹{row['entry_price']:,.2f}"
            )
    logger.info("Daily run complete.")
    logger.info("=" * 55)


if __name__ == "__main__":
    # Create logs dir if needed
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    run()

    # ── Optional: APScheduler (uncomment to run as daemon) ────────
    # from apscheduler.schedulers.blocking import BlockingScheduler
    # import pytz
    # scheduler = BlockingScheduler(timezone=pytz.timezone("Asia/Kolkata"))
    # scheduler.add_job(run, "cron", day_of_week="mon-fri", hour=16, minute=0)
    # logger.info("Scheduler started — running daily at 4:00 PM IST")
    # scheduler.start()
