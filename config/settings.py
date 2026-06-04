"""
config/settings.py
──────────────────
Single source of truth for all NSE-Alpha v1 parameters.
Change values here. Nothing else should have hardcoded constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "raw"))
OUTPUTS_DIR     = ROOT_DIR / "outputs" / "signals"
MODELS_DIR      = ROOT_DIR / "models" / "saved"

for _dir in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Market & Universe ──────────────────────────────────────────────────────
MARKET          = "NSE"
EXCHANGE_SUFFIX = ".NS"          # yfinance suffix for NSE tickers
INDEX_TICKER    = "^NSEI"        # Nifty 50 index
MARKET_OPEN     = "09:15"        # IST
MARKET_CLOSE    = "15:30"        # IST — signals generated after this


# ── Historical Data ────────────────────────────────────────────────────────
HISTORY_START   = "2014-01-01"
HISTORY_END     = "2024-12-31"
PRICE_COLUMN    = "Adj Close"    # always use adjusted close


# ── Prediction Target ──────────────────────────────────────────────────────
FORWARD_DAYS    = 5              # predict return over next N trading days
BUY_THRESHOLD   = 0.015         # +1.5% → Buy
SELL_THRESHOLD  = -0.015        # -1.5% → Sell
# Between thresholds → Hold


# ── Feature Engineering ────────────────────────────────────────────────────
# Technical indicator windows
RSI_WINDOW          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
BB_WINDOW           = 20
BB_STD              = 2.0
ATR_WINDOW          = 14
SMA_WINDOWS         = [20, 50, 200]
EMA_WINDOWS         = [12, 26]
MOMENTUM_WINDOWS    = [5, 10, 20]   # return over N days
VOLUME_MA_WINDOW    = 20            # volume ratio baseline

# Macro feature windows
VIX_CHANGE_WINDOW   = 5
SECTOR_RETURN_WINDOW= 5
MACRO_CHANGE_WINDOW = 5


# ── Macro / FRED ───────────────────────────────────────────────────────────
ENABLE_MACRO_FRED   = True          # set False to skip FRED entirely
FRED_API_KEY        = os.getenv("FRED_API_KEY", "")
FRED_SERIES = {
    "usd_inr":   "DEXINUS",         # USD/INR exchange rate (daily)
    "repo_rate": "INDIRLTLT01STM",  # India lending rate (monthly → ffill)
}

# NSE sector indices (yfinance tickers)
SECTOR_INDICES = {
    "bank":   "^NSEBANK",
    "it":     "^CNXIT",
    "fmcg":   "^CNXFMCG",
    "auto":   "^CNXAUTO",
    "pharma": "^CNXPHARMA",
}

# India VIX — downloaded from NSE as CSV
INDIA_VIX_URL = (
    "https://www.nseindia.com/api/historical/vixhistory"
    "?from={from_date}&to={to_date}"
)
INDIA_VIX_FALLBACK_TICKER = "^INDIAVIX"   # yfinance fallback if NSE direct fails


# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
TEST_SIZE       = 0.20           # final held-out test set
CV_FOLDS        = 5              # walk-forward folds
WF_WINDOW_DAYS  = 126            # ~6 months per out-of-sample window

LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "num_leaves":       31,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples":20,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}

# Class labels: 0=Sell, 1=Hold, 2=Buy  (sorted ascending)
CLASS_LABELS    = {0: "Sell", 1: "Hold", 2: "Buy"}
SIGNAL_COLUMN   = "signal"
CONFIDENCE_COL  = "confidence"
RETURN_PRED_COL = "expected_5d_return"


# ── Trading Rules ──────────────────────────────────────────────────────────
MIN_CONFIDENCE      = 0.60       # only trade signals with confidence >= this
STOP_LOSS_PCT       = -0.025     # -2.5%
TAKE_PROFIT_PCT     = 0.040      # +4.0%
HOLD_PERIOD_DAYS    = 5
POSITION_SIZE_PCT   = 0.10       # max 10% of capital per trade
MAX_OPEN_POSITIONS  = 10
SHORT_SELLING       = False      # disabled in v1
TRANSACTION_COST    = 0.001      # 0.10% per trade (brokerage + STT)
SLIPPAGE            = 0.0005     # 0.05% per trade


# ── Risk Limits ────────────────────────────────────────────────────────────
MAX_DRAWDOWN_PCT     = -0.15     # -15% portfolio drawdown → halt
MIN_CASH_BUFFER_PCT  =  0.20     # 20% always in cash
MAX_SECTOR_EXPOSURE  =  0.30     # 30% max per sector of deployed capital
CORR_FILTER_THRESHOLD=  0.80     # skip trade if corr with existing position > 0.80


# ── Evaluation Targets ─────────────────────────────────────────────────────
TARGET_DIRECTIONAL_ACCURACY = 0.55
TARGET_F1_SCORE             = 0.50
TARGET_SHARPE               = 1.50
TARGET_WIN_RATE             = 0.52
TARGET_PROFIT_FACTOR        = 1.30
TARGET_CAGR_ALPHA           = 0.03   # 3% above Nifty 50 benchmark


# ── Data Quality ───────────────────────────────────────────────────────────
MAX_MISSING_PCT     = 0.05       # drop stocks with > 5% missing days
GAP_FILL_LIMIT      = 2          # forward-fill gaps up to this many days
