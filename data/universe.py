"""
data/universe.py
────────────────
Nifty 50 + Nifty Next 50 ticker universe with sector mapping.
All tickers are in NSE format (appended with .NS for yfinance).

Constituents as of late 2024. NSE rebalances indices quarterly;
update this file after each rebalance if running live.
"""

from config.settings import EXCHANGE_SUFFIX

# ── Nifty 50 ──────────────────────────────────────────────────────────────
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

# ── Nifty Next 50 ─────────────────────────────────────────────────────────
NIFTY_NEXT_50 = [
    "ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "DMART",
    "BANKBARODA", "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "DLF", "DABUR", "GAIL",
    "GODREJCP", "GODREJPROP", "HAVELLS", "HAL", "ICICIGI",
    "ICICIPRULI", "IOC", "IRCTC", "INDIGO", "JINDALSTEL",
    "LICI", "LODHA", "MRF", "MUTHOOTFIN", "NHPC",
    "NMDC", "NAUKRI", "OFSS", "PIDILITIND", "PNB",
    "PIIND", "RECLTD", "SAIL", "SRF", "SIEMENS",
    "TATAPOWER", "TORNTPHARM", "TRENT", "UNITDSPR", "VBL",
    "VEDL", "VOLTAS", "ZOMATO", "ZYDUSLIFE", "MANKIND",
]

# ── Sector Mapping ─────────────────────────────────────────────────────────
# Maps NSE symbol → sector name used for sector exposure limits
SECTOR_MAP = {
    # Financials
    "HDFCBANK": "financials", "ICICIBANK": "financials", "KOTAKBANK": "financials",
    "AXISBANK": "financials", "SBIN": "financials", "INDUSINDBK": "financials",
    "BAJFINANCE": "financials", "BAJAJFINSV": "financials", "HDFCLIFE": "financials",
    "SBILIFE": "financials", "SHRIRAMFIN": "financials", "BANKBARODA": "financials",
    "CANBK": "financials", "PNB": "financials", "CHOLAFIN": "financials",
    "ICICIGI": "financials", "ICICIPRULI": "financials", "MUTHOOTFIN": "financials",
    "RECLTD": "financials", "LICI": "financials",

    # IT
    "TCS": "it", "INFY": "it", "HCLTECH": "it", "WIPRO": "it",
    "TECHM": "it", "LTIM": "it", "OFSS": "it",

    # Energy & Oil
    "RELIANCE": "energy", "ONGC": "energy", "BPCL": "energy",
    "IOC": "energy", "GAIL": "energy", "TATAPOWER": "energy",
    "ADANIGREEN": "energy", "ADANIPOWER": "energy", "NTPC": "energy",
    "POWERGRID": "energy", "NHPC": "energy", "NMDC": "energy",

    # Consumer
    "HINDUNILVR": "consumer", "ITC": "consumer", "NESTLEIND": "consumer",
    "BRITANNIA": "consumer", "DABUR": "consumer", "GODREJCP": "consumer",
    "COLPAL": "consumer", "TATACONSUM": "consumer", "DMART": "consumer",
    "VBL": "consumer", "UNITDSPR": "consumer", "ZOMATO": "consumer",
    "MANKIND": "consumer",

    # Pharma & Healthcare
    "SUNPHARMA": "pharma", "DRREDDY": "pharma", "CIPLA": "pharma",
    "DIVISLAB": "pharma", "APOLLOHOSP": "pharma", "TORNTPHARM": "pharma",
    "ZYDUSLIFE": "pharma", "PIIND": "pharma",

    # Auto
    "MARUTI": "auto", "TATAMOTORS": "auto", "M&M": "auto",
    "BAJAJ-AUTO": "auto", "HEROMOTOCO": "auto", "EICHERMOT": "auto",
    "BOSCHLTD": "auto",

    # Metals & Materials
    "TATASTEEL": "metals", "JSWSTEEL": "metals", "HINDALCO": "metals",
    "COALINDIA": "metals", "VEDL": "metals", "SAIL": "metals",
    "JINDALSTEL": "metals",

    # Cement & Construction
    "ULTRACEMCO": "cement", "GRASIM": "cement", "AMBUJACEM": "cement",
    "DLF": "realestate", "GODREJPROP": "realestate", "LODHA": "realestate",

    # Capital Goods & Industrials
    "LT": "industrials", "ABB": "industrials", "BEL": "industrials",
    "HAL": "industrials", "SIEMENS": "industrials", "HAVELLS": "industrials",

    # Paint & Chemicals
    "ASIANPAINT": "chemicals", "BERGEPAINT": "chemicals",
    "PIDILITIND": "chemicals", "SRF": "chemicals",

    # Diversified / Conglomerates
    "ADANIENT": "conglomerate", "ADANIPORTS": "conglomerate",
    "TITAN": "consumer", "MRF": "auto",

    # Telecom
    "BHARTIARTL": "telecom",

    # Others
    "IRCTC": "services", "INDIGO": "services", "NAUKRI": "services",
    "TRENT": "consumer", "TATAPOWER": "energy",
}


def get_universe(include_nifty50: bool = True,
                 include_next50: bool = True) -> list[str]:
    """
    Return the full ticker list in yfinance format (SYMBOL.NS).

    Parameters
    ----------
    include_nifty50 : bool
        Include Nifty 50 constituents.
    include_next50 : bool
        Include Nifty Next 50 constituents.

    Returns
    -------
    list[str]
        Sorted list of tickers e.g. ['ADANIENT.NS', 'ADANIPORTS.NS', ...]
    """
    tickers = []
    if include_nifty50:
        tickers += NIFTY_50
    if include_next50:
        tickers += NIFTY_NEXT_50

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return sorted([f"{t}{EXCHANGE_SUFFIX}" for t in unique])


def get_sector(symbol: str) -> str:
    """
    Return sector for a given NSE symbol (with or without .NS suffix).

    Parameters
    ----------
    symbol : str
        e.g. 'RELIANCE' or 'RELIANCE.NS'

    Returns
    -------
    str
        Sector name, or 'unknown' if not mapped.
    """
    clean = symbol.replace(EXCHANGE_SUFFIX, "")
    return SECTOR_MAP.get(clean, "unknown")


if __name__ == "__main__":
    universe = get_universe()
    print(f"Universe size: {len(universe)}")
    print(universe[:5], "...")
