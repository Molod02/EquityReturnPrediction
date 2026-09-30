"""Project-wide settings: paths, universe, dates, seed."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
REPORTS_DIR = ROOT / "reports"

RANDOM_SEED = 42
START = "2010-01-01"
END = "2026-09-01"          # fixed end date = reproducible results
FIRST_TEST_YEAR = 2015      # walk-forward starts here (earlier years are training only)
MARKET = "SPY"              # used for beta and idiosyncratic volatility

# 52 large-cap US stocks listed since before 2010, across 7 sectors.
# Caveat: these are TODAY's large caps, which creates survivorship bias (see README).
UNIVERSE = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "NVDA": "Tech",
    "ORCL": "Tech", "CSCO": "Tech", "INTC": "Tech", "IBM": "Tech", "ADBE": "Tech",
    "QCOM": "Tech", "TXN": "Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "C": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials", "USB": "Financials",
    # Health care
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "UNH": "Health",
    "ABT": "Health", "AMGN": "Health", "MDT": "Health", "BMY": "Health",
    # Consumer
    "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer", "WMT": "Consumer",
    "HD": "Consumer", "MCD": "Consumer", "NKE": "Consumer", "COST": "Consumer", "DIS": "Consumer",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "HON": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "GE": "Industrials", "LMT": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Utilities & telecom
    "NEE": "Utilities/Telecom", "DUK": "Utilities/Telecom", "VZ": "Utilities/Telecom", "T": "Utilities/Telecom",
}
TICKERS = list(UNIVERSE)
