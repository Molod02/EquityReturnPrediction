"""Download and cache daily prices and volumes."""
import pandas as pd

from src.config import DATA_DIR, TICKERS, MARKET, START, END

CLOSE_PATH = DATA_DIR / "raw" / "close.csv"
VOLUME_PATH = DATA_DIR / "raw" / "volume.csv"


def load_prices(tickers=TICKERS, market=MARKET, start=START, end=END, force=False):
    """Adjusted close and volume for the universe plus the market ETF.
    Downloads once, then loads from data/raw/ so every run uses identical data."""
    if CLOSE_PATH.exists() and VOLUME_PATH.exists() and not force:
        close = pd.read_csv(CLOSE_PATH, index_col=0, parse_dates=True)
        volume = pd.read_csv(VOLUME_PATH, index_col=0, parse_dates=True)
        return close, volume

    import yfinance as yf

    symbols = list(tickers) + [market]
    df = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"][symbols]
    volume = df["Volume"][symbols]

    CLOSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(CLOSE_PATH)
    volume.to_csv(VOLUME_PATH)
    return close, volume
