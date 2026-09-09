import yfinance as yf
import pandas as pd
import numpy as np

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

data = yf.download(tickers, start="2019-01-01", end="2024-01-01")
close_prices = data["Close"]
volume = data["Volume"]

# Daily returns
daily_returns = close_prices.pct_change()

# --- Feature engineering ---

# Momentum: cumulative return over the past 21 trading days (~1 month)
momentum = close_prices.pct_change(21)

# Volatility: rolling 21-day standard deviation of daily returns
volatility = daily_returns.rolling(21).std()

# Liquidity: rolling 21-day average trading volume
liquidity = volume.rolling(21).mean()

# --- Target variable ---
# What we're predicting: next month's forward return (21 trading days ahead)
forward_return = close_prices.pct_change(21).shift(-21)

# Combine everything into one long-format DataFrame (stock-date rows)
features = pd.concat({
    "momentum": momentum,
    "volatility": volatility,
    "liquidity": liquidity,
    "forward_return": forward_return
}, axis=1)

features = features.stack(level=1).reset_index()
features.columns = ["date", "ticker", "momentum", "volatility", "liquidity", "forward_return"]

# Drop rows with missing values (start/end of series where rolling windows aren't full)
features = features.dropna()

print(features.head(10))
print(features.shape)

# Save to CSV so we don't need to re-download every time
features.to_csv("features.csv", index=False)
print("Saved to features.csv")