"""Feature engineering: a monthly stock-by-date panel.

Why monthly: the original project predicted overlapping 21-day returns every day,
so neighbouring rows shared almost the same target. Sampling at month-ends and
predicting the NEXT month's return gives one non-overlapping target per stock-month.

Every feature at month-end t uses only data up to t.
"""
import numpy as np
import pandas as pd

from src.config import MARKET

FEATURES = ["mom_12_1", "rev_1m", "vol_1m", "vol_3m", "log_dollar_vol", "amihud", "beta_1y", "idio_vol"]
RANKED = [f + "_r" for f in FEATURES]

FEATURE_DESCRIPTIONS = {
    "mom_12_1": "Momentum: return from 12 months ago to 1 month ago (skips the last month)",
    "rev_1m": "Short-term reversal: return over the last month",
    "vol_1m": "Volatility: annualized std of daily returns, last 21 days",
    "vol_3m": "Volatility: annualized std of daily returns, last 63 days",
    "log_dollar_vol": "Liquidity: log of average daily dollar volume, last 21 days",
    "amihud": "Illiquidity (Amihud): log of average |return| / dollar volume, last 21 days",
    "beta_1y": "Market beta vs SPY, last 252 days (risk exposure)",
    "idio_vol": "Idiosyncratic volatility: residual vol after removing market exposure, last 63 days",
}


def month_end_dates(index):
    """Last trading day of each calendar month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().values)


def _rolling_beta(ret, mkt, window):
    return ret.rolling(window).cov(mkt).div(mkt.rolling(window).var(), axis=0)


def daily_features(close, volume, market=MARKET):
    """All features at daily frequency (one DataFrame per feature, stocks as columns)."""
    stocks = [c for c in close.columns if c != market]
    px, vol = close[stocks], volume[stocks]
    ret = px.pct_change()
    mkt = close[market].pct_change()
    dollar_vol = px * vol

    beta_63 = _rolling_beta(ret, mkt, 63)
    resid_var = ret.rolling(63).var() - beta_63 ** 2 * mkt.rolling(63).var().values[:, None]

    return {
        "mom_12_1": px.shift(21) / px.shift(252) - 1,
        "rev_1m": px / px.shift(21) - 1,
        "vol_1m": ret.rolling(21).std() * np.sqrt(252),
        "vol_3m": ret.rolling(63).std() * np.sqrt(252),
        "log_dollar_vol": np.log(dollar_vol.rolling(21).mean()),
        "amihud": np.log((ret.abs() / dollar_vol).rolling(21).mean() * 1e9),
        "beta_1y": _rolling_beta(ret, mkt, 252),
        "idio_vol": np.sqrt(resid_var.clip(lower=0)) * np.sqrt(252),
    }


def build_panel(close, volume, market=MARKET):
    """Long panel: one row per (month-end date, ticker) with features and next-month target.

    Targets:
      fwd_ret     = next month's return (close at next month-end / close today - 1)
      fwd_ret_xs  = fwd_ret minus that month's cross-sectional average
                    (we predict which stocks beat the others, not the market direction)
    Ranked features (*_r): each month, features are converted to cross-sectional
    ranks in [-0.5, 0.5]. This removes outliers and makes months comparable.
    """
    dates = month_end_dates(close.index)
    feats = daily_features(close, volume, market)
    stocks = [c for c in close.columns if c != market]

    monthly_px = close.loc[dates, stocks]
    fwd_ret = monthly_px.shift(-1) / monthly_px - 1

    panel = pd.concat({name: df.loc[dates] for name, df in feats.items()}, axis=1)
    panel = panel.stack(level=1, future_stack=True)
    panel.index.names = ["date", "ticker"]
    panel["fwd_ret"] = fwd_ret.stack(future_stack=True)

    panel = panel.dropna(subset=FEATURES)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    by_date = panel.groupby(level="date")
    for f in FEATURES:
        panel[f + "_r"] = by_date[f].rank(pct=True) - 0.5
    panel["fwd_ret_xs"] = panel["fwd_ret"] - by_date["fwd_ret"].transform("mean")
    return panel.sort_index()
