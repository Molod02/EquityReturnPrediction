"""How good are the predictions? The metrics quants actually use.

- Out-of-sample R²: share of return variance explained, vs predicting zero
- Information Coefficient (IC): each month, rank correlation between predicted and
  realized returns across stocks. Mean IC of 0.02-0.05 is typical for real signals.
- Long-short portfolio: buy the top quintile of predictions, short the bottom quintile.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERIODS_PER_YEAR = 12


def oos_r2(y, pred):
    """1 - SSE / sum(y²). Benchmark is a forecast of 0 (the cross-sectional average)."""
    y, pred = np.asarray(y), np.asarray(pred)
    return 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2)


def ic_series(preds, target):
    """Monthly Spearman rank IC for each prediction column. target: Series on same index."""
    df = preds.join(target.rename("_y")).dropna(subset=["_y"])
    out = {}
    for col in preds.columns:
        out[col] = df.groupby(level="date").apply(lambda g: spearmanr(g[col], g["_y"])[0])
    return pd.DataFrame(out)


def ic_summary(ic):
    """Mean IC, its t-statistic, share of months with IC > 0, and annualized IC information ratio."""
    n = ic.count()
    return pd.DataFrame({
        "mean_IC": ic.mean(),
        "t_stat": ic.mean() / (ic.std() / np.sqrt(n)),
        "hit_rate": (ic > 0).mean(),
        "IC_IR_ann": ic.mean() / ic.std() * np.sqrt(PERIODS_PER_YEAR),
        "months": n,
    })


def quintile_returns(pred, fwd_ret, q=5):
    """Average next-month return of each prediction quintile (1 = lowest predicted)."""
    df = pd.concat({"pred": pred, "ret": fwd_ret}, axis=1).dropna()
    df["q"] = df.groupby(level="date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False) + 1)
    return df.groupby(["date", "q"])["ret"].mean().unstack()


def long_short(pred, fwd_ret, q=5, cost_bps=10):
    """Monthly returns of an equal-weight top-minus-bottom quintile portfolio.

    Positions are formed at month-end t and earn the return to month-end t+1.
    Turnover: average fraction of each leg's names replaced that month (0 = none, 1 = all).
    Cost: replacing a name means a sell and a buy on each of 2 legs -> 4 x turnover x cost_bps.
    """
    qr = quintile_returns(pred, fwd_ret, q)
    gross = qr[q] - qr[1]

    df = pd.concat({"pred": pred}, axis=1).dropna()
    df["q"] = df.groupby(level="date")["pred"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False) + 1)
    dates = sorted(df.index.get_level_values("date").unique())
    turnover = {}
    prev_long = prev_short = None
    for d in dates:
        g = df.xs(d, level="date")
        longs, shorts = set(g.index[g.q == q]), set(g.index[g.q == 1])
        if prev_long is not None:
            turnover[d] = (len(longs - prev_long) / len(longs) + len(shorts - prev_short) / len(shorts)) / 2
        prev_long, prev_short = longs, shorts
    turnover = pd.Series(turnover).reindex(gross.index).fillna(1.0)  # first month: full build
    net = gross - turnover * 4 * cost_bps / 10_000                     # 2 legs x (sell + buy)
    return net.rename("long_short"), turnover


def performance(monthly_rets):
    """Annualized return, volatility, Sharpe (rf = 0) and max drawdown of monthly returns."""
    r = monthly_rets.dropna()
    wealth = (1 + r).cumprod()
    return pd.Series({
        "ann_return": r.mean() * PERIODS_PER_YEAR,
        "ann_vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR),
        "max_drawdown": (wealth / wealth.cummax() - 1).min(),
        "months": len(r),
    })


# ---------------------------------------------------------------- the original bug, reproduced
def leak_demo(close, market="SPY", seed=42):
    """Reproduce the original project's evaluation and show why it was wrong.

    Original setup: daily rows, target = next 21-day return (overlapping), features
    momentum/volatility/liquidity, rows shuffled with train_test_split.
    Compared with: the same data, but trained on the past and tested on the future.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    px = close.drop(columns=[market])
    ret = px.pct_change()
    feats = pd.concat({
        "momentum": px.pct_change(21),
        "volatility": ret.rolling(21).std(),
        "fwd": px.pct_change(21).shift(-21),
    }, axis=1).stack(level=1, future_stack=True).dropna()
    X, y = feats[["momentum", "volatility"]].values, feats["fwd"].values
    dates = feats.index.get_level_values(0)

    def gbm():
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    r2_random = r2_score(yte, gbm().fit(Xtr, ytr).predict(Xte))

    cut = np.sort(dates.unique())[int(0.8 * dates.nunique())]
    train = dates < cut - pd.Timedelta(days=35)       # gap so no target overlaps the test period
    test = dates >= cut
    r2_time = r2_score(y[test], gbm().fit(X[train], y[train]).predict(X[test]))
    return pd.Series({"Random shuffle (original method)": r2_random,
                      "Train on past, test on future": r2_time}, name="R2")
