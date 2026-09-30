"""Models and the walk-forward loop.

The key rule: a model is only ever trained on months whose outcome was already
known at the time of prediction. No random shuffling of rows, ever.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.config import RANDOM_SEED
from src.features import RANKED

TARGET = "fwd_ret_xs"
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]


# ---------------------------------------------------------------- helpers
def time_split(dates, val_fraction=0.2):
    """Boolean mask: True for the most recent val_fraction of MONTHS (validation set)."""
    months = np.sort(np.unique(dates))
    cutoff = months[int(len(months) * (1 - val_fraction))]
    return dates >= cutoff


# ---------------------------------------------------------------- models
class RidgeModel:
    """Regularized linear regression. Alpha chosen on a time-ordered validation set."""

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = alphas

    def fit(self, X, y, dates):
        val = time_split(dates)
        scores = {}
        for a in self.alphas:
            m = Ridge(alpha=a).fit(X[~val], y[~val])
            scores[a] = np.mean((m.predict(X[val]) - y[val]) ** 2)
        self.alpha_ = min(scores, key=scores.get)
        self.model_ = Ridge(alpha=self.alpha_).fit(X, y)   # refit on all training data
        return self

    def predict(self, X):
        return self.model_.predict(X)


class GBMModel:
    """Gradient boosting with shallow trees and a small learning rate (hard to overfit noise)."""

    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed

    def fit(self, X, y, dates):
        self.model_ = GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.02, subsample=0.8,
            min_samples_leaf=50, random_state=self.seed,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class MLPModel:
    """Small PyTorch feed-forward network with early stopping on a time-ordered validation set."""

    def __init__(self, hidden=(32, 16), lr=1e-3, weight_decay=1e-4, dropout=0.1,
                 max_epochs=200, patience=15, batch_size=256, seed=RANDOM_SEED):
        self.hidden, self.lr, self.weight_decay, self.dropout = hidden, lr, weight_decay, dropout
        self.max_epochs, self.patience, self.batch_size, self.seed = max_epochs, patience, batch_size, seed

    def _build(self, n_in):
        import torch.nn as nn
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, dates):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        val = time_split(dates)
        Xtr = torch.tensor(X[~val], dtype=torch.float32)
        ytr = torch.tensor(y[~val], dtype=torch.float32).view(-1, 1)
        Xva = torch.tensor(X[val], dtype=torch.float32)
        yva = torch.tensor(y[val], dtype=torch.float32).view(-1, 1)

        self.net_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            order = torch.randperm(len(Xtr))        # shuffling mini-batches INSIDE training is fine
            for i in range(0, len(Xtr), self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xva), yva).item()
            self.history_.append(val_loss)
            if val_loss < best_loss - 1e-8:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:     # early stopping
                    break

        self.net_.load_state_dict(best_state)
        self.epochs_ = epoch + 1
        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.tensor(X, dtype=torch.float32)).numpy().ravel()


class MomentumSignal:
    """Baseline with no fitting at all: predict that past winners keep winning."""

    def fit(self, X, y, dates):
        return self

    def predict(self, X):
        return X[:, RANKED.index("mom_12_1_r")]


MODELS = {"Momentum (no model)": MomentumSignal, "Ridge": RidgeModel, "GBM": GBMModel, "Neural net": MLPModel}


# ---------------------------------------------------------------- walk-forward
def walk_forward(panel, model_cls, first_test_year, features=RANKED, target=TARGET, verbose=False):
    """Expanding-window, annual refits.

    For test year Y, train on rows dated before Dec 1 of year Y-1: the target of a
    row dated at month-end t is realized at month-end t+1, so the last training
    row's outcome (November -> December) is known before January of year Y.
    Returns a DataFrame with the prediction for every test row, and the fitted models.
    """
    dates = panel.index.get_level_values("date")
    preds, fitted = [], {}
    for year in range(first_test_year, dates.max().year + 1):
        train = panel[dates < pd.Timestamp(f"{year - 1}-12-01")].dropna(subset=[target])
        test = panel[dates.year == year]
        if len(test) == 0 or len(train) == 0:
            continue
        model = model_cls().fit(train[features].values, train[target].values,
                                train.index.get_level_values("date").values)
        preds.append(pd.Series(model.predict(test[features].values), index=test.index))
        fitted[year] = model
        if verbose:
            print(f"{year}: trained on {train.index.get_level_values('date').nunique()} months")
    return pd.concat(preds).rename("pred"), fitted


def run_all(panel, first_test_year, models=MODELS, verbose=True):
    """Walk-forward for every model. Returns one column of predictions per model."""
    out, fitted = {}, {}
    for name, cls in models.items():
        if verbose:
            print(f"--- {name}")
        out[name], fitted[name] = walk_forward(panel, cls, first_test_year, verbose=verbose)
    return pd.DataFrame(out), fitted
