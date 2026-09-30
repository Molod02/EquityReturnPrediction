"""Latent risk factors: PCA on daily stock returns, then clustering stocks by their loadings.

PCA finds the few directions that explain most of how the 52 stocks move together.
The first component is usually "the market"; the next ones often look like sectors.
If clustering the loadings recovers the real sectors, the factors are meaningful.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.config import RANDOM_SEED, MARKET


def standardized_returns(close, market=MARKET):
    ret = close.drop(columns=[market]).pct_change().dropna(how="all").dropna(axis=1, how="any")
    return (ret - ret.mean()) / ret.std()


def pca_factors(close, n_components=10, market=MARKET):
    """Fit PCA on standardized daily returns (= PCA of the correlation matrix).
    Returns the fitted PCA, loadings (stocks x PCs) and factor return series (dates x PCs)."""
    z = standardized_returns(close, market)
    pca = PCA(n_components=n_components).fit(z.values)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(pca.components_.T, index=z.columns, columns=cols)
    factor_rets = pd.DataFrame(pca.transform(z.values), index=z.index, columns=cols)
    # PCA signs are arbitrary: flip each PC so that its average loading is positive
    sign = np.sign(loadings.mean()).replace(0, 1)
    return pca, loadings * sign, factor_rets * sign


def cluster_stocks(loadings, sectors, n_clusters=None, pcs=("PC2", "PC3", "PC4", "PC5", "PC6"), seed=RANDOM_SEED):
    """K-means on loadings of PC2..PC6 (PC1 = market, shared by all, so it's left out).
    Returns cluster labels and the adjusted Rand index vs true sectors (1 = perfect match, 0 = random)."""
    n_clusters = n_clusters or len(set(sectors.values()))
    X = loadings[list(pcs)].values
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)
    labels = pd.Series(labels, index=loadings.index, name="cluster")
    true = loadings.index.map(sectors)
    return labels, adjusted_rand_score(true, labels)
