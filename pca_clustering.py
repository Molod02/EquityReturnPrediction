import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = pd.read_csv("features.csv")

# Average each stock's factor exposures across the whole period
avg_exposures = features.groupby("ticker")[["momentum", "volatility", "liquidity"]].mean()

print("Average factor exposures per stock:")
print(avg_exposures)

# Standardize before PCA
scaler = StandardScaler()
scaled = scaler.fit_transform(avg_exposures)

# PCA to find latent factors
pca = PCA(n_components=2)
components = pca.fit_transform(scaled)

print("\nExplained variance ratio by component:", pca.explained_variance_ratio_)

# Cluster stocks based on their factor exposures
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled)

results = avg_exposures.copy()
results["PC1"] = components[:, 0]
results["PC2"] = components[:, 1]
results["cluster"] = clusters

print("\nStocks with PCA components and cluster assignment:")
print(results)