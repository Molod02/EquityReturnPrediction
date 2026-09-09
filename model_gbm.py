import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the features
features = pd.read_csv("features.csv")

X = features[["momentum", "volatility", "liquidity"]]
y = features["forward_return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient boosting doesn't need feature scaling
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Results")
print("Feature importances:", dict(zip(X.columns, model.feature_importances_)))
print("R^2 score on test set:", r2)