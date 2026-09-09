import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the features we built
features = pd.read_csv("features.csv")

X = features[["momentum", "volatility", "liquidity"]]
y = features["forward_return"]

# Split into train/test sets (80/20), keeping it simple for now
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (important for regularized regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regression
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression Results")
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("R^2 score on test set:", r2)