import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ml.churn_data import split_churn_data

# ---------- Step 1: split the data ----------
X_train, X_test, y_train, y_test = split_churn_data()

print("Training rows:", len(X_train), "| Test rows:", len(X_test))
print(f"Churn rate in training set: {y_train.mean():.3f}")
print(f"Churn rate in test set:     {y_test.mean():.3f}")
print("Features:", list(X_train.columns))
print()

# ---------- Step 2: scale the features (for logistic regression) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn the scaling from the training data only
X_test_scaled = scaler.transform(X_test)        # apply the same scaling to the test data

# ---------- Step 3: train two models ----------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

forest = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
forest.fit(X_train, y_train)  # trees do not need scaled features

# ---------- Step 4: a first look at the scores ----------
baseline = 1 - y_test.mean()  # accuracy of always predicting "stays"

print("=== ACCURACY ===")
print(f"Baseline (always predict 'stays'): {baseline:.3f}")
print(f"Logistic regression  - train: {log_reg.score(X_train_scaled, y_train):.3f} | test: {log_reg.score(X_test_scaled, y_test):.3f}")
print(f"Random forest        - train: {forest.score(X_train, y_train):.3f} | test: {forest.score(X_test, y_test):.3f}")
print()

# ---------- Step 5: what did each model learn? ----------
coefficients = pd.Series(log_reg.coef_[0], index=X_train.columns).sort_values()
print("=== LOGISTIC REGRESSION: points for each feature (negative = lowers churn risk) ===")
print(coefficients.round(2))
print()

importances = pd.Series(forest.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("=== RANDOM FOREST: how much each feature mattered ===")
print(importances.round(3))
