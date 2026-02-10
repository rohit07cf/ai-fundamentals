"""
Linear Regression with sklearn
================================
Minimal, clean, no showing off.
The goal: show that sklearn does in 4 lines what we did from scratch,
and tie the results back to the intuition.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Same fake data as the NumPy example
# ---------------------------------------------------------------------------
# True relationship: y = 3 + 2*x + noise

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 3 + 2 * X[:, 0] + np.random.randn(100) * 0.5

# Split into train/test — always do this before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# Fit the model
# ---------------------------------------------------------------------------
# That's it. Three lines. sklearn handles the Normal Equation internally.

model = LinearRegression()
model.fit(X_train, y_train)

print("=== Model Parameters ===")
print(f"Intercept (β₀): {model.intercept_:.4f}  (true: 3.0)")
print(f"Slope     (β₁): {model.coef_[0]:.4f}  (true: 2.0)")

# ---------------------------------------------------------------------------
# Evaluate on TEST data (not training data!)
# ---------------------------------------------------------------------------
# Remember from 02_metrics.md: training metrics can be misleading.
# Always report test metrics.

y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("\n=== Test Set Metrics ===")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}  (on average, we're off by ~{rmse:.2f} units)")
print(f"R²:   {r2:.4f}  (model explains {r2*100:.1f}% of test variance)")

# ---------------------------------------------------------------------------
# Tying it back to intuition
# ---------------------------------------------------------------------------
# The coefficient tells us: for every 1-unit increase in X,
# y increases by ~2 units (holding everything else constant).
# The intercept tells us: when X=0, the predicted y is ~3.
# Both are close to the true values (3 and 2), confirming our model works.

print("\n=== Interpretation ===")
print(f"For every 1-unit increase in X, y increases by ~{model.coef_[0]:.2f}")
print(f"When X = 0, predicted y = {model.intercept_:.2f}")
print(f"Our model captures ~{r2*100:.0f}% of the variance in the test data.")

# ---------------------------------------------------------------------------
# Quick residual check (sanity check from 03_assumptions.md)
# ---------------------------------------------------------------------------
residuals = y_test - y_pred_test
print("\n=== Residual Sanity Check ===")
print(f"Mean of residuals:  {np.mean(residuals):.4f}  (should be ~0)")
print(f"Std of residuals:   {np.std(residuals):.4f}  (should be ~0.5, our noise level)")
print(f"Max residual:       {np.max(np.abs(residuals)):.4f}")
print("(If mean ≈ 0 and std ≈ noise level, we're on track.)")
