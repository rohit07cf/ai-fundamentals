"""
Linear Regression from Scratch (NumPy only)
============================================
No sklearn. No magic. Just math you understand.

This implements both:
  1. The Normal Equation (closed-form solution)
  2. Gradient Descent (iterative solution)

Run it, read the comments, and make sure you can explain every line.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Generate some fake data
# ---------------------------------------------------------------------------
# True relationship: y = 3 + 2*x + noise
# We KNOW the answer, so we can verify our implementation works.

np.random.seed(42)
n_samples = 100

X_raw = 2 * np.random.rand(n_samples, 1)          # 100 points, 1 feature, range [0, 2]
y = 3 + 2 * X_raw[:, 0] + np.random.randn(n_samples) * 0.5  # true: intercept=3, slope=2

# Add a column of ones for the intercept term.
# Why? Because ŷ = β₀ + β₁x is the same as ŷ = [1, x] · [β₀, β₁]
# The column of ones lets us treat the intercept as just another weight.
X = np.column_stack([np.ones(n_samples), X_raw])   # shape: (100, 2)


# ---------------------------------------------------------------------------
# Method 1: Normal Equation
# ---------------------------------------------------------------------------
# β = (XᵀX)⁻¹ Xᵀy
#
# This gives the EXACT answer in one shot.
# Works great for small-to-medium datasets.
# Breaks when XᵀX is singular (perfect multicollinearity) or p is very large.

def normal_equation(X, y):
    """Solve for β using the closed-form Normal Equation."""
    # Step by step so you can follow on a whiteboard:
    XtX = X.T @ X            # (p x n) @ (n x p) = (p x p)  — feature correlations
    Xty = X.T @ y            # (p x n) @ (n x 1) = (p x 1)  — feature-target agreement
    beta = np.linalg.inv(XtX) @ Xty  # (p x p)⁻¹ @ (p x 1) = (p x 1)
    return beta

beta_normal = normal_equation(X, y)
print("=== Normal Equation ===")
print(f"Intercept (β₀): {beta_normal[0]:.4f}  (true: 3.0)")
print(f"Slope     (β₁): {beta_normal[1]:.4f}  (true: 2.0)")


# ---------------------------------------------------------------------------
# Method 2: Gradient Descent
# ---------------------------------------------------------------------------
# When p is large or data doesn't fit in memory, we can't invert XᵀX.
# Instead, we iteratively step toward the minimum.
#
# Update rule: β := β - α * gradient
# Gradient of MSE: (2/n) * Xᵀ(Xβ - y)

def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000, tolerance=1e-8):
    """Solve for β using batch gradient descent."""
    n = len(y)
    beta = np.zeros(X.shape[1])  # Start with all zeros (could be random too)

    for i in range(n_iterations):
        # Forward pass: make predictions
        y_pred = X @ beta

        # Compute the residuals (how wrong we are)
        residuals = y_pred - y

        # Compute the gradient of MSE with respect to β
        # This tells us: "which direction increases the loss?"
        gradient = (2 / n) * (X.T @ residuals)

        # Take a step in the OPPOSITE direction (downhill)
        beta = beta - learning_rate * gradient

        # Check for convergence: if the gradient is tiny, we're at the bottom
        if np.linalg.norm(gradient) < tolerance:
            print(f"  Converged at iteration {i}")
            break

    return beta

beta_gd = gradient_descent(X, y, learning_rate=0.1, n_iterations=1000)
print("\n=== Gradient Descent ===")
print(f"Intercept (β₀): {beta_gd[0]:.4f}  (true: 3.0)")
print(f"Slope     (β₁): {beta_gd[1]:.4f}  (true: 2.0)")


# ---------------------------------------------------------------------------
# Compare the two methods
# ---------------------------------------------------------------------------
# They should give (nearly) identical answers, because:
# - Both minimize the same loss function (MSE)
# - The loss is convex, so there's only one minimum
# - Normal Equation finds it exactly; GD converges to it

print("\n=== Comparison ===")
print(f"Max difference between methods: {np.max(np.abs(beta_normal - beta_gd)):.8f}")


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
# Tying back to the metrics file (02_metrics.md)

y_pred = X @ beta_normal

residuals = y - y_pred
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("\n=== Metrics ===")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}  (average prediction error in original units)")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r_squared:.4f}  (fraction of variance explained)")


# ---------------------------------------------------------------------------
# Verify the orthogonality property
# ---------------------------------------------------------------------------
# The residuals should be orthogonal to the column space of X.
# Meaning: Xᵀe ≈ 0 (within floating-point precision)

orthogonality_check = X.T @ residuals
print("\n=== Orthogonality Check ===")
print(f"Xᵀe = {orthogonality_check}")
print(f"(Should be ~0. This confirms the projection interpretation.)")
