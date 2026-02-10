"""
Logistic Regression from Scratch (NumPy only)
===============================================
No sklearn. No magic. Just the sigmoid, cross-entropy, and gradient descent.

This implements binary logistic regression step by step.
Read the comments — they explain WHY, not just WHAT.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Generate some fake data (two classes, linearly separable-ish)
# ---------------------------------------------------------------------------
# Class 0: centered around (-1, -1)
# Class 1: centered around (+1, +1)
# With some overlap, because real data is messy.

np.random.seed(42)
n_per_class = 100

X_class0 = np.random.randn(n_per_class, 2) + np.array([-1, -1])
X_class1 = np.random.randn(n_per_class, 2) + np.array([1, 1])

X_raw = np.vstack([X_class0, X_class1])           # (200, 2)
y = np.array([0] * n_per_class + [1] * n_per_class)  # (200,)

# Shuffle so the classes aren't in order
shuffle_idx = np.random.permutation(len(y))
X_raw = X_raw[shuffle_idx]
y = y[shuffle_idx]

# Add a column of ones for the intercept (bias) term.
# This lets us write z = Xβ where β includes the intercept.
X = np.column_stack([np.ones(len(y)), X_raw])  # (200, 3)


# ---------------------------------------------------------------------------
# The Sigmoid Function
# ---------------------------------------------------------------------------
# This is the heart of logistic regression.
# It takes any real number and squashes it into (0, 1).

def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z))"""
    # Clip z to avoid overflow in exp for very negative values
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# The Loss Function (Binary Cross-Entropy / Log Loss)
# ---------------------------------------------------------------------------
# Why not MSE? Because MSE + sigmoid = non-convex loss surface.
# Cross-entropy is convex → guaranteed global minimum.
# It also punishes confident wrong predictions SEVERELY.

def compute_loss(y, p):
    """
    Binary cross-entropy loss.
    L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]
    """
    n = len(y)
    # Clip predictions to avoid log(0) which is -infinity
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(1.0 / n) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------------------------
# The gradient of cross-entropy loss w.r.t. β is:
#   ∇L = (1/n) * Xᵀ(σ(Xβ) - y)
#
# Notice: this looks EXACTLY like the gradient for linear regression
# ((1/n) * Xᵀ(Xβ - y)) except Xβ is replaced by σ(Xβ).
# Beautiful, right?

def train_logistic_regression(X, y, learning_rate=0.1, n_iterations=1000):
    """Train logistic regression using batch gradient descent."""
    n, p = X.shape
    beta = np.zeros(p)  # Start with all zeros

    losses = []  # Track loss over time (should decrease!)

    for i in range(n_iterations):
        # Forward pass: compute predictions
        z = X @ beta              # Linear score
        p = sigmoid(z)            # Probability via sigmoid

        # Compute and store the loss
        loss = compute_loss(y, p)
        losses.append(loss)

        # Compute the gradient
        # This is the key equation: (1/n) * Xᵀ(predictions - labels)
        gradient = (1.0 / n) * (X.T @ (p - y))

        # Update weights (step downhill)
        beta = beta - learning_rate * gradient

        # Print progress every 200 iterations
        if i % 200 == 0:
            accuracy = np.mean((p >= 0.5) == y)
            print(f"  Iteration {i:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

    return beta, losses


# ---------------------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------------------
print("=== Training Logistic Regression from Scratch ===\n")
beta, losses = train_logistic_regression(X, y, learning_rate=0.5, n_iterations=1000)

print(f"\n=== Learned Parameters ===")
print(f"Intercept (β₀): {beta[0]:.4f}")
print(f"Weight x₁ (β₁): {beta[1]:.4f}")
print(f"Weight x₂ (β₂): {beta[2]:.4f}")


# ---------------------------------------------------------------------------
# Make predictions and evaluate
# ---------------------------------------------------------------------------
z = X @ beta
probs = sigmoid(z)
predictions = (probs >= 0.5).astype(int)

accuracy = np.mean(predictions == y)
print(f"\n=== Final Performance ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"Final loss: {losses[-1]:.4f}")

# Show some example predictions
print(f"\n=== Sample Predictions ===")
print(f"{'Features':>20s} | {'True':>5s} | {'P(y=1)':>7s} | {'Pred':>5s}")
print("-" * 50)
for i in range(10):
    features = f"({X_raw[i, 0]:+.2f}, {X_raw[i, 1]:+.2f})"
    print(f"{features:>20s} | {y[i]:>5d} | {probs[i]:>7.4f} | {predictions[i]:>5d}")


# ---------------------------------------------------------------------------
# Confusion matrix (by hand)
# ---------------------------------------------------------------------------
tp = np.sum((predictions == 1) & (y == 1))
fp = np.sum((predictions == 1) & (y == 0))
fn = np.sum((predictions == 0) & (y == 1))
tn = np.sum((predictions == 0) & (y == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Confusion Matrix ===")
print(f"              Pred 1   Pred 0")
print(f"  Actual 1:    {tp:4d}     {fn:4d}")
print(f"  Actual 0:    {fp:4d}     {tn:4d}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ---------------------------------------------------------------------------
# Decision boundary equation
# ---------------------------------------------------------------------------
# The boundary is where z = 0:  β₀ + β₁x₁ + β₂x₂ = 0
# Solving for x₂:  x₂ = -(β₀ + β₁x₁) / β₂
print(f"\n=== Decision Boundary ===")
print(f"Equation: {beta[0]:.3f} + {beta[1]:.3f}*x₁ + {beta[2]:.3f}*x₂ = 0")
if abs(beta[2]) > 1e-10:
    print(f"Solved:   x₂ = {-beta[0]/beta[2]:.3f} + {-beta[1]/beta[2]:.3f}*x₁")


# ---------------------------------------------------------------------------
# Verify: loss should have decreased monotonically
# ---------------------------------------------------------------------------
print(f"\n=== Loss Convergence Check ===")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Loss decreased monotonically: {all(losses[i] >= losses[i+1] - 1e-10 for i in range(len(losses)-1))}")
