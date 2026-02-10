"""
Logistic Regression with sklearn
==================================
Minimal, clean, no showing off.
The goal: show that sklearn does in a few lines what we did from scratch,
and tie the results back to the intuition.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Same data as the NumPy example
# ---------------------------------------------------------------------------
np.random.seed(42)
n_per_class = 100

X_class0 = np.random.randn(n_per_class, 2) + np.array([-1, -1])
X_class1 = np.random.randn(n_per_class, 2) + np.array([1, 1])

X = np.vstack([X_class0, X_class1])
y = np.array([0] * n_per_class + [1] * n_per_class)

# Train/test split — always do this before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Fit the model
# ---------------------------------------------------------------------------
# C=1.0 is the default regularization strength (C = 1/λ).
# Higher C = less regularization. Lower C = more regularization.
# max_iter increased because default (100) may not converge.

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("=== Model Parameters ===")
print(f"Intercept (β₀):  {model.intercept_[0]:.4f}")
print(f"Weight x₁ (β₁):  {model.coef_[0][0]:.4f}")
print(f"Weight x₂ (β₂):  {model.coef_[0][1]:.4f}")

# ---------------------------------------------------------------------------
# Predictions: probabilities AND class labels
# ---------------------------------------------------------------------------
# This is what makes logistic regression special:
# It gives you CALIBRATED PROBABILITIES, not just class labels.

y_probs = model.predict_proba(X_test)[:, 1]  # P(y=1) for each test point
y_pred = model.predict(X_test)                 # Class labels (0 or 1)

print("\n=== Sample Predictions ===")
print(f"{'x₁':>8s} {'x₂':>8s} | {'P(y=1)':>8s} | {'Pred':>5s} | {'True':>5s}")
print("-" * 45)
for i in range(8):
    print(f"{X_test[i, 0]:>8.3f} {X_test[i, 1]:>8.3f} | {y_probs[i]:>8.4f} | {y_pred[i]:>5d} | {y_test[i]:>5d}")

# ---------------------------------------------------------------------------
# Metrics — the full picture
# ---------------------------------------------------------------------------
print("\n=== Test Set Metrics ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}  (when I say 1, am I right?)")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}  (did I find all the 1s?)")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}  (harmonic mean of P and R)")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probs):.4f}  (ranking quality)")

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print(f"\n=== Confusion Matrix ===")
print(f"              Pred 0   Pred 1")
print(f"  Actual 0:    {cm[0, 0]:4d}     {cm[0, 1]:4d}")
print(f"  Actual 1:    {cm[1, 0]:4d}     {cm[1, 1]:4d}")

# ---------------------------------------------------------------------------
# Full classification report (sklearn's handy summary)
# ---------------------------------------------------------------------------
print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# ---------------------------------------------------------------------------
# Decision boundary interpretation
# ---------------------------------------------------------------------------
b0, b1, b2 = model.intercept_[0], model.coef_[0][0], model.coef_[0][1]
print(f"=== Decision Boundary ===")
print(f"Equation: {b0:.3f} + {b1:.3f}*x₁ + {b2:.3f}*x₂ = 0")
if abs(b2) > 1e-10:
    print(f"Solved:   x₂ = {-b0/b2:.3f} + {-b1/b2:.3f}*x₁")

# ---------------------------------------------------------------------------
# Coefficient interpretation (tying back to intuition)
# ---------------------------------------------------------------------------
print(f"\n=== Coefficient Interpretation ===")
for i, name in enumerate(["x₁", "x₂"]):
    coef = model.coef_[0][i]
    odds_ratio = np.exp(coef)
    print(f"  {name}: β = {coef:.4f}")
    print(f"       → 1-unit increase multiplies odds by e^{coef:.4f} = {odds_ratio:.4f}")
    if odds_ratio > 1:
        print(f"       → Odds increase by {(odds_ratio - 1) * 100:.1f}% per unit")
    else:
        print(f"       → Odds decrease by {(1 - odds_ratio) * 100:.1f}% per unit")

# ---------------------------------------------------------------------------
# Effect of regularization (quick demo)
# ---------------------------------------------------------------------------
print(f"\n=== Regularization Effect ===")
for C_val in [0.01, 1.0, 100.0]:
    m = LogisticRegression(C=C_val, max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    coef_norm = np.linalg.norm(m.coef_)
    print(f"  C={C_val:6.2f} | Accuracy: {acc:.4f} | ||β||: {coef_norm:.4f}  "
          f"{'(heavy reg)' if C_val < 0.1 else '(default)' if C_val == 1.0 else '(light reg)'}")
