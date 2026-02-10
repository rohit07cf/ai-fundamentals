# 06 — Regularization

> "Does logistic regression even overfit?"
> Yes. And when it does, the symptoms are sneaky.
> This file is about why it happens, how to stop it, and what to say when they ask.

---

## Why Logistic Regression Overfits

Three scenarios where vanilla logistic regression goes off the rails:

### 1. Too Many Features, Not Enough Data

With p features and n samples, if p is close to (or exceeds) n, the model can find a hyperplane that perfectly separates the training data — even if that separation is based on noise.

```
n = 50 samples, p = 200 features

The model has so many knobs to turn that it can fit ANYTHING.
Training accuracy: 100%
Test accuracy: 55%    ← basically random
```

### 2. Perfect or Near-Perfect Separation

When a feature (or combination of features) perfectly separates the classes, the coefficient goes to **infinity** — the sigmoid tries to become a step function.

```
    P(y=1)
   1.0 |            ________     ← Model wants a cliff, not a curve
       |           |
       |           |
       |           |
   0.0 |___________|
       +------------------------→ feature
```

The optimizer never converges because "bigger coefficient = better fit" has no limit.

### 3. Multicollinearity

Correlated features fight for credit, producing enormous coefficients with opposite signs. The model memorizes the training data through these fragile, cancelling weights.

💡 **The one-liner for interviews:** "Logistic regression overfits when it has too many degrees of freedom — too many features, too little data, or correlated features that let it exploit noise in the training set."

---

## The Fix: Regularization

Same idea as linear regression: **add a penalty for large coefficients**.

```
Regularized loss = Cross-entropy loss + λ × penalty(β)
```

The penalty discourages extreme weights, keeping the model from "trying too hard."

---

## L2 Regularization (Ridge)

```
Penalty = λ Σⱼ βⱼ²
```

### What it does

- Shrinks ALL coefficients toward zero (but never exactly to zero)
- Large coefficients get penalized more (quadratic penalty)
- Stabilizes estimates when features are correlated

### Effect on the decision boundary

```
No regularization:               L2 regularization:

     x₂                               x₂
      |  o  o                           |  o  o
      | o  /  o                         | o  / o
      |  /  o                           |  / o
      | /                               | /
      |/ x  x  x                        |/ x  x  x
      +----------→ x₁                   +----------→ x₁

The boundary might be at a             The boundary is more conservative.
wild angle driven by noise.            Coefficients are smaller, more stable.
```

### When to use L2

- Features are **correlated** (multicollinearity)
- You want to keep **all** features but control their influence
- Default choice when you need regularization

---

## L1 Regularization (Lasso)

```
Penalty = λ Σⱼ |βⱼ|
```

### What it does

- Shrinks coefficients toward zero AND can set them **exactly to zero**
- Performs automatic **feature selection**
- Creates **sparse** models (many coefficients = 0)

### Why it gives zeros (the geometry, briefly)

The L1 constraint region is a **diamond** in parameter space. The loss function's contour ellipses tend to hit the diamond at corners (where one or more parameters = 0).

```
L2 constraint (circle):         L1 constraint (diamond):

  β₂                              β₂
  |    ...                         |    /\
  |  .     .                       |   /  \
  | .   *   .                      |  / *  \   ← hits at corners
  |  .     .                       |  \    /     (some βⱼ = 0)
  |    ...                         |   \  /
  +----------→ β₁                  +----\/-----→ β₁

  * = solution on surface          * = solution at corner
  (both β nonzero)                 (one β may be zero)
```

### When to use L1

- You suspect only a **few features** truly matter
- You want **interpretable, sparse** models
- You need **automatic feature selection**

---

## L1 vs. L2 — The Complete Picture

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Penalty** | λ Σ \|βⱼ\| | λ Σ βⱼ² |
| **Coefficients** | Some become exactly 0 | All shrink, none exactly 0 |
| **Feature selection** | Yes (automatic) | No |
| **Correlated features** | Picks one, drops others | Distributes weight among them |
| **Sparsity** | Sparse model | Dense model |
| **Computational** | Needs special solvers (coordinate descent) | Works with standard gradient descent |
| **sklearn param** | `penalty='l1'` | `penalty='l2'` (default) |

### Elastic Net (The Hybrid)

```
Penalty = λ₁ Σ |βⱼ| + λ₂ Σ βⱼ²
```

Best of both worlds: sparse feature selection (L1) + stability with correlated features (L2).

In sklearn: `penalty='elasticnet'` with `l1_ratio` controlling the mix.

---

## Effect on Decision Boundary

```
λ = 0 (no reg):       λ = small:           λ = medium:          λ = large:

   o o | x x            o o | x x           o o | x x           o o   x x
  o o  | x x           o o  | x x          o o  | x x          o o | x x
  o o  | x x           o o  | x x          o o  | x x          o    | x x
 o o   |  x x         o o   |  x x        o o  |  x x          o o | x x
  o    |  x x          o    |  x x         o   |  x x           oo | x x

Boundary might be     Slight smoothing.     More conservative.   Very conservative.
at a weird angle.     Still flexible.        Better on new data.  May underfit.
Fits noise.
```

As λ increases:
1. Coefficients shrink → boundary becomes "simpler"
2. Training accuracy may decrease
3. Test accuracy first improves, then decreases (bias-variance tradeoff)

---

## Choosing λ (The C Parameter in sklearn)

⚠️ **sklearn uses `C` = 1/λ.** This is the inverse! Higher C = less regularization (more flexible). Lower C = more regularization (simpler model).

```
C = 1000   →  λ ≈ 0      →  Almost no regularization (overfit risk)
C = 1      →  λ = 1       →  Default, moderate regularization
C = 0.01   →  λ = 100     →  Heavy regularization (underfit risk)
```

**How to choose:** Cross-validation. Try a range of C values (e.g., [0.001, 0.01, 0.1, 1, 10, 100]) and pick the one with the best validation performance.

```python
# sklearn makes this easy:
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=10, cv=5)  # Tries 10 C values with 5-fold CV
```

---

## Regularization and Coefficient Interpretation

Important caveat: when you regularize, coefficients are **biased** (intentionally shrunk toward zero). They no longer have the clean "one unit increase in x → β change in log-odds" interpretation.

This is the **bias-variance tradeoff** in action:
- Add bias (shrink coefficients) → reduce variance (more stable predictions)
- The total error often decreases, even though individual coefficient estimates are "wrong"

💡 **When to mention this in interviews:** If asked to interpret regularized coefficients, say: "Regularized coefficients are biased toward zero, so their magnitudes understate the true effects. For interpretation, I'd either use the unregularized model or note that the coefficients reflect penalized estimates."

---

## Key Takeaways

- Logistic regression overfits with: too many features, perfect separation, or multicollinearity
- **L2 (Ridge):** Shrinks all coefficients, keeps all features. Default choice.
- **L1 (Lasso):** Zeros out some coefficients. Automatic feature selection.
- **Elastic Net:** Combines both. Good when you want sparsity + stability.
- **λ controls** the strength. In sklearn, **C = 1/λ** (inverse!).
- Choose λ/C via **cross-validation**.
- Regularized coefficients are **biased** — be careful interpreting them.

⚠️ **The interview move:** When asked "how do you prevent logistic regression from overfitting?", say: "Regularization — L2 to shrink coefficients, L1 if I also want feature selection. I'd tune the regularization strength with cross-validation. sklearn's default uses L2 with C=1, which is a reasonable starting point."
