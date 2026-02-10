# 05 — Regularization Preview

> This is a teaser, not the full movie.
> Just enough to sound smart when the conversation naturally flows from LR to regularization — which it almost always does in interviews.

---

## Why Linear Regression Overfits

"Wait, a simple linear model can overfit?"

**Yes.** Here's when:

1. **Too many features relative to data points** (p close to n, or p > n)
   - With enough features, you can fit training data perfectly — even noise
   - Extreme case: n=100 data points, p=100 features → perfect fit, zero training error, garbage on new data

2. **Multicollinearity**
   - Correlated features → huge, unstable coefficients that cancel each other out
   - Works on training data by accident, fails spectacularly on test data

3. **No constraint on coefficient size**
   - Vanilla LR is happy to give β = 10,000 if it minimizes training loss
   - Giant coefficients = model is "trying too hard" to fit the training data

💡 **The one-liner:** "Overfitting in linear regression happens when the model has too much freedom — too many features or no penalty for large coefficients."

---

## The Core Idea of Regularization

**Add a penalty for large coefficients to the loss function.**

```
Vanilla LR loss:     L = Σ(yᵢ - ŷᵢ)²              (just fit the data)

Regularized loss:    L = Σ(yᵢ - ŷᵢ)² + λ × penalty(β)   (fit the data BUT keep β small)
```

**λ (lambda)** controls the tradeoff:
- λ = 0 → no penalty → vanilla LR
- λ → ∞ → maximum penalty → all coefficients shrink toward zero
- λ in between → the sweet spot

---

## Ridge vs. Lasso — The Intuition

| | **Ridge (L2)** | **Lasso (L1)** |
|--|---------------|----------------|
| **Penalty** | λ Σ βⱼ² | λ Σ \|βⱼ\| |
| **What it does** | Shrinks all coefficients toward zero | Shrinks some coefficients exactly TO zero |
| **Effect** | Keeps all features, makes them smaller | **Feature selection** — kills unimportant features |
| **Geometry** | Circular constraint | Diamond constraint |
| **Best for** | Many correlated features (all somewhat useful) | Sparse models (few features actually matter) |

### The Geometry (Why Lasso Gives Zeros)

This is the part interviewers love to ask about.

```
Ridge (circle):                 Lasso (diamond):

  β₂                             β₂
  |    .....                      |    /\
  |  ..     ..                    |   /  \
  | .   *    .  ←constraint       |  / *  \  ←constraint
  |  ..     ..    region          |  \    /    region
  |    .....                      |   \  /
  +------------- β₁              +----\/------- β₁

  * = where the loss contour       * = where the loss contour
      touches the circle               touches the diamond

  → touches on the surface         → touches at a CORNER
  → β₁ and β₂ both nonzero         → one of them is exactly 0!
```

**The diamond has corners on the axes.** The loss function's contour ellipses are more likely to hit a corner, which means one coefficient is exactly zero. That's why Lasso does feature selection.

The circle has no corners, so both coefficients stay nonzero (just smaller). That's why Ridge shrinks but doesn't eliminate.

---

## When Interviewers Expect You to Bring This Up

1. **"Your model has 500 features. What do you do?"**
   → Mention regularization. Lasso for feature selection, Ridge if you think most features contribute.

2. **"Your coefficients are huge and unstable."**
   → Classic multicollinearity. Ridge regression fixes this by constraining coefficient sizes.

3. **"How do you prevent overfitting in linear regression?"**
   → Regularization (Ridge/Lasso), cross-validation for λ, or reducing features.

4. **"What's the difference between Ridge and Lasso?"**
   → L2 penalty shrinks everything, L1 penalty zeros out features. Mention the geometry.

---

## The Quick Reference

| Question | Ridge | Lasso |
|----------|-------|-------|
| Kill features entirely? | No | Yes |
| Handle multicollinearity? | Yes (great at it) | Picks one, drops the rest |
| Penalty math | Sum of β² | Sum of \|β\| |
| When to use | Many useful features | Sparse signal, want feature selection |
| Normal equation | β = (XᵀX + λI)⁻¹Xᵀy | No closed form, needs optimization |

💡 **The sentence that impresses:** "Ridge adds λI to XᵀX, which guarantees invertibility even with multicollinearity. That's actually why it's also called Tikhonov regularization — it was originally designed to stabilize ill-conditioned systems, not for machine learning."

---

## What's Coming (In Future Modules)

- Full Ridge regression deep dive
- Full Lasso deep dive
- Elastic Net (Ridge + Lasso hybrid)
- How to tune λ with cross-validation
- Regularization paths and coefficient plots

For now, you know enough to handle the regularization question in a linear regression interview. The deep dives come later.

---

## Key Takeaways

- LR overfits when: too many features, multicollinearity, or no coefficient constraints
- **Regularization** = add a penalty for large coefficients
- **Ridge (L2)** shrinks all coefficients, keeps all features
- **Lasso (L1)** shrinks some coefficients to exactly zero → feature selection
- **λ** controls the regularization strength
- Know the **diamond vs. circle geometry** — interviewers ask about it
- **Always mention regularization** when discussing overfitting in LR
