# 02 — Metrics

> Think of metrics like debugging tools for your model.
> You wouldn't ship code without testing it. You wouldn't ship a model without measuring it.
> Here's your toolkit.

---

## The Metrics Family

| Metric | Formula | What It Tells You |
|--------|---------|------------------|
| **MSE** | (1/n) Σ(yᵢ - ŷᵢ)² | Average squared error |
| **RMSE** | √MSE | Average error in the same units as y |
| **MAE** | (1/n) Σ\|yᵢ - ŷᵢ\| | Average absolute error |
| **R²** | 1 - (SS_res / SS_tot) | Fraction of variance explained |
| **Adjusted R²** | 1 - [(1-R²)(n-1)/(n-p-1)] | R² penalized for extra features |

---

## MSE — Mean Squared Error

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

**What it is:** Average of the squared residuals.

**Good for:** Optimization (it's what we minimize in least squares).

**Bad for:** Interpretation. If you're predicting house prices in dollars, MSE is in dollars². What does $50,000² even mean? Nothing intuitive.

**When big errors matter more:** MSE is your friend. Squaring means an error of 10 counts 100 times more than an error of 1.

---

## RMSE — Root Mean Squared Error

```
RMSE = √MSE
```

**What it is:** Square root of MSE. Now it's back in the same units as your target.

**Think of it as:** "On average, my predictions are off by about this much."

**The go-to metric** for regression problems in practice. When someone says "what's your model's error?" they usually want RMSE.

---

## MAE — Mean Absolute Error

```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

**What it is:** Average of the absolute residuals. No squaring.

**Key difference from RMSE:** MAE treats all errors equally. An error of 10 is just 10× worse than an error of 1 (not 100× like MSE).

**Use MAE when:** Outliers exist and you don't want them dominating your metric.

💡 **Interview gold:** "RMSE penalizes large errors more heavily than MAE because of the squaring. If your application can tolerate occasional big misses but wants good average performance, use MAE. If big errors are costly (like in financial models), use RMSE."

---

## R² — The One Everyone Asks About

```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(yᵢ - ŷᵢ)²     (sum of squared residuals — your model's errors)
  SS_tot = Σ(yᵢ - ȳ)²       (total variance — errors if you just predicted the mean)
```

### What it REALLY means

**R² answers: "How much better is my model than just predicting the average?"**

- **R² = 1.0** → Your model perfectly explains all the variance. Every prediction is exactly right.
- **R² = 0.0** → Your model is no better than predicting the mean for everything.
- **R² < 0** → Your model is actively **worse** than predicting the mean. Something is very wrong.

### The mental model

Think of it like a report card:
- **SS_tot** = the total variance in your data (the "problem difficulty")
- **SS_res** = the variance your model didn't explain (the "mistakes left over")
- **R²** = what fraction of the problem you actually solved

---

## Tiny Worked Example

Data: Three points, our model predicts ŷ.

| Actual (y) | Predicted (ŷ) | Residual | Residual² |
|-----------|--------------|----------|-----------|
| 2 | 2.5 | -0.5 | 0.25 |
| 4 | 3.5 | 0.5 | 0.25 |
| 6 | 5.5 | 0.5 | 0.25 |

**Mean of y:** (2 + 4 + 6) / 3 = 4

**SS_res** = 0.25 + 0.25 + 0.25 = **0.75**

**SS_tot** = (2-4)² + (4-4)² + (6-4)² = 4 + 0 + 4 = **8**

**MSE** = 0.75 / 3 = **0.25**

**RMSE** = √0.25 = **0.5**

**MAE** = (0.5 + 0.5 + 0.5) / 3 = **0.5**

**R²** = 1 - (0.75 / 8) = 1 - 0.094 = **0.906**

Translation: "Our model explains about 91% of the variance in y. On average, we're off by about 0.5 units."

---

## Why R² Can Lie to You

This is where interviews get spicy.

### Trap 1: R² ALWAYS increases when you add features

Even useless features will increase R² (or at least not decrease it). Why? Because the model has more degrees of freedom to fit the training data.

```
Model A: y ~ x₁             → R² = 0.80
Model B: y ~ x₁ + x_random  → R² = 0.81  (random noise "helped"!)
```

The model isn't better — it's just more flexible. This is why **Adjusted R²** exists.

### Trap 2: High R² doesn't mean your model is good

- R² = 0.99 on training data might just mean overfitting
- R² = 0.99 on a tiny dataset means almost nothing
- R² doesn't tell you if the **assumptions** hold

### Trap 3: R² doesn't tell you about individual predictions

An R² of 0.90 means 90% of variance explained. But some individual predictions could still be wildly off. Always look at residual plots.

⚠️ **The interview question:** "Is a higher R² always better?" The answer is **no**. A higher R² on training data might indicate overfitting. What matters is R² on **unseen test data**, and even then, you should check that assumptions hold and residuals look well-behaved.

---

## Adjusted R²

```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]

where:
  n = number of data points
  p = number of features
```

**What it does:** Penalizes R² for each feature you add. If a feature doesn't improve the model enough to justify its complexity, Adjusted R² will actually **decrease**.

**Use it for:** Comparing models with different numbers of features.

💡 **This is the sentence interviewers want to hear:**
> "R² always increases with more features, so it's unreliable for model comparison. Adjusted R² penalizes model complexity, making it better for feature selection. But for truly rigorous comparison, I'd use cross-validation."

---

## MSE vs MAE — When to Use Which

| Situation | Use | Why |
|-----------|-----|-----|
| Big errors are very costly | MSE / RMSE | Squaring amplifies big mistakes |
| Outliers present, you want robustness | MAE | Doesn't over-penalize outliers |
| You want interpretable units | RMSE or MAE | Both in same units as y |
| Optimization / training | MSE | Smooth, differentiable everywhere |
| You're comparing models | RMSE (standardized) | Most commonly used baseline metric |

---

## Quick Reference for Interviews

**"Explain R² to a non-technical person:"**
> "If R² is 0.85, it means our model captures 85% of the pattern in the data. The other 15% is noise or stuff we haven't accounted for."

**"What's the difference between MSE and MAE?"**
> "MSE squares the errors, so big mistakes get punished disproportionately. MAE treats all errors linearly. Use MSE when big errors are expensive; use MAE when you want robustness to outliers."

**"Can R² be negative?"**
> "Yes — it means the model is doing worse than just predicting the mean. This can happen with a bad model or when evaluating on test data with a model that overfit the training set."

---

## Key Takeaways

- **MSE** = average squared error. Great for optimization, bad for interpretation.
- **RMSE** = √MSE. Same units as y. Your default reporting metric.
- **MAE** = average absolute error. Robust to outliers.
- **R²** = fraction of variance explained. But it lies — always goes up with more features.
- **Adjusted R²** = R² with a complexity penalty. Better for comparing models.
- **Always check test data**, not just training metrics.
