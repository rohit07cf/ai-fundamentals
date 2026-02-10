# 03 — Assumptions & Diagnostics

> Linear regression has assumptions. Not suggestions. Assumptions.
> When they break, your model's results can go from "useful" to "garbage" quietly.
> Interviewers LOVE asking about these because it separates people who use LR from people who **understand** LR.

---

## The LINE Mnemonic

Remember these four letters: **L-I-N-E**

| Letter | Assumption | Plain English |
|--------|-----------|---------------|
| **L** | **Linearity** | The relationship between X and y is actually linear |
| **I** | **Independence** | The errors don't influence each other |
| **N** | **Normality** | The errors follow a normal distribution |
| **E** | **Equal variance (Homoscedasticity)** | The errors have the same spread everywhere |

There's a fifth one that people sometimes forget: **No perfect multicollinearity** (features aren't perfectly correlated with each other). We'll cover that in the multiple regression file.

---

## L — Linearity

### What it REALLY means
The true relationship between features and target is (approximately) linear. If the real pattern is a curve, forcing a straight line through it will give you systematically wrong predictions.

### How it breaks in real life
- Income vs. age: increases, then plateaus, then might decrease. Not linear.
- Temperature vs. ice cream sales: probably linear-ish in a range, then saturates.
- Any relationship with diminishing returns.

### How to detect
```
Residuals vs. Fitted Values Plot

If linear:                    If non-linear:
  e |  . . .  . .              e |     .  . .
    | .  .  . .  .               |  .        .
  --+---.--.--.--.---           --+---.---------.---
    | .  . .  .  .               |        .
    |  . .  . . .                |  . .      . .
    +---------------              +---------------
         ŷ                             ŷ

  Random scatter = GOOD         Pattern/curve = BAD
```

### What to do if broken
- **Add polynomial features** (x², x³) — lets the model capture curves
- **Apply transformations** (log, sqrt) to X or y
- **Use a non-linear model** (but that's often overkill if a transform works)

💡 **Interview tip:** "I always start by plotting residuals vs. fitted values. If I see a curve or pattern, I know the linearity assumption is violated and I'll try feature transformations before reaching for a more complex model."

---

## I — Independence of Errors

### What it REALLY means
Knowing the error for one data point tells you **nothing** about the error for another. Each residual is its own thing.

### How it breaks in real life
This breaks most often with **time series data** or **spatial data**:
- Stock prices: today's error is correlated with yesterday's error
- Temperature readings from nearby sensors: errors are spatially correlated
- Any data where observations have a natural ordering and nearby observations influence each other

### How to detect
- **Durbin-Watson test** — checks for autocorrelation in residuals (values near 2 = good, near 0 or 4 = bad)
- **Plot residuals vs. order** — look for waves or trends

```
Independent residuals:          Autocorrelated residuals:

e |  .   .                      e |  . .
  |    .    .  .                  |      . .
--+---.---.----.---             --+----------.-.---
  |  .  .    .                    |              . .
  |       .                       |                 .
  +--------------- order          +--------------- order

  Random = GOOD                 Wavy pattern = BAD
```

### What to do if broken
- **Add lagged variables** (include yesterday's value as a feature)
- **Use time-series models** (ARIMA, etc.) instead of plain LR
- **Cluster-robust standard errors** if you can't fix the structure

⚠️ **Common trap:** Most ML interview problems assume independence. But if someone gives you a time-series problem and asks you to apply linear regression, they're testing whether you'll mention this assumption.

---

## N — Normality of Errors

### What it REALLY means
The residuals (not the features, not y — the **residuals**) should be approximately normally distributed. Bell curve, centered at zero.

### How it breaks in real life
- **Skewed targets** (income, house prices): right-skewed residuals
- **Outliers**: fat tails in the residual distribution
- **Bounded targets** (percentages, counts): residuals can't be truly normal

### How to detect
- **Q-Q plot** (Quantile-Quantile): plot residual quantiles vs. theoretical normal quantiles

```
Q-Q Plot

Normal residuals:               Non-normal residuals:

Observed |    ./                 Observed |      ../
         |   ./                           |    ../
         |  ./                            |  /
         | ./                             | /
         |./                              |/...
         +---------- Theoretical          +---------- Theoretical

Points on line = GOOD           Curving off = BAD
```

- **Shapiro-Wilk test** — statistical test for normality

### What to do if broken
- **Transform y** (log(y) is the classic fix for right-skewed data)
- **Remove outliers** (if they're data errors)
- **Use robust regression** methods

### How much does it matter?

Honestly? **Less than you think** for predictions. Normality mainly affects:
- **Confidence intervals** and **p-values** (hypothesis testing)
- **Prediction intervals**

For just making predictions, normality is the least critical assumption. But interviewers still expect you to know it.

💡 **What to say:** "Normality of residuals matters most for inference — confidence intervals and hypothesis tests. For pure prediction tasks, it's less critical, but I'd still check for severe skewness or outliers that might indicate model problems."

---

## E — Equal Variance (Homoscedasticity)

### What it REALLY means
The spread of residuals should be roughly the same across all levels of prediction. The model shouldn't be more "uncertain" for high predictions than low ones (or vice versa).

### The name
- **Homoscedasticity** = same spread (what we want)
- **Heteroscedasticity** = different spread (the problem)

Don't let the words intimidate you. It's just "are the errors equally noisy everywhere?"

### How it breaks in real life

The classic example: **predicting income**.
- For people earning $30K, predictions might be off by $5K
- For people earning $300K, predictions might be off by $50K
- The error **scales with the prediction** — that's heteroscedasticity

### How to detect

```
Residuals vs. Fitted Values

Homoscedastic (GOOD):          Heteroscedastic (BAD):

e |  .   .  .                   e |              .  .
  |  . .   . .  .               e |          .  .
--+--.---.---.----.--           --+----.-.-------.---
  | . .  .  . .                   |   .  .
  |  .  .  .                      | .  .
  +-------------------            +-------------------
         ŷ                               ŷ

  Constant spread = GOOD        Fan/cone shape = BAD
```

### What to do if broken
- **Log-transform y** (often fixes fan-shaped residuals)
- **Weighted Least Squares** (give less weight to high-variance observations)
- **Use heteroscedasticity-robust standard errors** (HC standard errors / White's correction)

⚠️ **This is the assumption interviewers test most often** because the residual plot makes it visually obvious. Always mention checking for the "fan shape" or "megaphone pattern" in residual plots.

---

## Bonus: No Perfect Multicollinearity

### What it means
No feature can be a perfect linear combination of other features.

### Why it matters
If two features are perfectly correlated, (XᵀX) becomes singular (non-invertible). The Normal Equation literally can't be solved. Even near-perfect correlation causes **unstable, wildly fluctuating coefficients**.

### How to detect
- **VIF (Variance Inflation Factor)** — VIF > 10 is a red flag
- **Correlation matrix** — look for pairs with |r| > 0.9

### What to do
- **Drop one** of the correlated features
- **Combine them** (PCA or domain-knowledge-based combination)
- **Use regularization** (Ridge regression handles this gracefully)

We cover this more in `04_multiple_linear_regression.md`.

---

## The Master Diagnostic Checklist

When you build a linear regression model, check these in order:

```
1. Plot residuals vs. fitted values
   → Check for: patterns (non-linearity), fan shape (heteroscedasticity)

2. Q-Q plot of residuals
   → Check for: deviations from the diagonal (non-normality)

3. Plot residuals vs. each feature
   → Check for: non-linear patterns you might have missed

4. If time-series: plot residuals vs. order
   → Check for: autocorrelation (waves, trends)

5. Compute VIF for each feature
   → Check for: multicollinearity (VIF > 10)
```

---

## How Interviewers Test This Knowledge

**Question:** "You built a linear regression model and the R² is great. What do you check next?"

**Wrong answer:** "Nothing, it's working fine."

**Right answer:**
> "R² alone doesn't validate the model. I'd check the residual plots for linearity and homoscedasticity, a Q-Q plot for normality, and VIF scores for multicollinearity. A model can have high R² but violate assumptions — meaning the coefficients and confidence intervals are unreliable, even if predictions look decent."

---

## Key Takeaways

- **LINE**: Linearity, Independence, Normality, Equal variance
- **Residual plots are your best diagnostic tool** — always plot residuals vs. fitted values
- **Linearity** breaks → add polynomial terms or transform features
- **Independence** breaks → usually means time-series, use appropriate methods
- **Normality** matters for inference more than prediction — but still check
- **Homoscedasticity** breaks → log-transform y or use weighted least squares
- **Multicollinearity** → check VIF, drop or combine features, or use Ridge

⚠️ **The trap:** Don't just list assumptions. Explain what happens when they break and what you'd do about it. That's what separates a senior answer from a junior one.
