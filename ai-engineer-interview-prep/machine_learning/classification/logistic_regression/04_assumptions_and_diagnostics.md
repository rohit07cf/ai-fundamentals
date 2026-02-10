# 04 — Assumptions & Diagnostics

> "Logistic regression has assumptions? I thought it was non-parametric!"
> Nope. It's very much parametric. And knowing its assumptions is the difference between
> sounding competent and sounding experienced.

---

## The Big Three Assumptions

| # | Assumption | One-Liner |
|:-:|-----------|-----------|
| 1 | **Linearity in log-odds** | The log-odds of the outcome is a linear function of the features |
| 2 | **Independence of observations** | Each data point is independent of the others |
| 3 | **No extreme multicollinearity** | Features aren't highly correlated with each other |

Notice what's **NOT** on this list:
- ~~Normality of residuals~~ (that's linear regression)
- ~~Homoscedasticity~~ (that's linear regression)
- ~~Continuous target~~ (our target is binary — that's the whole point)

---

## 1. Linearity in Log-Odds (THE Important One)

### What it REALLY means

Logistic regression assumes that:

```
log(P / (1-P)) = β₀ + β₁x₁ + β₂x₂ + ...
```

The **log-odds** of the outcome have a **linear relationship** with each feature.

This does NOT mean the probability is linear. The S-curve is very much non-linear. But the **log-odds** are a straight line in feature space.

### Think of it like this

Imagine the effect of age on the probability of having high blood pressure:

```
GOOD (linear in log-odds):          BAD (non-linear in log-odds):

log-odds                              log-odds
    |      /                               |      _____
    |    /                                 |     /
    |  /                                   |    |
    |/                                     |   /
    +----------→ age                       | _/
                                           +----------→ age
    A straight line = assumption met.      A curve = assumption violated.
    Each year adds the same                The effect of age changes
    increment to the log-odds.             depending on how old you are.
```

### How it breaks in real life

- **Age and disease risk:** Risk might plateau or even decrease at very high ages (survivorship bias)
- **Income and loan default:** Low and high incomes might both have high default rates (U-shaped relationship)
- **Any non-monotonic relationship** between a feature and the outcome

### How to detect

1. **Box-Tidwell test** — tests for linearity in log-odds formally
2. **Binned plots:** Group a continuous feature into bins, compute the empirical log-odds for each bin, and plot. Should look roughly linear.

```
Empirical log-odds plot:

log-odds |        *
         |     *
         |   *         ← Roughly linear? Assumption met!
         | *
         |*
         +--------→ feature (binned)

log-odds |   *
         | *    *
         |*       *    ← Curved? Assumption violated!
         |         *
         |
         +--------→ feature (binned)
```

### What to do if broken

- **Transform the feature:** log(x), √x, x², etc.
- **Bin the feature:** Convert continuous to categorical (e.g., age groups)
- **Add polynomial terms:** x² or interaction terms
- **Use a different model:** Decision trees, GAMs, or neural networks don't require this assumption

💡 **Interview tip:** "Logistic regression assumes linearity in log-odds, not linearity in probability. I check this by plotting binned empirical log-odds against the feature. If the relationship is non-linear, I'll try feature transformations before reaching for a more complex model."

---

## 2. Independence of Observations

### What it means

Each row of data is independent. Knowing the outcome for one observation tells you nothing about another.

### How it breaks in real life

- **Repeated measurements:** Same patient measured at multiple time points
- **Clustered data:** Students within the same school, employees within the same company
- **Spatial/temporal data:** Nearby data points are correlated
- **Family data:** Siblings share genetics and environment

### How interviewers test this

They give you a scenario and ask: "What assumptions might be violated?"

**Example:** "We're predicting whether customers re-purchase, using data from 1,000 customers over 12 months, with one row per month."

**The answer they want:** "Independence is violated — each customer has multiple observations, so their rows are correlated. I'd use a mixed-effects model, GEE, or at minimum cluster-robust standard errors."

### What to do if broken

- **Mixed-effects logistic regression** (accounts for within-group correlation)
- **GEE (Generalized Estimating Equations)** — for population-level inference
- **Cluster-robust standard errors** — quick fix for standard errors
- **Aggregate** to one row per subject if possible

---

## 3. No Extreme Multicollinearity

### What it means

Features shouldn't be highly correlated with each other. The model can't figure out which correlated feature deserves the credit.

### How it breaks

Exactly the same as linear regression:
- **Unstable coefficients** that flip sign between runs
- **Inflated standard errors** — features appear "not significant" even when they are
- **Difficulty interpreting individual coefficients**

### How to detect

- **VIF (Variance Inflation Factor)** — VIF > 10 is a red flag
- **Correlation matrix** — check for |r| > 0.8 between feature pairs

### What to do

- **Drop one** of the correlated features
- **Combine features** (PCA, domain-knowledge averages)
- **Regularize** (L1/L2 — covered in the regularization file)

⚠️ **Important nuance:** Multicollinearity hurts **interpretation** (coefficients) more than **prediction** (class labels). If you only care about prediction accuracy, moderate multicollinearity is often fine.

---

## What Logistic Regression Does NOT Assume

This is just as important as knowing the assumptions:

| Not Required | Why People Think It Is |
|:---:|---|
| **Normal features** | Confusion with LDA or linear regression. LogReg uses the Bernoulli distribution. |
| **Normal residuals** | Residuals in logistic regression are binary (0 or 1 minus probability) — normality doesn't apply. |
| **Equal variance** | That's a linear regression thing. Binary outcomes don't have constant variance. |
| **Linear relationship between X and y** | The relationship is linear in **log-odds**, not in probability. The probability curve is S-shaped. |

💡 **Aha moment:** Logistic regression is actually a **generalized linear model** (GLM) with a Bernoulli distribution and logit link. It has fewer assumptions than linear regression because it doesn't require normality or constant variance of residuals.

---

## Sample Size Considerations

While not a formal "assumption," logistic regression needs enough data:

### Rule of Thumb: Events Per Variable (EPV)

- You need at least **10-20 events (positives) per feature**
- If you have 50 fraud cases and 200 features, your model is in trouble

```
   Features:  200
   Events:     50
   EPV:        50/200 = 0.25    ← Way too low!

   What happens: overfitting, unstable coefficients, perfect separation
```

### Perfect Separation (The Nightmare Scenario)

When a feature perfectly predicts the outcome, the coefficient goes to ±∞. The model "breaks" — it tries to make the sigmoid exactly 0 or 1.

**Example:** If every person with income > $100K got the loan and every person below didn't:
```
The model tries: β_income → ∞
The optimizer never converges.
```

**Fix:** Add regularization (L1 or L2). This constrains coefficients and prevents them from exploding.

---

## The Diagnostic Checklist

When you build a logistic regression model, check these:

```
1. Check class balance
   → Severe imbalance? Consider resampling, class weights, or a different metric.

2. Check linearity in log-odds
   → For each continuous feature, plot binned log-odds.
   → Non-linear? Transform the feature.

3. Check for multicollinearity
   → Compute VIF for each feature.
   → VIF > 10? Drop or combine features.

4. Check for independence
   → Is the data clustered or repeated? Use appropriate methods.

5. Check sample size
   → At least 10-20 events per feature.
   → Perfect separation? Add regularization.

6. Evaluate calibration
   → Are predicted probabilities well-calibrated?
   → Calibration plot (predicted P vs. observed frequency) should follow the diagonal.
```

---

## Calibration — The Often-Forgotten Check

A model can have great accuracy but terrible calibration:

```
Well-calibrated:                  Poorly calibrated:

Actual rate                       Actual rate
  1.0 |         /                  1.0 |              /
      |       /                        |            /
  0.5 |     /                    0.5 |     ----
      |   /                          |   /
  0.0 | /                        0.0 | /
      +----------→ Predicted P        +----------→ Predicted P

  Points on diagonal = GOOD       Curve = probabilities are off
```

**Calibration means:** When the model says "70% chance," it should be right about 70% of the time.

Logistic regression is generally **well-calibrated** out of the box (unlike random forests or SVMs). This is actually one of its strengths.

---

## Key Takeaways

- **Three assumptions:** Linearity in log-odds, independence, no extreme multicollinearity
- The linearity assumption is about **log-odds**, NOT about probability
- LogReg does NOT assume normal features, normal residuals, or homoscedasticity
- **Check VIF** for multicollinearity, **binned plots** for log-odds linearity
- Need at least **10-20 events per variable** to avoid instability
- **Perfect separation** → coefficients explode → add regularization
- Logistic regression is naturally **well-calibrated** — a real advantage

⚠️ **The question interviewers love:** "What are the assumptions of logistic regression?" Don't recite linear regression assumptions. Say: "Linearity in the log-odds — not in probability — independence of observations, and no extreme multicollinearity. It does NOT require normality or homoscedasticity."
