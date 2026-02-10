# 04 — Multiple Linear Regression

> Single-feature LR is cute. Real-world LR has dozens or hundreds of features.
> This is where things get interesting — and where interviewers start separating the strong from the shaky.

---

## "Holding Other Variables Constant"

This is the single most important phrase in multiple regression. Tattoo it on your brain.

**Each coefficient βⱼ tells you:**
> "If I increase feature xⱼ by one unit **while keeping ALL other features fixed**, the predicted y changes by βⱼ."

This is called the **ceteris paribus** interpretation (fancy Latin for "all else equal").

### Why this matters

In simple regression (one feature), the coefficient captures the TOTAL effect of that feature. In multiple regression, it captures the **partial** effect — what's left after accounting for all other features.

**Example:**
- Simple regression: salary ~ years_experience → β₁ = $8,000/year
- Multiple regression: salary ~ years_experience + education_level → β₁ = $5,000/year

The coefficient **dropped** because some of what we attributed to experience was actually explained by education. This is not a bug — it's the whole point of multiple regression.

💡 **Think of it like this:** Each feature gets credit only for the variance it **uniquely** explains, after the other features have had their say.

---

## Coefficient Interpretation — A Simple Example

**Model:** `house_price = 50,000 + 200 × sqft + 30,000 × bedrooms + 15,000 × has_garage`

| Coefficient | Interpretation |
|-------------|---------------|
| **50,000** (intercept) | Baseline price of a house with 0 sqft, 0 bedrooms, no garage (not realistic, but that's what it means mathematically) |
| **200** (sqft) | Each additional sq ft adds $200 to the price, **holding bedrooms and garage constant** |
| **30,000** (bedrooms) | Each additional bedroom adds $30K, **holding sqft and garage constant** |
| **15,000** (has_garage) | Having a garage adds $15K, **holding sqft and bedrooms constant** |

### Reading a prediction

A 1,500 sqft, 3-bedroom house with garage:
```
price = 50,000 + (200 × 1,500) + (30,000 × 3) + (15,000 × 1)
      = 50,000 + 300,000 + 90,000 + 15,000
      = $455,000
```

⚠️ **Common trap:** Don't say "200 means sqft is the most important feature." The coefficients depend on the **scale** of the features. sqft ranges in thousands while bedrooms ranges in single digits. You can't compare raw coefficients for importance without standardizing first.

---

## Feature Scaling — Why Gradient Descent Behaves Badly Otherwise

### The Problem

Imagine two features:
- `sqft`: ranges from 500 to 5,000
- `bedrooms`: ranges from 1 to 6

The loss surface looks like a **long, narrow valley**:

```
Without scaling:                With scaling:

β_bedrooms                      β_bedrooms
    |   /////                       |   ..
    |  /////                        |  ....
    | /////   ← GD zigzags!        | ......  ← GD goes straight!
    |/////                          |  ....
    +------------- β_sqft          +------------- β_sqft

Contours are elongated ellipses   Contours are circular
(different scales = different      (same scale = symmetric)
 curvature in each direction)
```

Gradient descent takes **zigzagging steps** in the elongated case because the gradient points toward the steepest direction, which isn't toward the minimum.

### The Fix: Standardization

```
x_standardized = (x - mean(x)) / std(x)
```

After standardization, every feature has:
- Mean = 0
- Standard deviation = 1

Now the loss surface is nicely circular, and GD converges much faster.

### Does the Normal Equation need scaling?

**No.** The Normal Equation gives the exact answer regardless of scaling. Scaling only matters for iterative methods like gradient descent.

But: scaling makes coefficients **comparable**. After standardization, a coefficient of 0.5 vs. 0.1 actually means the first feature is more important.

💡 **Interview tip:** "I always scale features before gradient descent for faster convergence. For the normal equation, scaling isn't required but helps with interpretability."

---

## One-Hot Encoding (Intuition Only)

When you have a **categorical feature** (like `city = {NYC, LA, Chicago}`), you can't feed it directly into a linear model. Models need numbers.

**One-hot encoding** creates a binary column for each category:

```
Original:           One-hot encoded:
| city    |         | is_NYC | is_LA | is_Chicago |
|---------|         |--------|-------|------------|
| NYC     |   →     |   1    |   0   |     0      |
| LA      |   →     |   0    |   1   |     0      |
| Chicago |   →     |   0    |   0   |     1      |
| NYC     |   →     |   1    |   0   |     0      |
```

### The dummy variable trap

If you include ALL categories, they're perfectly collinear (is_NYC + is_LA + is_Chicago = 1 always). So you **drop one category** (the "reference" category).

```
After dropping Chicago (reference):
| is_NYC | is_LA |
|--------|-------|
|   1    |   0   |  ← NYC
|   0    |   1   |  ← LA
|   0    |   0   |  ← Chicago (represented by both being 0)
```

Now:
- **β_NYC** = price difference between NYC and Chicago
- **β_LA** = price difference between LA and Chicago
- The intercept absorbs Chicago's baseline effect

⚠️ **Interview trap:** "Why do you drop one dummy variable?" Because including all of them creates perfect multicollinearity. The last category is implicitly represented when all other dummies are 0. Most libraries (sklearn) handle this automatically, but interviewers want to know you understand why.

---

## Multicollinearity — Explained Like a Human Problem

### The analogy

Imagine you're trying to figure out how much two employees contribute to a project, but they ALWAYS work together. They arrive together, leave together, and every task they complete, they do as a pair.

**Can you figure out who's contributing what?** No. Because their contributions are perfectly **confounded**. That's multicollinearity.

### What happens mathematically

When two features are highly correlated:
- The model can't tell which one deserves the credit
- **Coefficients become unstable** — small changes in data cause huge swings in β values
- **Standard errors inflate** — you lose statistical significance
- (XᵀX) becomes nearly singular — close to non-invertible

### Example

Feature A (sqft) and Feature B (sqft_in_meters) are perfectly correlated (one is just 0.093 × the other).

The model might give:
- Run 1: β_sqft = +500, β_sqft_meters = -300
- Run 2: β_sqft = -200, β_sqft_meters = +800

The PREDICTIONS might be fine (the effects cancel out), but the individual coefficients are meaningless.

### Detection: VIF

**Variance Inflation Factor** for feature j:

```
VIF_j = 1 / (1 - R²_j)

where R²_j is the R² from regressing feature j on all OTHER features
```

| VIF | Interpretation |
|-----|---------------|
| 1 | No correlation with other features |
| 1–5 | Moderate, usually OK |
| 5–10 | Getting concerning |
| > 10 | Serious multicollinearity — take action |

### What to do

1. **Drop one** of the correlated features (simplest)
2. **Combine them** into a single feature (domain knowledge or PCA)
3. **Use Ridge regression** — it handles multicollinearity by shrinking coefficients (see regularization preview)

💡 **The key insight:** Multicollinearity doesn't hurt **predictions** (ŷ is still fine). It hurts **interpretation** (individual coefficients are unreliable). So if you only care about prediction, it's less of a problem. But in interviews, always mention you'd check for it.

---

## Feature Importance in Multiple Regression

**Raw coefficients are NOT a measure of importance** unless features are standardized.

| Method | What It Does | Limitation |
|--------|-------------|------------|
| **Standardized coefficients** | Scale features first, then compare coefficients | Assumes linear, additive effects |
| **Drop-one-out** | Remove each feature, see how R² changes | Slow, doesn't capture interactions |
| **Permutation importance** | Shuffle one feature, measure performance drop | More robust, model-agnostic |

For an interview, standardized coefficients are the go-to answer for linear models.

---

## Key Takeaways

- Each coefficient = **partial effect** holding other features constant
- **Scale features** before GD (not needed for Normal Equation, but helps interpretability)
- **One-hot encode** categorical features; drop one category to avoid the dummy trap
- **Multicollinearity** makes coefficients unstable but doesn't hurt predictions
- Check **VIF** > 10 as a red flag
- Don't compare raw coefficients for importance — **standardize first**

⚠️ **Remember:** When an interviewer asks "interpret this coefficient," always include "holding all other variables constant." It's the magic phrase that shows you understand partial effects.
