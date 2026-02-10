# 00 — Intuition & Decision Boundary

> Logistic Regression is NOT regression — it's a classifier wearing a regression costume.
> The name trips everyone up. Don't let it trip you.

---

## Why Does Logistic Regression Exist?

Imagine you have this data:

| Hours Studied | Passed Exam? |
|:---:|:---:|
| 1 | No (0) |
| 2 | No (0) |
| 3 | No (0) |
| 4 | Yes (1) |
| 5 | Yes (1) |
| 6 | Yes (1) |

You want to predict: **Given hours studied, will the student pass?**

**What if we try linear regression?**

```
    y
  1.5|                    /
    |                  /
  1 | - - - - - - -o--o--o     ← Predictions > 1 ???
    |            / |
 0.5|          /   |
    |        /     |
  0 | o--o--o      |
    |  /           |
-0.5| /            |            ← Predictions < 0 ???
    +--------------------------- x (hours)
      1   2   3   4   5   6
```

Linear regression gives predictions **above 1 and below 0**. But we need probabilities! Probabilities must be between 0 and 1.

💡 **This is why logistic regression exists:** we need a model that outputs something between 0 and 1, interprets it as a probability, and makes a binary decision from it.

---

## Probabilities, Not Predictions

Linear regression says: "The answer is 3.7."

Logistic regression says: **"The probability of 'yes' is 0.87."**

This is a fundamentally different mindset:
- We don't predict a number. We predict a **probability**.
- Then we apply a **threshold** (usually 0.5) to make a decision.

```
P(pass) ≥ 0.5  →  Predict: PASS
P(pass) < 0.5  →  Predict: FAIL
```

**If you forget everything else, remember this:** Logistic regression computes a linear score, converts it to a probability with the sigmoid function, and thresholds it to classify.

---

## The Sigmoid — A Soft Switch

The sigmoid function is the bridge between "linear score" and "probability":

```
σ(z) = 1 / (1 + e^(-z))
```

It takes ANY number (−∞ to +∞) and squashes it into (0, 1).

```
    P(y=1)
   1.0 |                         ___________
       |                       /
   0.8 |                     /
       |                   /
   0.6 |                 /
       |               /
   0.5 | - - - - - - * - - - - - - - (threshold)
       |           /
   0.4 |         /
       |       /
   0.2 |     /
       |   /
   0.0 |__/
       +----+----+----+----+----+----+----→ z
        -6  -4   -2    0   +2   +4  +6

    z < 0  →  P < 0.5  →  Predict class 0
    z = 0  →  P = 0.5  →  Decision boundary!
    z > 0  →  P > 0.5  →  Predict class 1
```

### Think of it like this...

**The sigmoid is a "confidence dial."**
- z = −10 → "I'm almost certain it's class 0" (P ≈ 0.00)
- z = 0 → "I genuinely have no idea" (P = 0.50)
- z = +10 → "I'm almost certain it's class 1" (P ≈ 1.00)

The further from zero the linear score z is, the more confident the model becomes. The sigmoid translates that confidence into a proper probability.

### Key properties to know for interviews

| Property | Value | Why It Matters |
|----------|-------|----------------|
| σ(0) | 0.5 | The decision boundary lives at z = 0 |
| Range | (0, 1) | Always a valid probability |
| Symmetric | σ(−z) = 1 − σ(z) | Symmetry around the midpoint |
| Derivative | σ(z) · (1 − σ(z)) | Beautifully simple — makes gradient computation easy |

---

## Decision Boundary — Where the Magic Happens

The **decision boundary** is the set of points where the model says: **"I'm exactly 50/50."**

That happens when σ(z) = 0.5, which happens when **z = 0**.

Since z = β₀ + β₁x₁ + β₂x₂ + ..., the decision boundary is:

```
β₀ + β₁x₁ + β₂x₂ + ... = 0
```

That's the equation of a **line** (in 2D), a **plane** (in 3D), or a **hyperplane** (in higher dimensions).

### 2D Example

Two features: x₁ (exam 1 score) and x₂ (exam 2 score).
Trained model: z = −6 + 1.0·x₁ + 1.0·x₂

Decision boundary: −6 + x₁ + x₂ = 0 → **x₂ = 6 − x₁**

```
    x₂ (exam 2 score)
    |
  8 | o  o     o               o = class 1 (passed)
    |    o  o   o              x = class 0 (failed)
  6 | o   o\ o
    |    x  \  o    ← Decision boundary: x₂ = 6 - x₁
  4 |  x  x  \
    |    x  x  \
  2 |  x   x    \
    | x  x
  0 +------------------------→ x₁ (exam 1 score)
    0  2  4  6  8
```

**Points above the line:** z > 0 → P > 0.5 → Predicted class 1
**Points below the line:** z < 0 → P < 0.5 → Predicted class 0
**Points ON the line:** z = 0 → P = 0.5 → Coin flip

### Why the boundary is always linear

Because z is a **linear function** of the features. Setting z = 0 gives a linear equation. No curves, no wiggles — just a straight cut through feature space.

⚠️ **Common interview trap:** "Can logistic regression create a curved decision boundary?" **Not with raw features.** But if you add polynomial features (x², x₁·x₂, etc.), the boundary becomes curved in the original feature space — though it's still linear in the expanded feature space. This is a key distinction.

---

## A Simple Walk-Through

Let's say our trained model is: z = −3 + 1.5·x

For a student who studied x = 4 hours:

```
Step 1: Compute linear score
   z = -3 + 1.5(4) = -3 + 6 = 3

Step 2: Apply sigmoid
   P(pass) = 1 / (1 + e^(-3)) = 1 / (1 + 0.05) ≈ 0.95

Step 3: Apply threshold
   0.95 ≥ 0.5 → Predict: PASS ✓
```

For a student who studied x = 1 hour:

```
Step 1: z = -3 + 1.5(1) = -1.5
Step 2: P(pass) = 1 / (1 + e^(1.5)) = 1 / (1 + 4.48) ≈ 0.18
Step 3: 0.18 < 0.5 → Predict: FAIL ✗
```

For a student who studied x = 2 hours:

```
Step 1: z = -3 + 1.5(2) = 0
Step 2: P(pass) = 1 / (1 + e^0) = 1 / 2 = 0.50
Step 3: 0.50 = threshold → This is the decision boundary!
```

💡 **The decision boundary is at x = 2.** That's where z = 0, P = 0.5, and the model switches from predicting "fail" to "pass."

---

## Linear Regression vs. Logistic Regression

This comes up in every interview. Know the table cold.

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Output** | Continuous number | Probability (0 to 1) |
| **Use case** | Predict a value | Classify into categories |
| **Function** | ŷ = Xβ | P(y=1) = σ(Xβ) |
| **Loss function** | MSE (squared errors) | Log loss (cross-entropy) |
| **Decision rule** | None (just output ŷ) | Threshold the probability |
| **Boundary** | Not applicable | Linear hyperplane |
| **Solution** | Closed-form (Normal Eq.) | Iterative (no closed form) |

---

## What Interviewers Expect You to Say

When asked **"Explain logistic regression"**, here's what separates good from great:

**Good answer:**
> "It's a classification algorithm that uses the sigmoid function to predict probabilities."

**Great answer:**
> "Logistic regression models the log-odds of the positive class as a linear function of the features. It computes a linear score, passes it through the sigmoid to get a calibrated probability, and classifies based on a threshold. The decision boundary is linear in feature space — it's the hyperplane where the model is exactly 50% confident. We train it by minimizing cross-entropy loss via gradient descent, since unlike linear regression, there's no closed-form solution."

**The difference:** The good answer describes what it does. The great answer explains **the full pipeline, the geometry, and the training**.

---

## Key Takeaways

- Logistic regression is a **classifier**, not a regression model (despite the name)
- It outputs a **probability** via the sigmoid function, then thresholds to classify
- The **sigmoid** squashes (−∞, +∞) into (0, 1) — it's the "confidence dial"
- The **decision boundary** is where P = 0.5, which is where z = 0 — always a linear surface
- You can get curved boundaries by adding **polynomial features**

⚠️ **Don't say:** "Logistic regression fits a line to the data." That's linear regression. Logistic regression fits a **probability curve** to the data and draws a **decision boundary** between classes.
