# 01 — Math Core

> The goal here isn't to drown you in symbols.
> It's to make you **comfortable** with the math pipeline so you can whiteboard it without hesitation.
> Words first. Symbols second. Always.

---

## The Full Pipeline (Words First)

Here's everything logistic regression does, in plain English:

```
1. Take the features         →  "Here's what I know about this data point"
2. Compute a linear score    →  "Weight each feature, add them up"
3. Squeeze through sigmoid   →  "Convert that score into a probability"
4. Apply a threshold         →  "Make a yes/no decision"
```

That's the entire model. Four steps. Now let's add the math.

---

## Step 1: The Linear Combination — z = Xβ

**In words:** "Multiply each feature by its weight, add them up, include a bias term."

**In math:**

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ

Or in matrix form: z = Xβ
```

- **β₀** — the intercept (bias). Shifts the decision boundary left/right.
- **β₁, β₂, ...** — feature weights. How much each feature contributes to the score.
- **z** — the "log-odds" or "logit." Can be any real number.

This is identical to linear regression so far. The magic happens next.

---

## Step 2: The Sigmoid — σ(z) → Probability

**In words:** "Take that score and squeeze it into a probability between 0 and 1."

**In math:**

```
P(y = 1 | X) = σ(z) = 1 / (1 + e^(-z))
```

This function has the perfect properties for probability:
- **Always between 0 and 1** ✓
- **Monotonically increasing** — higher z = higher probability ✓
- **Smooth and differentiable** — gradient descent loves it ✓
- **σ(0) = 0.5** — natural midpoint ✓

💡 **Think of it like this:** z is the model's "raw opinion" (can be any number). The sigmoid is the translator that converts that raw opinion into a calibrated bet.

---

## Step 3: Why the Output Is Between 0 and 1

Let's trace through the math:

```
When z → +∞:  e^(-z) → 0    so σ(z) → 1/(1+0) = 1
When z → -∞:  e^(-z) → ∞    so σ(z) → 1/(1+∞) = 0
When z = 0:   e^0 = 1        so σ(z) → 1/(1+1) = 0.5
```

The sigmoid asymptotically approaches 0 and 1 but never actually reaches them. It's always a **probability**, never a certainty.

---

## The Log-Odds Connection (Interview Gold)

This is the part most people skip but interviewers love to probe.

The **odds** of class 1 are:

```
odds = P(y=1) / P(y=0) = P / (1 - P)
```

The **log-odds** (or **logit**) are:

```
log(P / (1 - P)) = z = β₀ + β₁x₁ + β₂x₂ + ...
```

**What this means:** Logistic regression assumes the **log-odds** of the positive class are a **linear function** of the features.

This is the fundamental assumption. Not that P is linear in X (it's not — it's an S-curve). But that **log(P/(1−P))** is linear in X.

### Coefficient Interpretation Through Odds

A coefficient of β₁ = 0.7 means:

> "For every 1-unit increase in x₁, the **log-odds** increase by 0.7, which means the **odds are multiplied by e^0.7 ≈ 2.01**."

In other words, the odds roughly **double** for each unit increase.

| Coefficient | Effect on Log-Odds | Effect on Odds |
|:-----------:|:-----------------:|:--------------:|
| β = 0 | No effect | Odds unchanged (× 1) |
| β = 0.5 | +0.5 | Odds × 1.65 |
| β = 1.0 | +1.0 | Odds × 2.72 |
| β = −1.0 | −1.0 | Odds × 0.37 (reduced) |

⚠️ **Do NOT confuse this with linear regression:** In linear regression, β₁ = 0.7 means "y increases by 0.7." In logistic regression, β₁ = 0.7 means "log-odds increase by 0.7" — the effect on probability is **non-linear** and depends on where you are on the sigmoid curve.

---

## The Full Mathematical Model

Putting it all together:

```
                    ┌─────────────┐
   Features         │  Linear     │         ┌──────────┐
   x₁, x₂, ...  →  │  Combination│  → z →  │ Sigmoid  │  → P(y=1)  → threshold → ŷ
                    │  z = Xβ     │         │ σ(z)     │
                    └─────────────┘         └──────────┘

   This part is               This part converts      This part
   EXACTLY linear              to probability          makes the
   regression                                          decision
```

### Writing it compactly

```
P(y = 1 | X) = σ(Xβ) = 1 / (1 + e^(-Xβ))

Decision: ŷ = 1  if  P ≥ 0.5  (i.e., z ≥ 0)
          ŷ = 0  if  P < 0.5  (i.e., z < 0)
```

---

## The Derivative of the Sigmoid (Why It's Beautiful)

You don't need to memorize the derivation, but know the result:

```
dσ/dz = σ(z) · (1 - σ(z))
```

**Why this matters:**
- It makes gradient computation elegant
- The derivative is highest at z = 0 (steepest part of the S-curve) — the model learns fastest near the decision boundary
- The derivative approaches 0 at extreme z values — this is the **vanishing gradient** problem in deep learning (a connection interviewers might probe)

```
   Sigmoid:                    Its derivative:
   1.0 |        ___            0.25|
       |      /                    |     .
   0.5 |    /                  0.12|   .   .
       |  /                        | .       .
   0.0 |/                     0.0 |.           .
       +----------→ z              +-----------→ z

   S-shaped                    Bell-shaped (peaks at z=0)
```

💡 **Aha moment:** The derivative is maximized at the decision boundary (z = 0). This means the model adjusts its weights most aggressively for data points it's least sure about. Intuitively, it focuses its learning effort where it matters most.

---

## Quick Numerical Walk-Through

Model: z = −2 + 3x₁ + (−1)x₂

| x₁ | x₂ | z | σ(z) | Prediction |
|:---:|:---:|:--:|:----:|:----------:|
| 0 | 0 | −2 | 0.12 | Class 0 |
| 1 | 0 | +1 | 0.73 | Class 1 |
| 1 | 1 | 0 | 0.50 | Boundary! |
| 2 | 1 | +3 | 0.95 | Class 1 (confident) |
| 0 | 3 | −5 | 0.01 | Class 0 (very confident) |

Decision boundary: −2 + 3x₁ − x₂ = 0 → **x₂ = 3x₁ − 2**

---

## What You Need on the Whiteboard

If asked to write the logistic regression model, write exactly this:

```
1.  z = β₀ + β₁x₁ + ... + βₚxₚ         (linear score)
2.  P(y=1|X) = 1 / (1 + e^(-z))          (sigmoid → probability)
3.  log(P/(1-P)) = z                      (log-odds are linear!)
4.  Loss: L = -Σ [yᵢ log(pᵢ) + (1-yᵢ)log(1-pᵢ)]   (cross-entropy)
5.  Optimize with gradient descent         (no closed-form solution)
```

That's five lines. That's the whole algorithm on a whiteboard. Clean, complete, impressive.

---

## Key Takeaways

- **z = Xβ** computes a linear score (same as linear regression)
- **σ(z)** converts the score to a probability between 0 and 1
- The model assumes **log-odds are linear** in the features
- Coefficients affect **odds multiplicatively** (e^β), not probabilities additively
- Sigmoid derivative = σ(z)(1 − σ(z)) — peaks at the decision boundary
- **No closed-form solution** — we must use iterative optimization

⚠️ **The biggest interview mistake:** Saying "logistic regression predicts a line." It doesn't. It predicts an **S-curve** of probabilities. The decision boundary is a line, but the output function is non-linear.
