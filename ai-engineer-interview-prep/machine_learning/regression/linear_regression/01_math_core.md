# 01 — Math Core

> The goal here isn't to make you a mathematician.
> It's to make you **comfortable** with the math so you can explain it on a whiteboard without sweating.
> Words first, symbols second. Always.

---

## ŷ = Xβ — In Words First

Before the formula, the idea:

**"My prediction is a weighted sum of the features."**

For a single data point:
> ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ

- **β₀** is the intercept (prediction when all features are zero)
- **β₁, β₂, ...** are weights that say "for each unit increase in this feature, the prediction changes by this much"
- **x₁, x₂, ...** are the feature values

For ALL data points at once (matrix form):

```
ŷ = Xβ

where:
  X = [n × (p+1)] matrix    (n data points, p features + 1 column of ones for intercept)
  β = [(p+1) × 1] vector    (the weights we're solving for)
  ŷ = [n × 1] vector        (all predictions stacked up)
```

💡 **Think of it like this:** X is a big spreadsheet of your data. β is a recipe that says "multiply each column by this number and add them up." The result ŷ is your predictions.

---

## Residuals — "How Wrong We Are"

The residual vector:

```
e = y - ŷ = y - Xβ
```

- **e** is the vector of all mistakes
- Each element eᵢ = yᵢ - ŷᵢ (actual minus predicted)
- Positive eᵢ → we undershot
- Negative eᵢ → we overshot

**Our goal:** Find β that makes e as small as possible.

---

## Least Squares — "Punish Big Mistakes More"

We minimize the **sum of squared residuals** (also called RSS, SSE, or "the loss"):

```
L(β) = Σᵢ (yᵢ - ŷᵢ)² = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)
```

Why not just sum the errors?
- They'd cancel out (positive + negative = 0). Useless.

Why not absolute errors?
- Not differentiable at zero. Harder to optimize.

Squaring is the Goldilocks choice: it's smooth, differentiable, convex, and has a unique minimum.

---

## The Normal Equation

**If you forget everything else, remember this:**

```
β = (XᵀX)⁻¹ Xᵀy
```

### What it means in English

1. **Xᵀy** — how much each feature "agrees" with the target (dot products between feature columns and y)
2. **XᵀX** — how much the features agree with each other (their correlations and magnitudes)
3. **(XᵀX)⁻¹** — "undo" the feature correlations so each feature gets proper credit
4. **The whole thing** — give each feature the right weight after accounting for all the other features

### Where it comes from

Remember from the geometry file: residuals are orthogonal to the column space of X.

```
Xᵀe = 0
Xᵀ(y - Xβ) = 0
Xᵀy - XᵀXβ = 0
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy     ← Just solve for β
```

That's it. Four lines. The entire derivation is just writing down the orthogonality condition and solving.

### When it works and when it doesn't

| Works when... | Fails when... |
|---------------|---------------|
| XᵀX is invertible | Features are perfectly collinear (XᵀX is singular) |
| Dataset fits in memory | n or p is very large (matrix inversion is O(p³)) |
| Number of features < samples | p > n (more features than data points) |

⚠️ **Do NOT over-derive this in interviews.** Write the Normal Equation. Explain where it comes from (orthogonality). Move on. Nobody wants to watch you expand matrix transposes for 5 minutes.

---

## Gradient Descent — Why We Need It at Scale

The Normal Equation is elegant, but it requires **inverting a (p+1) × (p+1) matrix**, which is O(p³). When p is large (thousands of features), this is slow.

Enter **Gradient Descent**: an iterative approach that takes baby steps toward the minimum.

### The Idea

```
1. Start with random β
2. Compute the gradient (which direction is "downhill"?)
3. Take a step in that direction
4. Repeat until you converge
```

### The Update Rule

```
β := β - α · ∇L(β)

where:
  α = learning rate (step size)
  ∇L(β) = -2Xᵀ(y - Xβ) = gradient of the loss
```

### What the gradient means

- **∇L(β)** points in the direction of **steepest increase** in the loss
- We move in the **opposite direction** (that's the minus sign) to decrease the loss
- **α** controls how big each step is

### The Learning Rate Tradeoff

```
Too small α:                    Too large α:

L |                             L |
  |  \                            |  \      /\    /\
  |   \                           |   \    /  \  /
  |    \                          |    \  /    \/
  |     \                         |     \/
  |      \____                    |
  |           \_____              | (bouncing! never converges)
  +------------------ step        +------------------ step
   (works but painfully slow)
```

💡 **Interview tip:** If asked "Normal Equation vs Gradient Descent," say:
> "Normal Equation gives the exact answer in one step but costs O(p³). Gradient Descent is iterative but scales better to large p and large n, especially with stochastic variants. In practice, for small-to-medium problems I'd use the closed-form; for large-scale problems, SGD."

---

## Gradient Descent Variants (Brief)

| Variant | Uses | Tradeoff |
|---------|------|----------|
| **Batch GD** | All n data points per step | Stable but slow per iteration |
| **Stochastic GD (SGD)** | 1 random point per step | Noisy but fast, good for large n |
| **Mini-batch GD** | k random points per step | Best of both worlds, most common in practice |

⚠️ **Common interview trap:** "Does gradient descent always find the global minimum for linear regression?" **Yes** — the loss function is convex (bowl-shaped), so any local minimum is the global minimum. This is NOT true for neural networks.

---

## Putting It All Together

Here's the complete picture:

```
Start with data: X (features) and y (target)
                    |
                    v
           +-----------------+
           | Choose approach: |
           +-----------------+
            /              \
   Small p, data          Large p or n,
   fits in memory         streaming data
        |                      |
        v                      v
  Normal Equation         Gradient Descent
  β = (XᵀX)⁻¹Xᵀy        β := β - α·∇L(β)
        |                      |
        v                      v
  Exact answer            Approximate answer
  in one step             (converges to exact)
        |                      |
        +----------+-----------+
                   |
                   v
            ŷ = Xβ (predictions)
            e = y - ŷ (residuals)
```

---

## Key Takeaways

- **ŷ = Xβ** is just "prediction = weighted sum of features"
- **Residuals** are how wrong we are: e = y - ŷ
- **Least Squares** minimizes squared residuals — penalizes big errors more
- **Normal Equation** β = (XᵀX)⁻¹Xᵀy — comes from orthogonality, gives exact answer
- **Gradient Descent** — iterative alternative that scales to large problems
- For LR, GD always converges to the global minimum (convex loss)

⚠️ **Remember:** The Normal Equation assumes XᵀX is invertible. If features are perfectly collinear, it breaks. That's one reason regularization (Ridge) exists — it adds λI to XᵀX, making it always invertible.
