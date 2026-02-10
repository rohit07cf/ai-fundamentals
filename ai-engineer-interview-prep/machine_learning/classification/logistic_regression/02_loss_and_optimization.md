# 02 — Loss & Optimization

> This is the file where you learn WHY we train logistic regression the way we do.
> The sigmoid is the model. The loss function is the teacher. Gradient descent is the student doing homework.

---

## Why MSE Is WRONG for Classification

This is a question interviewers love because it tests whether you think or just memorize.

**What happens if we use MSE (Mean Squared Error) for logistic regression?**

```
MSE Loss = Σ (yᵢ - σ(zᵢ))²
```

Two problems:

### Problem 1: Non-Convex Loss Surface

When you plug the sigmoid into MSE, the resulting loss function has **multiple local minima**. Gradient descent might get stuck in a bad spot and never find the best answer.

```
MSE + sigmoid loss surface:

    Loss |
         |  \     /\
         |   \   /  \      ← Multiple valleys!
         |    \ /    \        Gradient descent might
         |     v      \       get stuck here instead
         |              \     of finding the true minimum
         |               \___
         +--------------------→ β

    NOT convex. Bad for optimization.
```

### Problem 2: Saturated Gradients

When the sigmoid outputs something close to 0 or 1, the MSE gradient becomes **tiny**. The model learns extremely slowly when it's very wrong — exactly when it should be learning fastest.

**Think of it like this:** If the model confidently predicts 0.99 but the true label is 0, MSE barely punishes it (gradient is small). That's backwards.

💡 **This is the sentence interviewers want to hear:**
> "MSE creates a non-convex loss surface when combined with the sigmoid, leading to local minima. Cross-entropy loss is convex for logistic regression, guaranteeing a global optimum, and it also penalizes confident wrong predictions much more heavily."

---

## Log Loss (Cross-Entropy) — The Right Choice

### The Intuition

The loss should say:

- **"If the true label is 1 and you predicted P close to 1 → small loss (good job)"**
- **"If the true label is 1 and you predicted P close to 0 → HUGE loss (terrible mistake)"**
- Same logic for true label 0, in reverse.

We need a loss that **explodes** when the model is confidently wrong.

### The Formula

```
For a single data point:

L = -[y · log(p) + (1-y) · log(1-p)]

where:
  y = true label (0 or 1)
  p = predicted probability P(y=1)
```

For the whole dataset:

```
L = -(1/n) Σᵢ [yᵢ · log(pᵢ) + (1-yᵢ) · log(1-pᵢ)]
```

### Why This Works — Breaking It Down

**When y = 1** (true positive case):
```
L = -log(p)

p = 0.99  →  L = -log(0.99) = 0.01    (almost no loss — great!)
p = 0.50  →  L = -log(0.50) = 0.69    (moderate loss — uncertain)
p = 0.01  →  L = -log(0.01) = 4.61    (MASSIVE loss — you were very wrong)
```

**When y = 0** (true negative case):
```
L = -log(1-p)

p = 0.01  →  L = -log(0.99) = 0.01    (almost no loss — great!)
p = 0.50  →  L = -log(0.50) = 0.69    (moderate loss — uncertain)
p = 0.99  →  L = -log(0.01) = 4.61    (MASSIVE loss — you were very wrong)
```

See the pattern? **The loss rockets to infinity as the prediction approaches the wrong answer.** The model is severely punished for being confidently wrong.

```
    Loss
    |
  5 |  \                          /
    |   \                        /
  4 |    \                      /
    |     \                    /
  3 |      \                  /
    |       \                /
  2 |        \              /
    |         \            /
  1 |          \          /
    |           \________/
  0 +-----|-----|-----|-----→ Predicted P
    0   0.25  0.5  0.75   1

    Left curve:  -log(p)    loss when y=1
    Right curve: -log(1-p)  loss when y=0
```

---

## The Likelihood View (High-Level)

If someone asks "where does log loss come from?", the answer is **maximum likelihood estimation**.

**The idea:** We want to find β that makes the observed data **most probable**.

```
For each data point:
  If yᵢ = 1:  the likelihood of seeing this is pᵢ
  If yᵢ = 0:  the likelihood of seeing this is (1 - pᵢ)

Combined (using a neat trick):
  P(yᵢ | xᵢ) = pᵢ^yᵢ · (1-pᵢ)^(1-yᵢ)

For all data points (assuming independence):
  L(β) = Π pᵢ^yᵢ · (1-pᵢ)^(1-yᵢ)
```

Taking the log (because products are hard, sums are easy):

```
log L(β) = Σ [yᵢ · log(pᵢ) + (1-yᵢ) · log(1-pᵢ)]
```

**Maximizing** log-likelihood = **minimizing** negative log-likelihood = **minimizing cross-entropy loss**.

They're the same thing. Three names for one idea.

💡 **Aha moment:** Log loss isn't some arbitrary choice. It falls out naturally from asking: "What parameters make my observed data most likely?" That's maximum likelihood estimation — one of the most fundamental ideas in statistics.

---

## Why the Loss Is Convex (Interview Gold)

When you combine the sigmoid with log loss, the resulting optimization problem is **convex**:

```
Cross-entropy + sigmoid loss surface:

    Loss |
         |\
         | \
         |  \
         |   \
         |    \___
         |        \___
         |            \____
         +-------------------→ β

    Convex! One global minimum. Gradient descent guaranteed to find it.
```

This is NOT obvious and it's a common interview question:

⚠️ **"Is the loss function for logistic regression convex?"** — **Yes.** Binary cross-entropy loss with the sigmoid is convex with respect to the model parameters. This guarantees gradient descent finds the global optimum.

---

## Gradient Descent for Logistic Regression

Since there's no closed-form solution (no "Normal Equation" for logistic regression), we use gradient descent.

### The Gradient

The gradient of the log loss with respect to β is:

```
∇L = (1/n) · Xᵀ(σ(Xβ) - y)

Or per parameter:
∂L/∂βⱼ = (1/n) Σᵢ (pᵢ - yᵢ) · xᵢⱼ
```

**In words:** For each parameter, the gradient is the average of (prediction error) × (feature value).

💡 **Notice something beautiful?** This looks almost identical to the gradient for linear regression: (1/n) · Xᵀ(Xβ - y). The only difference is that Xβ is replaced by σ(Xβ). Same structure, different model.

### The Update Rule

```
β := β - α · (1/n) · Xᵀ(σ(Xβ) - y)

where α = learning rate
```

### Step by Step

```
1. Initialize β (zeros or small random values)
2. Compute predictions: p = σ(Xβ)
3. Compute errors: e = p - y
4. Compute gradient: g = (1/n) · Xᵀe
5. Update: β = β - α · g
6. Repeat until convergence
```

### Intuitive Explanation of the Gradient

For a single data point where y = 1 and p = 0.3 (model is underconfident):
- Error = 0.3 - 1 = -0.7 (negative → model needs to increase prediction)
- If xⱼ = 2, the gradient for βⱼ is -0.7 × 2 = -1.4
- Update: βⱼ increases (we subtract a negative → moves up)
- Result: next time, z will be larger → σ(z) will be larger → closer to correct!

The model adjusts each weight proportional to:
1. **How wrong it was** (p - y)
2. **How much that feature was "on"** (xⱼ)

Features that were active when the model was wrong get the biggest corrections.

---

## Why No Closed-Form Solution?

In linear regression, we set the gradient to zero and solve: β = (XᵀX)⁻¹Xᵀy.

In logistic regression, the gradient involves σ(Xβ), which is **non-linear** in β:

```
∇L = (1/n) · Xᵀ(σ(Xβ) - y) = 0

Can't isolate β algebraically because σ is non-linear!
```

So we **must** iterate. But the convexity guarantee means we'll get to the global minimum regardless of where we start. It just takes some patience.

---

## Quick Reference: Loss Functions Compared

| Loss | Formula | Used For | Convex for LR? |
|------|---------|----------|:-:|
| **MSE** | (y - p)² | Regression | No (with sigmoid) |
| **Log Loss** | -[y·log(p) + (1-y)·log(1-p)] | Classification | **Yes** |
| **Hinge Loss** | max(0, 1 - y·z) | SVM | Yes |

---

## Key Takeaways

- **MSE is wrong** for logistic regression: non-convex loss surface + saturated gradients
- **Log loss (cross-entropy)** is the right choice: convex, heavily penalizes confident mistakes
- Log loss comes from **maximum likelihood estimation** — it's principled, not arbitrary
- **No closed-form solution** — we use gradient descent (convex → guaranteed to converge)
- The gradient has a beautiful form: **(1/n) · Xᵀ(predictions - labels)**
- Each weight update is proportional to (how wrong) × (how much that feature was on)

⚠️ **Interview trap:** "What loss function does logistic regression use?" Don't just say "cross-entropy." Say: "Binary cross-entropy, also called log loss. It comes from maximum likelihood estimation and is convex for logistic regression, which guarantees a unique global optimum."
