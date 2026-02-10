# 00 — Intuition & Geometry

> Linear Regression is just finding the best straight line... but let's really understand what that means.

---

## The Setup

You've got data. Some input features (X) and a target (y) you want to predict.

Linear regression says: **"I bet there's a roughly linear relationship here. Let me find the best line (or plane, or hyperplane) that describes it."**

That's it. That's the whole idea.

But what does "best" mean? That's where it gets interesting.

---

## Errors as Vertical Distances

Look at this scatter plot with a line through it:

```
    y
    |
  8 |           o
    |         /
  6 |    o  /  o
    |     /  |
  4 |   / o  |  <-- This vertical gap is the RESIDUAL
    |  /     |      (how wrong the model is for this point)
  2 | /  o
    |/
  0 +------------- x
    0  2  4  6  8
```

Each data point has a **residual** — the vertical distance between where the point actually is and where the line says it should be.

- Point above the line? **Positive residual** (we underestimated).
- Point below the line? **Negative residual** (we overestimated).

**If you forget everything else, remember this:** Linear regression picks the line that makes these vertical gaps as small as possible — specifically, it minimizes the **sum of squared residuals**.

---

## Why Squared? Why Not Just Sum the Errors?

Great question. Three reasons:

1. **Positive and negative errors would cancel out.** A line through the middle of your data with equal errors above and below would have a total error of zero. That's useless.

2. **Squaring punishes big mistakes more.** An error of 10 contributes 100 to the total, while an error of 1 contributes just 1. The model really doesn't want to be wildly wrong anywhere.

3. **It gives us a smooth, differentiable function.** This means we can use calculus to find the exact minimum. No sharp corners, no ambiguity. One clean answer.

💡 **"Why squared and not absolute value?"** is a classic interview question. The answer: absolute value has a corner at zero (not differentiable), which makes optimization harder. Squared gives a smooth, convex function with a unique global minimum.

---

## Why the "Best" Line is Special

The "best" line (in the least-squares sense) has a remarkable property:

**The residuals are uncorrelated with the predictions.**

Think about what this means: after the model has extracted all the linear signal it can, whatever's left (the residuals) has no linear pattern the model could still exploit. It's squeezed out every drop of linear information.

---

## The Geometry Intuition (Projection)

This is the part that separates "knows linear regression" from "deeply understands linear regression."

### Setup
- You have a target vector **y** (all your actual values, stacked up as a vector in n-dimensional space)
- You have feature vectors (columns of **X**) — each one is also a vector in n-dimensional space
- Linear regression asks: **"What linear combination of my feature vectors gets me as close to y as possible?"**

### The Answer: Projection

```
                      y (actual target)
                     /|
                    / |
                   /  |  <-- residual (e)
                  /   |      This is PERPENDICULAR to
                 /    |      the column space of X
                /     |
    ŷ ---------+      |
    (projection of y  |
     onto column
     space of X)

    ============ Column space of X ============
```

**ŷ is the projection of y onto the column space of X.**

The residual vector **e = y - ŷ** is **perpendicular** (orthogonal) to the column space. This is why:
- `X^T * e = 0` (the residuals are orthogonal to every feature column)
- This orthogonality condition directly gives us the **Normal Equation**

💡 **This is the deepest insight:** Linear regression is just an **orthogonal projection**. The residual is the part of y that lives outside the column space of X — the part that no linear combination of features can explain.

---

## Tiny 3-Point Example

Let's make this concrete. Three data points:

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |

We want: `ŷ = b₀ + b₁x`

**Step 1: What line minimizes the squared residuals?**

Using the formulas (we'll derive them in the next file):
- b₁ = (mean of x*y - mean(x)*mean(y)) / (mean of x² - mean(x)²)
- b₁ = (2+8+15)/3 - (2)(11/3)) / ((1+4+9)/3 - (2)²)
- b₁ = (25/3 - 22/3) / (14/3 - 4) = (3/3) / (2/3) = 1.5
- b₀ = mean(y) - b₁ * mean(x) = 11/3 - 1.5 * 2 = 11/3 - 3 = 2/3 ≈ 0.67

**Our line: ŷ = 0.67 + 1.5x**

**Step 2: Check the predictions and residuals.**

| x | y (actual) | ŷ (predicted) | residual (y - ŷ) |
|---|-----------|--------------|------------------|
| 1 | 2 | 2.17 | -0.17 |
| 2 | 4 | 3.67 | +0.33 |
| 3 | 5 | 5.17 | -0.17 |

Notice: residuals sum to approximately zero (they always do when there's an intercept). And they're small — no data point is wildly off.

```
    y
  6 |
  5 |              o (3,5)
    |            /
  4 |      o   / (2,4)
    |        /
  3 |      /
    |    /
  2 |  o (1,2)
    |  /
  1 | /
    |/
  0 +------------- x
    0  1  2  3  4
```

---

## The Column Space Picture (Interview Gold)

Here's what you're really doing in higher dimensions:

```
    Imagine all possible predictions you could make
    with linear combinations of your features.
    That's a SUBSPACE (a flat surface through the origin).

    Your actual target y probably doesn't live in that subspace.
    So you find the CLOSEST point in the subspace to y.
    That closest point is ŷ.
    The "error" is the perpendicular drop from y to ŷ.

    It's literally the same as dropping a perpendicular
    from a point to a plane in 3D geometry.
```

---

## What Interviewers Expect You to Say

When asked **"Explain linear regression"**, here's what separates good from great:

**Good answer:**
> "It fits a line by minimizing the sum of squared errors."

**Great answer:**
> "Linear regression finds the linear combination of features that's closest to the target vector in terms of Euclidean distance. Geometrically, it projects the target onto the column space of the feature matrix. The residuals are orthogonal to that column space, which is why X-transpose times the residual vector equals zero — and that orthogonality condition gives us the normal equation."

**The difference:** The good answer describes what it does. The great answer explains **why it works**.

---

## Key Takeaways

- Linear regression minimizes the **sum of squared residuals** (vertical distances)
- Squaring penalizes big errors more and gives a smooth optimization surface
- Geometrically, LR is **projecting y onto the column space of X**
- Residuals are **orthogonal** to the feature space — the model has extracted all linear signal
- The **Normal Equation** comes directly from this orthogonality condition

⚠️ **Common trap:** Don't say "it minimizes the distance between points and the line." Be specific — it minimizes the **vertical** distances (residuals), not the perpendicular distances (that's a different method called Total Least Squares).
