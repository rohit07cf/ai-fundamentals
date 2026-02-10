# Diagram: Bias-Variance Tradeoff for Logistic Regression

## The Classic Curve

```
    Error
    ^
    |
    | \                                    ___________
    |  \  Total Error                  ___/
    |   \  (what we                ___/
    |    \  actually see)      ___/
    |     \                ___/
    |      \           ___/
    |       \      ___/
    |        \ ___/
    |         X  ← SWEET SPOT (minimum total error)
    |        /\___
    |       /     \___
    |      /          \___________  Bias²
    |     /
    |    / Variance
    |   /
    |  /
    | /
    |/___________________________________
    +---------------------------------------------> Model Complexity
    Simple                                 Complex
    (high regularization,                  (no regularization,
     few features)                          many features)
```

## What This Means for Logistic Regression

```
HIGH REGULARIZATION (C small)          LOW REGULARIZATION (C large)
━━━━━━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    x₂                                     x₂
     |  o o x x x                            |  o o     x x
     |  o o x x x                            |  o o\    x x
     |  o o x x x                            |  o o  \  x x
     |  o o | x x   ← almost vertical        |  o o    \x x   ← captures the
     |  o o x x x     too simple!            |  o o  x  \x x    true pattern
     +──────────→ x₁                         +───────────→ x₁

    HIGH BIAS:                              LOW BIAS:
    - Boundary is too rigid                 - Boundary fits the data well
    - Misses the true pattern               - Captures the true pattern
    - Stable across datasets                - May wobble with different data

    LOW VARIANCE:                           HIGH VARIANCE:
    - Same boundary every time              - Different data → different boundary
    - Doesn't adapt to noise                - Overly sensitive to individual points
```

## The Regularization Knob

```
    C = 0.001          C = 0.1            C = 1              C = 1000
    (max reg)          (strong reg)       (default)          (no reg)

     x₂                x₂                x₂                  x₂
     | o o|x x         | o o |x x        | o o \x x         | o o  \  x
     | o o|x x         | o o |x x        | o o  \x x        | o o  |\ x
     | o o|x x         | o o  |x x       | o o   \x x       | o o  |  \x
     | o o|x x         | o o  | x x      | o o    \x x      | o   /|  x
     +─────→ x₁        +──────→ x₁       +────────→ x₁      +──/───→ x₁

     Nearly vertical   Still rigid        Balanced            Wiggly (overfitting
     (underfitting)    but better         (sweet spot)        to noise)

     High bias         ←──── Bias decreases, Variance increases ────→
     Low variance      ←──── Moving right = more complex model ─────→
```

## The Dartboard Analogy (For Classification)

```
    Low Bias, Low Variance:     Low Bias, High Variance:
    (What we want)              (Overfitting)

    Boundary close to truth     Boundary close to truth ON AVERAGE
    and STABLE across datasets. but wiggles differently each time.

    Dataset 1: ─── \            Dataset 1: ─── \
    Dataset 2: ─── \            Dataset 2: ──── /\
    Dataset 3: ─── \            Dataset 3: ── \  /
    (same every time)           (different every time!)


    High Bias, Low Variance:    High Bias, High Variance:
    (Underfitting)              (Worst case)

    Boundary consistently       Boundary is wrong AND unstable.
    in the wrong place.

    Dataset 1: ── |             Dataset 1: ── / ←wrong
    Dataset 2: ── |             Dataset 2: ── | ←wrong differently
    Dataset 3: ── |             Dataset 3: ── \ ←wrong again
    (always wrong, same way)    (always wrong, different ways)
```

## How to Navigate the Tradeoff in Practice

```
    Step 1: Start with moderate regularization (C=1, default)
              |
              v
    Step 2: Check training vs. validation performance
              |
              ├─── Training HIGH, Validation LOW?
              │    → Overfitting! Decrease C (more regularization)
              │    → Or reduce features
              │
              ├─── Training LOW, Validation LOW?
              │    → Underfitting! Increase C (less regularization)
              │    → Or add features / polynomial terms
              │
              └─── Training ≈ Validation, both reasonable?
                   → You're in the sweet spot! ✓
                   → Fine-tune with cross-validation
```

## The Math (Brief)

For any classifier, the expected error decomposes as:

```
Expected Error = Bias² + Variance + Irreducible Noise

Bias²:     How far the average prediction is from the truth.
           High when model is too simple.

Variance:  How much predictions vary across different training sets.
           High when model is too complex.

Noise:     Randomness in the data. Can't be reduced.
           Sets the floor for your error.
```

For logistic regression specifically:
- **Increasing regularization (↓ C):** Increases bias, decreases variance
- **Decreasing regularization (↑ C):** Decreases bias, increases variance
- **Adding features:** Decreases bias (more flexible), increases variance (more to overfit)
- **More training data:** Decreases variance (more evidence), doesn't change bias

## What to Say in Interviews

> "The bias-variance tradeoff in logistic regression is controlled primarily by regularization. High regularization constrains the model, increasing bias but reducing variance — the decision boundary becomes simpler and more stable. Low regularization lets the model fit more closely to the training data, reducing bias but increasing variance — the boundary adapts to noise. I tune the regularization parameter C using cross-validation to find the point where total error is minimized."

> Follow-up: "Adding more features reduces bias but increases variance. More training data reduces variance without affecting bias. So if I'm underfitting, I add features or reduce regularization. If I'm overfitting, I increase regularization, reduce features, or get more data."
