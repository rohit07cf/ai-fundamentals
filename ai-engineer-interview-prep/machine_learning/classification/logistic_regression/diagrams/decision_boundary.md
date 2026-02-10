# Diagram: Decision Boundary in 2D

## The Picture

```
    x₂ (feature 2)
     |
  10 |  o     o    o                      o = Class 1 (positive)
     |     o    o      o                  x = Class 0 (negative)
   8 |  o    o    o
     |    o  o    o
   6 | o    o  \    o
     |   o     \  o                   \ = Decision boundary
   4 |    x  x  \                         β₀ + β₁x₁ + β₂x₂ = 0
     |  x   x  x \
   2 |    x  x     \
     | x    x   x
   0 +--x---x---x---x-----------→ x₁ (feature 1)
     0   2   4   6   8  10

     Below the line: z < 0  →  P < 0.5  →  Class 0
     Above the line: z > 0  →  P > 0.5  →  Class 1
     ON the line:    z = 0  →  P = 0.5  →  Coin flip
```

## Probability Shading (How Confidence Changes)

```
    x₂
     |
     |   ████████████  P ≈ 0.95  (very confident class 1)
     |   ▓▓▓▓▓▓▓▓▓▓▓▓  P ≈ 0.80
     |   ░░░░░░░░░░░░░  P ≈ 0.60
     |   ─────────────  P = 0.50  ← DECISION BOUNDARY
     |   ░░░░░░░░░░░░░  P ≈ 0.40
     |   ▓▓▓▓▓▓▓▓▓▓▓▓  P ≈ 0.20
     |   ████████████  P ≈ 0.05  (very confident class 0)
     +──────────────────→ x₁

     The further from the boundary, the more confident the model.
     The boundary itself is where uncertainty is maximum (P = 0.5).
```

## How Coefficients Affect the Boundary

```
    Changing β₀ (intercept):     Changing β₁, β₂ (slopes):

    x₂                           x₂
     |    /    /    /              |  \       |       /
     |   /    /    /               |   \      |      /
     |  /    /    /                |    \     |     /
     | /    /    /                 |     \    |    /
     |/    /    /                  |      \   |   /
     +────────────→ x₁            +──────────────→ x₁

    Shifts the boundary              Rotates the boundary
    left/right (parallel)            (changes the angle)
```

- **β₀** shifts the boundary without changing its angle
- **β₁, β₂** determine the orientation (angle) of the boundary
- **Magnitude** of coefficients affects how quickly probability changes (steepness of sigmoid)

## Non-Linear Boundaries (With Feature Engineering)

```
    Original features only:       With polynomial features (x₁², x₂², x₁·x₂):

    x₂                            x₂
     |  o o    x x                  |  o o    x x
     | o o  /  x x                  | o o ..   x x
     |o o  / x x x                  |o o .' '. x x
     |o o /  x x                    |o o '.  .' x x
     | o / x x x                    | o o  ''  x x
     +──/──────────→ x₁            +──────────────→ x₁

    Linear boundary only           Curved boundary!
    (might not separate well)      (still "linear" in the expanded feature space)
```

⚠️ **Key insight:** The boundary is always linear in the feature space the model sees. If you add x₁² as a feature, the boundary is linear in (x₁, x₂, x₁²) space — which projects as a curve in the original (x₁, x₂) space.

## What to Say in Interviews

> "The decision boundary in logistic regression is a linear hyperplane defined by where the model's output probability equals 0.5 — equivalently, where the linear score z equals zero. Points on one side are classified positive, the other side negative. The confidence increases with distance from the boundary. The boundary is always linear in the given feature space, but you can create non-linear boundaries in the original space by adding polynomial or interaction features."
