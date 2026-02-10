# Diagram: Least Squares as Projection

## The Picture

```
    This lives in n-dimensional space (one dimension per data point).
    We're drawing a simplified 3D version.

                        y (actual target vector)
                       /|
                      / |
                     /  |
                    /   |  e = y - ŷ
                   /    |  (residual vector)
                  /     |
                 /      |  This vector is PERPENDICULAR
                /       |  to the column space of X
               /        |
    ŷ --------+         |
    (projection)        |
              |         |
    ==========|=========|========================
    |         |                                  |
    |    Column Space of X                       |
    |    (all possible linear combinations       |
    |     of your feature vectors)               |
    |                                            |
    ==============================================

    ŷ lives IN the column space (it's a linear combo of features)
    e is PERPENDICULAR to the column space
    y = ŷ + e  (actual = projection + residual)
```

## A More Concrete View

```
    Imagine you're in a room. The FLOOR is the column space of X.

                   y  ← you're pointing at a spot on the ceiling
                   |
                   |  ← residual (straight down)
                   |
    ───────────────ŷ────────────────  ← floor (column space)

    ŷ is the spot on the floor directly below y.
    That's the closest point on the floor to y.
    The residual is the vertical drop.
    That drop is perpendicular to the floor.
```

## Why This Matters

The perpendicularity isn't a coincidence — it's the **definition** of "closest."

If the residual e weren't perpendicular to the column space, there would exist some direction within the column space that could get us closer to y. The perpendicularity condition tells us we've found the best possible linear combination.

**Mathematically:** e ⊥ column space of X means:
```
Xᵀe = 0
Xᵀ(y - Xβ) = 0
XᵀXβ = Xᵀy
β = (XᵀX)⁻¹Xᵀy    ← The Normal Equation!
```

## The Projection Matrix

For bonus points, the projection onto the column space is:

```
H = X(XᵀX)⁻¹Xᵀ       (called the "hat matrix" because it puts the hat on y → ŷ)

ŷ = Hy                  (project y onto column space)
e = (I - H)y            (what's left over)
```

Properties of H:
- Hᵀ = H (symmetric)
- H² = H (idempotent — projecting twice is the same as projecting once)

## What to Say in Interviews

> "Linear regression is geometrically a projection of the target vector onto the column space of the feature matrix. The predicted values ŷ are the closest point in that subspace to y, and the residual vector is orthogonal to it. The Normal Equation comes directly from this orthogonality condition: X-transpose times the residual equals zero."

> Bonus: "The hat matrix H = X(XᵀX)⁻¹Xᵀ is the projection operator. It's idempotent — projecting twice gives the same result — which makes geometric sense."
