# Diagram: Gradient Descent Path

## The Loss Surface (Bird's Eye View)

```
    β₁
    ^
    |
    |     ___________________
    |    /                   \
    |   /    ___________      \
    |  /    /           \      \
    | /    /    _____    \      \
    |/    /    /     \    \      \
    |    /    /   *   \    \     |    * = global minimum
    |    \    \       /    /     |
    |     \    \_____/    /      |
    |      \             /      /
    |       \___________/      /
    |        \                /
    |         \______________/
    +-----------------------------------> β₀

    Contour lines = all (β₀, β₁) combos giving the same loss
    Inner contours = lower loss
    * = minimum loss (best β₀, β₁)
```

## Gradient Descent Path (Well-Scaled Features)

```
    β₁
    ^
    |
    |            __________
    |           /          \
    |          /   ______   \
    |    S    /   /      \   \
    |    ↓   /   /   *    \   \
    |     ↓ /   /  ↗      \   \
    |      ↓   / ↗         \   |
    |       ↓ /↗            \  |
    |        ↓               \ |
    |                          |
    +-----------------------------------> β₀

    S = starting point
    ↓ ↗ = gradient descent steps
    * = converged!

    With well-scaled features, the path is smooth
    and heads mostly straight toward the minimum.
```

## Gradient Descent Path (Poorly-Scaled Features)

```
    β₁ (small-range feature)
    ^
    |
    |  S
    |  ↓
    |  → ↓
    |    → ↓
    |  ←   → ↓           These elongated contours
    |    ←   → ↓         cause ZIGZAGGING!
    |  ←   ←   → ↓
    |    ←     → ↓
    |      ← → ↓
    |        ↓ *
    |
    +-----------------------------------> β₀ (large-range feature)

    The contours are elongated ellipses because
    features have very different scales.
    GD overshoots in the narrow direction,
    then corrects, then overshoots again.
    Wastes many iterations.
```

## Learning Rate Effects

```
    Good learning rate:        Too small:              Too large:

    L(β)                       L(β)                    L(β)
    |  \                       |  \                     |  \      /\
    |   \                      |   \                    |   \    /  \
    |    \                     |    \                   |    \  /    \
    |     \                    |     \                  |     \/      \
    |      \___                |      \                 |           ↑
    |          \____           |       \                |     DIVERGING!
    |               \___*      |        \               |
    +------------- step        |         \____          +------------- step
                               |              \
    Converges nicely           +------------- step
                               Takes forever
```

## Batch vs. Stochastic vs. Mini-Batch

```
    Batch GD:                   SGD:                    Mini-batch GD:

    β₁                         β₁                      β₁
    |  S                        |  S                     |  S
    |   \                       |  | /                   |   \
    |    \                      | / |                    |    \/
    |     \                     ||  / \                  |     \
    |      \                    | \|   |                 |      \/
    |       *                   |  |\ /                  |       \
    +---------> β₀              |  * /                   |        *
                                +---------> β₀           +---------> β₀
    Smooth, direct path         Noisy, bouncy path      Best of both:
    (uses ALL data each step)   (uses 1 point/step)     moderate noise,
                                                        faster convergence
```

## The Algorithm Step by Step

```
    Initialize β randomly
          |
          v
    ┌─────────────────────┐
    │ Compute predictions: │
    │   ŷ = Xβ            │
    │                      │
    │ Compute gradient:    │
    │   g = -2Xᵀ(y - ŷ)  │──────→ (this tells us "which way is uphill")
    │                      │
    │ Update weights:      │
    │   β = β - α·g       │──────→ (step DOWNHILL by learning rate α)
    │                      │
    │ Check: converged?    │
    └──────────┬───────────┘
               │
          No ──┤── Yes → Done! β is your answer.
               │
               └──→ (go back to top)
```

## What to Say in Interviews

> "Gradient descent iteratively moves β in the direction of steepest decrease of the loss function. For linear regression, the loss is convex, so we're guaranteed to reach the global minimum. Feature scaling is critical because unscaled features create elongated contours that cause zigzagging. In practice, I'd use mini-batch SGD for large datasets — it's a good balance between the stability of batch GD and the speed of stochastic GD."

> Follow-up: "The learning rate α is the most important hyperparameter. Too small and it converges slowly; too large and it diverges. In practice, adaptive methods like Adam adjust the learning rate per-parameter."
