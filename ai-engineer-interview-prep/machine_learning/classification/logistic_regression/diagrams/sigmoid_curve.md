# Diagram: The Sigmoid Curve

## The Picture

```
    P(y = 1)

   1.0 |                                    _______________
       |                                  /
       |                                /
   0.8 |                              /
       |                            /
       |                          /
   0.6 |                        /
       |                      /
       |                    /
   0.5 | - - - - - - - - -* - - - - - - - - - - -  (threshold)
       |                /
       |              /
   0.4 |            /
       |          /
       |        /
   0.2 |      /
       |    /
       |  /
   0.0 |/_______________
       +----+----+----+----+----+----+----+----+→ z (linear score)
        -8  -6   -4   -2    0   +2   +4   +6

                         * = σ(0) = 0.5  ← the tipping point

       ◄── Confident 0 ──►◄── Uncertain ──►◄── Confident 1 ──►
```

## What's Happening

The sigmoid function σ(z) = 1 / (1 + e^(−z)) converts the linear score z into a probability:

- **z is very negative** (left side): The model is confident the answer is class 0. P ≈ 0.
- **z is around zero** (middle): The model is uncertain. P ≈ 0.5. This is the decision boundary.
- **z is very positive** (right side): The model is confident the answer is class 1. P ≈ 1.

The curve is **S-shaped** — the most rapid change happens around z = 0, and the output saturates (flattens) at the extremes.

## Key Values to Remember

```
σ(-∞) = 0.0    "Absolutely certain it's class 0"
σ(-2)  ≈ 0.12   "Pretty sure it's class 0"
σ(-1)  ≈ 0.27   "Leaning toward class 0"
σ(0)   = 0.50   "Complete coin flip — the decision boundary"
σ(1)   ≈ 0.73   "Leaning toward class 1"
σ(2)   ≈ 0.88   "Pretty sure it's class 1"
σ(+∞) = 1.0    "Absolutely certain it's class 1"
```

## The Derivative (Bonus)

```
    dσ/dz

  0.25 |              ___
       |            /     \
  0.20 |          /         \
       |        /             \
  0.15 |      /                 \
       |    /                     \
  0.10 |   /                       \
       |  /                         \
  0.05 | /                           \
       |/                             \
  0.0  +----+----+----+----+----+----+→ z
        -6  -4   -2    0   +2   +4

  dσ/dz = σ(z) · (1 - σ(z))

  Peaks at z = 0 (maximum slope = 0.25)
  → The model learns FASTEST at the decision boundary
  → Near-certain predictions produce tiny gradients (vanishing gradient)
```

## What to Say in Interviews

> "The sigmoid function converts the linear score into a calibrated probability between 0 and 1. At z=0 it outputs 0.5, which defines the decision boundary. Its derivative is σ(z)·(1−σ(z)), which peaks at the decision boundary — meaning the model focuses its learning on the uncertain region. At the extremes, the gradient vanishes, which is the same vanishing gradient problem we see in deep neural networks with sigmoid activations."
