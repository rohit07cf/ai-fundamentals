# Diagram: Best Fit Line + Residuals

## The Picture

```
    y (target)
    |
 10 |                          o  (8, 10)
    |                        / |
  9 |                      /   |  residual = +1.2
    |                    /     |
  8 |               o-/-------+  ŷ = 8.8
    |              (6, 8)
  7 |            /   |
    |          /     | residual = -0.6
  6 |    o   /       |
    |  (3,6) |-------+  ŷ = 6.6
  5 |      / |
    |    /   | residual = +0.4
  4 |  /     |
    | / o----+  ŷ = 3.6
  3 |/ (2, 4)
    |/
  2 |  o (1, 2)
    |  |-----+  ŷ = 2.4
  1 |  | residual = -0.4
    | /
  0 +-------------------------------- x (feature)
    0   1   2   3   4   5   6   7   8

    ──── = best fit line (ŷ = 1.2x + 1.2)
    |    = residuals (vertical distances)
    o    = actual data points
```

## What's Happening

- Each **o** is an actual data point
- The **line** is the model's prediction at each x value
- The **vertical bars** are the **residuals** — the gap between reality and prediction
- Linear regression finds the line that makes the **sum of these squared gaps as small as possible**

## Key Details

- Residuals are **vertical** distances, not perpendicular distances
- Some residuals are positive (point above line), some negative (point below)
- The line passes through the point (mean(x), mean(y)) — always
- With an intercept, residuals always sum to zero

## What to Say in Interviews

> "Linear regression minimizes the sum of squared vertical distances between the data points and the fitted line. These vertical distances are the residuals. I always plot residuals after fitting to check for patterns — if they show a curve or fan shape, the model assumptions are violated."
