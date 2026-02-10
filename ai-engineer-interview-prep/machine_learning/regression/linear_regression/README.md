# Linear Regression — The Complete Interview Guide

> This is the most important topic you'll be asked about in an ML interview.
> Not because it's the most powerful model — but because **how well you explain it reveals how deeply you think.**
> An interviewer who asks about Linear Regression isn't testing your memory. They're testing your understanding.

---

## What You'll Master Here

- The **geometric intuition** behind why LR works (not just the formula)
- The **math**, explained so you can re-derive it on a whiteboard without panic
- **Metrics** — what they mean, when they lie, and what interviewers want to hear
- **Assumptions** — the ones that matter, and what breaks when they're violated
- **Multiple regression** — coefficients, scaling, encoding, and multicollinearity
- **Regularization preview** — just enough to sound smart when the conversation goes there
- **30 interview Q&As** — practiced, punchy, ready to speak out loud

---

## 60-Second Interview Pitch

*Practice saying this out loud. Seriously.*

> "Linear regression models the relationship between features and a continuous target by fitting a hyperplane that minimizes the sum of squared residuals. Geometrically, it's projecting the target vector onto the column space of the feature matrix — the residuals are orthogonal to that space, which is why the normal equation involves X-transpose-X. In practice, I use the closed-form solution for small datasets and gradient descent when scaling up. The key assumptions are linearity, independence, homoscedasticity, and normality of residuals — and I always check those with residual plots before trusting the model. When features are correlated, I look at VIF scores and consider regularization with Ridge or Lasso."

That's it. That's the whole pitch. If you can say this fluently, you're ahead of 90% of candidates.

---

## What Linear Regression is REALLY Doing

Forget formulas for a second. Here's the mental model:

**You have a cloud of data points. Linear regression finds the flat surface (line, plane, hyperplane) that passes through the middle of that cloud in the best possible way.**

What's "best"? The surface where the **total squared vertical distance** from every point to the surface is as small as possible.

Why squared? Because:
- It **penalizes big mistakes** more than small ones
- It gives a **unique, smooth** solution (no corners, no ambiguity)
- It has a **beautiful closed-form** answer (the Normal Equation)

Think of it like this: if every data point had a spring connecting it vertically to the line, linear regression finds the position of the line where **total spring tension is minimized**.

---

## File Guide

| File | What's Inside | Read Time |
|------|--------------|-----------|
| [00 — Intuition & Geometry](00_intuition_and_geometry.md) | Why LR works, visual intuition, projection | 8 min |
| [01 — Math Core](01_math_core.md) | Normal equation, gradient descent, derivations | 10 min |
| [02 — Metrics](02_metrics.md) | MSE, RMSE, MAE, R², Adjusted R² | 6 min |
| [03 — Assumptions & Diagnostics](03_assumptions_and_diagnostics.md) | What to check, what breaks, how to fix | 8 min |
| [04 — Multiple Linear Regression](04_multiple_linear_regression.md) | Many features, coefficients, scaling, encoding | 8 min |
| [05 — Regularization Preview](05_regularization_preview.md) | Ridge, Lasso, why LR overfits | 4 min |
| [06 — Interview Q&A](06_interview_qna.md) | 30 sharp questions with interview-ready answers | 12 min |
| [diagrams/](diagrams/) | ASCII diagrams with plain-English explanations | 5 min |
| [example_numpy.py](example_numpy.py) | LR from scratch in NumPy | 5 min |
| [example_sklearn.py](example_sklearn.py) | LR with sklearn, tying code to intuition | 3 min |

---

## If You're Revising 10 Minutes Before the Interview

Read these in this order. Nothing else.

1. **Re-read the 60-second pitch above.** Say it out loud twice.
2. **Skim [06_interview_qna.md](06_interview_qna.md)** — pick 5 random questions, answer them out loud.
3. **Glance at the [bias-variance diagram](diagrams/bias_variance.md)** — remember the tradeoff curve.
4. **Remember these three things:**
   - LR minimizes sum of squared residuals
   - The Normal Equation is: **beta = (X^T X)^{-1} X^T y**
   - Assumptions: Linearity, Independence, Homoscedasticity, Normality (mnemonic: **LINE**)

**You've got this.** Go be dangerous.
