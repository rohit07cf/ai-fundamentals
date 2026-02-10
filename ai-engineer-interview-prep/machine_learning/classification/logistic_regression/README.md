# Logistic Regression — The Complete Interview Guide

> If Linear Regression is the "Hello World" of ML, Logistic Regression is the "Hello World" of classification.
> It looks simple. It IS simple. But the depth of understanding you can show with it is enormous.
> This is the topic where interviewers decide: "Does this person really get ML, or do they just import sklearn?"

---

## What You'll Master Here

- **Why** logistic regression exists (and why linear regression can't do this job)
- The **sigmoid function** — what it does, why it's perfect, and how to draw it on a whiteboard
- The **decision boundary** — what it is, what shapes it can take, and why it's always linear (in feature space)
- **Log loss** (cross-entropy) — why MSE is wrong and what we use instead
- **Classification metrics** — accuracy, precision, recall, F1, ROC-AUC, and when each one lies
- **Assumptions** — the ones that matter and the ones interviewers always ask about
- **Multiclass** — OvR, OvO, and softmax at a high level
- **Regularization** — why logistic regression overfits and how L1/L2 fix it
- **30 interview Q&As** — practiced, punchy, ready to speak out loud

---

## 60-Second Interview Pitch

*Practice saying this out loud. Twice. Right now.*

> "Logistic regression is a linear classifier that models the probability of a binary outcome. It computes a linear combination of features, passes it through a sigmoid function to get a probability between 0 and 1, and classifies based on a threshold — typically 0.5. The key insight is that it's modeling the log-odds of the positive class as a linear function of the features. We train it by maximizing the likelihood of the observed labels, which is equivalent to minimizing the binary cross-entropy loss. I use gradient descent since there's no closed-form solution. The decision boundary is linear in feature space, and I regularize with L1 or L2 depending on whether I want feature selection or just coefficient shrinkage."

If you can say that fluidly, you own this topic.

---

## What Logistic Regression is REALLY Doing

Strip away the formulas. Here's the mental model:

**You have data points that belong to two groups.** Logistic regression draws a straight line (or flat hyperplane) between them — the **decision boundary** — and for every point, it tells you **how confident it is** about which side that point belongs to.

Points far from the boundary → high confidence (close to 0% or 100%).
Points near the boundary → low confidence (close to 50%).

That's it. It's a **confidence-calibrated linear separator**.

The sigmoid function is just the machinery that converts "distance from the boundary" into "probability."

---

## File Guide

| File | What's Inside | Read Time |
|------|--------------|-----------|
| [00 — Intuition & Decision Boundary](00_intuition_and_decision_boundary.md) | Why it exists, sigmoid, boundary intuition | 8 min |
| [01 — Math Core](01_math_core.md) | z = Xβ, sigmoid, log-odds, the full pipeline | 8 min |
| [02 — Loss & Optimization](02_loss_and_optimization.md) | Why MSE fails, log loss, gradient descent | 8 min |
| [03 — Metrics](03_metrics.md) | Accuracy, precision, recall, F1, ROC-AUC | 10 min |
| [04 — Assumptions & Diagnostics](04_assumptions_and_diagnostics.md) | What to check, what breaks, how to fix | 7 min |
| [05 — Multiclass & Extensions](05_multiclass_and_extensions.md) | OvR, OvO, softmax | 5 min |
| [06 — Regularization](06_regularization.md) | L1, L2, overfitting, sparsity | 6 min |
| [07 — Interview Q&A](07_interview_qna.md) | 30 sharp questions with interview-ready answers | 12 min |
| [diagrams/](diagrams/) | ASCII diagrams with plain-English explanations | 5 min |
| [example_numpy.py](example_numpy.py) | Logistic regression from scratch in NumPy | 5 min |
| [example_sklearn.py](example_sklearn.py) | Logistic regression with sklearn, tying code to intuition | 3 min |

---

## If You're Revising 10 Minutes Before the Interview

Read these in this order. Nothing else.

1. **Re-read the 60-second pitch above.** Say it out loud twice.
2. **Skim [07_interview_qna.md](07_interview_qna.md)** — pick 5 random questions, answer them out loud.
3. **Glance at the [sigmoid diagram](diagrams/sigmoid_curve.md)** — remember the S-curve and what it converts.
4. **Remember these four things:**
   - Logistic regression models **P(y=1|X)** using the sigmoid of a linear combination
   - Loss function is **log loss (cross-entropy)**, NOT MSE
   - Decision boundary is **linear** in feature space
   - Assumptions: linearity in **log-odds**, independence, no extreme multicollinearity

**You've got this.** Go be confident.
