# Machine Learning

> The bread and butter of AI Engineer interviews.
> If deep learning is the flashy sports car, ML fundamentals are the engine.
> Interviewers want to know you understand **why** things work, not just how to call `.fit()`.

---

## Topic Map

| Area | Topics | Status |
|------|--------|--------|
| **Regression** | Linear Regression, Ridge, Lasso, Elastic Net | Linear Regression done |
| **Classification** | Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost | Logistic Regression done |
| **Clustering** | K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models | Planned |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP, SVD | Planned |
| **Evaluation & Regularization** | Cross-validation, Bias-Variance, Hyperparameter Tuning | Planned |

---

## When to Use Classification vs. Regression (Plain English)

Use **regression** when your **target variable is a number** — something you can measure on a scale.

- Predicting house prices? **Regression.**
- Predicting someone's salary? **Regression.**

Use **classification** when your **target is a category** — a label, a bucket, a yes/no.

- Will the customer churn? **Classification.**
- Is this email spam? **Classification.**
- Predicting whether someone clicks a button? **Classification.**

The simplest mental test: **"Can I average two predictions and get a meaningful answer?"**
- Average of two prices = makes sense → **regression**.
- Average of "spam" and "not spam" = nonsense → **classification**.

---

## Top ML Interview Traps

These trip up smart people all the time. Don't be one of them.

**1. "What's the difference between L1 and L2 regularization?"**
> They're NOT asking for the formula. They want: L1 gives **sparse** solutions (feature selection), L2 gives **small** weights (feature shrinkage). Know *why* — the diamond vs. circle constraint geometry.

**2. "Is more data always better?"**
> No. More data helps with **variance** (overfitting), but if your model has **high bias** (underfitting), more data won't save you. Fix the model first.

**3. "When would you NOT use a neural network?"**
> Small data, need for interpretability, tabular data (where XGBoost often wins), or when a simpler model performs equally well. Occam's razor still applies.

**4. "What's the bias-variance tradeoff?"**
> Bias = how wrong your model is on average. Variance = how much your model changes with different training data. You can't minimize both at once — you're looking for the sweet spot.

**5. "How do you handle missing data?"**
> Don't just say "drop it" or "impute the mean." Ask: **Is it missing at random?** Mean/median imputation, model-based imputation, or sometimes missingness itself is a feature.

**6. "Explain overfitting to a non-technical person."**
> "The model memorized the answers instead of learning the patterns. It's like studying only the practice exam — you'll ace it, but fail the real test."
