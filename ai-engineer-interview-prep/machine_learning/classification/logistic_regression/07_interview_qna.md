# 07 — Interview Q&A

> 30 questions. Sharp answers. Spoken-tone, confident phrasing.
> Practice saying these OUT LOUD. Reading them silently is 30% as effective.
> Each answer is 2–4 lines — exactly what you'd say in an interview.

---

## Fundamentals

**Q1: What is logistic regression?**
> It's a linear classifier that models the probability of a binary outcome. It computes a linear combination of features, passes it through the sigmoid function to get a probability, and classifies based on a threshold. Despite the name, it's classification, not regression.

**Q2: Why is it called "regression" if it's a classifier?**
> Because it regresses the log-odds onto the features — the log-odds ARE a continuous quantity modeled linearly. The "logistic" part refers to the logistic (sigmoid) function that converts log-odds to probabilities. The classification happens only when we apply a threshold.

**Q3: What does the sigmoid function do?**
> It takes any real number and squashes it into the range (0, 1), making it interpretable as a probability. σ(z) = 1/(1+e^(−z)). At z=0 it outputs 0.5, and it approaches 0 and 1 asymptotically at the extremes. It's smooth and differentiable, which makes gradient-based optimization possible.

**Q4: What is the decision boundary in logistic regression?**
> It's the set of points where the model predicts exactly 50% probability — where the linear score z equals zero. In 2D it's a line, in 3D a plane, in general a hyperplane. It's always linear in the original feature space, though you can get curved boundaries by adding polynomial features.

**Q5: What's the difference between logistic regression and linear regression?**
> Linear regression predicts a continuous value by minimizing MSE. Logistic regression predicts a probability using the sigmoid and minimizes cross-entropy loss. Linear regression has a closed-form solution; logistic regression requires iterative optimization. Linear regression outputs any real number; logistic regression outputs a probability in (0,1).

**Q6: What are the "log-odds" and why do they matter?**
> The log-odds is log(P/(1−P)), also called the logit. Logistic regression's core assumption is that log-odds are a linear function of features. This means a unit increase in a feature adds a constant amount to the log-odds, which multiplies the odds by e^β. This is how you interpret coefficients.

---

## Loss Function & Training

**Q7: Why can't you use MSE for logistic regression?**
> MSE combined with the sigmoid creates a non-convex loss surface with local minima — gradient descent might get stuck. Cross-entropy loss is convex for logistic regression, guaranteeing a unique global minimum. Cross-entropy also penalizes confident wrong predictions much more heavily, giving better gradients.

**Q8: What loss function does logistic regression use?**
> Binary cross-entropy, also called log loss: L = −Σ[y·log(p) + (1−y)·log(1−p)]. It comes from maximum likelihood estimation. When the model is confidently wrong, the loss goes to infinity, creating a strong gradient signal. It's convex, so gradient descent always finds the global optimum.

**Q9: Is there a closed-form solution for logistic regression?**
> No. Unlike linear regression, the sigmoid makes the gradient equation non-linear in the parameters, so you can't solve it algebraically. We use iterative methods — typically gradient descent, Newton's method, or L-BFGS. But the loss is convex, so convergence is guaranteed.

**Q10: How does gradient descent work for logistic regression?**
> Compute predictions with the sigmoid, calculate the error (prediction minus label), then update each weight by the learning rate times the average of (error × feature value). The gradient is (1/n)·Xᵀ(σ(Xβ) − y) — same structure as linear regression but with the sigmoid applied to the predictions.

---

## Metrics & Evaluation

**Q11: When is accuracy a bad metric?**
> On imbalanced datasets. If 99% of emails are not spam, a model that always predicts "not spam" gets 99% accuracy but catches zero spam. In such cases, precision, recall, F1, or ROC-AUC give a much more meaningful picture of model performance.

**Q12: Explain precision and recall.**
> Precision: of everything the model predicted positive, what fraction actually was? It measures false alarm rate. Recall: of everything actually positive, what fraction did the model catch? It measures miss rate. There's a tradeoff — raising the threshold increases precision but decreases recall.

**Q13: When would you optimize for precision vs. recall?**
> Optimize precision when false positives are costly — like a spam filter marking real emails as spam. Optimize recall when false negatives are dangerous — like cancer screening where missing a case could be fatal. The choice depends on the business cost of each error type.

**Q14: What is F1-score?**
> The harmonic mean of precision and recall: 2×(P×R)/(P+R). It's zero if either precision or recall is zero, so both must be reasonable for a good F1. Use it when you care about both precision and recall and classes are imbalanced. It's more informative than accuracy in most real-world scenarios.

**Q15: What does ROC-AUC measure?**
> ROC-AUC measures the model's ability to rank positive examples above negative ones, across all classification thresholds. An AUC of 1.0 means perfect ranking, 0.5 means random. Intuitively, it's the probability that a randomly chosen positive has a higher predicted score than a randomly chosen negative.

**Q16: Can you have high AUC but low accuracy?**
> Yes, with very imbalanced classes and a suboptimal threshold. AUC evaluates ranking ability across all thresholds, while accuracy depends on the specific threshold chosen. A model might rank well overall but still misclassify at the default 0.5 threshold. Adjusting the threshold based on the ROC curve can fix this.

---

## Assumptions & Diagnostics

**Q17: What are the assumptions of logistic regression?**
> Linearity in log-odds — not in probability — independence of observations, and no extreme multicollinearity. It does NOT require normality of features or residuals, or homoscedasticity. These are common misconceptions from confusing it with linear regression.

**Q18: What does "linearity in log-odds" mean in practice?**
> It means each feature has a constant additive effect on the log-odds of the outcome. A one-unit increase in a feature always changes the log-odds by the same amount, regardless of the feature's current value. If this doesn't hold — say the effect of age plateaus — you need feature transformations or a different model.

**Q19: How do you interpret a logistic regression coefficient?**
> A coefficient β means a one-unit increase in that feature increases the log-odds by β, or equivalently multiplies the odds by e^β, holding all other features constant. Note: this is NOT a linear change in probability — the effect on probability depends on where you are on the sigmoid curve.

**Q20: Does logistic regression handle multicollinearity?**
> Not well without regularization. Correlated features cause unstable, inflated coefficients. Predictions may still be OK, but individual coefficients become uninterpretable. Fix with L2 regularization (distributes weight), L1 (selects one feature), or manually dropping/combining correlated features.

---

## Regularization & Overfitting

**Q21: Can logistic regression overfit?**
> Absolutely. It overfits when there are too many features relative to samples, when features are highly correlated, or when there's perfect separation in the data. The model finds a hyperplane that perfectly separates training data but generalizes poorly. Regularization is the standard fix.

**Q22: What's the difference between L1 and L2 regularization for logistic regression?**
> L2 shrinks all coefficients toward zero but keeps them nonzero — it's good for correlated features and is the default in sklearn. L1 can shrink coefficients exactly to zero, performing automatic feature selection — it's good when you suspect only a few features matter. Elastic Net combines both.

**Q23: What does the C parameter mean in sklearn's LogisticRegression?**
> C is the inverse of regularization strength: C = 1/λ. Higher C means less regularization (more flexible, risk of overfitting). Lower C means more regularization (simpler model, risk of underfitting). The default is C=1. I'd tune it with cross-validation over a logarithmic grid.

**Q24: What is perfect separation and how do you fix it?**
> Perfect separation happens when a feature perfectly predicts the outcome. The model tries to set that feature's coefficient to infinity, making the sigmoid a step function. The optimizer never converges. Fix it with regularization — even a small L2 penalty constrains coefficients and stabilizes training.

---

## Multiclass & Extensions

**Q25: How do you extend logistic regression to multiple classes?**
> Two approaches: One-vs-Rest trains K binary classifiers, one per class, and picks the most confident. Softmax regression trains one model with K weight vectors and uses the softmax function to produce probabilities that sum to 1. Softmax is more principled for mutually exclusive classes; OvR works when classes can overlap.

**Q26: What's the relationship between sigmoid and softmax?**
> Softmax is the generalization of sigmoid to K classes. When K=2, softmax reduces exactly to the sigmoid. Both convert raw scores into probabilities, but sigmoid outputs one probability (the other is 1 minus it), while softmax outputs K probabilities that sum to 1.

**Q27: How does logistic regression relate to neural networks?**
> A logistic regression model is exactly a single-layer neural network with sigmoid (binary) or softmax (multiclass) activation. No hidden layers, no non-linear feature interactions — just input directly to output through a linear transformation plus activation. It's the simplest possible neural network.

---

## Practical & Tricky

**Q28: You're building a fraud detection model. 1% of transactions are fraud. What do you do?**
> First, don't use accuracy — it'll be 99% even if you predict "not fraud" for everything. Use precision-recall metrics or AUC-PR. Second, consider class weights (sklearn's `class_weight='balanced'`), SMOTE, or adjusting the decision threshold. Third, focus on recall to catch fraud, but watch precision so you don't flag every transaction.

**Q29: Your logistic regression model gives great training accuracy but poor test accuracy. What's happening?**
> Overfitting. Too many features, insufficient data, or no regularization. I'd add L2 or L1 regularization, reduce features through selection or PCA, check for data leakage, and use cross-validation to tune the regularization parameter. I'd also verify there isn't a data issue like temporal leakage.

**Q30: Why would you choose logistic regression over a more complex model like XGBoost or a neural network?**
> Interpretability — coefficients have clear meaning as log-odds multipliers. Speed — it trains quickly even on large datasets. Calibration — it naturally outputs well-calibrated probabilities. Baseline — it's the standard first model I'd try for any classification problem. If it performs well enough, there's no need for complexity. And in regulated industries (finance, healthcare), interpretability is often a requirement.

---

## Bonus Lightning Round

**Q: What's the time complexity of training logistic regression?**
> O(n·p) per gradient descent iteration, where n is samples and p is features. With L-BFGS or Newton's method, convergence is typically fast (quadratic for Newton). Overall much cheaper than tree ensembles or neural networks.

**Q: Can logistic regression output probabilities greater than 1?**
> No. The sigmoid guarantees output is strictly between 0 and 1. This is exactly why we use it instead of linear regression for classification — linear regression can and will predict outside [0,1].

**Q: How does the threshold affect the model?**
> The threshold doesn't change the model at all — it only changes the decision rule applied to the model's probabilities. Lowering the threshold increases recall (catches more positives) but decreases precision (more false alarms). The model itself is fixed after training.

---

## How to Use This File

1. **Pick 5 random questions.** Answer them out loud in your own words.
2. **Record yourself** — do you sound confident or hesitant?
3. **Focus on the ones you stumble on** — those are your weak spots.
4. **The night before:** Read Q5, Q7, Q12, Q17, Q22 — these are the five most common interview questions about logistic regression.
