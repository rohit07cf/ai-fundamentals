# 06 — Interview Q&A

> 30 questions. Sharp answers. Spoken-tone, confident phrasing.
> Practice saying these OUT LOUD. Reading them silently is 30% as effective.
> Each answer is 2–4 lines — exactly what you'd say in an interview.

---

## Fundamentals

**Q1: What is linear regression?**
> It models the relationship between features and a continuous target by finding the hyperplane that minimizes the sum of squared residuals. Geometrically, it's projecting the target vector onto the column space of the feature matrix.

**Q2: What does "linear" in linear regression actually mean?**
> Linear in the **parameters**, not the features. y = β₁x² + β₂x is still linear regression because it's linear in β₁ and β₂. y = x^β would NOT be linear regression.

**Q3: Why do we minimize squared errors instead of absolute errors?**
> Squared errors give a smooth, differentiable, convex loss function with a unique global minimum and a clean closed-form solution. Absolute errors have a corner at zero, making optimization harder — though they're more robust to outliers.

**Q4: What's the Normal Equation?**
> β = (XᵀX)⁻¹Xᵀy. It comes from setting the gradient of the loss to zero, or equivalently, from the orthogonality condition that residuals must be perpendicular to the column space of X. It gives the exact solution in one step.

**Q5: When would you use Gradient Descent instead of the Normal Equation?**
> When the number of features is large. The Normal Equation requires inverting a p×p matrix, which is O(p³). Gradient descent is iterative but scales better — especially stochastic GD for large datasets. For small-to-medium problems, the normal equation is fine.

**Q6: Does gradient descent always find the global minimum for linear regression?**
> Yes. The loss function for linear regression is convex — it's a bowl shape with exactly one minimum. Any local minimum is the global minimum. This is NOT true for neural networks.

---

## Metrics

**Q7: Explain R² to a non-technical person.**
> If R² is 0.85, it means our model captures 85% of the pattern in the data. The other 15% is noise or stuff we haven't accounted for. An R² of zero means the model is no better than just guessing the average every time.

**Q8: Can R² be negative?**
> Yes. It means the model performs worse than simply predicting the mean. This typically happens when evaluating on test data with an overfit model, or when the model is fundamentally wrong for the data.

**Q9: Is a higher R² always better?**
> No. R² on training data always increases with more features — even random ones. That's why we use Adjusted R² (which penalizes complexity) or, better yet, cross-validated metrics. High training R² can simply mean overfitting.

**Q10: What's the difference between MSE, RMSE, and MAE?**
> MSE is the average squared error — great for optimization but in squared units. RMSE is the square root of MSE, so it's back in the original units and is the standard reporting metric. MAE is the average absolute error — it's more robust to outliers because it doesn't square the errors.

**Q11: When would you use MAE over RMSE?**
> When outliers are present and you don't want them dominating the metric. MAE treats all errors linearly, while RMSE penalizes large errors disproportionately. In domains where occasional big misses are acceptable but average accuracy matters, MAE is better.

---

## Assumptions

**Q12: What are the assumptions of linear regression?**
> LINE: Linearity (relationship is linear), Independence (errors are uncorrelated), Normality (errors are normally distributed), and Equal variance or homoscedasticity (error variance is constant). Plus no perfect multicollinearity among features.

**Q13: Which assumption is most important?**
> Linearity. If the true relationship isn't linear, your model is fundamentally misspecified — no amount of data or diagnostics can fix a straight line trying to model a curve. Always check residual plots first.

**Q14: What happens if homoscedasticity is violated?**
> Predictions can still be unbiased, but standard errors become unreliable — meaning confidence intervals and p-values are wrong. The fix is log-transforming y, using weighted least squares, or using heteroscedasticity-robust standard errors.

**Q15: How do you check model assumptions in practice?**
> Plot residuals vs. fitted values — look for patterns (non-linearity) and fan shapes (heteroscedasticity). Q-Q plot for normality. Durbin-Watson test for autocorrelation. VIF for multicollinearity. Residual plots are the single most important diagnostic.

**Q16: Does normality of residuals matter for predictions?**
> Not much. Normality mainly affects inference — confidence intervals and hypothesis tests. For pure prediction, it's the least critical assumption. But severe non-normality often signals other problems like outliers or model misspecification.

---

## Multiple Regression & Features

**Q17: How do you interpret a coefficient in multiple regression?**
> It's the expected change in y for a one-unit increase in that feature, **holding all other features constant**. This partial effect can differ from the simple regression coefficient because other features absorb some of the variance.

**Q18: Can you compare coefficients to determine feature importance?**
> Not with raw coefficients — they depend on feature scales. A coefficient of 500 on "sqft" isn't necessarily more important than 30,000 on "bedrooms." Standardize features first, then compare coefficients. Or use permutation importance for a model-agnostic approach.

**Q19: What is multicollinearity and why is it a problem?**
> It's when features are highly correlated with each other. The model can't distinguish their individual effects, so coefficients become unstable and inflate in magnitude. Predictions may still be fine, but interpreting individual coefficients becomes unreliable. Check VIF — values above 10 are concerning.

**Q20: Why do you drop one category in one-hot encoding?**
> To avoid the dummy variable trap — perfect multicollinearity. If all dummy columns sum to 1 (which they do), the last one is redundant. The dropped category becomes the reference level that other categories are compared against.

**Q21: Why should you scale features before gradient descent?**
> Unscaled features create an elongated loss surface where GD zigzags inefficiently. Scaling makes contours circular, so GD converges much faster. The normal equation doesn't need scaling for correctness, but scaling helps coefficient interpretability.

---

## Overfitting & Regularization

**Q22: Can linear regression overfit?**
> Absolutely. When you have many features relative to observations, or correlated features, the model can fit noise in the training data. The coefficients grow large and unstable. Regularization (Ridge/Lasso) constrains coefficient sizes to prevent this.

**Q23: What's the difference between Ridge and Lasso?**
> Ridge uses an L2 penalty (sum of squared coefficients) — it shrinks all coefficients toward zero but never exactly to zero. Lasso uses an L1 penalty (sum of absolute coefficients) — it can shrink coefficients exactly to zero, effectively doing feature selection. Use Ridge when most features are useful; use Lasso when you want sparsity.

**Q24: Why does Lasso give sparse solutions but Ridge doesn't?**
> It's about constraint geometry. Lasso's L1 constraint forms a diamond with corners on the axes. The loss function's contour ellipses tend to hit these corners, setting some coefficients exactly to zero. Ridge's L2 constraint is a circle with no corners, so coefficients get small but stay nonzero.

**Q25: How do you choose the regularization parameter λ?**
> Cross-validation. Fit the model for many λ values, evaluate on held-out folds, and pick the λ that minimizes the validation error. Libraries like sklearn's RidgeCV and LassoCV do this automatically.

---

## Practical & Tricky

**Q26: You built a linear regression model and R² is 0.95 on training data. Are you happy?**
> Not yet. I'd check R² on held-out test data first — 0.95 on training data could be overfitting. Then I'd check residual plots for assumption violations, and look at Adjusted R² to see if all features are pulling their weight.

**Q27: Your model has large coefficients with opposite signs on correlated features. What's happening?**
> Classic multicollinearity. The correlated features are "fighting" — one gets a large positive coefficient, the other a large negative one. They cancel out for predictions, but individually they're meaningless. Fix with Ridge regression, feature selection, or dropping one of the correlated features.

**Q28: When would you NOT use linear regression?**
> When the relationship is fundamentally non-linear and transformations don't help, when the target is categorical (use logistic regression or classifiers), when you have complex interactions that a linear model can't capture, or when you need a model that captures non-additive effects.

**Q29: What's the difference between linear regression and logistic regression?**
> Linear regression predicts a continuous value by fitting a hyperplane to minimize squared errors. Logistic regression predicts probabilities for classification by fitting a linear model inside a sigmoid function and maximizing likelihood. Despite the name, logistic regression is a classifier.

**Q30: Explain the bias-variance tradeoff in the context of linear regression.**
> A simple LR model (few features) has high bias — it underfits, missing real patterns. A complex LR model (many features, no regularization) has high variance — it overfits, fitting noise. Regularization adds controlled bias to reduce variance. The sweet spot minimizes total error, which is bias² + variance + irreducible noise.

---

## Bonus Lightning Round

**Q: What's the time complexity of the Normal Equation?**
> O(p³) for the matrix inversion, O(np²) for computing XᵀX. Dominated by whichever is larger.

**Q: What does the intercept represent?**
> The predicted value when ALL features are zero. Often not meaningful in practice (a house with 0 sqft), but it anchors the regression plane.

**Q: Can you use linear regression for time series?**
> You can, but the independence assumption is usually violated. Residuals will be autocorrelated. You'd need to add lagged features or use proper time-series methods like ARIMA.

---

## How to Use This File

1. **Pick 5 random questions.** Answer them out loud in your own words.
2. **Record yourself** answering — do you sound confident or hesitant?
3. **Focus on the ones you stumble on** — those are your weak spots.
4. **Sleep on it** — memory consolidates overnight. Review once more in the morning.
