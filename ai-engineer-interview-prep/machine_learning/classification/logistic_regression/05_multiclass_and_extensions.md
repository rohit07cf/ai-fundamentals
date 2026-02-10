# 05 — Multiclass & Extensions

> Standard logistic regression handles two classes: yes/no, spam/not spam, cat/dog.
> But what if you have cat/dog/fish? Three classes. Or ten. Or a thousand.
> That's where multiclass strategies come in.
> This is a short file — just enough to handle the interview question confidently.

---

## The Problem

Binary logistic regression outputs: P(y = 1) and P(y = 0) = 1 − P(y = 1).

But with K > 2 classes, we need: P(y = class₁), P(y = class₂), ..., P(y = classₖ).

And they need to **sum to 1**.

Two main approaches: **One-vs-Rest** and **Softmax**.

---

## One-vs-Rest (OvR) — The Simple Strategy

### The idea

Train **K separate binary classifiers**, one for each class:

```
Class A vs. (not A)   →   Classifier 1  →  P(A)
Class B vs. (not B)   →   Classifier 2  →  P(B)
Class C vs. (not C)   →   Classifier 3  →  P(C)

Prediction: whichever classifier gives the highest probability wins.
```

### Example: Cat / Dog / Fish

```
Classifier 1:  "Is it a cat?"    →  P(cat)   = 0.7
Classifier 2:  "Is it a dog?"    →  P(dog)   = 0.3
Classifier 3:  "Is it a fish?"   →  P(fish)  = 0.1

Prediction: CAT (highest score)
```

### Pros and cons

| Pros | Cons |
|------|------|
| Simple to implement | Probabilities don't sum to 1 (they're from different models) |
| Each classifier is just standard binary LogReg | Can have ambiguous regions where no class is confident |
| Scales OK with many classes | K models to train |
| Handles imbalanced classes per classifier | Assumes classes are independent |

💡 **This is what sklearn uses by default** when you call `LogisticRegression()` with multiclass data (well, technically it defaults to the `multinomial` solver now, but OvR is the conceptual foundation).

---

## One-vs-One (OvO) — Brief Mention

### The idea

Train a classifier for **every pair** of classes: K(K−1)/2 classifiers.

```
For 3 classes (A, B, C):
  Classifier 1:  A vs. B
  Classifier 2:  A vs. C
  Classifier 3:  B vs. C

Prediction: majority vote across all classifiers.
```

### When to use

- Good for **SVMs** (where training on small subsets is efficient)
- Rarely used for logistic regression in practice
- Know it exists, but don't dwell on it in interviews

⚠️ **OvO scales poorly:** K(K-1)/2 classifiers. For K=100, that's 4,950 models. Ouch.

---

## Softmax Regression — The Elegant Approach

### The idea (also called Multinomial Logistic Regression)

Instead of K separate models, train **one model** with K sets of weights, and use the **softmax function** to get proper probabilities that sum to 1.

### The softmax function

```
P(y = k | X) = e^(zₖ) / Σⱼ e^(zⱼ)

where zₖ = Xβₖ for each class k
```

**In words:** Compute a score for each class, exponentiate them all, then divide by the total. The result: valid probabilities that sum to 1.

### How it works

```
3 classes. Input features produce scores:

z_cat  = 2.0   →   e^2.0 = 7.39
z_dog  = 1.0   →   e^1.0 = 2.72
z_fish = 0.5   →   e^0.5 = 1.65

Total = 7.39 + 2.72 + 1.65 = 11.76

P(cat)  = 7.39 / 11.76 = 0.63
P(dog)  = 2.72 / 11.76 = 0.23
P(fish) = 1.65 / 11.76 = 0.14
                          ────
                          1.00 ✓  (sums to 1!)
```

### The connection to binary logistic regression

When K = 2, softmax reduces exactly to the sigmoid function. Softmax is the **generalization** of sigmoid to multiple classes.

```
K = 2:  softmax(z₁, z₂)  =  e^z₁ / (e^z₁ + e^z₂)  =  1 / (1 + e^(z₂-z₁))  =  σ(z₁ - z₂)
```

### Loss function: Categorical Cross-Entropy

The binary cross-entropy generalizes to:

```
L = -Σᵢ Σₖ yᵢₖ · log(pᵢₖ)

where yᵢₖ = 1 if sample i belongs to class k, else 0
```

---

## OvR vs. Softmax — Quick Comparison

| Aspect | OvR | Softmax |
|--------|-----|---------|
| **Models trained** | K separate binary | 1 unified model |
| **Probabilities** | Don't sum to 1 | Sum to 1 ✓ |
| **Assumption** | Classes independent | Classes compete (mutually exclusive) |
| **When to use** | Classes NOT mutually exclusive | Classes ARE mutually exclusive |
| **Speed** | Can parallelize K models | One model, but larger |
| **sklearn** | `multi_class='ovr'` | `multi_class='multinomial'` |

💡 **Interview insight:** "If classes are mutually exclusive (a photo is either cat, dog, or fish — not two at once), softmax is more principled because it ensures probabilities sum to 1 and classes compete for probability mass. If classes can overlap (an article can be both 'politics' and 'economics'), OvR is more appropriate."

---

## When Interviewers Expect You to Mention This

1. **"How would you extend logistic regression to multiple classes?"**
   → Mention OvR (simple, K binary classifiers) and softmax/multinomial (single model, probabilities sum to 1).

2. **"What's the relationship between sigmoid and softmax?"**
   → "Softmax is the generalization of sigmoid to K classes. When K=2, softmax reduces to sigmoid."

3. **"When would you use OvR vs. softmax?"**
   → "OvR when classes may overlap. Softmax when classes are mutually exclusive."

4. **"How does logistic regression connect to neural networks?"**
   → "A logistic regression model is mathematically identical to a single-layer neural network with a sigmoid (binary) or softmax (multiclass) output activation. It's the simplest possible neural network."

---

## Key Takeaways

- **OvR:** Train K binary classifiers, pick the most confident one. Simple but probabilities don't sum to 1.
- **Softmax:** One model, K weight vectors, softmax converts scores to proper probabilities. More principled for mutually exclusive classes.
- **Softmax = generalization of sigmoid** to K classes.
- **OvR for overlapping classes, softmax for mutually exclusive classes.**
- Logistic regression is a **single-layer neural network** — this connection comes up in deep learning interviews.
