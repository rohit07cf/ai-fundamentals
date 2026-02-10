# 03 — Metrics

> In regression you have MSE and R². Clean, simple, one number.
> Classification? Welcome to the jungle.
> You've got accuracy, precision, recall, F1, ROC-AUC, and each one tells a different story.
> The trick is knowing WHICH story matters for YOUR problem.

---

## The Confusion Matrix — Your Home Base

Everything in classification metrics starts from this 2×2 table. Memorize it.

```
                        Predicted
                    Positive    Negative
                 +------------+------------+
    Actual  Pos  |     TP     |     FN     |   ← Actual positives
                 +------------+------------+
    Actual  Neg  |     FP     |     TN     |   ← Actual negatives
                 +------------+------------+
                   ↑             ↑
                Predicted      Predicted
                positive       negative
```

- **TP (True Positive):** Said "yes," was right. ("Caught the spam.")
- **FP (False Positive):** Said "yes," was wrong. ("Cried wolf.")
- **FN (False Negative):** Said "no," was wrong. ("Let the spam through.")
- **TN (True Negative):** Said "no," was right. ("Correctly ignored a good email.")

**Memory trick:** The second word tells you the **prediction**. The first word tells you if it was **right or wrong**.
- **True** Positive = correctly predicted positive
- **False** Positive = incorrectly predicted positive

---

## Accuracy — The Obvious (and Often Lying) Metric

```
Accuracy = (TP + TN) / (TP + TN + FP + FN) = correct / total
```

**What it says:** "What fraction of ALL predictions were correct?"

**When it's useful:** When classes are roughly balanced and all errors are equally bad.

**When it LIES:**

⚠️ **The class imbalance trap.** This is the #1 interview question about accuracy.

Imagine a fraud detection system. 99% of transactions are legitimate, 1% are fraud.

A model that **always predicts "not fraud"** gets:
```
Accuracy = 99%    ← Looks amazing!
Recall = 0%       ← Catches zero fraud. Completely useless.
```

**If you forget everything else, remember this:** Accuracy is meaningless on imbalanced datasets. Always ask about class distribution first.

---

## Precision — "When I Say Yes, Am I Right?"

```
Precision = TP / (TP + FP)
```

**In plain English:** Of all the things I **predicted** as positive, how many actually were?

**Think of it like this:** Precision answers: **"How much can I trust a positive prediction?"**

**When precision matters most:**
- **Spam filter:** If you mark a real email as spam (FP), the user misses something important. High precision = few false alarms.
- **Criminal sentencing:** If you convict an innocent person (FP), the cost is enormous.
- Any time **false positives are expensive**.

---

## Recall (Sensitivity) — "Did I Find Everything?"

```
Recall = TP / (TP + FN)
```

**In plain English:** Of all the **actual** positives, how many did I catch?

**Think of it like this:** Recall answers: **"Am I missing anything important?"**

**When recall matters most:**
- **Cancer screening:** Missing a positive case (FN) could be fatal. High recall = catch everything.
- **Fraud detection:** A missed fraud (FN) costs real money.
- Any time **false negatives are dangerous**.

---

## The Precision–Recall Tradeoff

You can't maximize both. Here's why:

```
Lower threshold (say 0.3):
  → More things get labeled positive
  → You CATCH more positives (recall ↑)
  → But you also get more false alarms (precision ↓)

Higher threshold (say 0.8):
  → Fewer things get labeled positive
  → The ones you label are more likely correct (precision ↑)
  → But you MISS more positives (recall ↓)
```

```
               Precision vs. Recall as threshold changes

    1.0 |  Precision
        |  \
    0.8 |   \         ← At some point, one goes up
        |    \           as the other goes down
    0.6 |     \___
        |         \
    0.4 |    ___   \
        |   /   \   \  Recall
    0.2 |  /     \   \
        | /       \   \
    0.0 |/         \___\
        +-----|-----|-----|-----→ Threshold
        0   0.25  0.5  0.75   1
```

💡 **The interview move:** When asked about precision vs. recall, don't just define them. Say: "They're in tension. Lowering the threshold increases recall but decreases precision. The right balance depends on the cost of false positives vs. false negatives in the business context."

---

## F1-Score — The Compromise

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**What it is:** The **harmonic mean** of precision and recall. Balances both.

**Why harmonic mean?** Because the regular average is too forgiving. If precision = 1.0 and recall = 0.0, the average is 0.5 (seems OK), but F1 = 0.0 (correctly says this is terrible).

| Precision | Recall | Average | F1 |
|:---------:|:------:|:-------:|:---:|
| 1.0 | 0.0 | 0.50 | **0.00** |
| 0.9 | 0.1 | 0.50 | **0.18** |
| 0.7 | 0.7 | 0.70 | **0.70** |
| 1.0 | 1.0 | 1.00 | **1.00** |

**When to use F1:** When you care about both precision and recall roughly equally, and classes are imbalanced (so accuracy is unreliable).

---

## ROC-AUC — The Threshold-Free Metric

### ROC Curve

The ROC curve plots **True Positive Rate** (recall) vs. **False Positive Rate** at every possible threshold.

```
TPR (Recall) = TP / (TP + FN)     "Of actual positives, how many caught?"
FPR          = FP / (FP + TN)     "Of actual negatives, how many falsely flagged?"
```

```
    TPR (Recall)
  1.0 |          ___________
      |         /
  0.8 |        /
      |      /        ← Good model (curves toward top-left)
  0.6 |     /
      |    /
  0.4 |   /
      |  / .  .  .  . ← Random model (diagonal)
  0.2 | /
      |/
  0.0 +-----|-----|-----|-----→ FPR
      0   0.25  0.5  0.75   1
```

### AUC (Area Under the Curve)

- **AUC = 1.0** → Perfect classifier (top-left corner)
- **AUC = 0.5** → Random guessing (diagonal line)
- **AUC < 0.5** → Worse than random (flip your predictions!)

**What AUC intuitively means:** If you pick a random positive and a random negative, AUC is the probability that the model ranks the positive higher than the negative.

### When to use ROC-AUC

- When you want to evaluate the model **across all thresholds**
- When you want a single number for model comparison
- Works well even with moderate class imbalance

⚠️ **Trap:** ROC-AUC can be misleadingly high on very imbalanced datasets (because TN dominates FPR). In extreme imbalance, prefer **Precision-Recall AUC** instead.

---

## A Complete Worked Example

**Scenario:** Email spam detector. 10 emails, 3 are actually spam.

| Email | Actual | Predicted P | Predicted (threshold=0.5) |
|:-----:|:------:|:-----------:|:------------------------:|
| 1 | Spam | 0.9 | Spam ✓ |
| 2 | Spam | 0.8 | Spam ✓ |
| 3 | Spam | 0.3 | Not Spam ✗ |
| 4 | Not Spam | 0.6 | Spam ✗ |
| 5 | Not Spam | 0.1 | Not Spam ✓ |
| 6 | Not Spam | 0.2 | Not Spam ✓ |
| 7 | Not Spam | 0.05 | Not Spam ✓ |
| 8 | Not Spam | 0.15 | Not Spam ✓ |
| 9 | Not Spam | 0.4 | Not Spam ✓ |
| 10 | Not Spam | 0.1 | Not Spam ✓ |

**Confusion Matrix:**
```
              Predicted Spam    Predicted Not
Actual Spam       2 (TP)            1 (FN)
Actual Not        1 (FP)            6 (TN)
```

**Metrics:**
```
Accuracy  = (2+6)/(2+6+1+1) = 8/10 = 0.80
Precision = 2/(2+1) = 0.67     "67% of flagged emails were actually spam"
Recall    = 2/(2+1) = 0.67     "We caught 67% of spam"
F1        = 2×(0.67×0.67)/(0.67+0.67) = 0.67
```

**What if we lower the threshold to 0.25?** Email #3 (P=0.3) now becomes "Spam."

```
New: TP=3, FP=1, FN=0, TN=6
Precision = 3/4 = 0.75    ← went UP
Recall = 3/3 = 1.00       ← went UP (caught ALL spam!)
F1 = 2×(0.75×1.0)/(0.75+1.0) = 0.86
```

Better threshold for this problem! This is why threshold tuning matters.

---

## Which Metric When?

| Situation | Best Metric | Why |
|-----------|-------------|-----|
| Balanced classes, equal error cost | **Accuracy** | Simple, interpretable |
| Imbalanced classes | **F1** or **Precision-Recall AUC** | Accuracy lies here |
| False positives are expensive | **Precision** | Minimize false alarms |
| False negatives are dangerous | **Recall** | Catch everything |
| Need threshold-independent evaluation | **ROC-AUC** | Evaluates all thresholds |
| Very imbalanced + care about positives | **Precision-Recall AUC** | ROC-AUC can be misleading |
| Comparing multiple models overall | **ROC-AUC** or **F1** | Standard comparison metrics |

---

## Key Takeaways

- **Accuracy lies** on imbalanced datasets — always ask about class distribution
- **Precision** = "when I say yes, am I right?" → minimize false positives
- **Recall** = "did I find them all?" → minimize false negatives
- **F1** = harmonic mean of precision and recall — use when both matter
- **ROC-AUC** = threshold-free metric, great for model comparison
- There's always a **precision-recall tradeoff** — the threshold controls it
- **No single metric is best** — the right one depends on the business cost of each error type

⚠️ **The ultimate interview answer:** "The choice of metric depends on the cost of false positives vs. false negatives. In fraud detection, I'd optimize for recall. In email spam filtering, I'd balance precision and recall. I'd never rely on accuracy alone without first checking class balance."
