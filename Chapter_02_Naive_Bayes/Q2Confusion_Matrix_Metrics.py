import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# =====================================================================
# STEP 1: DATA LOAD & SPLIT
# =====================================================================
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data.tolist(), iris.target.tolist(), test_size=0.2, random_state=42
)


# =====================================================================
# QUESTION 1: GAUSSIAN NAÏVE BAYES CLASSIFIER (FROM SCRATCH)
# =====================================================================
def mean(lst):
  return sum(lst) / len(lst)


def stdev(lst):
  m = mean(lst)
  return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


# Class-wise Mean aur Stdev Calculate karna (Training Phase)
stats = {
    c: [
        (
            mean([X_train[i][j] for i in range(len(X_train)) if y_train[i] == c]),
            stdev([
                X_train[i][j] for i in range(len(X_train)) if y_train[i] == c
            ]),
        )
        for j in range(4)
    ]
    for c in set(y_train)
}


# Gaussian Probability Density Function
def prob(x, m, s):
  return (1 / (math.sqrt(2 * math.pi) * s)) * math.exp(
      -((x - m) ** 2) / (2 * s**2)
  )


# Test sample ke liye Prediction karna
def predict(sample):
  scores = {
      c: math.prod(
          prob(sample[j], stats[c][j][0], stats[c][j][1]) for j in range(4)
      )
      for c in stats
  }
  return max(scores, key=scores.get)


# Test dataset par Prediction run karna
y_pred = [predict(s) for s in X_test]


# =====================================================================
# QUESTION 2: PERFORMANCE EVALUATION OF THE CLASSIFIER (FROM SCRATCH)
# =====================================================================
classes = [0, 1, 2]

# 1. Confusion Matrix Generation (3x3 Table)
cm = [[0] * 3 for _ in range(3)]
for actual, predicted in zip(y_test, y_pred):
  cm[actual][predicted] += 1

# 2. Overall Accuracy Calculation
accuracy = (sum(cm[i][i] for i in classes) / len(y_test)) * 100

# OUTPUTS PRINTING
print("=" * 50)
print("Q1: GAUSSIAN NAÏVE BAYES PREDICTIONS")
print("=" * 50)
print("Actual Labels:   ", y_test)
print("Predicted Labels:", y_pred)

print("\n" + "=" * 50)
print("Q2: PERFORMANCE EVALUATION METRICS")
print("=" * 50)
print("Confusion Matrix:")
for row in cm:
  print("  ", row)

print(f"\nOverall Accuracy: {accuracy:.2f}%")
print("-" * 50)
print("Class-wise Detailed Metrics:")
print("-" * 50)

# Class-wise Metrics: Precision, Recall (TPR), TNR, FPR, FNR, F1-Score
for c in classes:
  TP = cm[c][c]
  FP = sum(cm[i][c] for i in classes if i != c)
  FN = sum(cm[c][i] for i in classes if i != c)
  TN = len(y_test) - (TP + FP + FN)

  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  tpr = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall / Sensitivity
  tnr = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
  fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
  fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
  f1 = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0

  class_name = iris.target_names[c]
  print(f"Class {c} ({class_name}):")
  print(f"  * Precision : {prec:.4f}")
  print(f"  * Recall/TPR: {tpr:.4f}")
  print(f"  * Specificity/TNR: {tnr:.4f}")
  print(f"  * FPR       : {fpr:.4f}")
  print(f"  * FNR       : {fnr:.4f}")
  print(f"  * F1-Score  : {f1:.4f}\n")