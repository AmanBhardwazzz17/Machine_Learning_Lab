import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Data Load & Split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data.tolist(), iris.target.tolist(), test_size=0.2, random_state=42
)


# 2. Q1: Gaussian Naive Bayes from Scratch
def mean(lst):
  return sum(lst) / len(lst)


def stdev(lst):
  m = mean(lst)
  return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))


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


def prob(x, m, s):
  return (1 / (math.sqrt(2 * math.pi) * s)) * math.exp(
      -((x - m) ** 2) / (2 * s**2)
  )


def predict(sample):
  scores = {
      c: math.prod(
          prob(sample[j], stats[c][j][0], stats[c][j][1]) for j in range(4)
      )
      for c in stats
  }
  return max(scores, key=scores.get)


y_pred = [predict(s) for s in X_test]