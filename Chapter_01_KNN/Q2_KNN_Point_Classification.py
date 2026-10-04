import math

# Dataset: X, Y, Class[cite: 5]
dataset = [
    [1, 2, "A"],
    [2, 3, "A"],
    [3, 4, "A"],
    [6, 7, "B"],
    [7, 8, "B"],
    [8, 9, "B"],
]

# Test Point: (5,6), K = 3[cite: 5]
test_point = [5, 6]
K = 3


def euclidean_distance(p1, p2):
  return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


distances = []
for row in dataset:
  dist = euclidean_distance(test_point, row[:2])
  distances.append((dist, row[2]))

distances.sort(key=lambda x: x[0])
k_nearest = [label for _, label in distances[:K]]
prediction = max(set(k_nearest), key=k_nearest.count)

print("Q2 - Predicted Class for (5,6):", prediction)