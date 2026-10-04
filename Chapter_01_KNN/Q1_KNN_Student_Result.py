
import math

# Dataset: Study Hours, Attendance (%), Result
dataset = [
    [2, 60, "Fail"],
    [3, 65, "Fail"],
    [4, 70, "Pass"],
    [5, 75, "Pass"],
    [6, 80, "Pass"],
]

# Test Point: Study Hours = 4.5, Attendance = 72%, K = 3
test_point = [4.5, 72]
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

print("Q1 - Predicted Result:", prediction)