import math

# Dataset: Area, Bedrooms, Bathrooms, Age, Distance, Price[cite: 5]
dataset = [
    [1000, 2, 1, 15, 12, 42],
    [1200, 2, 2, 10, 10, 50],
    [1400, 3, 2, 8, 9, 60],
    [1600, 3, 2, 6, 8, 68],
    [1800, 3, 3, 5, 7, 78],
    [2000, 4, 3, 4, 6, 90],
    [2200, 4, 3, 3, 5, 105],
    [2400, 4, 4, 2, 4, 120],
    [2600, 5, 4, 2, 3, 135],
    [2800, 5, 4, 1, 2, 150],
    [3000, 5, 5, 1, 1, 170],
]

# Test Point: Area=1900, Bedrooms=3, Bathrooms=3, Age=5, Distance=6, K = 3[cite: 5]
test_point = [1900, 3, 3, 5, 6]
K = 3


def euclidean_distance(p1, p2):
  return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


distances = []
for row in dataset:
  dist = euclidean_distance(test_point, row[:5])
  distances.append((dist, row[5]))

distances.sort(key=lambda x: x[0])
k_nearest_prices = [price for _, price in distances[:K]]
predicted_price = sum(k_nearest_prices) / K

print(f"Q4 - Predicted House Price: ₹{predicted_price:.2f} Lakh")