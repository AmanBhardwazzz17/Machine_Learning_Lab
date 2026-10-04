import math

# Dataset: Area, Bedrooms, Age, Price (₹ lakh)[cite: 5]
dataset = [
    [1000, 2, 10, 45],
    [1200, 2, 8, 50],
    [1500, 3, 5, 65],
    [1800, 3, 4, 75],
    [2000, 4, 3, 90],
    [2200, 4, 2, 100],
    [2500, 4, 1, 120],
    [2800, 5, 2, 135],
]

# Test Data: 1700 sq.ft., 3 Bedroom, 5 years age, K = 3[cite: 5]
test_point = [1700, 3, 5]
K = 3


def euclidean_distance(p1, p2):
  return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


distances = []
for row in dataset:
  dist = euclidean_distance(test_point, row[:3])
  distances.append((dist, row[3]))

distances.sort(key=lambda x: x[0])
k_nearest_prices = [price for _, price in distances[:K]]
predicted_price = sum(k_nearest_prices) / K

print(f"Q3 - Predicted House Price: ₹{predicted_price:.2f} Lakh")