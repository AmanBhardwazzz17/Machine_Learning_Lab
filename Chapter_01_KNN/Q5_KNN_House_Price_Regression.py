x = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]  #[cite: 8]
y = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]  #[cite: 8]


def knn_predict(x_values, y_values, test_points, k=3):  #[cite: 8]
  results = {}  #[cite: 8]
  for t in test_points:  #[cite: 8]
    distances = [
        (abs(x_values[i] - t), y_values[i]) for i in range(len(x_values))
    ]  #[cite: 8]
    distances.sort(key=lambda d: d[0])  #[cite: 8]
    neighbors = distances[:k]  #[cite: 8]
    avg_price = sum(price for _, price in neighbors) / k  #[cite: 8]
    results[t] = avg_price  #[cite: 8]
  return results  #[cite: 8]


test = [3000, 2000, 1500]  #[cite: 8]
predictions = knn_predict(x, y, test, k=3)  #[cite: 8]

for sq_ft, price in predictions.items():  #[cite: 8]
  print(f"{sq_ft} sq.ft -> ${price * 1000:.2f}")  #[cite: 8]