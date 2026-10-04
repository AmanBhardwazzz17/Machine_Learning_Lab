# Dataset: Square Feet (X) vs Price in $1000s (Y)
X = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
Y = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]

# Mean Calculation
n = len(X)
mean_x = sum(X) / n
mean_y = sum(Y) / n

# Calculating Slope (m) and Intercept (c) -> Y = m*X + c
numerator = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

m = numerator / denominator
c = mean_y - (m * mean_x)

print(f"Linear Regression Equation: Y = {m:.4f} * X + {c:.4f}\n")


# Prediction Function
def predict_price(sq_ft):
  return m * sq_ft + c


# Test Predictions
test_sizes = [3000, 2000, 1500]
print("--- LINEAR REGRESSION PREDICTIONS ---")
for size in test_sizes:
  predicted_price = predict_price(size)
  print(f"{size} sq.ft -> ${predicted_price * 1000:.2f}")