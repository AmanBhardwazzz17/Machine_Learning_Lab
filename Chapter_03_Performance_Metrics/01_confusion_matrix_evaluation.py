from sklearn.metrics import confusion_matrix, classification_report

# Actual and Predicted values
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Confusion Matrix calculation
cm = confusion_matrix(y_true, y_pred)
print("--- Confusion Matrix ---")
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))