try:
    import pandas as pd  # type: ignore[import-not-found]
    from sklearn.metrics import accuracy_score  # type: ignore[import-not-found]
    from sklearn.preprocessing import LabelEncoder  # type: ignore[import-not-found]
    from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit("Install the required packages first: pip install pandas scikit-learn") from exc

# 1. Load the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal'],
    'Windy': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# 2. Calculate Gini Impurity for Target Variable[cite: 6]
total = len(df)
yes_cnt = (df['Play Tennis'] == 'Yes').sum()
no_cnt = (df['Play Tennis'] == 'No').sum()
gini_target = 1 - ((yes_cnt / total) ** 2 + (no_cnt / total) ** 2)

# Label Encoding for Decision Tree
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['Play Tennis']

# 3. Build Decision Tree Classifier using Gini Impurity[cite: 6]
clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X, y)

# 4. Predict for new example: Outlook=Sunny, Temperature=Cool, Humidity=Normal, Windy=Yes[cite: 6]
new_sample = pd.DataFrame([{
    'Outlook': encoders['Outlook'].transform(['Sunny'])[0],
    'Temperature': encoders['Temperature'].transform(['Cool'])[0],
    'Humidity': encoders['Humidity'].transform(['Normal'])[0],
    'Windy': encoders['Windy'].transform(['Yes'])[0]
}])

pred_encoded = clf.predict(new_sample)
prediction = encoders['Play Tennis'].inverse_transform(pred_encoded)[0]

# Display Results
print("--- ASSIGNMENT 1: PLAY TENNIS DECISION TREE ---")
print(f"Target Variable Gini Impurity: {gini_target:.4f}")
print(f"Prediction for new example: Play Tennis = {prediction}")

# Display Accuracy on Training Data[cite: 6]
y_pred = clf.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")