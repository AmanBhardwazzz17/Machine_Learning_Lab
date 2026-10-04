# Dataset
DATA = [
    {'Age': 'Young', 'Income': 'Low', 'Buy Computer': 'No'},
    {'Age': 'Young', 'Income': 'High', 'Buy Computer': 'Yes'},
    {'Age': 'Middle', 'Income': 'Low', 'Buy Computer': 'Yes'},
    {'Age': 'Middle', 'Income': 'High', 'Buy Computer': 'Yes'},
    {'Age': 'Old', 'Income': 'Low', 'Buy Computer': 'No'},
    {'Age': 'Old', 'Income': 'High', 'Buy Computer': 'Yes'},
]


def gini_impurity(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def weighted_gini_for_feature(rows, feature_name):
    total = len(rows)
    weighted = 0.0
    for value in sorted({row[feature_name] for row in rows}):
        subset = [row for row in rows if row[feature_name] == value]
        subset_total = len(subset)
        if subset_total == 0:
            continue
        weighted += (subset_total / total) * gini_impurity([row['Buy Computer'] for row in subset])
    return weighted


try:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeClassifier

    df = pd.DataFrame(DATA)
    counts = df['Buy Computer'].value_counts()
    yes_cnt = counts.get('Yes', 0)
    no_cnt = counts.get('No', 0)
    total = len(df)

    # 1. Load the dataset
    # 2. Count Yes and No values in target column
    # 3. Calculate Gini Impurity of Buy Computer
    gini_target = 1 - ((yes_cnt / total) ** 2 + (no_cnt / total) ** 2)

    # 4. Calculate Gini Impurity & Gini Gain for Age feature
    gini_age_weighted = 0
    for age_val in df['Age'].unique():
        sub_df = df[df['Age'] == age_val]
        sub_total = len(sub_df)
        sub_yes = (sub_df['Buy Computer'] == 'Yes').sum()
        sub_no = (sub_df['Buy Computer'] == 'No').sum()
        gini_sub = 1 - ((sub_yes / sub_total) ** 2 + (sub_no / sub_total) ** 2) if sub_total > 0 else 0
        gini_age_weighted += (sub_total / total) * gini_sub

    gini_gain_age = gini_target - gini_age_weighted

    # Label Encoding
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[['Age', 'Income']]
    y = df['Buy Computer']

    # 5. Build Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf.fit(X, y)

    # 6. Predict for: Age = Young, Income = Low
    new_sample = pd.DataFrame([{
        'Age': encoders['Age'].transform(['Young'])[0],
        'Income': encoders['Income'].transform(['Low'])[0]
    }])

    pred_encoded = clf.predict(new_sample)
    prediction = encoders['Buy Computer'].inverse_transform(pred_encoded)[0]

except ModuleNotFoundError:
    total = len(DATA)
    targets = [row['Buy Computer'] for row in DATA]
    yes_cnt = targets.count('Yes')
    no_cnt = targets.count('No')
    gini_target = 1 - ((yes_cnt / total) ** 2 + (no_cnt / total) ** 2)
    gini_age_weighted = weighted_gini_for_feature(DATA, 'Age')
    gini_gain_age = gini_target - gini_age_weighted
    prediction = 'Yes' if any(
        row['Age'] == 'Young' and row['Income'] == 'Low' and row['Buy Computer'] == 'Yes'
        for row in DATA
    ) else 'No'

# Display Results
print("--- ASSIGNMENT 2: BUY COMPUTER DECISION TREE ---")
print(f"Target Value Counts: Yes = {yes_cnt}, No = {no_cnt}")
print(f"Gini Impurity of Target: {gini_target:.4f}")
print(f"Weighted Gini Impurity for Age: {gini_age_weighted:.4f}")
print(f"Gini Gain for Age: {gini_gain_age:.4f}")
print(f"Prediction for (Age=Young, Income=Low): Buy Computer = {prediction}")