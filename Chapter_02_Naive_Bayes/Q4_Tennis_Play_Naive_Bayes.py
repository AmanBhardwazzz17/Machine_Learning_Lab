# Dataset: Weather, Play Tennis
dataset = [
    ["Sunny", "No"],
    ["Sunny", "No"],
    ["Overcast", "Yes"],
    ["Rain", "Yes"],
    ["Rain", "Yes"],
    ["Rain", "No"],
    ["Overcast", "Yes"],
    ["Sunny", "No"],
    ["Sunny", "Yes"],
    ["Rain", "Yes"],
]

total_samples = len(dataset)

# 1. Prior Probabilities P(Yes) and P(No)[cite: 5]
yes_count = sum(1 for row in dataset if row[1] == "Yes")
no_count = sum(1 for row in dataset if row[1] == "No")

p_yes = yes_count / total_samples
p_no = no_count / total_samples

# 2. Likelihoods P(Sunny|Yes) and P(Sunny|No)[cite: 5]
sunny_given_yes = sum(
    1 for row in dataset if row[0] == "Sunny" and row[1] == "Yes"
)
sunny_given_no = sum(
    1 for row in dataset if row[0] == "Sunny" and row[1] == "No"
)

p_sunny_given_yes = sunny_given_yes / yes_count
p_sunny_given_no = sunny_given_no / no_count

# 3. Posterior Probabilities P(Yes|Sunny) and P(No|Sunny)[cite: 5]
p_yes_sunny = p_sunny_given_yes * p_yes
p_no_sunny = p_sunny_given_no * p_no

# 4. Final Prediction[cite: 5]
prediction = "Yes" if p_yes_sunny > p_no_sunny else "No"

# Display Intermediate Calculations[cite: 5]
print("--- INTERMEDIATE CALCULATIONS ---")
print(f"P(Yes) = {p_yes:.2f}, P(No) = {p_no:.2f}")
print(
    f"P(Sunny|Yes) = {p_sunny_given_yes:.2f}, P(Sunny|No) ="
    f" {p_sunny_given_no:.2f}"
)
print(f"P(Yes|Sunny) Score = {p_yes_sunny:.4f}")
print(f"P(No|Sunny) Score  = {p_no_sunny:.4f}")
print("---------------------------------")
print(f"Final Prediction for Weather = Sunny: {prediction}")