# pyright: reportMissingModuleSource=false
import matplotlib.pyplot as plt
import numpy as np  # Fixed: Missing import in original screenshot[cite: 6]
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
iris = load_iris()  #[cite: 6]
df = pd.DataFrame(
    data=iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)  #[cite: 6]
df["species"] = iris.target  #[cite: 6]

X = df[
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
]  #[cite: 6]
Y = df[["species"]]  #[cite: 6]

# Heatmap before PCA
plt.figure(figsize=(15, 15))  #[cite: 6]
p = sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")  #[cite: 6]
plt.show()

# 2. Standardize Features
X_scaled = StandardScaler().fit_transform(X)

# 3. Covariance Matrix & Eigen Values/Vectors
cov_matrix = np.cov(X_scaled, rowvar=False)
print("Covariance Matrix:\n", cov_matrix)

values, vectors = np.linalg.eig(cov_matrix)
values = np.real_if_close(values)
vectors = np.real_if_close(vectors)

# Sort eigenpairs in descending order of explained variance
sorted_index = np.argsort(values)[::-1]
values = values[sorted_index]
vectors = vectors[:, sorted_index]

# 4. Explained Variance Ratio
explained_variances = values / np.sum(values)
print("Explained Variances:\n", explained_variances)

# Plot Explained Variance
plt.figure(figsize=(6, 4))
plt.bar(
    range(1, len(explained_variances) + 1),
    explained_variances,
    alpha=0.5,
    align="center",
    label="individual explained variance",
)
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal components")
plt.xticks(range(1, len(explained_variances) + 1))
plt.show()

# 5. Projection onto PC1 and PC2
projected_1 = X_scaled @ vectors[:, 0]
projected_2 = X_scaled @ vectors[:, 1]

res = pd.DataFrame({"PC1": projected_1, "PC2": projected_2, "species": Y["species"].values})
print(res.head())

# Heatmap after PCA
plt.figure(figsize=(10, 10))
X_pca = res[["PC1", "PC2"]]
p = sns.heatmap(X_pca.corr(), annot=True, cmap="RdYlGn")
plt.show()

# Scatter Plot of Principal Components
plt.figure(figsize=(8, 6))
for species_id, species_name in enumerate(iris.target_names):
    subset = res[res["species"] == species_id]
    plt.scatter(subset["PC1"], subset["PC2"], label=species_name, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris dataset in PCA space")
plt.legend()
plt.show()