# PCA on Iris using the in-built sklearn.decomposition.PCA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# 1. Load Dataset
iris = load_iris()
df = pd.DataFrame(
    data=iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
df["species"] = iris.target

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
Y = df["species"]

# 2. Standardize Features
X_scaled = StandardScaler().fit_transform(X)

# 3. PCA with 2 components
pca = PCA(n_components=2)
X_new = pca.fit_transform(X_scaled)

print("Shape before:", X.shape, "-> after:", X_new.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance     :", np.cumsum(pca.explained_variance_ratio_))
print("Principal components:\n", pca.components_)

# 4. Cross-check with from-scratch eigen decomposition (file 1)
cov = np.cov(X_scaled, rowvar=False)
values, vectors = np.linalg.eigh(cov)
idx = np.argsort(values)[::-1]
scratch = X_scaled @ vectors[:, idx][:, :2]
print("Matches from-scratch result (up to sign):", np.allclose(np.abs(scratch), np.abs(X_new)))

# 5. Visuals
plt.figure(figsize=(7, 6))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn", fmt=".2f")
plt.title("Iris - Correlation heatmap (before PCA)")
plt.tight_layout(); plt.savefig(f"{OUT}/iris_corr_heatmap.png", dpi=150); plt.show()

plt.figure(figsize=(7, 6))
for k, c in zip(range(3), ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    plt.scatter(X_new[Y == k, 0], X_new[Y == k, 1], c=c, edgecolor="k", s=50,
                label=iris.target_names[k])
ev = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({ev[0]:.1f}%)"); plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
plt.title("Iris - PCA (sklearn, 4D -> 2D)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/iris_pca_sklearn.png", dpi=150); plt.show()