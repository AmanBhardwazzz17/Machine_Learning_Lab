# PCA on Iris using SVD (scipy.linalg.svd)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
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

# 2. Standardize (centering is required, otherwise SVD != PCA)
X_scaled = StandardScaler().fit_transform(X)

# 3. SVD:  X = U . Sigma . V^T
u, s, v = svd(X_scaled)
sigma = np.diag(s)
print("Sigma (singular values on diagonal):\n", sigma)

# 4. Project onto top-2 components  (U[:, :4] . Sigma[:, :2])
X_new = np.dot(u[:, :4], sigma[:, :2])
print("X_new shape:", X_new.shape)
print(X_new[:5])

# 5. Explained variance = s^2 / sum(s^2)
evr = s**2 / np.sum(s**2)
print("Explained variance ratio:", evr)
print("Cumulative (2 comps)    :", evr[:2].sum())

# 6. Cross-check with sklearn PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)
print("Matches sklearn PCA (up to sign):", np.allclose(np.abs(X_new), np.abs(X_pca)))

# 7. Visuals
plt.figure(figsize=(7, 6))
for k, c in zip(range(3), ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    plt.scatter(X_new[Y == k, 0], X_new[Y == k, 1], c=c, edgecolor="k", s=50,
                label=iris.target_names[k])
plt.xlabel("Component 1"); plt.ylabel("Component 2")
plt.title("Iris - PCA via SVD (4D -> 2D)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/iris_pca_svd.png", dpi=150); plt.show()

plt.figure(figsize=(6, 4))
plt.bar(range(1, 5), evr * 100, color="#4c72b0")
plt.plot(range(1, 5), np.cumsum(evr) * 100, "o-", color="#c44e52")
plt.xticks(range(1, 5)); plt.xlabel("Component"); plt.ylabel("Variance (%)")
plt.title("Iris - variance from singular values"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/iris_svd_variance.png", dpi=150); plt.show()