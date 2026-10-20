"""
ML Lab Assignment 5 - PCA on the Wine dataset (13 features -> 2 components)
Tasks: load -> split X/y -> standardize -> PCA(2) -> explained variance
       -> scatter plot -> original vs reduced / information lost
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ---- Task 1 & 2: load data, separate X and y ----
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
names = wine.target_names
print("Shape:", X.shape, "| Classes:", [str(n) for n in names], "| Counts:", np.bincount(y))

# ---- Task 3: standardize ----
X_scaled = StandardScaler().fit_transform(X)

# ---- Task 4: PCA 13 -> 2 ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Reduced shape:", X_pca.shape)

# ---- Task 5: explained variance ----
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
for i, (e, c) in enumerate(zip(evr, cum), 1):
    print(f"PC{i}: explained = {e*100:.2f}%  cumulative = {c*100:.2f}%")
print(f"Information lost: {(1-cum[-1])*100:.2f}%")

full_evr = PCA().fit(X_scaled).explained_variance_ratio_  # for scree plot

# ---- Task 6: scatter plot ----
plt.figure(figsize=(8, 6))
for k, c in enumerate(COLORS):
    plt.scatter(X_pca[y == k, 0], X_pca[y == k, 1], c=c, label=names[k],
                edgecolor="k", s=55, alpha=0.85)
plt.xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
plt.title("PCA of Wine Dataset (13D -> 2D)")
plt.legend(title="Wine class"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{OUT}/wine_pca_scatter.png", dpi=150); plt.show()

# scree + cumulative variance
plt.figure(figsize=(8, 5))
idx = np.arange(1, 14)
plt.bar(idx, full_evr * 100, color="#4c72b0", label="Individual")
plt.plot(idx, np.cumsum(full_evr) * 100, "o-", color="#c44e52", label="Cumulative")
plt.axhline(95, ls="--", c="gray"); plt.text(1, 96.5, "95% threshold", color="gray")
plt.xticks(idx); plt.xlabel("Principal component"); plt.ylabel("Explained variance (%)")
plt.title("Scree plot & cumulative variance"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/wine_scree_plot.png", dpi=150); plt.show()

# loadings heatmap
plt.figure(figsize=(7, 6))
load = pca.components_.T
plt.imshow(load, cmap="coolwarm", aspect="auto", vmin=-.6, vmax=.6)
plt.colorbar(label="Loading")
plt.yticks(range(13), wine.feature_names); plt.xticks([0, 1], ["PC1", "PC2"])
for i in range(13):
    for j in range(2):
        plt.text(j, i, f"{load[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Feature loadings"); plt.tight_layout()
plt.savefig(f"{OUT}/wine_loadings_heatmap.png", dpi=150); plt.show()

# ---- Task 7: original vs reduced, information lost ----
X_back = pca.inverse_transform(X_pca)
mse = np.mean((X_scaled - X_back) ** 2)
per_feat = np.mean((X_scaled - X_back) ** 2, axis=0)
print(f"Original shape {X_scaled.shape} -> reduced {X_pca.shape}")
print(f"Reconstruction MSE (standardized units): {mse:.4f}")
print("Per-feature reconstruction error:")
for n, e in sorted(zip(wine.feature_names, per_feat), key=lambda t: -t[1]):
    print(f"  {n:32s} {e:.3f}")

plt.figure(figsize=(8, 4.5))
order = np.argsort(per_feat)[::-1]
plt.barh([wine.feature_names[i] for i in order][::-1], per_feat[order][::-1], color="#dd8452")
plt.xlabel("Reconstruction error (standardized units)")
plt.title("Information lost per feature after 13D -> 2D")
plt.tight_layout(); plt.savefig(f"{OUT}/wine_information_loss.png", dpi=150); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for k, c in enumerate(COLORS):
    ax[0].scatter(X_scaled[y == k, 0], X_scaled[y == k, 6], c=c, edgecolor="k", s=40, label=names[k])
    ax[1].scatter(X_pca[y == k, 0], X_pca[y == k, 1], c=c, edgecolor="k", s=40, label=names[k])
ax[0].set(xlabel="alcohol (scaled)", ylabel="flavanoids (scaled)", title="Original: 2 of 13 features")
ax[1].set(xlabel="PC1", ylabel="PC2", title="PCA: 2 components (all 13 features combined)")
for a in ax:
    a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/wine_original_vs_pca.png", dpi=150); plt.show()

# ---- Bonus 1: same result via SVD ----
U, s, Vt = svd(X_scaled, full_matrices=False)
X_svd = X_scaled @ Vt[:2].T
print("SVD explained variance ratio:", np.round((s**2 / np.sum(s**2))[:2], 4))
print("SVD matches sklearn PCA (up to sign):", np.allclose(np.abs(X_svd), np.abs(X_pca)))

# ---- Bonus 2: eigen-decomposition of covariance matrix ----
vals, vecs = np.linalg.eigh(np.cov(X_scaled.T))
vals = vals[::-1]
print("Eigen EVR:", np.round((vals / vals.sum())[:2], 4))

# ---- Bonus 3: does 2D keep the class information? ----
a_full = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y, cv=5).mean()
a_pca = cross_val_score(LogisticRegression(max_iter=1000), X_pca, y, cv=5).mean()
print(f"LogReg 5-fold accuracy: original 13D = {a_full:.4f}, PCA 2D = {a_pca:.4f}")