
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
OUTDIR = "seeds_pca_results"
os.makedirs(OUTDIR, exist_ok=True)

col_names = [
    'area', 'perimeter', 'compactness',
    'length_kernel', 'width_kernel',
    'asymmetry_coef', 'length_groove', 'class'
]

df = pd.read_csv(DATA_URL, sep='\s+', header=None, names=col_names)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df['class'].value_counts())

X = df.drop(columns=['class']).copy()
y = df['class'].copy()

print("\nMissing values per column:\n", X.isna().sum())

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_centered = X_scaled - np.mean(X_scaled, axis=0)

cov = np.cov(X_centered, rowvar=False)  # shape (7,7)

eig_vals, eig_vecs = np.linalg.eig(cov)

idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx].real
eig_vecs_sorted = eig_vecs[:, idx].real

PC_manual_2 = X_centered.dot(eig_vecs_sorted[:, :2])
PC_manual_3 = X_centered.dot(eig_vecs_sorted[:, :3])

total_variance = eig_vals_sorted.sum()
explained_variance_ratio_manual = eig_vals_sorted / total_variance
cumulative_explained = np.cumsum(explained_variance_ratio_manual)

print("\nEigenvalues (sorted):\n", eig_vals_sorted)
print("Explained variance ratio (manual):\n", explained_variance_ratio_manual)
print("Cumulative explained (manual):\n", cumulative_explained)

pca = PCA(n_components=7)
scores_sklearn = pca.fit_transform(X_scaled)
explained_ratio_sklearn = pca.explained_variance_ratio_
cum_explained_sklearn = np.cumsum(explained_ratio_sklearn)

print("\nExplained variance ratio (sklearn):\n", explained_ratio_sklearn)
print("Cumulative explained (sklearn):\n", cum_explained_sklearn)

PC_sklearn_2 = scores_sklearn[:, :2]
PC_sklearn_3 = scores_sklearn[:, :3]

import matplotlib
matplotlib.use('Agg')

def plot_2d(pc_scores, labels, title, outpath):
    plt.figure(figsize=(7,6))
    classes = np.unique(labels)
    markers = ['o', '^', 's']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, cls in enumerate(classes):
        mask = labels == cls
        plt.scatter(pc_scores[mask,0], pc_scores[mask,1],
                    label=str(cls), marker=markers[i%len(markers)], alpha=0.8)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(title)
    plt.legend(title='class')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

plot_2d(PC_manual_2, y.values, "PCA Manual (first 2 PCs)", os.path.join(OUTDIR, "manual_pca_2d.png"))
plot_2d(PC_sklearn_2, y.values, "PCA sklearn (first 2 PCs)", os.path.join(OUTDIR, "sklearn_pca_2d.png"))

def plot_3d(pc_scores, labels, title, outpath):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    classes = np.unique(labels)
    markers = ['o', '^', 's']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, cls in enumerate(classes):
        mask = labels == cls
        ax.scatter(pc_scores[mask,0], pc_scores[mask,1], pc_scores[mask,2],
                   label=str(cls), marker=markers[i%len(markers)], alpha=0.8)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend(title='class')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

plot_3d(PC_manual_3, y.values, "PCA Manual (first 3 PCs)", os.path.join(OUTDIR, "manual_pca_3d.png"))
plot_3d(PC_sklearn_3, y.values, "PCA sklearn (first 3 PCs)", os.path.join(OUTDIR, "sklearn_pca_3d.png"))

def variance_loss_by_k(eigvals_sorted, k):
    total = eigvals_sorted.sum()
    lost = eigvals_sorted[k:].sum()
    return lost / total

for k in (1,2,3,4,5,6,7):
    lost_frac = variance_loss_by_k(eig_vals_sorted, k)
    print(f"Using {k} components -> lost variance fraction = {lost_frac:.6f} ({lost_frac*100:.3f}%)")

def reconstruction_mse(X_scaled, pca_full, k):
    components_k = pca_full.components_[:k]
    mean = pca_full.mean_
    scores = (X_scaled - mean).dot(components_k.T)
    X_rec = scores.dot(components_k) + mean
    mse = np.mean((X_scaled - X_rec)**2)
    return mse

for k in (1,2,3,4,5,6,7):
    mse = reconstruction_mse(X_scaled, pca, k)
    print(f"Reconstruction MSE with {k} components: {mse:.6f}")

results = {
    "eig_vals_sorted": eig_vals_sorted.tolist(),
    "explained_variance_ratio_manual": explained_variance_ratio_manual.tolist(),
    "explained_variance_ratio_sklearn": explained_ratio_sklearn.tolist(),
    "cumulative_manual": cumulative_explained.tolist(),
    "cumulative_sklearn": cum_explained_sklearn.tolist()
}
with open(os.path.join(OUTDIR, "pca_results_summary.json"), "w") as f:
    import json
    json.dump(results, f, indent=2)

print("\nPlots and summary saved to:", OUTDIR)
