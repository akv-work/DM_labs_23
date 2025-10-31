import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Визуализация ===
def plot_pca(X_reduced, y):
    for class_value in np.unique(y):
        plt.scatter(X_reduced[y == class_value, 0],
                    X_reduced[y == class_value, 1],
                    label=f'Class {class_value}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA — 2D проекция')
    plt.show()

def plot_pca_3d(X_reduced_3, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_classes = np.unique(y)
    colors = plt.get_cmap('viridis', len(unique_classes))
    for i, class_value in enumerate(unique_classes):
        ax.scatter(X_reduced_3[y == class_value, 0],
                   X_reduced_3[y == class_value, 1],
                   X_reduced_3[y == class_value, 2],
                   label=f'Class {class_value}',
                   color=colors(i))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.title('PCA — 3D проекция')
    plt.show()

# === Загрузка и очистка данных ===
data = pd.read_csv(r"C:\Users\User\Desktop\Studing-7sem\IAD\lab1\Exasens.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print("Столбцы после очистки:", data.columns.tolist())

# === Разделение признаков и меток ===
y = data["Diagnosis"]
X = data.drop(columns=["Diagnosis", "ID"])

# Преобразуем категориальные переменные
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Нормализация
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Добавляем это ---
y = y.astype(str)   # <--- Преобразуем метки в строки
# ----------------------

# PCA вручную
cov_matrix = np.cov(X_scaled, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

# 2D и 3D проекции
eigenvector_subset_2 = sorted_eigenvectors[:, 0:2]
eigenvector_subset_3 = sorted_eigenvectors[:, 0:3]

X_reduced_2 = np.dot(X_scaled, eigenvector_subset_2)
X_reduced_3 = np.dot(X_scaled, eigenvector_subset_3)

# Визуализация
plot_pca(X_reduced_2, y)
plot_pca_3d(X_reduced_3, y)

# === PCA через sklearn ===
pca = PCA(n_components=2)
X_reduced_sklearn_2 = pca.fit_transform(X_scaled)

pca_3 = PCA(n_components=3)
X_reduced_sklearn_3 = pca_3.fit_transform(X_scaled)

plot_pca(X_reduced_sklearn_2, y)
plot_pca_3d(X_reduced_sklearn_3, y)

# === Объяснённая дисперсия ===
explained_variance_2 = np.sum(sorted_eigenvalue[:2]) / np.sum(sorted_eigenvalue)
explained_variance_3 = np.sum(sorted_eigenvalue[:3]) / np.sum(sorted_eigenvalue)

print(f"\nОбъяснённая дисперсия для 2D проекции: {explained_variance_2 * 100:.2f}%")
print(f"Объяснённая дисперсия для 3D проекции: {explained_variance_3 * 100:.2f}%")
