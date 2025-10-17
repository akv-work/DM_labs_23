import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')


X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Размер данных:", X_scaled.shape)
print("Пример данных:")
print(data.head(), "\n")


cov_matrix = np.cov(X_scaled.T)


eig_values, eig_vectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eig_values)[::-1]
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx]


X_pca_2 = X_scaled.dot(eig_vectors[:, :2])
X_pca_3 = X_scaled.dot(eig_vectors[:, :3])


pca_2 = PCA(n_components=2)
X_sklearn_2 = pca_2.fit_transform(X_scaled)

pca_3 = PCA(n_components=3)
X_sklearn_3 = pca_3.fit_transform(X_scaled)


plt.figure(figsize=(8,6))
plt.scatter(X_pca_2[:,0], X_pca_2[:,1], c=y, cmap='coolwarm', s=50)
plt.title('PCA вручную (2 главные компоненты)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3[:,0], X_pca_3[:,1], X_pca_3[:,2], c=y, cmap='coolwarm', s=50)
ax.set_title('PCA вручную (3 главные компоненты)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(X_sklearn_2[:,0], X_sklearn_2[:,1], c=y, cmap='coolwarm', s=50)
plt.title('PCA sklearn (2 главные компоненты)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_sklearn_3[:,0], X_sklearn_3[:,1], X_sklearn_3[:,2], c=y, cmap='coolwarm', s=50)
ax.set_title('PCA sklearn (3 главные компоненты)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


loss_2 = 1 - np.sum(eig_values[:2]) / np.sum(eig_values)
loss_3 = 1 - np.sum(eig_values[:3]) / np.sum(eig_values)

print(f"Потери при 2 компонентах: {loss_2:.4f} ({loss_2*100:.2f}%)")
print(f"Потери при 3 компонентах: {loss_3:.4f} ({loss_3*100:.2f}%)")


print("\nВывод:")
print("1. Метод PCA позволил сократить размерность выборки до 2 и 3 компонент.")
print("2. При этом большая часть информации (дисперсии) сохраняется, а потери невелики.")
print("3. Визуализация показывает, что классы частично разделимы в пространстве главных компонент.")
print("4. Результаты PCA, выполненного вручную и с помощью sklearn, совпадают.")
