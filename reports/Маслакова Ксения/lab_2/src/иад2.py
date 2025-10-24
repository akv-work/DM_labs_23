import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("winequality-white.csv", sep=';')

print("Размер данных:", data.shape)
print("Пример данных:")
print(data.head())

X = data.drop('quality', axis=1).values
y = data['quality'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(model, loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch_X, _ in loader:
            optimizer.zero_grad()
            decoded, _ = model(batch_X)
            loss = criterion(decoded, batch_X)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Эпоха [{epoch+1}/{epochs}], Потеря: {loss.item():.6f}")


autoencoder_2 = Autoencoder(X.shape[1], 2)
autoencoder_3 = Autoencoder(X.shape[1], 3)

print("\nОбучение автоэнкодера с 2 нейронами...")
train_autoencoder(autoencoder_2, loader, epochs=100)

print("\nОбучение автоэнкодера с 3 нейронами...")
train_autoencoder(autoencoder_3, loader, epochs=100)

with torch.no_grad():
    _, encoded_2 = autoencoder_2(X_tensor)
    _, encoded_3 = autoencoder_3(X_tensor)

encoded_2 = encoded_2.numpy()
encoded_3 = encoded_3.numpy()

plt.figure(figsize=(7, 5))
scatter = plt.scatter(encoded_2[:, 0], encoded_2[:, 1], c=y, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label='Класс (качество вина)')
plt.title('Автоэнкодер: 2 нейрона (2D)')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(encoded_3[:, 0], encoded_3[:, 1], encoded_3[:, 2], c=y, cmap='rainbow', alpha=0.7)
fig.colorbar(p, ax=ax, label='Класс (качество вина)')
ax.set_title('Автоэнкодер: 3 нейрона (3D)')
plt.show()

print("\nВыполняется t-SNE...")
tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label='Класс (качество вина)')
plt.title('t-SNE: 2 компоненты')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], c=y, cmap='rainbow', alpha=0.7)
fig.colorbar(p, ax=ax, label='Класс (качество вина)')
ax.set_title('t-SNE: 3 компоненты')
plt.show()

pca_2 = PCA(n_components=2)
pca_3 = PCA(n_components=3)
X_pca_2 = pca_2.fit_transform(X_scaled)
X_pca_3 = pca_3.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y, cmap='rainbow', alpha=0.7)
plt.colorbar(scatter, label='Класс (качество вина)')
plt.title('PCA: 2 компоненты')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y, cmap='rainbow', alpha=0.7)
fig.colorbar(p, ax=ax, label='Класс (качество вина)')
ax.set_title('PCA: 3 компоненты')
plt.show()

explained_var_2 = np.sum(pca_2.explained_variance_ratio_)
explained_var_3 = np.sum(pca_3.explained_variance_ratio_)
print(f"\nПотери при 2 компонентах PCA: {1 - explained_var_2:.4f} ({(1 - explained_var_2) * 100:.2f}%)")
print(f"Потери при 3 компонентах PCA: {1 - explained_var_3:.4f} ({(1 - explained_var_3) * 100:.2f}%)")

