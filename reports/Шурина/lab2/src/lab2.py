import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import arff

# -------------------------------
# 1. Загрузка данных из .arff
# -------------------------------
data_path = r"C:\Users\User\Desktop\Studing-7sem\IAD\lab2\Rice_Cammeo_Osmancik.arff"
data_arff = arff.loadarff(data_path)
data_df = pd.DataFrame(data_arff[0])

# Предполагаем, что последняя колонка — это класс, возможно в формате bytes
X = data_df.iloc[:, :-1].values
y = data_df.iloc[:, -1].values

# Если классы в байтах, конвертируем в строки
if y.dtype.kind == 'S':
    y = np.array([val.decode('utf-8') for val in y])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 2. Автоэнкодер
# -------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(X_scaled, bottleneck_size=2, epochs=100, lr=0.001):
    input_size = X_scaled.shape[1]
    model = Autoencoder(input_size, bottleneck_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_scaled)
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded, decoded = model(X_tensor)
        loss = criterion(decoded, X_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    return model

# Пример для 2D и 3D узкого слоя
ae_2d = train_autoencoder(X_scaled, bottleneck_size=2)
ae_3d = train_autoencoder(X_scaled, bottleneck_size=3)

# -------------------------------
# 3. Визуализация кодировки автоэнкодера
# -------------------------------
def plot_encoded(encoded_data, labels, title="Autoencoder 2D", dim=2):
    encoded_data = encoded_data.detach().numpy()
    if dim == 2:
        plt.figure(figsize=(8,6))
        for cls in np.unique(labels):
            plt.scatter(encoded_data[labels==cls, 0], encoded_data[labels==cls, 1], label=f'Class {cls}')
        plt.legend()
        plt.title(title)
        plt.show()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(labels):
            ax.scatter(encoded_data[labels==cls, 0], encoded_data[labels==cls, 1],
                       encoded_data[labels==cls, 2], label=f'Class {cls}')
        ax.set_title(title)
        ax.legend()
        plt.show()

encoded_2d, _ = ae_2d(torch.FloatTensor(X_scaled))
encoded_3d, _ = ae_3d(torch.FloatTensor(X_scaled))

plot_encoded(encoded_2d, y, title="Autoencoder 2D", dim=2)
plot_encoded(encoded_3d, y, title="Autoencoder 3D", dim=3)

# -------------------------------
# 4. t-SNE (2D и 3D)
# -------------------------------
def plot_tsne(X_scaled, labels, dim=2, perplexity=30):
    tsne = TSNE(n_components=dim, perplexity=perplexity, init='pca', random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    if dim == 2:
        plt.figure(figsize=(8,6))
        for cls in np.unique(labels):
            plt.scatter(X_tsne[labels==cls, 0], X_tsne[labels==cls, 1], label=f'Class {cls}')
        plt.legend()
        plt.title(f"t-SNE 2D (perplexity={perplexity})")
        plt.show()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(labels):
            ax.scatter(X_tsne[labels==cls, 0], X_tsne[labels==cls, 1], X_tsne[labels==cls, 2], label=f'Class {cls}')
        ax.set_title(f"t-SNE 3D (perplexity={perplexity})")
        ax.legend()
        plt.show()

plot_tsne(X_scaled, y, dim=2, perplexity=30)
plot_tsne(X_scaled, y, dim=3, perplexity=30)

# -------------------------------
# 5. PCA (2D и 3D)
# -------------------------------
def plot_pca(X_scaled, labels, dim=2):
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(X_scaled)
    if dim == 2:
        plt.figure(figsize=(8,6))
        for cls in np.unique(labels):
            plt.scatter(X_pca[labels==cls, 0], X_pca[labels==cls, 1], label=f'Class {cls}')
        plt.legend()
        plt.title("PCA 2D")
        plt.show()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for cls in np.unique(labels):
            ax.scatter(X_pca[labels==cls, 0], X_pca[labels==cls, 1], X_pca[labels==cls, 2], label=f'Class {cls}')
        ax.set_title("PCA 3D")
        ax.legend()
        plt.show()

plot_pca(X_scaled, y, dim=2)
plot_pca(X_scaled, y, dim=3)
