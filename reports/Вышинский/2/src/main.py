import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo


SAVE_FIGS = True
FIG_PATH = "./figs/"


rice_data = fetch_ucirepo(id=545)
X = rice_data.data.features
y = rice_data.data.targets.values  # shape (n_samples, 1)


le = LabelEncoder()
y_encoded = le.fit_transform(y.ravel())  # ravel(): (n,1) -> (n,)
target_names = le.classes_.tolist()  # ['Cammeo', 'Osmancik']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        hidden_dim = max(32, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(model, data_loader, epochs=50, lr=0.001, device="cpu"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in data_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg:.6f}")
    model.to("cpu")
    return model

dataset = TensorDataset(X_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

ae_2d = Autoencoder(input_dim=X.shape[1], latent_dim=2)
train_autoencoder(ae_2d, data_loader, epochs=50, lr=1e-3)
X_ae_2d = ae_2d.encode(X_tensor).detach().numpy()

ae_3d = Autoencoder(input_dim=X.shape[1], latent_dim=3)
train_autoencoder(ae_3d, data_loader, epochs=50, lr=1e-3)
X_ae_3d = ae_3d.encode(X_tensor).detach().numpy()


def plot_2d_markers(X_proj, y, title, save_as=None):
    plt.figure(figsize=(8,6))
    markers = ['o', '^', 's', 'P', 'X', 'D']  # на случай >2 классов
    unique_labels = np.unique(y)
    for i, lab in enumerate(unique_labels):
        idx = (y == lab)
        plt.scatter(X_proj[idx,0], X_proj[idx,1],
                    label=str(target_names[int(lab)]),
                    marker=markers[i % len(markers)],
                    alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(alpha=0.2)
    if save_as and SAVE_FIGS:
        plt.savefig(save_as, dpi=200, bbox_inches='tight')
    plt.show()

def plot_3d_markers(X_proj, y, title, save_as=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', '^', 's', 'P', 'X', 'D']
    unique_labels = np.unique(y)
    for i, lab in enumerate(unique_labels):
        idx = (y == lab)
        ax.scatter(X_proj[idx,0], X_proj[idx,1], X_proj[idx,2],
                   label=str(target_names[int(lab)]),
                   marker=markers[i % len(markers)],
                   alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    if save_as and SAVE_FIGS:
        plt.savefig(save_as, dpi=200, bbox_inches='tight')
    plt.show()


plot_2d_markers(X_ae_2d, y_encoded, 'Autoencoder 2D (latent=2)', save_as=FIG_PATH+"ae_2d.png")
plot_3d_markers(X_ae_3d, y_encoded, 'Autoencoder 3D (latent=3)', save_as=FIG_PATH+"ae_3d.png")


perplexities = [20, 30, 40, 50, 60]
best_perplexity = 50

tsne_2d = TSNE(n_components=2, perplexity=best_perplexity, init='pca', random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)
plot_2d_markers(X_tsne_2d, y_encoded, f"t-SNE 2D (perplexity={best_perplexity})", save_as=FIG_PATH+f"tsne2d_p{best_perplexity}.png")

tsne_3d = TSNE(n_components=3, perplexity=best_perplexity, init='pca', random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)
plot_3d_markers(X_tsne_3d, y_encoded, f"t-SNE 3D (perplexity={best_perplexity})", save_as=FIG_PATH+f"tsne3d_p{best_perplexity}.png")

for perp in perplexities:
    X_temp = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42).fit_transform(X_scaled)
    plot_2d_markers(X_temp, y_encoded, f"t-SNE 2D (perplexity={perp})", save_as=FIG_PATH+f"tsne2d_p{perp}.png")


pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print("Explained variance ratio (2D PCA):", pca_2d.explained_variance_ratio_)
plot_2d_markers(X_pca_2d, y_encoded, "PCA 2D Projection", save_as=FIG_PATH+"pca2d.png")

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
print("Explained variance ratio (3D PCA):", pca_3d.explained_variance_ratio_)
plot_3d_markers(X_pca_3d, y_encoded, "PCA 3D Projection", save_as=FIG_PATH+"pca3d.png")
