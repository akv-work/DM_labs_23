
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=186)

X = dataset.data.features
y = dataset.data.targets

print("Размер данных:", X.shape)
print("Первые 5 строк:")
print(X.head())

X = X.fillna(X.mean())
y = y.fillna(y.mean())

le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X.loc[:, col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = X_train.shape[1]
autoencoder = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {loss.item():.4f}")

with torch.no_grad():
    X_train_encoded = autoencoder.encoder(X_train)
    X_test_encoded = autoencoder.encoder(X_test)

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.fc(x)

model = RegressionModel(6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_encoded)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Эпоха [{epoch + 1}/{num_epochs}], Потеря: {loss.item():.4f}")
model.eval()
with torch.no_grad():
    preds = model(X_test_encoded).numpy()
    true = y_test.numpy()
    mse = mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    mape = mean_absolute_percentage_error(true, preds)

print("\n=== Результаты ===")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape * 100:.2f}%")
