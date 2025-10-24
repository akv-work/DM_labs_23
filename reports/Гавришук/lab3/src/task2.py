import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "aveOralF"
LR_AE = 1e-3
LR_REG = 1e-4
EPOCHS_AE = 100
EPOCHS_REG = 100
BATCH_SIZE = 32
LATENT_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Fetching dataset from UCI...")
infrared = fetch_ucirepo(id=925)
X_df = infrared.data.features.copy()
y_df = infrared.data.targets.copy()

if TARGET not in y_df.columns:
    raise ValueError(f"Target {TARGET} not found. Available: {list(y_df.columns)}")

y = y_df[TARGET]
data = pd.concat([X_df, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
X_df = data.drop(columns=[TARGET])
y = data[TARGET].values

X_df = pd.get_dummies(X_df, drop_first=True)

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X_df)

y_mean, y_std = y.mean(), y.std()
y_scaled = (y - y_mean) / y_std

X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)

n_features = X_train.shape[1]
print(f"Dataset ready: {n_features} features, Train={len(X_train)}, Val={len(X_val)}")

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

autoencoder = Autoencoder(n_features).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=LR_AE)

print("\nPretraining autoencoder...")
for epoch in range(EPOCHS_AE):
    autoencoder.train()
    optimizer.zero_grad()
    outputs = autoencoder(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        val_loss = criterion(autoencoder(X_val), X_val).item()
        print(f"Epoch {epoch+1}/{EPOCHS_AE} | TrainLoss={loss.item():.6f} | ValLoss={val_loss:.6f}")

class EncoderRegressor(nn.Module):
    def __init__(self, pretrained_encoder, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = pretrained_encoder
        self.reg_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.reg_head(z)

regressor = EncoderRegressor(autoencoder.encoder, latent_dim=LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(regressor.parameters(), lr=LR_REG)

print("\nTraining regression model (with fine-tuning)...")
for epoch in range(EPOCHS_REG):
    regressor.train()
    optimizer.zero_grad()
    preds = regressor(X_train)
    loss = criterion(preds, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        val_loss = criterion(regressor(X_val), y_val).item()
        print(f"Epoch {epoch+1}/{EPOCHS_REG} | Train MSE={loss.item():.6f} | Val MSE={val_loss:.6f}")

regressor.eval()
with torch.no_grad():
    y_pred = regressor(X_val).cpu().numpy().ravel()
    y_true = y_val.cpu().numpy().ravel()

y_pred = y_pred * y_std + y_mean
y_true = y_true * y_std + y_mean

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\n📊 Evaluation Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

print("\nPredictions sample:")
for i in range(5):
    print(f"Real: {y_true[i]:.3f} | Pred: {y_pred[i]:.3f}")
