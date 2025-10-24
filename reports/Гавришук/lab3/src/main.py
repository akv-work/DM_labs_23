import math
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

TARGET = "aveOralF"
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Fetching dataset from UCI...")
infrared = fetch_ucirepo(id=925)
X_df = infrared.data.features.copy()
y_df = infrared.data.targets.copy()

if TARGET not in y_df.columns:
    raise ValueError(f"Target {TARGET} not found. Available: {list(y_df.columns)}")

y = y_df[TARGET]
data = pd.concat([X_df, y], axis=1).replace([np.inf, -np.inf], np.nan)
data = data.dropna()

X_df = data.drop(columns=[TARGET])
y = data[TARGET].values

print(f"Dataset cleaned: X={X_df.shape}, y={y.shape}")

y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / (y_std + 1e-8)

categorical_cols = []
numeric_cols = []
for col in X_df.columns:
    if X_df[col].dtype == object:
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)
for c in ["Gender", "Age", "Ethnicity"]:
    if c in X_df.columns and c not in categorical_cols:
        categorical_cols.append(c)
        if c in numeric_cols:
            numeric_cols.remove(c)

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols[:10], " (total:", len(numeric_cols), ")")

cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"))
])
num_pipe = Pipeline([
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y_norm, test_size=0.2, random_state=RANDOM_SEED)
X_train = preprocessor.fit_transform(X_train_df)
X_test = preprocessor.transform(X_test_df)

print("Feature matrix shapes after preprocessing:", X_train.shape, X_test.shape)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_train, y_train)
test_ds = TabularDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

in_features = X_train.shape[1]
print("Input features:", in_features)

class FCRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = FCRegressor(in_features).to(DEVICE)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)

def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn):
    model.eval()
    preds_list, targets_list = [], []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    preds_arr = np.vstack(preds_list).flatten()
    targets_arr = np.vstack(targets_list).flatten()
    return total_loss / len(loader.dataset), preds_arr, targets_arr

best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, _, _ = evaluate(model, test_loader, criterion)
    scheduler.step(val_loss)
    if not math.isnan(val_loss) and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

if best_state is None:
    print("⚠️ No valid best_state found. Using last model state.")
    best_state = model.state_dict()

model.load_state_dict(best_state)

_, preds, targets = evaluate(model, test_loader, criterion)

preds_real = preds * (y_std + 1e-8) + y_mean
targets_real = targets * (y_std + 1e-8) + y_mean

def mape(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

mae = mean_absolute_error(targets_real, preds_real)
rmse = math.sqrt(mean_squared_error(targets_real, preds_real))
r2 = r2_score(targets_real, preds_real)
mape_val = mape(targets_real, preds_real)

print("\n--- Final test metrics ---")
print(f"Target: {TARGET}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape_val:.3f}%")

res_df = pd.DataFrame({"y_true": targets_real, "y_pred": preds_real})
print("\nSample predictions (first 10):")
print(res_df.head(10).to_string(index=False))
