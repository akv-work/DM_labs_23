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

RBM_LR = 1e-3
RBM_CD_K = 1
RBM_EPOCHS_FIRST = 50
RBM_EPOCHS_OTHER = 30

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

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y_norm, test_size=0.2, random_state=RANDOM_SEED
)
X_train = preprocessor.fit_transform(X_train_df)
X_test = preprocessor.transform(X_test_df)

print("Feature matrix shapes after preprocessing:", X_train.shape, X_test.shape)

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


train_ds = TabularDataset(X_train, y_train)
test_ds = TabularDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
train_x_loader = DataLoader(TabularDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

in_features = X_train.shape[1]
print("Input features:", in_features)

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, visible_type='bernoulli', device=None):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.visible_type = visible_type
        self.device = device if device is not None else torch.device('cpu')

        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        pre = torch.matmul(v, self.W) + self.h_bias
        p_h = torch.sigmoid(pre)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        pre = torch.matmul(h, self.W.t()) + self.v_bias
        if self.visible_type == 'bernoulli':
            p_v = torch.sigmoid(pre)
            return p_v, torch.bernoulli(p_v)
        elif self.visible_type == 'gaussian':
            mean = pre
            sample = mean + torch.randn_like(mean)
            return mean, sample
        else:
            raise ValueError("visible_type must be 'bernoulli' or 'gaussian'")

    def free_energy(self, v):
        if self.visible_type == 'gaussian':
            vbias_term = ((v - self.v_bias) ** 2).sum(dim=1)
        else:
            vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W) + self.h_bias
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term

    def forward(self, v):
        p_h, _ = self.sample_h(v)
        return p_h

    def contrastive_divergence(self, v0, k=1, lr=1e-3):
        batch_size = v0.size(0)
        v = v0
        p_h0 = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        h0 = torch.bernoulli(p_h0)

        hk = h0
        vk = None
        for step in range(k):
            mean_vk, vk = self.sample_v(hk)
            p_hk, hk = self.sample_h(vk)

        pos_assoc = torch.matmul(v.t(), p_h0)
        neg_assoc = torch.matmul(vk.t(), p_hk)

        dW = (pos_assoc - neg_assoc) / batch_size
        dv_bias = torch.mean(v - vk, dim=0)
        dh_bias = torch.mean(p_h0 - p_hk, dim=0)

        self.W.data += lr * dW
        self.v_bias.data += lr * dv_bias
        self.h_bias.data += lr * dh_bias

        if self.visible_type == 'gaussian':
            recon_error = torch.mean((v0 - mean_vk) ** 2).item()
        else:
            recon_error = torch.mean((v0 - vk) ** 2).item()

        return recon_error


def pretrain_rbms(X_numpy, layer_sizes, device):
    rbms = []
    representations = []
    current_data = torch.tensor(X_numpy, dtype=torch.float32, device=device)

    for i, h_dim in enumerate(layer_sizes):
        v_dim = current_data.shape[1]
        if i == 0:
            vis_type = 'gaussian'
            epochs = RBM_EPOCHS_FIRST
        else:
            vis_type = 'bernoulli'
            epochs = RBM_EPOCHS_OTHER

        print(f"\nPretraining RBM layer {i + 1}/{len(layer_sizes)}: visible={v_dim}, hidden={h_dim}, visible_type={vis_type}, epochs={epochs}")
        rbm = RBM(n_visible=v_dim, n_hidden=h_dim, visible_type=vis_type, device=device).to(device)

        ds = TabularDataset(current_data.cpu().numpy())
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        for ep in range(1, epochs + 1):
            epoch_err = 0.0
            nb = 0
            for batch in loader:
                v0 = batch.to(device)
                err = rbm.contrastive_divergence(v0, k=RBM_CD_K, lr=RBM_LR)
                epoch_err += err * v0.size(0)
                nb += v0.size(0)
            epoch_err /= nb
            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(f"  RBM layer {i + 1} epoch {ep:3d} | recon MSE: {epoch_err:.6f}")

        rbms.append(rbm)

        with torch.no_grad():
            p_h = torch.sigmoid(torch.matmul(current_data, rbm.W) + rbm.h_bias)
            current_data = p_h
            representations.append(p_h.cpu().numpy())

    return rbms, representations


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


layer_sizes = [128, 64, 32, 16]
rbms, representations = pretrain_rbms(X_train, layer_sizes, DEVICE)
print("\nFinished pretraining RBMs.")

model = FCRegressor(in_features).to(DEVICE)

linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
assert len(linear_layers) >= len(rbms), "Linear layers less than RBMs - architecture mismatch"

for i, rbm in enumerate(rbms):
    lin = linear_layers[i]
    W_t = rbm.W.t().cpu().detach()
    h_b = rbm.h_bias.cpu().detach()
    if lin.weight.shape == W_t.shape:
        lin.weight.data.copy_(W_t)
    else:
        print(f"Warning: shape mismatch when assigning weights to layer {i}: lin.weight {lin.weight.shape}, W_t {W_t.shape}")
    if lin.bias is not None and lin.bias.shape[0] == h_b.shape[0]:
        lin.bias.data.copy_(h_b)
    else:
        print(f"Warning: bias mismatch for layer {i}")

print("Initialized FC model weights from RBMs.")


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
    print(" No valid best_state found. Using last model state.")
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