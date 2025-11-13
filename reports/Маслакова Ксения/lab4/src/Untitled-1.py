import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
print("Device:", DEVICE)
TARGET = "aveOralF"
TEST_SIZE = 0.20
RANDOM_STATE = SEED

# архитектура (умеренная для слабых машин)
H1, H2, H3 = 32, 16, 8

# эпохи (умеренные)
EPOCHS_BASE = 60
EPOCHS_AE_PRETRAIN = 40
EPOCHS_AE_FINETUNE = 60
EPOCHS_RBM = 40
LR = 1e-3
BATCH_SIZE = 64

# --------------------------
# Загрузка и предобработка данных
# --------------------------
print("Загружаем датасет (ucimlrepo id=925)...")
from ucimlrepo import fetch_ucirepo
infra = fetch_ucirepo(id=925)

X_df = infra.data.features.copy()
y_df = infra.data.targets.copy()

if TARGET not in y_df.columns:
    raise RuntimeError(f"Целевой столбец {TARGET} не найден в наборе. Доступны: {list(y_df.columns)}")

y_ser = y_df[TARGET]
X = X_df.copy()
y = y_ser.copy()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# числовые: fillna mean
if len(num_cols) > 0:
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# категориальные: fillna mode, then codes
for c in cat_cols:
    if X[c].isna().any():
        modes = X[c].mode()
        fillv = modes.iloc[0] if len(modes) > 0 else "missing"
        X.loc[:, c] = X[c].fillna(fillv)
    # кодируем
    X.loc[:, c] = pd.Categorical(X[c]).codes

# целевая — заполним средним если есть NaN
y = y.fillna(y.mean())

# Преобразуем в numpy float32
X_np = X.to_numpy(dtype=np.float32)
y_np = y.to_numpy(dtype=np.float32).reshape(-1, 1)

# масштабирование
scaler_X = StandardScaler().fit(X_np)
X_scaled = scaler_X.transform(X_np)
scaler_y = StandardScaler().fit(y_np)
y_scaled = scaler_y.transform(y_np)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Перевод в тензоры
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

print("Shapes:", X_train_t.shape, y_train_t.shape, X_test_t.shape, y_test_t.shape)
# Вспомогательные функции для обучения / оценки
def train_regressor(model, X_t, y_t, X_val=None, y_val=None, epochs=50, lr=1e-3, batch_size=64, verbose=False):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = X_t.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= n
        if verbose and (ep % 10 == 0 or ep == epochs-1):
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    vloss = nn.MSELoss()(model(X_val), y_val).item()
                print(f"Epoch {ep+1}/{epochs} train_loss={epoch_loss:.6f} val_loss={vloss:.6f}")
            else:
                print(f"Epoch {ep+1}/{epochs} train_loss={epoch_loss:.6f}")
    return model

def eval_regressor(model, X_t, y_t, scaler_y):
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    preds_inv = scaler_y.inverse_transform(preds)
    y_true_inv = scaler_y.inverse_transform(y_t.cpu().numpy())
    mse = mean_squared_error(y_true_inv, preds_inv)
    mae = mean_absolute_error(y_true_inv, preds_inv)
    mape = mean_absolute_percentage_error(y_true_inv, preds_inv)
    return mse, mae, mape, preds_inv, y_true_inv
# 1) (без предобучения)
class RegressorBase(nn.Module):
    def __init__(self, input_dim, h1=H1, h2=H2, h3=H3):
        super().__init__()
        self.l1 = nn.Linear(input_dim, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, 1)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        return self.out(x)

print("\n--- Training baseline (from scratch) ---")
reg_base = RegressorBase(X_train_t.shape[1], H1, H2, H3).to(DEVICE)
reg_base = train_regressor(reg_base, X_train_t, y_train_t, epochs=EPOCHS_BASE, lr=LR,
                           batch_size=BATCH_SIZE, verbose=True)
mse_base, mae_base, mape_base, preds_base, y_true = eval_regressor(reg_base, X_test_t, y_test_t, scaler_y)
print(f"Baseline test MSE={mse_base:.6f}, MAE={mae_base:.6f}, MAPE={mape_base*100:.2f}%")

# 2) Pretraining via autoencoder (layer-wise weight transfer)
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, h1=H1, h2=H2, h3=H3):
        super().__init__()
        # encoder
        self.e1 = nn.Linear(input_dim, h1)
        self.e2 = nn.Linear(h1, h2)
        self.e3 = nn.Linear(h2, h3)
        # decoder (mirror)
        self.d3 = nn.Linear(h3, h2)
        self.d2 = nn.Linear(h2, h1)
        self.d1 = nn.Linear(h1, input_dim)
    def forward(self, x):
        x1 = torch.relu(self.e1(x))
        x2 = torch.relu(self.e2(x1))
        x3 = torch.relu(self.e3(x2))
        # decode
        y2 = torch.relu(self.d3(x3))
        y1 = torch.relu(self.d2(y2))
        recon = self.d1(y1)
        return recon

def train_autoencoder_full(ae, X_t, epochs=30, lr=1e-3, batch_size=64, verbose=False):
    ae.to(DEVICE)
    opt = optim.Adam(ae.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = X_t.size(0)
    for ep in range(epochs):
        ae.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            opt.zero_grad()
            rec = ae(xb)
            loss = crit(rec, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= n
        if verbose and (ep % 10 == 0 or ep == epochs-1):
            print(f"AE epoch {ep+1}/{epochs} loss={epoch_loss:.6f}")
    return ae

print("\n--- Autoencoder pretraining (full encoder trained) ---")
ae = DeepAutoencoder(X_train_t.shape[1], H1, H2, H3)
ae = train_autoencoder_full(ae, X_train_t, epochs=EPOCHS_AE_PRETRAIN, lr=LR,
                            batch_size=BATCH_SIZE, verbose=True)

reg_ae = RegressorBase(X_train_t.shape[1], H1, H2, H3).to(DEVICE)
with torch.no_grad():
    reg_ae.l1.weight.copy_(ae.e1.weight)
    reg_ae.l1.bias.copy_(ae.e1.bias)
    reg_ae.l2.weight.copy_(ae.e2.weight)
    reg_ae.l2.bias.copy_(ae.e2.bias)
    reg_ae.l3.weight.copy_(ae.e3.weight)
    reg_ae.l3.bias.copy_(ae.e3.bias)

reg_ae = train_regressor(reg_ae, X_train_t, y_train_t, epochs=EPOCHS_AE_FINETUNE, lr=LR,
                        batch_size=BATCH_SIZE, verbose=True)
mse_ae, mae_ae, mape_ae, preds_ae, _ = eval_regressor(reg_ae, X_test_t, y_test_t, scaler_y)
print(f"AE-pretrained test MSE={mse_ae:.6f}, MAE={mae_ae:.6f}, MAPE={mape_ae*100:.2f}%")

# 3) RBM pretraining (layer-wise), simple Gaussian-Bernoulli RBM
class RBM_simple:
    def __init__(self, n_vis, n_hid, lr=1e-3):
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = torch.randn(n_vis, n_hid) * 0.01
        self.vbias = torch.zeros(n_vis)
        self.hbias = torch.zeros(n_hid)
        self.lr = lr

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def sample_h(self, v):
        logits = torch.matmul(v, self.W) + self.hbias  
        p_h = torch.sigmoid(logits)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h):
        # Gaussian visible: mean = h @ W.T + vbias
        v_mean = torch.matmul(h, self.W.t()) + self.vbias
        # sample with Gaussian noise
        v_sample = v_mean + torch.randn_like(v_mean) * 0.01
        return v_mean, v_sample

    def cd1_update(self, v0):
        if not isinstance(v0, torch.Tensor):
            v0 = torch.tensor(v0, dtype=torch.float32)
        p_h0, h0 = self.sample_h(v0)
        v1_mean, v1 = self.sample_v(h0)
        p_h1, h1 = self.sample_h(v1_mean)
        dW = torch.matmul(v0.t(), p_h0) - torch.matmul(v1_mean.t(), p_h1)
        dv = torch.sum(v0 - v1_mean, dim=0)
        dh = torch.sum(p_h0 - p_h1, dim=0)
        self.W += (self.lr / v0.shape[0]) * dW
        self.vbias += (self.lr / v0.shape[0]) * dv
        self.hbias += (self.lr / v0.shape[0]) * dh
        recon_err = torch.mean((v0 - v1_mean) ** 2).item()
        return recon_err

    def train(self, X_numpy, epochs=20, batch_size=64, verbose=False):
        n = X_numpy.shape[0]
        errors = []
        for ep in range(epochs):
            perm = np.random.permutation(n)
            tot_err = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                xb = X_numpy[idx]
                err = self.cd1_update(xb)
                tot_err += err * xb.shape[0]
            tot_err /= n
            errors.append(tot_err)
            if verbose and (ep % 10 == 0 or ep == epochs-1):
                print(f"RBM {self.n_vis}->{self.n_hid} epoch {ep+1}/{epochs} recon_err={tot_err:.6f}")
        return errors

    def transform(self, X_numpy):
        X_t = torch.tensor(X_numpy, dtype=torch.float32)
        p_h = torch.sigmoid(torch.matmul(X_t, self.W) + self.hbias)
        return p_h.numpy()

print("\n--- RBM layer-wise pretraining ---")
rbm1 = RBM_simple(X_train.shape[1], H1, lr=1e-3)
err1 = rbm1.train(X_train, epochs=EPOCHS_RBM, batch_size=BATCH_SIZE, verbose=True)
H1_train = rbm1.transform(X_train)
H1_test = rbm1.transform(X_test)

rbm2 = RBM_simple(H1, H2, lr=1e-3)
err2 = rbm2.train(H1_train, epochs=EPOCHS_RBM, batch_size=BATCH_SIZE, verbose=True)
H2_train = rbm2.transform(H1_train)
H2_test = rbm2.transform(H1_test)

rbm3 = RBM_simple(H2, H3, lr=1e-3)
err3 = rbm3.train(H2_train, epochs=EPOCHS_RBM, batch_size=BATCH_SIZE, verbose=True)
H3_train = rbm3.transform(H2_train)
H3_test = rbm3.transform(H2_test)
reg_rbm = RegressorBase(X_train_t.shape[1], H1, H2, H3).to(DEVICE)
with torch.no_grad():
    reg_rbm.l1.weight.copy_(rbm1.W.t())
    reg_rbm.l1.bias.copy_(rbm1.hbias)
    reg_rbm.l2.weight.copy_(rbm2.W.t())
    reg_rbm.l2.bias.copy_(rbm2.hbias)
    reg_rbm.l3.weight.copy_(rbm3.W.t())
    reg_rbm.l3.bias.copy_(rbm3.hbias)

reg_rbm = train_regressor(reg_rbm, X_train_t, y_train_t, epochs=EPOCHS_BASE, lr=LR,
                          batch_size=BATCH_SIZE, verbose=True)
mse_rbm, mae_rbm, mape_rbm, preds_rbm, _ = eval_regressor(reg_rbm, X_test_t, y_test_t, scaler_y)
print(f"RBM-pretrained test MSE={mse_rbm:.6f}, MAE={mae_rbm:.6f}, MAPE={mape_rbm*100:.2f}%")

# Сводная таблица результатов
results = pd.DataFrame({
    "Модель": ["Baseline (no pretrain)", "AE pretrain (weight transfer)", "RBM pretrain (layer-wise)"],
    "MSE": [mse_base, mse_ae, mse_rbm],
    "MAE": [mae_base, mae_ae, mae_rbm],
    "MAPE (%)": [mape_base * 100, mape_ae * 100, mape_rbm * 100]
})
print("\n=== Сравнение результатов ===")
print(results.to_string(index=False))
