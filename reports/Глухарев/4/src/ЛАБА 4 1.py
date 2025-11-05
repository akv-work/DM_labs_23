import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import random

csv_path = "CTG.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ctg(csv_path):
    df = pd.read_csv(csv_path)
    target_col = None
    for c in df.columns:
        if "NSP" in c or "CLASS" in c or "class" in c.lower():
            target_col = c
            break
    if target_col is None:
        raise RuntimeError("Target column (NSP/CLASS) not found in csv")
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    y = df[target_col].astype(int).values - 1
    return X.values, y

class RBM:
    def __init__(self, n_visible, n_hidden, k=1, lr=1e-3, use_cuda=False):
        self.nv = n_visible
        self.nh = n_hidden
        self.k = k
        self.lr = lr
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        W = torch.randn(n_visible, n_hidden) * 0.01
        self.W = W.to(self.device)
        self.v_bias = torch.zeros(n_visible, device=self.device)
        self.h_bias = torch.zeros(n_hidden, device=self.device)

    def sample_h(self, v):
        prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return prob, torch.bernoulli(prob)

    def contrastive_divergence(self, v0):
        v = v0.to(self.device)
        ph_prob, ph_sample = self.sample_h(v)
        nv = v
        for _ in range(self.k):
            _, h = self.sample_h(nv)
            nv_prob, nv = self.sample_v(h)
        nh_prob, _ = self.sample_h(nv)
        pos_grad = torch.matmul(v.t(), ph_prob)
        neg_grad = torch.matmul(nv.t(), nh_prob)
        batch_size = v.size(0)
        self.W += self.lr * (pos_grad - neg_grad) / batch_size
        self.v_bias += self.lr * torch.mean(v - nv, dim=0)
        self.h_bias += self.lr * torch.mean(ph_prob - nh_prob, dim=0)
        loss = torch.mean((v - nv_prob) ** 2).item()
        return loss

    def transform(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        h_prob = torch.sigmoid(torch.matmul(X_t, self.W) + self.h_bias)
        return h_prob.cpu().numpy()

class AEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)

class AutoencoderFull(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        enc_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(nn.ReLU())
        dec_layers = []
        for i in range(len(hidden_dims)-1, -1, -1):
            dec_layers.append(nn.Linear(dims[i+1], dims[i]))
            if i != 0:
                dec_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

def train_classifier(model, train_loader, val_loader, epochs=30, lr=1e-3):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            ys.extend(yb.numpy())
    return np.array(ys), np.array(preds), model

def run_experiment(csv_path):
    X, y = load_ctg(csv_path)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]; y = y[mask]
    num_classes = len(np.unique(y))
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_n = int(0.8 * n)
    tr_idx = idx[:train_n]; te_idx = idx[train_n:]
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    scaler_clf = StandardScaler().fit(X_train)
    Xtr_clf = scaler_clf.transform(X_train)
    Xte_clf = scaler_clf.transform(X_test)
    scaler_rbm = MinMaxScaler().fit(X_train)
    Xtr_rbm = scaler_rbm.transform(X_train)
    Xte_rbm = scaler_rbm.transform(X_test)
    Xtr_tensor = torch.tensor(Xtr_clf, dtype=torch.float32)
    Xte_tensor = torch.tensor(Xte_clf, dtype=torch.float32)
    ytr_tensor = torch.tensor(y_train, dtype=torch.long)
    yte_tensor = torch.tensor(y_test, dtype=torch.long)
    train_ds = TensorDataset(Xtr_tensor, ytr_tensor)
    test_ds = TensorDataset(Xte_tensor, yte_tensor)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    input_dim = X.shape[1]
    hidden_dims = [128, 64]
    mlp_hidden = hidden_dims

    print("\n--- Baseline (no pretraining) ---")
    model_base = MLP(input_dim, mlp_hidden, num_classes)
    y_true_base, y_pred_base, model_base = train_classifier(model_base, train_loader, test_loader, epochs=40, lr=1e-3)
    f1_base = f1_score(y_true_base, y_pred_base, average='macro')
    print("Baseline F1 (macro):", f1_base)
    print(confusion_matrix(y_true_base, y_pred_base))

    print("\n--- Autoencoder stacked pretraining ---")
    ae = AutoencoderFull(input_dim, hidden_dims)
    ae = ae.to(device)
    ae_opt = optim.Adam(ae.parameters(), lr=1e-3)
    ae_crit = nn.MSELoss()
    Xtr_ae = torch.tensor(Xtr_clf, dtype=torch.float32).to(device)
    ae_epochs = 30
    for ep in range(ae_epochs):
        ae.train()
        ae_opt.zero_grad()
        rec = ae(Xtr_ae)
        loss = ae_crit(rec, Xtr_ae)
        loss.backward()
        ae_opt.step()
    model_ae = MLP(input_dim, mlp_hidden, num_classes)
    with torch.no_grad():
        enc_layers = [l for l in ae.encoder if isinstance(l, nn.Linear)]
        mlp_lin = [l for l in model_ae.net if isinstance(l, nn.Linear)]
        for i in range(len(enc_layers)):
            mlp_lin[i].weight.data = enc_layers[i].weight.data.clone()
            mlp_lin[i].bias.data = enc_layers[i].bias.data.clone()
    y_true_ae, y_pred_ae, model_ae = train_classifier(model_ae, train_loader, test_loader, epochs=40, lr=1e-3)
    f1_ae = f1_score(y_true_ae, y_pred_ae, average='macro')
    print("AE-pretrain F1 (macro):", f1_ae)
    print(confusion_matrix(y_true_ae, y_pred_ae))

    print("\n--- RBM stacked pretraining ---")
    rbm1 = RBM(n_visible=input_dim, n_hidden=hidden_dims[0], k=1, lr=0.01, use_cuda=False)
    epochs_rbm = 20
    batch_size = 64
    Xrbm = Xtr_rbm
    for ep in range(epochs_rbm):
        perm = np.random.permutation(len(Xrbm))
        losses = []
        for i in range(0, len(Xrbm), batch_size):
            batch = torch.tensor(Xrbm[perm[i:i+batch_size]], dtype=torch.float32, device=rbm1.device)
            loss = rbm1.contrastive_divergence(batch)
            losses.append(loss)
        if (ep+1)%5==0:
            print(f"RBM1 epoch {ep+1}, recon_loss={np.mean(losses):.6f}")
    H1 = rbm1.transform(Xrbm)
    rbm2 = RBM(n_visible=hidden_dims[0], n_hidden=hidden_dims[1], k=1, lr=0.01, use_cuda=False)
    for ep in range(epochs_rbm):
        perm = np.random.permutation(len(H1))
        losses = []
        for i in range(0, len(H1), batch_size):
            batch = torch.tensor(H1[perm[i:i+batch_size]], dtype=torch.float32, device=rbm2.device)
            loss = rbm2.contrastive_divergence(batch)
            losses.append(loss)
        if (ep+1)%5==0:
            print(f"RBM2 epoch {ep+1}, recon_loss={np.mean(losses):.6f}")
    model_rbm = MLP(input_dim, mlp_hidden, num_classes)
    with torch.no_grad():
        model_rbm.net[0].weight.data = rbm1.W.t().clone()
        model_rbm.net[0].bias.data = rbm1.h_bias.clone()
        model_rbm.net[2].weight.data = rbm2.W.t().clone()
        model_rbm.net[2].bias.data = rbm2.h_bias.clone()
    y_true_rbm, y_pred_rbm, model_rbm = train_classifier(model_rbm, train_loader, test_loader, epochs=40, lr=1e-3)
    f1_rbm = f1_score(y_true_rbm, y_pred_rbm, average='macro')
    print("RBM-pretrain F1 (macro):", f1_rbm)
    print(confusion_matrix(y_true_rbm, y_pred_rbm))

    print("\n=== SUMMARY (F1 macro) ===")
    print(f"Baseline: {f1_base:.4f}")
    print(f"AE pretrain: {f1_ae:.4f}")
    print(f"RBM pretrain: {f1_rbm:.4f}")
    print("\nBaseline report:\n", classification_report(y_true_base, y_pred_base))
    print("\nAE report:\n", classification_report(y_true_ae, y_pred_ae))
    print("\nRBM report:\n", classification_report(y_true_rbm, y_pred_rbm))

if __name__ == "__main__":
    csv_path = "CTG.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found in working directory.")
    run_experiment(csv_path)