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

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RBM:
    def __init__(self, n_visible, n_hidden, k=1, lr=1e-3, use_cuda=False):
        self.nv = n_visible; self.nh = n_hidden; self.k = k; self.lr = lr
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

class AutoencoderFull(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        enc = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            enc.append(nn.Linear(dims[i], dims[i+1])); enc.append(nn.ReLU())
        dec = []
        for i in range(len(hidden_dims)-1, -1, -1):
            dec.append(nn.Linear(dims[i+1], dims[i]));
            if i!=0: dec.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
    def forward(self,x):
        z = self.encoder(x); return self.decoder(z)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1])); layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def load_wholesale(csv_path="wholesale.csv"):
    df = pd.read_csv(csv_path)
    if "Channel" in df.columns:
        target = "Channel"
    elif "Region" in df.columns:
        target = "Region"
    else:
        df["TargetBin"] = (df.select_dtypes(include=[np.number]).sum(axis=1) > df.select_dtypes(include=[np.number]).sum(axis=1).median()).astype(int)
        target = "TargetBin"
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).values
    y = df[target].astype(int).values - 1
    return X, y

def train_classifier(model, Xtr, ytr, Xte, yte, epochs=40, batch_size=32, lr=1e-3):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    ds_tr = TensorDataset(torch.tensor(Xtr,dtype=torch.float32), torch.tensor(ytr,dtype=torch.long))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        for Xb, yb in dl_tr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
    model.eval()
    preds=[]; truths=[]
    ds_te = TensorDataset(torch.tensor(Xte,dtype=torch.float32), torch.tensor(yte,dtype=torch.long))
    dl_te = DataLoader(ds_te, batch_size=128)
    with torch.no_grad():
        for Xb,yb in dl_te:
            Xb = Xb.to(device)
            out = model(Xb)
            preds.extend(out.argmax(1).cpu().numpy())
            truths.extend(yb.numpy())
    return np.array(truths), np.array(preds), model

def run(csv_path="wholesale.csv"):
    X,y = load_wholesale(csv_path)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]; y = y[mask]
    num_classes = len(np.unique(y))
    n = len(y)
    idx = np.arange(n); np.random.shuffle(idx)
    tr = int(0.8*n)
    train_idx, test_idx = idx[:tr], idx[tr:]
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    print("Classes:", np.unique(y, return_counts=True))
    scaler_clf = StandardScaler().fit(Xtr)
    Xtr_clf = scaler_clf.transform(Xtr); Xte_clf = scaler_clf.transform(Xte)
    scaler_rbm = MinMaxScaler().fit(Xtr)
    Xtr_rbm = scaler_rbm.transform(Xtr); Xte_rbm = scaler_rbm.transform(Xte)
    input_dim = Xtr.shape[1]
    hidden_dims = [128, 64]
    mlp_hidden = hidden_dims
    print("\n--- Baseline ---")
    model_base = MLP(input_dim, mlp_hidden, num_classes)
    y_true_b, y_pred_b, model_base = train_classifier(model_base, Xtr_clf, ytr, Xte_clf, yte, epochs=40, lr=1e-3)
    f1_b = f1_score(y_true_b, y_pred_b, average='macro')
    print("Baseline F1:", f1_b); print(confusion_matrix(y_true_b, y_pred_b))
    print("\n--- AE pretrain ---")
    ae = AutoencoderFull(input_dim, hidden_dims).to(device)
    ae_opt = optim.Adam(ae.parameters(), lr=1e-3); ae_crit = nn.MSELoss()
    Xtr_tensor = torch.tensor(Xtr_clf, dtype=torch.float32).to(device)
    for ep in range(30):
        ae.train()
        ae_opt.zero_grad()
        rec = ae(Xtr_tensor)
        loss = ae_crit(rec, Xtr_tensor)
        loss.backward(); ae_opt.step()
    model_ae = MLP(input_dim, mlp_hidden, num_classes)
    with torch.no_grad():
        enc_layers = [l for l in ae.encoder if isinstance(l, nn.Linear)]
        mlp_lin = [l for l in model_ae.net if isinstance(l, nn.Linear)]
        for i in range(len(enc_layers)):
            mlp_lin[i].weight.data = enc_layers[i].weight.data.clone()
            mlp_lin[i].bias.data = enc_layers[i].bias.data.clone()
    y_true_ae, y_pred_ae, model_ae = train_classifier(model_ae, Xtr_clf, ytr, Xte_clf, yte, epochs=40)
    f1_ae = f1_score(y_true_ae, y_pred_ae, average='macro')
    print("AE F1:", f1_ae); print(confusion_matrix(y_true_ae, y_pred_ae))
    print("\n--- RBM pretrain ---")
    rbm1 = RBM(n_visible=input_dim, n_hidden=hidden_dims[0], k=1, lr=0.01)
    epochs_rbm = 20; batch_size = 64
    for ep in range(epochs_rbm):
        perm = np.random.permutation(len(Xtr_rbm))
        losses=[]
        for i in range(0,len(Xtr_rbm),batch_size):
            batch = torch.tensor(Xtr_rbm[perm[i:i+batch_size]], dtype=torch.float32, device=rbm1.device)
            l = rbm1.contrastive_divergence(batch); losses.append(l)
        if (ep+1)%5==0:
            print(f"RBM1 ep {ep+1}, loss {np.mean(losses):.6f}")
    H1 = rbm1.transform(Xtr_rbm)
    rbm2 = RBM(n_visible=hidden_dims[0], n_hidden=hidden_dims[1], k=1, lr=0.01)
    for ep in range(epochs_rbm):
        perm = np.random.permutation(len(H1))
        losses=[]
        for i in range(0,len(H1),batch_size):
            batch = torch.tensor(H1[perm[i:i+batch_size]], dtype=torch.float32, device=rbm2.device)
            l = rbm2.contrastive_divergence(batch); losses.append(l)
        if (ep+1)%5==0:
            print(f"RBM2 ep {ep+1}, loss {np.mean(losses):.6f}")
    model_rbm = MLP(input_dim, mlp_hidden, num_classes)
    with torch.no_grad():
        model_rbm.net[0].weight.data = rbm1.W.t().clone()
        model_rbm.net[0].bias.data = rbm1.h_bias.clone()
        model_rbm.net[2].weight.data = rbm2.W.t().clone()
        model_rbm.net[2].bias.data = rbm2.h_bias.clone()
    y_true_r, y_pred_r, model_rbm = train_classifier(model_rbm, Xtr_clf, ytr, Xte_clf, yte, epochs=40)
    f1_r = f1_score(y_true_r, y_pred_r, average='macro')
    print("RBM F1:", f1_r); print(confusion_matrix(y_true_r, y_pred_r))
    print("\n=== SUMMARY ===")
    print(f"Baseline F1: {f1_b:.4f}")
    print(f"AE F1:       {f1_ae:.4f}")
    print(f"RBM F1:      {f1_r:.4f}")
    print("\nBaseline report:\n", classification_report(y_true_b, y_pred_b))
    print("\nAE report:\n", classification_report(y_true_ae, y_pred_ae))
    print("\nRBM report:\n", classification_report(y_true_r, y_pred_r))

if __name__ == "__main__":
    if not os.path.exists("wholesale.csv"):
        raise FileNotFoundError("wholesale.csv not found in working dir")
    run("wholesale.csv")
