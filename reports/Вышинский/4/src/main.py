import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os
import datetime


torch.manual_seed(42)
np.random.seed(42)

SAVE_RESULTS = True
RESULTS_DIR = "./results/"
os.makedirs(RESULTS_DIR, exist_ok=True)



def load_maternal_health_data():
    data = fetch_ucirepo(id=863)
    X = data.data.features
    y = data.data.targets["RiskLevel"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y



def load_rice_data():
    data = fetch_ucirepo(id=545)
    X = data.data.features
    y = data.data.targets["Class"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y



class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ClassificationNet, self).__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x



class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.k = k

    def sample_h(self, v):
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h):
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v, torch.bernoulli(prob_v)

    def contrastive_divergence(self, v, lr=0.01):
        prob_h0, h0 = self.sample_h(v)
        v_k = v
        for _ in range(self.k):
            prob_v_k, v_k = self.sample_v(h0)
            prob_h_k, h_k = self.sample_h(v_k)

        self.W.data += lr * ((torch.matmul(prob_h0.t(), v) - torch.matmul(prob_h_k.t(), v_k)) / v.size(0))
        self.v_bias.data += lr * torch.mean(v - v_k, dim=0)
        self.h_bias.data += lr * torch.mean(prob_h0 - prob_h_k, dim=0)

        loss = torch.mean((v - v_k) ** 2)
        return loss



def pretrain_layers(input_data, hidden_sizes, epochs=50, lr=0.01):
    pretrained_weights = []
    current_input = input_data

    for h_size in hidden_sizes:
        ae = Autoencoder(current_input.shape[1], h_size)
        optimizer = optim.Adam(ae.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = TensorDataset(current_input, current_input)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        prev_loss = float('inf')
        patience = 5
        wait = 0

        for epoch in range(epochs):
            total_loss = 0
            for data, target in loader:
                optimizer.zero_grad()
                output = ae(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss > prev_loss - 1e-5:
                wait += 1
                if wait >= patience:
                    break
            else:
                wait = 0
            prev_loss = avg_loss

        with torch.no_grad():
            current_input = torch.relu(ae.encoder(current_input))
        pretrained_weights.append((ae.encoder.weight.data.clone(), ae.encoder.bias.data.clone()))

    return pretrained_weights



def pretrain_rbm_layers(input_data, hidden_sizes, epochs=30, lr=0.01):
    pretrained_weights = []
    current_input = input_data

    for h_size in hidden_sizes:
        rbm = RBM(current_input.shape[1], h_size)
        dataset = DataLoader(current_input, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataset:
                batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                loss = rbm.contrastive_divergence(batch, lr=lr)
                total_loss += loss.item()
            print(f"RBM layer {h_size} — epoch {epoch+1}, loss={total_loss / len(dataset):.6f}")

        with torch.no_grad():
            prob_h, _ = rbm.sample_h(current_input)
            current_input = prob_h
            pretrained_weights.append((rbm.W.data.clone(), rbm.h_bias.data.clone()))

    return pretrained_weights



def init_with_pretrain(net, pretrained_weights):
    i = 0
    for layer in net.network:
        if isinstance(layer, nn.Linear) and i < len(pretrained_weights):
            w, b = pretrained_weights[i]
            layer.weight.data = w
            layer.bias.data = b
            i += 1



def train_model(net, X_train, y_train, X_test, y_test, epochs=100, lr=0.001, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))

    net.eval()
    with torch.no_grad():
        y_pred = torch.argmax(net(X_test), dim=1).cpu().numpy()
        f1 = f1_score(y_test.cpu().numpy(), y_pred, average='weighted')
        cm = confusion_matrix(y_test.cpu().numpy(), y_pred)
    return f1, cm, losses



def process_dataset(name, loader_func):
    print(f"\n===== {name} Dataset =====")
    X, y = loader_func()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    input_size = X_train.shape[1]
    hidden_sizes = [64, 32, 16]
    output_size = len(np.unique(y))


    net_no_pre = ClassificationNet(input_size, hidden_sizes, output_size)
    f1_no, cm_no, losses_no = train_model(net_no_pre, X_train, y_train, X_test, y_test)
    print("\nWithout pretraining:")
    print(f"F1-score: {f1_no:.4f}")


    pretrained = pretrain_layers(X_train, hidden_sizes)
    net_pre = ClassificationNet(input_size, hidden_sizes, output_size)
    init_with_pretrain(net_pre, pretrained)
    f1_pre, cm_pre, losses_pre = train_model(net_pre, X_train, y_train, X_test, y_test)
    print("\nWith Autoencoder pretraining:")
    print(f"F1-score: {f1_pre:.4f}")


    pretrained_rbm = pretrain_rbm_layers(X_train, hidden_sizes)
    net_rbm = ClassificationNet(input_size, hidden_sizes, output_size)
    init_with_pretrain(net_rbm, pretrained_rbm)
    f1_rbm, cm_rbm, losses_rbm = train_model(net_rbm, X_train, y_train, X_test, y_test)
    print("\nWith RBM pretraining:")
    print(f"F1-score: {f1_rbm:.4f}")


    if SAVE_RESULTS:
        plt.figure(figsize=(10, 5))
        plt.plot(losses_no, label="No Pretrain")
        plt.plot(losses_pre, label="Autoencoder")
        plt.plot(losses_rbm, label="RBM")
        plt.title(f"Loss Curves — {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}_loss.png"), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n===== Summary for {name} =====")
    print(f"No pretrain:  {f1_no:.4f}")
    print(f"Autoencoder:  {f1_pre:.4f}")
    print(f"RBM:          {f1_rbm:.4f}")

    return f1_no, f1_pre, f1_rbm


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")

    f1_no_mh, f1_pre_mh, f1_rbm_mh = process_dataset("Maternal_Health_Risk", load_maternal_health_data)
    f1_no_rice, f1_pre_rice, f1_rbm_rice = process_dataset("Rice_Cammeo_Osmancik", load_rice_data)

    print("\n===== Final Comparison =====")
    print(f"Maternal Health — No: {f1_no_mh:.4f}, Autoenc: {f1_pre_mh:.4f}, RBM: {f1_rbm_mh:.4f}")
    print(f"Rice — No: {f1_no_rice:.4f}, Autoenc: {f1_pre_rice:.4f}, RBM: {f1_rbm_rice:.4f}")

    if SAVE_RESULTS:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== Run at {timestamp} =====\n")
            f.write(f"Maternal Health — No: {f1_no_mh:.4f}, Autoenc: {f1_pre_mh:.4f}, RBM: {f1_rbm_mh:.4f}\n")
            f.write(f"Rice — No: {f1_no_rice:.4f}, Autoenc: {f1_pre_rice:.4f}, RBM: {f1_rbm_rice:.4f}\n")
            f.write("=" * 40 + "\n")
