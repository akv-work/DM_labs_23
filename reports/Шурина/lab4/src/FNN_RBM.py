import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data(file_path):
    data = pd.read_csv(file_path)

    X = data.drop(columns=["RiskLevel"])
    y = data["RiskLevel"]

    mapping = {
        "low risk": 0,
        "mid risk": 1,
        "high risk": 2
    }
    y = y.map(mapping).values

    return X, y

class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def sample_h(self, v):
        prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return torch.bernoulli(prob)

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            h = self.sample_h(v)
            v = self.sample_v(h)
        return v0, v

    def train_rbm(self, data, lr=0.01, batch_size=64, epochs=200):
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]

                v0, vk = self.contrastive_divergence(batch)
                loss = torch.mean((v0 - vk) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 20 == 0:
                print(f"RBM Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")

        print("RBM pretraining finished.\n")


class MLP(nn.Module):
    def __init__(self, input_size, rbm_weights):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc1.weight.data = rbm_weights.clone()

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)  # 3 класса

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(model, criterion, optimizer, X_train, y_train, epochs=300):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"MLP Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    print("MLP training finished.\n")

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    return acc

if __name__ == "__main__":
    # !!! укажи путь к датасету !!!
    X, y = load_data("Maternal Health Risk Data Set.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    rbm = RBM(
        visible_units=X_train.shape[1],
        hidden_units=64
    )
    rbm.train_rbm(X_train_tensor, lr=0.01, epochs=200)

    # --- MLP ---
    model = MLP(
        input_size=X_train.shape[1],
        rbm_weights=rbm.W
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(
        model,
        criterion,
        optimizer,
        X_train_tensor,
        y_train_tensor,
        epochs=300
    )

    accuracy = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor
    )

    print(f"Accuracy with RBM pretraining: {accuracy:.4f}")
