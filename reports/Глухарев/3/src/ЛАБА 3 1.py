import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# ===== 1. Загрузка данных =====
file_path = "CTG.csv"  # <-- укажи точное имя файла
df = pd.read_csv(file_path)

# Определяем столбец с меткой
target_col = None
for col in df.columns:
    if "NSP" in col or "CLASS" in col or "class" in col.lower():
        target_col = col
        break

if target_col is None:
    raise ValueError("Не найден столбец с целевой переменной (CLASS или NSP)!")

# Приведение типов и очистка
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')  # нечисловые -> NaN
df = df.dropna(subset=[target_col])  # убираем строки без метки

# Отделяем признаки и метку
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
y = df[target_col].astype(int) - 1  # NSP: 1,2,3 → 0,1,2

# Проверим баланс классов
print("Классы в данных:", np.unique(y, return_counts=True))

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# Разделение train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)


# ===== 2. Определяем модели =====
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ===== 3. Обучение =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tensor.shape[1]
output_dim = len(y.unique())

# ---- A) Без предобучения ----
model_plain = Classifier(input_dim, output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_plain.parameters(), lr=0.001)

for epoch in range(30):
    model_plain.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model_plain(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Эпоха [{epoch+1}/30] | Потеря (без предобучения): {total_loss/len(train_loader):.4f}")

# ---- B) С автоэнкодерным предобучением ----
autoenc = Autoencoder(input_dim).to(device)
ae_criterion = nn.MSELoss()
ae_opt = optim.Adam(autoenc.parameters(), lr=0.001)

for epoch in range(20):
    autoenc.train()
    total_loss = 0
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)
        ae_opt.zero_grad()
        reconstructed = autoenc(X_batch)
        loss = ae_criterion(reconstructed, X_batch)
        loss.backward()
        ae_opt.step()
        total_loss += loss.item()
    print(f"Эпоха [{epoch+1}/20] | Потеря автоэнкодера: {total_loss/len(train_loader):.4f}")

# Используем encoder для инициализации классификатора
model_pretrained = Classifier(input_dim, output_dim=output_dim).to(device)
with torch.no_grad():
    model_pretrained.model[0].weight = nn.Parameter(autoenc.encoder[0].weight.clone())
    model_pretrained.model[0].bias = nn.Parameter(autoenc.encoder[0].bias.clone())

optimizer2 = optim.Adam(model_pretrained.parameters(), lr=0.001)

for epoch in range(30):
    model_pretrained.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer2.zero_grad()
        outputs = model_pretrained(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
    print(f"Эпоха [{epoch+1}/30] | Потеря (с предобучением): {total_loss/len(train_loader):.4f}")


# ===== 4. Оценка =====
def evaluate(model):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y_batch.numpy())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    cm = confusion_matrix(targets, preds)
    return acc, f1, cm


acc1, f11, cm1 = evaluate(model_plain)
acc2, f12, cm2 = evaluate(model_pretrained)

print("\n=== Результаты на Cardiotocography ===")
print(f"Без предобучения:  Accuracy={acc1:.3f}, F1={f11:.3f}")
print(f"С автоэнкодером:   Accuracy={acc2:.3f}, F1={f12:.3f}")
print("\nМатрица ошибок (до):\n", cm1)
print("\nМатрица ошибок (после):\n", cm2)