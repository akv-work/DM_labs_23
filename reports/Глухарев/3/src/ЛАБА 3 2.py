import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
import numpy as np

# ===== 1. Загрузка и подготовка данных =====
file_path = "wholesale.csv"  # <-- укажи точное имя файла
df = pd.read_csv(file_path)

# Определяем целевую колонку (зависит от твоего CSV)
# В оригинале UCI Wholesale Customers — 'Channel' или 'Region'
target_col = 'Channel' if 'Channel' in df.columns else 'Region'

# Удаляем строки с пропусками
df = df.dropna(subset=[target_col])

# Признаки и целевая переменная
X = df.drop(columns=[target_col])
y = df[target_col].astype(int) - 1  # делаем классы от 0

num_classes = len(np.unique(y))
print(f"Количество классов: {num_classes}")
print("Баланс классов:", np.unique(y, return_counts=True))

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Преобразуем в тензоры
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# ===== 2. Разделение на train/test =====
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

input_dim = X.shape[1]


# ===== 3. Модель классификатора =====
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# ===== 4. Обучение без предобучения =====
model = Classifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Обучение без предобучения ===")
for epoch in range(50):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/50], Потеря: {total_loss:.4f}")


# ===== 5. Оценка эффективности =====
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted.numpy())

f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)
print("\nF1-score (без предобучения):", f1)
print("Матрица ошибок:\n", cm)


# ===== 6. Автоэнкодер для предобучения =====
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(input_dim)
ae_criterion = nn.MSELoss()
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

print("\n=== Предобучение автоэнкодера ===")
for epoch in range(50):
    autoencoder.train()
    total_loss = 0
    for X_batch, _ in train_loader:
        ae_optimizer.zero_grad()
        reconstructed = autoencoder(X_batch)
        loss = ae_criterion(reconstructed, X_batch)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/50], Потеря: {total_loss:.4f}")


# ===== 7. Модель с предобучением =====
model_pretrained = Classifier(input_dim, num_classes)
with torch.no_grad():
    model_pretrained.fc1.weight = autoencoder.encoder[0].weight
    model_pretrained.fc1.bias = autoencoder.encoder[0].bias
    model_pretrained.fc2.weight = autoencoder.encoder[2].weight
    model_pretrained.fc2.bias = autoencoder.encoder[2].bias
    model_pretrained.fc3.weight = autoencoder.encoder[4].weight
    model_pretrained.fc3.bias = autoencoder.encoder[4].bias

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pretrained.parameters(), lr=0.001)

print("\n=== Обучение с предобучением ===")
for epoch in range(50):
    model_pretrained.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model_pretrained(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/50], Потеря: {total_loss:.4f}")


# ===== 8. Оценка эффективности после предобучения =====
model_pretrained.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model_pretrained(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted.numpy())

f1_pretrained = f1_score(y_true, y_pred, average='weighted')
cm_pretrained = confusion_matrix(y_true, y_pred)
print("\nF1-score (с предобучением):", f1_pretrained)
print("Матрица ошибок:\n", cm_pretrained)

# ===== 9. Сравнение =====
print("\n=== Сравнение результатов ===")
print(f"Без предобучения: F1 = {f1:.4f}")
print(f"С предобучением:  F1 = {f1_pretrained:.4f}")