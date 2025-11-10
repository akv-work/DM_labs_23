
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. Загрузка данных ---
dataset = fetch_ucirepo(id=925)
X = dataset.data.features.copy()
y = dataset.data.targets.copy()

if 'aveOralF' in y.columns:
    y = y['aveOralF']
else:
    y = y[y.columns[0]]

# Обработка категориальных признаков
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# Заполняем пропуски средними значениями
X = X.fillna(X.mean())

# --- 2. Нормализация ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# --- 3. Разделение ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. Преобразуем в тензоры ---
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# --- 5. Определяем автоэнкодер ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 6. Определяем основную регрессионную сеть ---
class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# --- 7. Обучение автоэнкодера ---
def train_autoencoder(X_train, input_dim, hidden_dim=8, epochs=30, lr=0.005):
    model = Autoencoder(input_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        output = model(X_train)
        loss = criterion(output, X_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха [{epoch+1}/{epochs}], Потеря автоэнкодера: {loss.item():.6f}")
    return model, losses

# --- 8. Обучение регрессора ---
def train_regressor(model, X_train, y_train, epochs=50, lr=0.005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        preds = model(X_train)
        loss = criterion(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха [{epoch+1}/{epochs}], Потеря модели: {loss.item():.6f}")
    return losses

# --- 9. Обучение без предобучения ---
print("\n=== Обучение без предобучения ===")
input_dim = X_train.shape[1]
model_vanilla = Regressor(input_dim)
losses_vanilla = train_regressor(model_vanilla, X_train, y_train)
preds_vanilla = model_vanilla(X_test).detach().numpy()
mape_vanilla = mean_absolute_percentage_error(y_test, preds_vanilla)
print(f"MAPE без предобучения: {mape_vanilla:.4f}")

# --- 10. Предобучение автоэнкодером ---
print("\n=== Предобучение автоэнкодером ===")
ae, losses_ae = train_autoencoder(X_train, input_dim)
encoded_train = ae.encoder(X_train).detach()
encoded_test = ae.encoder(X_test).detach()

# --- 11. Обучение регрессора на предобученных признаках ---
model_pretrained = Regressor(encoded_train.shape[1])
losses_pretrained = train_regressor(model_pretrained, encoded_train, y_train)
preds_pretrained = model_pretrained(encoded_test).detach().numpy()
mape_pretrained = mean_absolute_percentage_error(y_test, preds_pretrained)
print(f"MAPE с предобучением: {mape_pretrained:.4f}")

# --- 12. Графики потерь ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses_ae, label='Autoencoder Loss', color='orange')
plt.title('Обучение автоэнкодера')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(losses_vanilla, label='Без предобучения')
plt.plot(losses_pretrained, label='С предобучением')
plt.title('Потери регрессора')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(y_test, preds_vanilla, label='Без предобучения', alpha=0.6)
plt.scatter(y_test, preds_pretrained, label='С предобучением', alpha=0.6)
plt.title('Сравнение реальных и предсказанных')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные')
plt.legend()

plt.tight_layout()
plt.show()

# --- 13. Итог ---
print("\n=== Итог ===")
print(f"MAPE без предобучения: {mape_vanilla:.4f}")
print(f"MAPE с предобучением: {mape_pretrained:.4f}")
if mape_pretrained < mape_vanilla:
    print("✅ Предобучение улучшило качество модели!")
else:
    print("⚠️ Предобучение не дало улучшения (попробуй увеличить эпохи или hidden_dim).")
