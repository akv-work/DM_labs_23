
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# -------------------------
# 1. Загрузка и подготовка данных
# -------------------------
df = pd.read_csv("maternal.csv")

le = LabelEncoder()
df['RiskLevel'] = le.fit_transform(df['RiskLevel'])

X = df.drop('RiskLevel', axis=1).values
y = df['RiskLevel'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 2. Базовая MLP модель (без предобучения)
# -------------------------
model = models.Sequential([
    Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Обучаем базовую MLP модель...")
history_mlp = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

# Оценка
y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------
# 3. Автоэнкодер
# -------------------------
input_dim = X.shape[1]
autoencoder_input = Input(shape=(input_dim,))
encoded = layers.Dense(32, activation='relu')(autoencoder_input)
encoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(autoencoder_input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("\nОбучаем автоэнкодер...")
history_auto = autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

# -------------------------
# 4. Модель с предобучением (encoder часть автоэнкодера)
# -------------------------
encoder = models.Model(autoencoder_input, encoded)
X_encoded_train = encoder.predict(X_train, verbose=0)
X_encoded_test = encoder.predict(X_test, verbose=0)

classifier = models.Sequential([
    Input(shape=(16,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nОбучаем классификатор на закодированных данных...")
history_pre = classifier.fit(X_encoded_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

y_pred_enc = classifier.predict(X_encoded_test, verbose=0).argmax(axis=1)
print("=== Результаты с предобучением автоэнкодером ===")
print("F1-score (macro):", f1_score(y_test, y_pred_enc, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_enc))

# -------------------------
# 5. Визуализация
# -------------------------
plt.figure(figsize=(12, 5))

# --- Потери MLP ---
plt.subplot(1, 2, 1)
plt.plot(history_mlp.history['loss'], label='train_loss (MLP)')
plt.plot(history_mlp.history['val_loss'], label='val_loss (MLP)')
plt.title('Loss MLP (без предобучения)')
plt.xlabel('Эпоха')
plt.ylabel('Потери (Loss)')
plt.legend()
plt.grid(True)

# --- Потери автоэнкодера ---
plt.subplot(1, 2, 2)
plt.plot(history_auto.history['loss'], label='train_loss (Autoencoder)')
plt.plot(history_auto.history['val_loss'], label='val_loss (Autoencoder)')
plt.title('Loss автоэнкодера (предобучение)')
plt.xlabel('Эпоха')
plt.ylabel('Потери (MSE)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------
# 6. Confusion matrices
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix (без предобучения)")
axes[0].set_xlabel("Предсказано")
axes[0].set_ylabel("Истинное значение")

sns.heatmap(confusion_matrix(y_test, y_pred_enc), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Confusion Matrix (c автоэнкодером)")
axes[1].set_xlabel("Предсказано")
axes[1].set_ylabel("Истинное значение")

plt.tight_layout()
plt.show()
