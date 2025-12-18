import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("Загрузка датасета Parkinsons Telemonitoring...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
try:
    data = pd.read_csv(url)
    print("Данные успешно загружены!")
except:
    print("Ошибка загрузки. Использую локальную копию...")
    data = pd.read_csv('/content/parkinsons_updrs.data')

print(f"Размер датасета: {data.shape}")

print("Пропущенные значения:")
print(data.isnull().sum())

print("Статистика целевой переменной motor_UPDRS:")
print(f"Минимум: {data['motor_UPDRS'].min():.2f}")
print(f"Максимум: {data['motor_UPDRS'].max():.2f}")
print(f"Среднее: {data['motor_UPDRS'].mean():.2f}")
print(f"Стандартное отклонение: {data['motor_UPDRS'].std():.2f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(data['motor_UPDRS'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Распределение motor_UPDRS')
plt.xlabel('motor_UPDRS')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

X = data.drop(['motor_UPDRS', 'total_UPDRS', 'subject#'], axis=1, errors='ignore')

if 'subject#' in X.columns:
    X = X.drop('subject#', axis=1)

y = data['motor_UPDRS']

print(f"Количество признаков: {X.shape[1]}")

categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print(f"Размерность после обработки: X{X.shape}, y{y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_deep_nn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='linear')
    ])

    return model

input_dim = X_train_scaled.shape[1]
model = create_deep_nn(input_dim)

print("Архитектура модели:")
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.0001,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=True
)

print("Обучение завершено!")

print("\n" + "="*50)
print("ОЦЕНКА ЭФФЕКТИВНОСТИ МОДЕЛИ")
print("="*50)

y_pred = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("МЕТРИКИ КАЧЕСТВА:")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}")

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('График обучения: Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('График обучения: MAE')
plt.xlabel('Эпоха')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Реальные значения motor_UPDRS')
plt.ylabel('Предсказанные значения motor_UPDRS')
plt.title(f'Предсказания vs Реальные значения\nR2 = {r2:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
errors = y_pred - y_test
plt.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title('Распределение ошибок предсказания')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
first_layer_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
feature_names = X.columns

top_indices = np.argsort(feature_importance)[-10:]
plt.barh(range(len(top_indices)), feature_importance[top_indices])
plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
plt.xlabel('Важность признака')
plt.title('Топ-10 важных признаков')
plt.grid(True, alpha=0.3, axis='x')

plt.subplot(2, 3, 6)
plt.hist(y_test, bins=30, alpha=0.7, label='Реальные', color='blue', edgecolor='black')
plt.hist(y_pred, bins=30, alpha=0.7, label='Предсказанные', color='red', edgecolor='black')
plt.xlabel('motor_UPDRS')
plt.ylabel('Частота')
plt.title('Сравнение распределений')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*50)

bins = np.linspace(y_test.min(), y_test.max(), 5)
bin_errors = []

for i in range(len(bins)-1):
    mask = (y_test >= bins[i]) & (y_test < bins[i+1])
    if mask.sum() > 0:
        bin_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        bin_errors.append(bin_mae)
        print(f"Диапазон [{bins[i]:.1f}, {bins[i+1]:.1f}]: {mask.sum()} samples, MAE = {bin_mae:.3f}")

print(f"Средняя ошибка: {errors.mean():.3f}")
print(f"Стандартное отклонение ошибок: {errors.std():.3f}")
print(f"Максимальная положительная ошибка: {errors.max():.3f}")
print(f"Максимальная отрицательная ошибка: {errors.min():.3f}")

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
print("Ошибки по квантилям:")
for q in quantiles:
    q_error = np.quantile(np.abs(errors), q)
    print(f"{q*100:.0f}% ошибок <= {q_error:.3f}")

baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("СРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ (предсказание средним):")
print(f"Базовая MAE: {baseline_mae:.4f}")
print(f"Наша MAE: {mae:.4f}")
print(f"Улучшение: {((baseline_mae - mae) / baseline_mae * 100):.1f}%")

print(f"Базовая RMSE: {baseline_rmse:.4f}")
print(f"Наша RMSE: {rmse:.4f}")
print(f"Улучшение: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")
