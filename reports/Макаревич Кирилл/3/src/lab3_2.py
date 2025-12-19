import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
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

X = data.drop(['motor_UPDRS', 'total_UPDRS', 'subject#'], axis=1, errors='ignore')
if 'subject#' in X.columns:
    X = X.drop('subject#', axis=1)

y = data['motor_UPDRS']

categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print(f"Количество признаков: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Масштабирование завершено")

input_dim = X_train_scaled.shape[1]
encoding_dims = [64, 32, 16]

print(f"\n" + "="*50)
print("ПАРАМЕТРЫ АВТОЭНКОДЕРНОГО ПРЕДОБУЧЕНИЯ")
print("="*50)
print(f"Входная размерность: {input_dim}")
print(f"Архитектура автоэнкодеров: {input_dim} -> {encoding_dims}")

def create_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='relu', name=f'encoder_{encoding_dim}')(input_layer)

    decoded = Dense(input_dim, activation='linear', name=f'decoder_{encoding_dim}')(encoded)

    encoder = Model(input_layer, encoded, name=f'encoder_{encoding_dim}')
    autoencoder = Model(input_layer, decoded, name=f'autoencoder_{encoding_dim}')

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return encoder, autoencoder

def pretrain_autoencoders(X_train, encoding_dims, epochs_per_autoencoder=50):
    encoders = []
    current_data = X_train

    print("\n" + "="*50)
    print("НАЧАЛО ПРЕДОБУЧЕНИЯ АВТОЭНКОДЕРОВ")
    print("="*50)

    for i, encoding_dim in enumerate(encoding_dims):
        print(f"\n--- Обучение автоэнкодера {i+1}/{len(encoding_dims)} ---")
        print(f"Входная размерность: {current_data.shape[1]}")
        print(f"Целевая размерность: {encoding_dim}")

        encoder, autoencoder = create_autoencoder(current_data.shape[1], encoding_dim)

        print(f"Обучение автоэнкодера... (эпох: {epochs_per_autoencoder})")
        history = autoencoder.fit(
            current_data, current_data,
            epochs=epochs_per_autoencoder,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            shuffle=True
        )

        encoders.append(encoder)

        current_data = encoder.predict(current_data, verbose=0)

        print(f"Автоэнкодер {i+1} обучен. Final loss: {history.history['loss'][-1]:.4f}")

    return encoders

encoders = pretrain_autoencoders(X_train_scaled, encoding_dims, epochs_per_autoencoder=50)

print("\n" + "="*50)
print("СОЗДАНИЕ МОДЕЛИ С ПРЕДОБУЧЕННЫМИ ВЕСАМИ")
print("="*50)

def create_pretrained_model(encoders, input_dim):
    model = Sequential()

    current_dim = input_dim
    for i, encoder in enumerate(encoders):
        trained_weights, trained_biases = encoder.layers[1].get_weights()

        new_layer = Dense(
            units=trained_weights.shape[1],
            activation='relu',
            input_shape=(current_dim,) if i == 0 else None,
            name=f'pretrained_dense_{i+1}'
        )

        model.add(new_layer)

        if i == 0:
            model.build((None, current_dim))

        model.layers[-1].set_weights([trained_weights, trained_biases])

        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))

        current_dim = trained_weights.shape[1]

    model.add(Dense(8, activation='relu', name='final_dense_1'))
    model.add(Dropout(0.1, name='final_dropout'))
    model.add(Dense(1, activation='linear', name='output'))

    return model

pretrained_model = create_pretrained_model(encoders, input_dim)

print("Архитектура модели с предобученными весами:")
pretrained_model.summary()

for layer in pretrained_model.layers:
    if 'pretrained' in layer.name:
        layer.trainable = False

pretrained_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

print("Начало обучения с замороженными предобученными слоями...")

history_stage1 = pretrained_model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1,
    shuffle=True
)

print("Разморозка предобученных слоев для тонкой настройки...")

for layer in pretrained_model.layers:
    layer.trainable = True

pretrained_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
    metrics=['mae']
)

history_stage2 = pretrained_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    ],
    verbose=1,
    shuffle=True
)

print("Обучение завершено!")

def create_standard_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(8, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

print("\n" + "="*50)
print("ОБУЧЕНИЕ СТАНДАРТНОЙ МОДЕЛИ ДЛЯ СРАВНЕНИЯ")
print("="*50)

standard_model = create_standard_model(input_dim)

history_standard = standard_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    ],
    verbose=1,
    shuffle=True
)

print("Оценка моделей...")

y_pred_pretrained = pretrained_model.predict(X_test_scaled).flatten()
y_pred_standard = standard_model.predict(X_test_scaled).flatten()

mae_pretrained = mean_absolute_error(y_test, y_pred_pretrained)
mse_pretrained = mean_squared_error(y_test, y_pred_pretrained)
rmse_pretrained = np.sqrt(mse_pretrained)
r2_pretrained = r2_score(y_test, y_pred_pretrained)
mape_pretrained = mean_absolute_percentage_error(y_test, y_pred_pretrained)

mae_standard = mean_absolute_error(y_test, y_pred_standard)
mse_standard = mean_squared_error(y_test, y_pred_standard)
rmse_standard = np.sqrt(mse_standard)
r2_standard = r2_score(y_test, y_pred_standard)
mape_standard = mean_absolute_percentage_error(y_test, y_pred_standard)

print("\n" + "="*50)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*50)

print("\nМОДЕЛЬ С ПРЕДОБУЧЕНИЕМ:")
print(f"MAE: {mae_pretrained:.4f}")
print(f"MSE: {mse_pretrained:.4f}")
print(f"RMSE: {rmse_pretrained:.4f}")
print(f"R2: {r2_pretrained:.4f}")
print(f"MAPE: {mape_pretrained:.4f}")

print("\nСТАНДАРТНАЯ МОДЕЛЬ:")
print(f"MAE: {mae_standard:.4f}")
print(f"MSE: {mse_standard:.4f}")
print(f"RMSE: {rmse_standard:.4f}")
print(f"R2: {r2_standard:.4f}")
print(f"MAPE: {mape_standard:.4f}")

print("\nРАЗНИЦА:")
print(f"Улучшение MAE: {((mae_standard - mae_pretrained) / mae_standard * 100):.1f}%")
print(f"Улучшение R2: {((r2_pretrained - r2_standard) / abs(r2_standard) * 100):.1f}%")

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_pretrained, alpha=0.6, color='blue', label='С предобучением')
plt.scatter(y_test, y_pred_standard, alpha=0.6, color='red', label='Стандартная')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение предсказаний моделей')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(history_stage1.history['val_loss'] + history_stage2.history['val_loss'],
         label='С предобучением (val)', linewidth=2)
plt.plot(history_standard.history['val_loss'],
         label='Стандартная (val)', linewidth=2)
plt.title('Сравнение потерь на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
errors_pretrained = y_pred_pretrained - y_test
errors_standard = y_pred_standard - y_test

plt.hist(errors_pretrained, bins=50, alpha=0.7, label='С предобучением', color='blue')
plt.hist(errors_standard, bins=50, alpha=0.7, label='Стандартная', color='red')
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title('Распределение ошибок')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
metrics = ['MAE', 'RMSE', 'MAPE']
values_pretrained = [mae_pretrained, rmse_pretrained, mape_pretrained]
values_standard = [mae_standard, rmse_standard, mape_standard]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, values_pretrained, width, label='С предобучением', color='blue')
plt.bar(x + width/2, values_standard, width, label='Стандартная', color='red')
plt.xlabel('Метрики')
plt.ylabel('Значение')
plt.title('Сравнение метрик качества')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
models = ['С предобучением', 'Стандартная']
r2_scores = [r2_pretrained, r2_standard]

bars = plt.bar(models, r2_scores, color=['blue', 'red'])
plt.ylabel('R2 Score')
plt.title('Сравнение R2 Score')
plt.ylim(0, 1)
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ОБУЧЕНИЕ С ПРЕДОБУЧЕНИЕМ ЗАВЕРШЕНО")
print("="*70)
