import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ПРЕДОБУЧЕНИЕ АВТОЭНКОДЕРОМ ДЛЯ ДАТАСЕТА MUSHROOM")
print("=" * 70)

# Создание синтетических данных mushroom
print("Создание демонстрационных данных mushroom...")
np.random.seed(42)
n_samples = 2000

columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Создаем данные с зависимостью от класса
data = {
    'class': np.random.choice(['edible', 'poisonous'], n_samples, p=[0.6, 0.4]),
}

for i, col in enumerate(columns[1:]):
    if i % 3 == 0:
        # Сильная корреляция с классом
        data[col] = np.where(data['class'] == 'edible',
                            np.random.choice(['a', 'b', 'c'], n_samples, p=[0.7, 0.2, 0.1]),
                            np.random.choice(['a', 'b', 'c'], n_samples, p=[0.1, 0.2, 0.7]))
    elif i % 3 == 1:
        # Средняя корреляция
        data[col] = np.where(data['class'] == 'edible',
                            np.random.choice(['x', 'y', 'z'], n_samples, p=[0.6, 0.3, 0.1]),
                            np.random.choice(['x', 'y', 'z'], n_samples, p=[0.2, 0.4, 0.4]))
    else:
        # Слабая корреляция
        data[col] = np.random.choice(['m', 'n', 'o', 'p'], n_samples)

data = pd.DataFrame(data)
print("Демонстрационные данные созданы!")

print(f"Размер датасета: {data.shape}")
print(f"Распределение классов:")
print(data['class'].value_counts())

# Подготовка данных
X = data.drop('class', axis=1)
y = data['class']

# Кодирование категориальных признаков
label_encoders = {}
X_encoded = X.copy()

print("\nКодирование категориальных признаков...")
for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le
    print(f"  {column}: {len(le.classes_)} категорий")

# Кодирование целевой переменной
y_encoded = LabelEncoder().fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"\nРазмерность после обработки: X{X_encoded.shape}, y{y_categorical.shape}")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_categorical, test_size=0.2, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Валидационная выборка: {X_val.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Масштабирование завершено")

# Параметры автоэнкодеров
input_dim = X_train_scaled.shape[1]
encoding_dims = [32, 16, 8]  # Размерности скрытых слоев

print(f"\n" + "="*50)
print("ПАРАМЕТРЫ АВТОЭНКОДЕРНОГО ПРЕДОБУЧЕНИЯ")
print("="*50)
print(f"Входная размерность: {input_dim}")
print(f"Архитектура автоэнкодеров: {input_dim} -> {encoding_dims}")
print(f"Количество классов: {y_categorical.shape[1]}")

def create_autoencoder(input_dim, encoding_dim):
    """Создает автоэнкодер для предобучения одного слоя"""
    # Входной слой
    input_layer = Input(shape=(input_dim,))

    # Энкодер
    encoded = Dense(encoding_dim, activation='relu', name=f'encoder_{encoding_dim}')(input_layer)

    # Декодер
    decoded = Dense(input_dim, activation='linear', name=f'decoder_{encoding_dim}')(encoded)

    # Модели
    encoder = Model(input_layer, encoded, name=f'encoder_{encoding_dim}')
    autoencoder = Model(input_layer, decoded, name=f'autoencoder_{encoding_dim}')

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return encoder, autoencoder

def pretrain_autoencoders(X_train, encoding_dims, epochs_per_autoencoder=30):
    """Последовательно предобучает автоэнкодеры"""
    encoders = []
    current_data = X_train

    print("\n" + "="*50)
    print("НАЧАЛО ПРЕДОБУЧЕНИЯ АВТОЭНКОДЕРОВ")
    print("="*50)

    for i, encoding_dim in enumerate(encoding_dims):
        print(f"\n--- Обучение автоэнкодера {i+1}/{len(encoding_dims)} ---")
        print(f"Входная размерность: {current_data.shape[1]}")
        print(f"Целевая размерность: {encoding_dim}")

        # Создание автоэнкодера
        encoder, autoencoder = create_autoencoder(current_data.shape[1], encoding_dim)

        # Обучение автоэнкодера
        print(f"Обучение автоэнкодера... (эпох: {epochs_per_autoencoder})")
        history = autoencoder.fit(
            current_data, current_data,
            epochs=epochs_per_autoencoder,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            shuffle=True
        )

        # Сохранение обученного энкодера
        encoders.append(encoder)

        # Преобразование данных для следующего автоэнкодера
        current_data = encoder.predict(current_data, verbose=0)

        print(f"Автоэнкодер {i+1} обучен. Final loss: {history.history['loss'][-1]:.4f}")

    return encoders

# Предобучение автоэнкодеров
encoders = pretrain_autoencoders(X_train_scaled, encoding_dims, epochs_per_autoencoder=30)

print("\n" + "="*50)
print("СОЗДАНИЕ КЛАССИФИКАТОРА С ПРЕДОБУЧЕННЫМИ ВЕСАМИ")
print("="*50)

def create_pretrained_classifier(encoders, input_dim, num_classes):
    """Создает классификатор с предобученными весами"""
    model = Sequential()

    # Добавляем предобученные слои
    current_dim = input_dim
    for i, encoder in enumerate(encoders):
        # Получаем веса обученного слоя
        trained_weights, trained_biases = encoder.layers[1].get_weights()

        # Создаем новый слой с такими же параметрами
        new_layer = Dense(
            units=trained_weights.shape[1],
            activation='relu',
            input_shape=(current_dim,) if i == 0 else None,
            name=f'pretrained_dense_{i+1}'
        )

        # Добавляем слой в модель
        model.add(new_layer)

        # Устанавливаем обученные веса
        if i == 0:
            model.build((None, current_dim))
        model.layers[-1].set_weights([trained_weights, trained_biases])

        # Добавляем BatchNormalization и Dropout
        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(Dropout(0.3, name=f'dropout_{i+1}'))

        current_dim = trained_weights.shape[1]

    # Добавляем финальные слои для классификации
    model.add(Dense(8, activation='relu', name='final_dense_1'))
    model.add(BatchNormalization(name='final_bn'))
    model.add(Dropout(0.2, name='final_dropout'))
    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model

# Создание классификатора с предобученными весами
num_classes = y_categorical.shape[1]
pretrained_model = create_pretrained_classifier(encoders, input_dim, num_classes)

print("Архитектура классификатора с предобученными весами:")
pretrained_model.summary()

# Замораживаем предобученные слои на первых эпохах
for layer in pretrained_model.layers:
    if 'pretrained' in layer.name:
        layer.trainable = False

# Компиляция с замороженными слоями
pretrained_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Начало обучения с замороженными предобученными слоями...")

# Первый этап обучения с замороженными слоями
history_stage1 = pretrained_model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    verbose=1,
    shuffle=True
)

print("Разморозка предобученных слоев для тонкой настройки...")

# Размораживаем все слои для тонкой настройки
for layer in pretrained_model.layers:
    layer.trainable = True

# Перекомпиляция с меньшим learning rate для тонкой настройки
pretrained_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Второй этап обучения с размороженными слоями
history_stage2 = pretrained_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    ],
    verbose=1,
    shuffle=True
)

print("Обучение завершено!")

# Создание обычной модели для сравнения
def create_standard_classifier(input_dim, num_classes):
    """Создает обычный классификатор без предобучения"""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(8, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(8, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

print("\n" + "="*50)
print("ОБУЧЕНИЕ СТАНДАРТНОГО КЛАССИФИКАТОРА ДЛЯ СРАВНЕНИЯ")
print("="*50)

standard_model = create_standard_classifier(input_dim, num_classes)

history_standard = standard_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
    ],
    verbose=1,
    shuffle=True
)

print("Оценка моделей...")

# Предсказания
y_pred_pretrained = pretrained_model.predict(X_test_scaled)
y_pred_standard = standard_model.predict(X_test_scaled)

# Преобразование в метки классов
y_pred_pretrained_labels = np.argmax(y_pred_pretrained, axis=1)
y_pred_standard_labels = np.argmax(y_pred_standard, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

class_names = ['edible', 'poisonous']

# Метрики для предобученной модели
accuracy_pretrained = accuracy_score(y_test_labels, y_pred_pretrained_labels)
f1_pretrained = f1_score(y_test_labels, y_pred_pretrained_labels, average='weighted')

# Метрики для стандартной модели
accuracy_standard = accuracy_score(y_test_labels, y_pred_standard_labels)
f1_standard = f1_score(y_test_labels, y_pred_standard_labels, average='weighted')

print("\n" + "="*50)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ")
print("="*50)

print("\nМОДЕЛЬ С ПРЕДОБУЧЕНИЕМ:")
print(f"Accuracy: {accuracy_pretrained:.4f}")
print(f"F1-Score: {f1_pretrained:.4f}")
print("\nОтчет классификации:")
print(classification_report(y_test_labels, y_pred_pretrained_labels, target_names=class_names))

print("\nСТАНДАРТНАЯ МОДЕЛЬ:")
print(f"Accuracy: {accuracy_standard:.4f}")
print(f"F1-Score: {f1_standard:.4f}")
print("\nОтчет классификации:")
print(classification_report(y_test_labels, y_pred_standard_labels, target_names=class_names))

print("\nРАЗНИЦА:")
print(f"Улучшение Accuracy: {((accuracy_pretrained - accuracy_standard) / accuracy_standard * 100):.1f}%")
print(f"Улучшение F1-Score: {((f1_pretrained - f1_standard) / f1_standard * 100):.1f}%")

# Визуализация результатов
plt.figure(figsize=(20, 12))

# 1. Матрицы ошибок
plt.subplot(2, 3, 1)
cm_pretrained = confusion_matrix(y_test_labels, y_pred_pretrained_labels)
sns.heatmap(cm_pretrained, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок: С предобучением')
plt.ylabel('Реальный класс')
plt.xlabel('Предсказанный класс')

plt.subplot(2, 3, 2)
cm_standard = confusion_matrix(y_test_labels, y_pred_standard_labels)
sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Матрица ошибок: Стандартная')
plt.ylabel('Реальный класс')
plt.xlabel('Предсказанный класс')

# 2. Графики обучения
plt.subplot(2, 3, 3)
plt.plot(history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
         label='С предобучением (val)', linewidth=2)
plt.plot(history_standard.history['val_accuracy'],
         label='Стандартная (val)', linewidth=2)
plt.title('Сравнение точности на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Сравнение потерь
plt.subplot(2, 3, 4)
plt.plot(history_stage1.history['val_loss'] + history_stage2.history['val_loss'],
         label='С предобучением (val)', linewidth=2)
plt.plot(history_standard.history['val_loss'],
         label='Стандартная (val)', linewidth=2)
plt.title('Сравнение потерь на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Сравнение метрик
plt.subplot(2, 3, 5)
metrics = ['Accuracy', 'F1-Score']
values_pretrained = [accuracy_pretrained, f1_pretrained]
values_standard = [accuracy_standard, f1_standard]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, values_pretrained, width, label='С предобучением', color='blue')
plt.bar(x + width/2, values_standard, width, label='Стандартная', color='red')
plt.xlabel('Метрики')
plt.ylabel('Значение')
plt.title('Сравнение метрик классификации')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Accuracy сравнение
plt.subplot(2, 3, 6)
models = ['С предобучением', 'Стандартная']
acc_scores = [accuracy_pretrained, accuracy_standard]

bars = plt.bar(models, acc_scores, color=['blue', 'red'])
plt.ylabel('Accuracy')
plt.title('Сравнение Accuracy')
plt.ylim(0, 1)
for bar, score in zip(bars, acc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ важности признаков через автоэнкодер
print("\n" + "="*50)
print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
print("="*50)

# Получаем веса первого автоэнкодера
first_encoder_weights = encoders[0].layers[1].get_weights()[0]
feature_importance = np.mean(np.abs(first_encoder_weights), axis=1)

# Топ-10 самых важных признаков
top_indices = np.argsort(feature_importance)[-10:][::-1]
top_features = [X.columns[i] for i in top_indices]
top_importance = feature_importance[top_indices]

print("Топ-10 важных признаков по автоэнкодеру:")
for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
    print(f"{i+1}. {feature}: {importance:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importance[::-1])
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel('Важность признака')
plt.title('Топ-10 важных признаков (по автоэнкодеру)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ОБУЧЕНИЕ С ПРЕДОБУЧЕНИЕМ ДЛЯ КЛАССИФИКАЦИИ ЗАВЕРШЕНО")
print("="*70)
