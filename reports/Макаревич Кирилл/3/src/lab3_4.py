import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

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

print("Пропущенные значения:")
print(data.isnull().sum())

print("Статистика целевой переменной class:")
class_counts = data['class'].value_counts()
print(class_counts)
print(f"Съедобные: {class_counts['edible']} ({class_counts['edible']/len(data)*100:.1f}%)")
print(f"Ядовитые: {class_counts['poisonous']} ({class_counts['poisonous']/len(data)*100:.1f}%)")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.bar(class_counts.index, class_counts.values, color=['green', 'red'], alpha=0.7)
plt.title('Распределение классов')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.grid(True, alpha=0.3)

X = data.drop('class', axis=1)
y = data['class']

print(f"Количество признаков: {X.shape[1]}")

# Кодирование категориальных признаков
label_encoders = {}
X_encoded = X.copy()

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Кодирование целевой переменной
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"Размерность после обработки: X{X_encoded.shape}, y{y_categorical.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_categorical, test_size=0.2, random_state=42, shuffle=True
)

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_deep_nn(input_dim, num_classes):
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

        Dense(num_classes, activation='softmax')
    ])

    return model

input_dim = X_train_scaled.shape[1]
num_classes = y_categorical.shape[1]
model = create_deep_nn(input_dim, num_classes)

print("Архитектура модели:")
model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
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

y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print("МЕТРИКИ КАЧЕСТВА:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nОтчет классификации:")
print(classification_report(y_true, y_pred, target_names=le_target.classes_))

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('График обучения: Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('График обучения: Accuracy')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title('Матрица ошибок')
plt.ylabel('Реальный класс')
plt.xlabel('Предсказанный класс')

plt.subplot(2, 3, 4)
# Вероятности для положительного класса (poisonous)
if le_target.classes_[1] == 'poisonous':
    positive_class_probs = y_pred_proba[:, 1]
else:
    positive_class_probs = y_pred_proba[:, 0]

plt.hist(positive_class_probs[y_true == 0], bins=30, alpha=0.7, label='Съедобные', color='green')
plt.hist(positive_class_probs[y_true == 1], bins=30, alpha=0.7, label='Ядовитые', color='red')
plt.xlabel('Вероятность ядовитости')
plt.ylabel('Частота')
plt.title('Распределение вероятностей')
plt.legend()
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
# Анализ уверенности модели
confidence = np.max(y_pred_proba, axis=1)
correct_predictions = (y_pred == y_true)

plt.hist(confidence[correct_predictions], bins=30, alpha=0.7, label='Правильные', color='green')
plt.hist(confidence[~correct_predictions], bins=30, alpha=0.7, label='Ошибки', color='red')
plt.xlabel('Уверенность модели')
plt.ylabel('Частота')
plt.title('Распределение уверенности предсказаний')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*50)

# Анализ по классам
print("АНАЛИЗ ПО КЛАССАМ:")
for i, class_name in enumerate(le_target.classes_):
    class_mask = y_true == i
    if class_mask.sum() > 0:
        class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
        print(f"{class_name}: {class_mask.sum()} samples, Accuracy = {class_accuracy:.3f}")

# Анализ уверенности предсказаний
confidence_ranges = [0.0, 0.5, 0.7, 0.9, 1.0]
print("\nАНАЛИЗ УВЕРЕННОСТИ ПРЕДСКАЗАНИЙ:")
for i in range(len(confidence_ranges)-1):
    low, high = confidence_ranges[i], confidence_ranges[i+1]
    mask = (confidence >= low) & (confidence < high)
    if mask.sum() > 0:
        range_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        print(f"Уверенность [{low:.1f}, {high:.1f}): {mask.sum()} samples, Accuracy = {range_accuracy:.3f}")

# Сравнение с базовой моделью (предсказание наиболее частого класса)
baseline_pred = np.full_like(y_true, np.argmax(np.bincount(y_true)))
baseline_accuracy = accuracy_score(y_true, baseline_pred)

print(f"\nСРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ (предсказание наиболее частого класса):")
print(f"Базовая Accuracy: {baseline_accuracy:.4f}")
print(f"Наша Accuracy: {accuracy:.4f}")
print(f"Улучшение: {((accuracy - baseline_accuracy) / baseline_accuracy * 100):.1f}%")

# Анализ ошибок
errors_mask = (y_pred != y_true)
if errors_mask.sum() > 0:
    print(f"\nАНАЛИЗ ОШИБОК:")
    print(f"Всего ошибок: {errors_mask.sum()} ({errors_mask.sum()/len(y_true)*100:.1f}%)")

    error_probs = confidence[errors_mask]
    print(f"Средняя уверенность при ошибках: {error_probs.mean():.3f}")
    print(f"Минимальная уверенность при ошибках: {error_probs.min():.3f}")

    # Анализ типов ошибок
    error_types = []
    for true_class, pred_class in zip(y_true[errors_mask], y_pred[errors_mask]):
        error_types.append(f"{le_target.classes_[true_class]} -> {le_target.classes_[pred_class]}")

    error_counts = pd.Series(error_types).value_counts()
    print("\nТипы ошибок:")
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count}")

print("\n" + "="*70)
print("МОДЕЛЬ УСПЕШНО ОБУЧЕНА И ПРОТЕСТИРОВАНА")
print("="*70)
