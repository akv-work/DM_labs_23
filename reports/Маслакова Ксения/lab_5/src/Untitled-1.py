
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# 1. Загрузка набора данных
# ------------------------------
digits = load_digits()
X, y = digits.data, digits.target

print(f"Размер выборки: {X.shape}")
print(f"Количество классов: {len(np.unique(y))}")

# ------------------------------
# 2. Разделение данных
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# ------------------------------
# 3. Обучение моделей
# ------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
}

# Попробуем импортировать XGBoost и CatBoost, если они установлены
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
except ImportError:
    print("⚠️ XGBoost не установлен, пропускаем...")

try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=50,
        depth=5,
        learning_rate=0.1,
        verbose=0,
        random_state=42
    )
except ImportError:
    print("⚠️ CatBoost не установлен, пропускаем...")

# ------------------------------
# 4. Обучение и оценка
# ------------------------------
results = {}

for name, model in models.items():
    print(f"\n=== Обучение модели: {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Точность: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    results[name] = acc

# ------------------------------
# 5. Сравнение моделей
# ------------------------------
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title("Сравнение точности моделей")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.ylim(0.8, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ------------------------------
# 6. Выводы
# ------------------------------
best_model = max(results, key=results.get)
print(f"\n🏆 Лучшая модель: {best_model} с точностью {results[best_model]:.4f}")
