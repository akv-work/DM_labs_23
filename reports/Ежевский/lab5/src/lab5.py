import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Загрузка IRIS CSV формата Kaggle ===
df = pd.read_csv("iris.csv")

# Удаляем Id
df = df.drop("Id", axis=1)

# Целевая переменная
X = df.drop("Species", axis=1)

# Кодировка
le = LabelEncoder()
y = le.fit_transform(df["Species"])

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Модели
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss', use_label_encoder=False),
    "CatBoost": CatBoostClassifier(iterations=200, verbose=0, random_seed=42)
}

results = {}

# Обучение
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "acc": acc, "pred": y_pred}

# График точности
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=[r["acc"] for r in results.values()], palette="viridis")
plt.title("Сравнение точности моделей", fontsize=14)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()

# Матрицы ошибок
for name, res in results.items():
    cm = confusion_matrix(y_test, res["pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

# Decision Tree структура
plt.figure(figsize=(14, 14))
plot_tree(
    results["Decision Tree"]["model"],
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree Structure")
plt.show()

# Feature importance
def plot_feature_importance(model, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 4))
        sns.barplot(x=importances[indices], y=np.array(X.columns)[indices], palette="mako")
        plt.title(f"Feature Importance: {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

for name, res in results.items():
    plot_feature_importance(res["model"], name)

# Сравнение точности
print("Сравнение точности:")
for name, res in results.items():
    print(f"{name:15s} — Accuracy: {res['acc']:.4f}")

# Лучшая модель
best_model = max(results.items(), key=lambda x: x[1]['acc'])
print(f"\nЛучшая модель: {best_model[0]} с точностью {best_model[1]['acc']:.4f}")

# Classification reports
print("\n=== Classification Reports ===")
for name, res in results.items():
    print(f"\n--- {name} ---")
    print(classification_report(y_test, res["pred"], target_names=le.classes_))
