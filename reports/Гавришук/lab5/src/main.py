from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=False,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n {name}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Примеры изображений и предсказаний (Decision Tree)", fontsize=14)

y_pred_sample = models["Decision Tree"].predict(X_test)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"Pred: {y_pred_sample[i]}\nTrue: {y_test[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.barh(list(results.keys()), list(results.values()), color="skyblue")
plt.xlabel("Accuracy")
plt.title("Сравнение точности моделей")
plt.xlim(0, 1)
for i, v in enumerate(results.values()):
    plt.text(v + 0.01, i, f"{v:.3f}", va="center")
plt.show()