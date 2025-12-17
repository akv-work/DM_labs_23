import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def load_data(file_path):
    columns = [
        "buying", "maint", "doors", "persons",
        "lug_boot", "safety", "class"
    ]
    data = pd.read_csv(file_path, header=None, names=columns)
    return data

def preprocess_data(data):
    X = data.drop(columns=["class"])
    y = data["class"]

    # Кодирование признаков
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)

    # Кодирование целевой переменной
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded, X.columns, label_encoder

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    return model, acc

if __name__ == "__main__":

    data = load_data("car.data")

    X, y, feature_names, label_encoder = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Обучение моделей ---\n")

    # 1️⃣ Decision Tree
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )
    dt_model, dt_acc = train_and_evaluate(
        dt, X_train, X_test, y_train, y_test, "Decision Tree"
    )

    # 2️⃣ Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model, rf_acc = train_and_evaluate(
        rf, X_train, X_test, y_train, y_test, "Random Forest"
    )

    # 3️⃣ AdaBoost
    ada = AdaBoostClassifier(
        n_estimators=100,
        algorithm="SAMME",
        random_state=42
    )
    ada_model, ada_acc = train_and_evaluate(
        ada, X_train, X_test, y_train, y_test, "AdaBoost"
    )

    # 4️⃣ XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42
    )
    xgb_model, xgb_acc = train_and_evaluate(
        xgb, X_train, X_test, y_train, y_test, "XGBoost"
    )

    # 5️⃣ CatBoost
    cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_state=42
    )
    cat_model, cat_acc = train_and_evaluate(
        cat, X_train, X_test, y_train, y_test, "CatBoost"
    )

    print("\n--- Важность признаков (Decision Tree) ---")
    for name, importance in zip(feature_names, dt_model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    print("\n--- Важность признаков (Random Forest) ---")
    for name, importance in zip(feature_names, rf_model.feature_importances_):
        print(f"{name}: {importance:.4f}")
