import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('winequality-white.csv', sep=';')

# Создаём бинарную целевую переменную: "good" = 1 если quality >= 7, иначе 0
df['good'] = (df['quality'] >= 7).astype(int)

X = df.drop(columns=['quality', 'good'])
y = df['good'].values

print("Размер датасета:", df.shape)
print("Распределение классов (0 = ordinary, 1 = good):")
print(pd.Series(y).value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'AdaBoost (trees)': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=100, random_state=42
    ),
    'XGBoost': XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(iterations=200, random_state=42, verbose=False)
}


results = []
detailed_reports = {}

for name, model in models.items():
    print(f"\nОбучаем модель: {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, pos_label=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    report = classification_report(y_test, y_pred, target_names=['ordinary','good'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'model': name,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })
    detailed_reports[name] = {
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'model_obj': model
    }

    print(f"  F1 (good=1): {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")

results_df = pd.DataFrame(results).sort_values('f1', ascending=False).reset_index(drop=True)
print("\nСравнение моделей по F1 (по убыванию):")
print(results_df)

best = results_df.iloc[0]
print(f"\nЛучшая модель по F1: {best['model']} (F1 = {best['f1']:.4f}, Precision = {best['precision']:.4f}, Recall = {best['recall']:.4f})")

# Покажем classification_report и матрицу ошибок для лучшей модели
best_name = best['model']
print(f"\nClassification report для лучшей модели ({best_name}):")
print(classification_report(y_test, detailed_reports[best_name]['y_pred'], target_names=['ordinary','good']))

print("Confusion matrix (rows: true, cols: predicted):")
print(detailed_reports[best_name]['confusion_matrix'])

plt.figure(figsize=(8,5))
sns.barplot(x='model', y='f1', data=results_df)
plt.title('F1-score (класс good=1) для моделей')
plt.xticks(rotation=45)
plt.ylim(0,1)
for i, v in enumerate(results_df['f1']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
df_pr = results_df.melt(id_vars=['model'], value_vars=['precision','recall'], var_name='metric', value_name='value')
sns.barplot(x='model', y='value', hue='metric', data=df_pr)
plt.title('Precision и Recall (класс good=1)')
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.tight_layout()
plt.show()

results_df.to_csv('wine_models_f1_comparison.csv', index=False)
print("\nТаблица результатов сохранена в wine_models_f1_comparison.csv")
