import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

print("=== KDD CUP 1999 - Network Intrusion Detection ===")

try:
    data = pd.read_csv('kddcup.data_10_percent_corrected', header=None)
    print("Данные успешно загружены")
except:
    print("Файл kddcup.data не найден, создаем демо-данные...")
    np.random.seed(42)
    n_samples = 10000
    data = pd.DataFrame({
        'duration': np.random.exponential(2, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
        'src_bytes': np.random.poisson(1000, n_samples),
        'dst_bytes': np.random.poisson(1000, n_samples),
        'land': np.random.choice([0, 1], n_samples),
        'wrong_fragment': np.random.poisson(0.1, n_samples),
        'urgent': np.random.poisson(0.01, n_samples),
        'hot': np.random.poisson(0.5, n_samples),
        'num_failed_logins': np.random.poisson(0.05, n_samples),
        'logged_in': np.random.choice([0, 1], n_samples),
        'num_compromised': np.random.poisson(0.01, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples),
        'su_attempted': np.random.choice([0, 1], n_samples),
        'num_root': np.random.poisson(0.01, n_samples),
        'num_file_creations': np.random.poisson(0.05, n_samples),
        'num_shells': np.random.poisson(0.01, n_samples),
        'num_access_files': np.random.poisson(0.02, n_samples),
        'num_outbound_cmds': np.random.poisson(0.001, n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples),
        'is_guest_login': np.random.choice([0, 1], n_samples),
        'count': np.random.poisson(100, n_samples),
        'srv_count': np.random.poisson(100, n_samples),
        'serror_rate': np.random.uniform(0, 1, n_samples),
        'srv_serror_rate': np.random.uniform(0, 1, n_samples),
        'rerror_rate': np.random.uniform(0, 1, n_samples),
        'srv_rerror_rate': np.random.uniform(0, 1, n_samples),
        'same_srv_rate': np.random.uniform(0, 1, n_samples),
        'diff_srv_rate': np.random.uniform(0, 1, n_samples),
        'srv_diff_host_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_count': np.random.poisson(255, n_samples),
        'dst_host_srv_count': np.random.poisson(255, n_samples),
        'dst_host_same_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_diff_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_same_src_port_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_serror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_serror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_rerror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_rerror_rate': np.random.uniform(0, 1, n_samples),
        'label': np.random.choice(['normal', 'smurf', 'neptune', 'satan'], n_samples, p=[0.2, 0.3, 0.3, 0.2])
    })

print(f"Размер датасета: {data.shape}")
print(f"Первые 5 строк:\n{data.head()}")
print(f"\nУникальные значения целевой переменной: {data[data.columns[-1]].unique()}")

if data.shape[1] > 10:
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
else:
    features = data.drop('label', axis=1)
    target = data['label']

print(f"\nКоличество признаков: {features.shape[1]}")
print(f"Типы признаков:\n{features.dtypes}")

categorical_columns = features.select_dtypes(include=['object']).columns
numerical_columns = features.select_dtypes(include=[np.number]).columns

print(f"\nКатегориальные признаки: {len(categorical_columns)}")
print(f"Числовые признаки: {len(numerical_columns)}")

features_encoded = features.copy()
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    features_encoded[col] = le.fit_transform(features[col].astype(str))
    label_encoders[col] = le

target_binary = target.apply(lambda x: 0 if str(x).strip().replace('.', '') == 'normal' else 1)

print(f"\nРаспределение целевой переменной:")
normal_count = (target_binary == 0).sum()
attack_count = (target_binary == 1).sum()
total_count = len(target_binary)
print(f"Нормальные подключения: {normal_count} ({normal_count / total_count * 100:.1f}%)")
print(f"Атаки: {attack_count} ({attack_count / total_count * 100:.1f}%)")

if normal_count == 0:
    print("ВНИМАНИЕ: В данных отсутствуют нормальные подключения! Добавляем синтетические данные...")
    synthetic_normal = features_encoded.iloc[:1000].copy()
    for col in numerical_columns:
        synthetic_normal[col] = np.random.normal(synthetic_normal[col].mean(), synthetic_normal[col].std(), 1000)
    synthetic_target = pd.Series([0] * 1000)

    features_encoded = pd.concat([features_encoded, synthetic_normal], ignore_index=True)
    target_binary = pd.concat([target_binary, synthetic_target], ignore_index=True)

    normal_count = (target_binary == 0).sum()
    attack_count = (target_binary == 1).sum()
    total_count = len(target_binary)
    print(f"Новое распределение:")
    print(f"Нормальные подключения: {normal_count} ({normal_count / total_count * 100:.1f}%)")
    print(f"Атаки: {attack_count} ({attack_count / total_count * 100:.1f}%)")

scaler = StandardScaler()
features_scaled = features_encoded.copy()
features_scaled[numerical_columns] = scaler.fit_transform(features_encoded[numerical_columns])

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target_binary, test_size=0.3, random_state=42, stratify=target_binary
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(random_state=42, n_estimators=100, verbose=False)
}

results = {}

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

for name, model in models.items():
    print(f"\nОбучение {name}...")
    start_time = time.time()

    try:
        if name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        recall_attack = recall_score(y_test, y_pred, pos_label=1)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'training_time': training_time,
            'recall_attack': recall_attack,
            'accuracy': accuracy,
            'model': model
        }

        print(f"Время обучения: {training_time:.2f} сек")
        print(f"Recall для класса 'атака': {recall_attack:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Ошибка при обучении {name}: {e}")
        results[name] = {
            'training_time': None,
            'recall_attack': None,
            'accuracy': None,
            'model': None
        }

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Training Time (s)': [results[name]['training_time'] for name in results],
    'Recall Attack': [results[name]['recall_attack'] for name in results],
    'Accuracy': [results[name]['accuracy'] for name in results]
}).sort_values('Recall Attack', ascending=False)

print(results_df)

print("\n" + "=" * 60)
print("ДЕТАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"Лучшая модель: {best_model_name}")
print(f"Recall для атак: {results[best_model_name]['recall_attack']:.4f}")

y_pred_best = best_model.predict(X_test)
print(f"\nОтчет по классификации для {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=['normal', 'attack']))

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

models_names = list(results.keys())
recall_scores = [results[name]['recall_attack'] if results[name]['recall_attack'] is not None else 0 for name in
                 models_names]
training_times = [results[name]['training_time'] if results[name]['training_time'] is not None else 0 for name in
                  models_names]

bars1 = ax1.bar(models_names, recall_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
ax1.set_title('Recall для класса "атака" по моделям')
ax1.set_ylabel('Recall Score')
ax1.set_ylim(0, 1)
for bar, score in zip(bars1, recall_scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.3f}',
             ha='center', va='bottom')

bars2 = ax2.bar(models_names, training_times, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
ax2.set_title('Время обучения моделей')
ax2.set_ylabel('Время (секунды)')
for bar, time_val in zip(bars2, training_times):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{time_val:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()