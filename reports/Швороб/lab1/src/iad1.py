import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)


def load_and_preprocess_data():
    try:
        df = pd.read_csv('hcvdat0.csv')

        print("Информация о данных:")
        print(df.info())
        print("\nПервые 5 строк:")
        print(df.head())

        print(f"\nКолонки в файле: {df.columns.tolist()}")
        print(f"Размерность данных: {df.shape}")

        print("\nПропущенные значения:")
        print(df.isnull().sum())

        categories = df['Category'].copy()

        columns_to_drop = ['Category', 'Unnamed: 0']
        df_for_pca = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        print(f"\nПризнаки для PCA: {df_for_pca.columns.tolist()}")

        if 'Sex' in df_for_pca.columns:
            print("Кодируем переменную 'Sex'...")
            df_for_pca = pd.get_dummies(df_for_pca, columns=['Sex'], drop_first=True)

        print("Замена пропущенных значений...")
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df_for_pca),
                                  columns=df_for_pca.columns)

        print("Стандартизация данных...")
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_imputed)

        print(f"Размерность после предобработки: {df_scaled.shape}")

        return df_scaled, categories, df_imputed.columns

    except FileNotFoundError:
        print("Ошибка: Файл hcvdat0.csv не найден в текущей директории!")
        print("Убедитесь, что файл находится в той же папке, что и скрипт Python.")
        return None, None, None
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None, None


def manual_pca(X, n_components=3):
    X_centered = X

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    components = eigenvectors_sorted[:, :n_components]
    X_pca = X_centered @ components

    return X_pca, eigenvalues_sorted, eigenvectors_sorted


def sklearn_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X)
    all_eigenvalues = pca_full.explained_variance_

    return X_pca, all_eigenvalues, pca.components_


def plot_pca_results(X_manual_2d, X_sklearn_2d, X_manual_3d, X_sklearn_3d, categories, eigenvalues_manual,
                     eigenvalues_sklearn):
    unique_categories = categories.unique()[:8]  # Ограничиваем количество цветов
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, category in enumerate(unique_categories):
        mask = categories == category
        if mask.sum() > 0:  # Проверяем, что есть точки этой категории
            axes[0, 0].scatter(X_manual_2d[mask, 0], X_manual_2d[mask, 1],
                               c=[colors[i]], label=str(category), alpha=0.7, s=50)
    axes[0, 0].set_title('PCA (ручной метод) - 2 компоненты')
    axes[0, 0].set_xlabel('Главная компонента 1')
    axes[0, 0].set_ylabel('Главная компонента 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    for i, category in enumerate(unique_categories):
        mask = categories == category
        if mask.sum() > 0:
            axes[0, 1].scatter(X_sklearn_2d[mask, 0], X_sklearn_2d[mask, 1],
                               c=[colors[i]], label=str(category), alpha=0.7, s=50)
    axes[0, 1].set_title('PCA (sklearn) - 2 компоненты')
    axes[0, 1].set_xlabel('Главная компонента 1')
    axes[0, 1].set_ylabel('Главная компонента 2')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)

    ax1 = fig.add_subplot(2, 2, 3, projection='3d')
    for i, category in enumerate(unique_categories):
        mask = categories == category
        if mask.sum() > 0:
            ax1.scatter(X_manual_3d[mask, 0], X_manual_3d[mask, 1], X_manual_3d[mask, 2],
                        c=[colors[i]], label=str(category), alpha=0.7, s=50)
    ax1.set_title('PCA (ручной метод) - 3 компоненты')
    ax1.set_xlabel('Главная компонента 1')
    ax1.set_ylabel('Главная компонента 2')
    ax1.set_zlabel('Главная компонента 3')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    for i, category in enumerate(unique_categories):
        mask = categories == category
        if mask.sum() > 0:
            ax2.scatter(X_sklearn_3d[mask, 0], X_sklearn_3d[mask, 1], X_sklearn_3d[mask, 2],
                        c=[colors[i]], label=str(category), alpha=0.7, s=50)
    ax2.set_title('PCA (sklearn) - 3 компоненты')
    ax2.set_xlabel('Главная компонента 1')
    ax2.set_ylabel('Главная компонента 2')
    ax2.set_zlabel('Главная компонента 3')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    plot_explained_variance(eigenvalues_manual, eigenvalues_sklearn)


def plot_explained_variance(eigenvalues_manual, eigenvalues_sklearn):
    eigenvalues_manual = np.real(eigenvalues_manual)
    eigenvalues_sklearn = np.real(eigenvalues_sklearn)

    explained_variance_manual = eigenvalues_manual / np.sum(eigenvalues_manual)
    explained_variance_sklearn = eigenvalues_sklearn / np.sum(eigenvalues_sklearn)

    cumulative_variance_manual = np.cumsum(explained_variance_manual)
    cumulative_variance_sklearn = np.cumsum(explained_variance_sklearn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    n_components_show = min(10, len(explained_variance_manual), len(explained_variance_sklearn))

    components_range = range(1, n_components_show + 1)
    ax1.bar(components_range, explained_variance_manual[:n_components_show], alpha=0.6, label='Объясненная дисперсия')
    ax1.plot(components_range, cumulative_variance_manual[:n_components_show], 'r-', marker='o',
             label='Накопленная дисперсия')
    ax1.set_title('Объясненная дисперсия (ручной метод)')
    ax1.set_xlabel('Главные компоненты')
    ax1.set_ylabel('Доля объясненной дисперсии')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(components_range, explained_variance_sklearn[:n_components_show], alpha=0.6, label='Объясненная дисперсия')
    ax2.plot(components_range, cumulative_variance_sklearn[:n_components_show], 'r-', marker='o',
             label='Накопленная дисперсия')
    ax2.set_title('Объясненная дисперсия (sklearn)')
    ax2.set_xlabel('Главные компоненты')
    ax2.set_ylabel('Доля объясненной дисперсии')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_information_loss(eigenvalues, n_components_2d=2, n_components_3d=3):
    eigenvalues = np.real(eigenvalues)
    total_variance = np.sum(eigenvalues)

    variance_2d = np.sum(eigenvalues[:n_components_2d])
    loss_2d = 1 - (variance_2d / total_variance)

    variance_3d = np.sum(eigenvalues[:n_components_3d])
    loss_3d = 1 - (variance_3d / total_variance)

    return loss_2d, loss_3d, variance_2d / total_variance, variance_3d / total_variance


def analyze_feature_importance(eigenvectors, feature_names, n_components=3):
    print("\nАнализ важности признаков в главных компонентах:")
    print("=" * 50)

    for i in range(n_components):
        print(f"\nГлавная компонента {i + 1}:")
        component_weights = np.real(eigenvectors[:, i])
        feature_importance = pd.DataFrame({
            'Признак': feature_names,
            'Вес': component_weights,
            'Абсолютный вес': np.abs(component_weights)
        })
        feature_importance = feature_importance.sort_values('Абсолютный вес', ascending=False)

        for _, row in feature_importance.head(5).iterrows():
            print(f"  {row['Признак']}: {row['Вес']:.3f}")


def main():

    print("1. Загрузка и предобработка данных...")
    X, categories, feature_names = load_and_preprocess_data()

    if X is None:
        print("Не удалось загрузить данные. Завершение работы.")
        return

    print(f"Размерность данных после предобработки: {X.shape}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Количество наблюдений: {X.shape[0]}")
    print(f"Уникальные категории: {categories.unique()}")

    print("\n2. Применение PCA...")

    # Ручной метод
    print("2.1 Ручной метод с numpy.linalg.eig...")
    X_manual_2d, eigenvalues_manual, eigenvectors_manual = manual_pca(X, n_components=2)
    X_manual_3d, _, _ = manual_pca(X, n_components=3)

    # Метод sklearn
    print("2.2 Метод с sklearn.decomposition.PCA...")
    X_sklearn_2d, eigenvalues_sklearn, eigenvectors_sklearn = sklearn_pca(X, n_components=2)
    X_sklearn_3d, _, _ = sklearn_pca(X, n_components=3)

    # Визуализация
    print("\n3. Визуализация результатов...")
    plot_pca_results(X_manual_2d, X_sklearn_2d, X_manual_3d, X_sklearn_3d,
                     categories, eigenvalues_manual, eigenvalues_sklearn)

    print("\n4. Расчет потерь информации...")
    loss_2d_manual, loss_3d_manual, var_2d_manual, var_3d_manual = calculate_information_loss(eigenvalues_manual)
    loss_2d_sklearn, loss_3d_sklearn, var_2d_sklearn, var_3d_sklearn = calculate_information_loss(eigenvalues_sklearn)

    print("\nРезультаты анализа потерь информации:")
    print("=" * 50)
    print("Ручной метод:")
    print(f"  - Объясненная дисперсия (2 компоненты): {var_2d_manual:.3f} ({var_2d_manual * 100:.1f}%)")
    print(f"  - Потери информации (2 компоненты): {loss_2d_manual:.3f} ({loss_2d_manual * 100:.1f}%)")
    print(f"  - Объясненная дисперсия (3 компоненты): {var_3d_manual:.3f} ({var_3d_manual * 100:.1f}%)")
    print(f"  - Потери информации (3 компоненты): {loss_3d_manual:.3f} ({loss_3d_manual * 100:.1f}%)")

    print("\nSklearn метод:")
    print(f"  - Объясненная дисперсия (2 компоненты): {var_2d_sklearn:.3f} ({var_2d_sklearn * 100:.1f}%)")
    print(f"  - Потери информации (2 компоненты): {loss_2d_sklearn:.3f} ({loss_2d_sklearn * 100:.1f}%)")
    print(f"  - Объясненная дисперсия (3 компоненты): {var_3d_sklearn:.3f} ({var_3d_sklearn * 100:.1f}%)")
    print(f"  - Потери информации (3 компоненты): {loss_3d_sklearn:.3f} ({loss_3d_sklearn * 100:.1f}%)")

    analyze_feature_importance(eigenvectors_manual, feature_names)

if __name__ == "__main__":
    main()