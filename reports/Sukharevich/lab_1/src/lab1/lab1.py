import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import perf_counter

'''
===========================================Загрузка данных из датасета==============================================
'''
fn = "seeds_dataset.txt"

if os.path.exists(fn):
    try:
        ''' Датасет Seeds состоит из 7 признаков, характеризующих свойства пшеницы и 1 стлобца для визуализации
            Делим столбцы с помощью pandas. fn - файл, sep=r"\s+" - регулярное выражение для разделителя по табу,
            header=None - нет строки заголовков, engine=python - движок для чтения, умеет работать с regExp'''
        data = pd.read_csv(fn, sep=r"\s+", header=None, engine="python")
    except Exception:
        try:
            data = pd.read_csv(fn, header=None)
        except Exception:
            pass

cols = ["area", "perimeter", "compactness", "length of kernel",
        "width of kernel", "asymmetry coefficient", "length of kernel groove", "class"]
data.columns = cols[:data.shape[1]]

print("Seeds dataset preview:")
print(data.head(15))

'''
==========================================Разделение Таблицы=======================================================
'''
''' 
    iloc - индексирование по номерам строк и столбцов. Берёт все строки(:) и все столбцы, кроме последнего(:-1)
    .values превращает это в numpy-массив
'''
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype(int).values


''' 
=====================Масштабирование данных через объект StandardScaler и его медот для масштабирования============
'''
# Standardize features (mean=0, var=1) — classic PCA preprocessing
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

'''
=====================================Реализация PCA вручную ч-з numpy.linalg.eig==================================
'''

''' 
    Ставит отметку времени до начала вычислений (высокоточный таймер). Позже manual_time = t1 - t0 даствремя выполнения блока.
'''
t0 = perf_counter()

''' Ковариационная матрица - для данных с p признаками ковариационная матрица — это квадратная матрица размера p × p, 
    где элемент в строке i, столбце j равен ковариации между признаком i и признаком j.
    Интуиция:
    - На диагонали — дисперсии каждого признака.
    - Вне диагонали — насколько два признака изменяются совместно (коррелируют).
    - Ковариационная матрица показывает структуру взаимосвязей и направление «наибольшей вариации» в данных.
    
    rowvar=False т.к. по умолчанию np.cov ожидает, что каждая строка — переменная, а столбцы — наблюдения.
    У нас обычная форма X_std — строки = наблюдения, столбцы = признаки.
'''
cov_mat = np.cov(X_std, rowvar=False)

''' 
    Вычисляет собственные значения и собственные векторы ковариационной матрицы. PCA ищет ортогональные направления 
    (главные компоненты), вдоль которых дисперсия данных максимально. И именно собственные векторы ковариационной 
    матрицы дают эти направления; собственные значения дают «величину» (сколько дисперсии объясняет соответствующая 
    компонента).
    Так как cov_mat — симметричная матрица, лучше использовать np.linalg.eigh (специализированная для эрмитовых/
    симметричных матриц) — она численно устойчивее и гарантирует вещественные собственные значения.
'''
eigvals, eigvecs = np.linalg.eig(cov_mat)

''' 
    Сортировка. np.argsort(eigvals) — индексы, сортирующие массив собственных значений по возрастанию. [::-1] — разворачиваем,
    чтобы получить убывающий порядок.
    После сортировки первые элементы — крупнейшие собственные значения (их направления объясняют наибольшую долю дисперсии).
    .real — избавляемся от возможных чисто численных мнимых частей, которые иногда появляются при np.linalg.eig.
     
    eigvals_sorted — вектор длины p (одна собственная величина на компоненту).
    eigvecs_sorted — матрица p × p, где каждый столбец — собственный вектор, соответствующий eigvals_sorted[i].   
'''
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx].real
eigvecs_sorted = eigvecs[:, idx].real

'''
    Проекции 
    eigvecs_sorted[:, :2] — матрица p × 2 из первых двух собственных векторов (первые 2 главные компоненты).
    Умножением X_std @ W (где W — матрица компонент) мы проецируем каждое наблюдение на новое базисное пространство. 
    Если X_std имеет форму (n, p), то Z2_manual будет (n, 2) — каждая строка = координаты одного образца в 
    пространстве первых двух ПК.
'''
Z2_manual = X_std @ eigvecs_sorted[:, :2]
Z3_manual = X_std @ eigvecs_sorted[:, :3]

'''
    Фиксируем время окончания и вычисляем продолжительность выполнения блока (в секундах). Удобно для сравнения
    времени с sklearn.PCA или профилирования.
'''
t1 = perf_counter()
manual_time = t1 - t0

'''
=========================================Реализация PCA ч-з sklearn===============================================
'''

''' Засекаем время начала выполнения (для сравнения с ручным методом через eig). '''
t0 = perf_counter()

'''
    Создаём объект метода главных компонент (PCA) из библиотеки scikit-learn.
    n_components=2 → мы хотим оставить только две главные компоненты (для 2D-проекции).
'''
pca2 = PCA(n_components=2)

''' 
    fit → находит собственные значения/векторы ковариационной матрицы (т.е. подбирает главные компоненты).
    transform → проецирует данные X_std на первые 2 главные компоненты.
    Z2_sklearn — это новая матрица данных формы (N, 2), где N = число образцов.
    Каждая строка теперь описана не 7 признаками, а 2 новыми координатами
'''
Z2_sklearn = pca2.fit_transform(X_std)

''' То же самое, но теперь берём три главные компоненты. '''
pca3 = PCA(n_components=3)
''' Получаем матрицу (N, 3), где каждая строка — это точка в новом 3D-пространстве. '''
Z3_sklearn = pca3.fit_transform(X_std)

''' Засекаем время окончания. '''
t1 = perf_counter()
''' Вычисляем время выполнения метода PCA через sklearn. '''
sk_time = t1 - t0

'''
==================================================Визуализация===================================================
'''
def scatter_by_class_2d(Z, y, title, fname):
    plt.figure()
    classes = np.unique(y)
    markers = ['o', '^', 's', 'D', 'P', 'X']
    for i, c in enumerate(classes):
        mask = y == c
        plt.scatter(Z[mask, 0], Z[mask, 1], marker=markers[i % len(markers)], label=f"class {c}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

def scatter_by_class_3d(Z, y, title, fname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    classes = np.unique(y)
    markers = ['o', '^', 's', 'D', 'P', 'X']
    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], marker=markers[i % len(markers)], label=f"class {c}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

scatter_by_class_2d(Z2_manual, y, "PCA (manual eig) — first 2 PCs", "data/pca_manual_2d.png")
scatter_by_class_2d(Z2_sklearn, y, "PCA (sklearn) — first 2 PCs", "data/pca_sklearn_2d.png")

scatter_by_class_3d(Z3_manual, y, "PCA (manual eig) — first 3 PCs", "data/pca_manual_3d.png")
scatter_by_class_3d(Z3_sklearn, y, "PCA (sklearn) — first 3 PCs", "data/pca_sklearn_3d.png")

'''
=========================================Вычисление потерь========================================================
'''
'''
    Доля объяснённой дисперсии
    eigvals_sorted — собственные значения ковариационной матрицы, отсортированные по убыванию.
    Каждое собственное значение = доля дисперсии, которую объясняет соответствующая главная компонента
    total_var — сумма всех собственных значений (это общая дисперсия в данных).
    explained_var_ratio — массив, показывающий, какую долю дисперсии объясняет каждая компонента.
'''
total_var = eigvals_sorted.sum().real
explained_var_ratio = eigvals_sorted / total_var

''' 
    Накопленная доля объяснённой дисперсии
    cum_exp_2 — сколько дисперсии сохраняется, если оставить 2 главные компоненты.
    cum_exp_3 — то же самое для 3 компонент.
    loss_var_2, loss_var_3 — какая часть информации теряется (остаточная дисперсия, которую "выкинули").
'''
cum_exp_2 = explained_var_ratio[:2].sum().real
cum_exp_3 = explained_var_ratio[:3].sum().real
loss_var_2 = 1.0 - cum_exp_2
loss_var_3 = 1.0 - cum_exp_3

'''
    Восстановление данных из проекции
    W2, W3 — матрицы из первых 2 и 3 собственных векторов (главных компонент).
    Z2_manual и Z3_manual — проекции исходных данных на 2D и 3D пространство.
    Умножение Z @ W.T возвращает данные обратно в исходное пространство признаков (приближённо, т.к. мы выкинули часть компонент).
    X_rec_2 — восстановленные данные после PCA с 2 компонентами.
    X_rec_3 — восстановленные данные после PCA с 3 компонентами.       
'''
W2 = eigvecs_sorted[:, :2]
W3 = eigvecs_sorted[:, :3]

X_rec_2 = (Z2_manual @ W2.T)
X_rec_3 = (Z3_manual @ W3.T)

'''
    Ошибка восстановления
    Считаем среднеквадратичную ошибку (MSE) между исходными нормализованными данными (X_std) и восстановленными (X_rec_2, X_rec_3).
    Чем меньше MSE → тем лучше PCA с данным числом компонент сохраняет структуру данных    
'''
mse_rec_2 = np.mean((X_std - X_rec_2)**2)
mse_rec_3 = np.mean((X_std - X_rec_3)**2)

'''
==============================================Автоотчёт============================================================
'''
report = f"""# PCA Report — Seeds Dataset (Variant 6)

**Data**: Seeds (features: 7, classes: last column).  
**Preprocessing**: Standardization (z-score).

## Two independent PCA implementations
1. **Manual (NumPy)**: covariance → `numpy.linalg.eig` → sort → project.  
   Runtime: {manual_time:.6f} s.
2. **sklearn.decomposition.PCA**: direct fit/transform (2D & 3D).  
   Runtime: {sk_time:.6f} s.

## Explained variance (manual eig)
- Eigenvalues (descending): {np.array2string(eigvals_sorted, precision=4)}  
- Explained variance ratio: {np.array2string(explained_var_ratio, precision=4)}
- Cumulative (2 PCs): **{cum_exp_2:.4f}** → variance *retained*  
  ⇒ Variance *lost* (2 PCs): **{loss_var_2:.4f}**
- Cumulative (3 PCs): **{cum_exp_3:.4f}** → variance *retained*  
  ⇒ Variance *lost* (3 PCs): **{loss_var_3:.4f}**

## Reconstruction loss (MSE in standardized space)
- Using 2 PCs: **{mse_rec_2:.6f}**
- Using 3 PCs: **{mse_rec_3:.6f}**

## Figures
- 2D (manual): `pca_manual_2d.png`
- 2D (sklearn): `pca_sklearn_2d.png`
- 3D (manual): `pca_manual_3d.png`
- 3D (sklearn): `pca_sklearn_3d.png`

## Notes
- If only class **1** samples are present (as in the minimal demo data), the class-wise separation plots will naturally show a single marker.  
  For full analysis across the 3 classes (*Kama*, *Rosa*, *Canadian*), place the full dataset file as 
  `/mnt/data/seeds dataset.txt` (whitespace-separated, 8 columns) and rerun this notebook cell.
- The manual and sklearn projections should be identical up to possible sign flips per component (a standard PCA ambiguity).

## How to run locally
- Ensure Python with `numpy`, `pandas`, `matplotlib`, `scikit-learn` is installed.
- Place the data file (8 columns, last is class label) and update the `possible_filenames` list if needed.
- Execute the script or notebook to regenerate figures and statistics.
"""

report_path = "data/pca_seeds_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

summary = {
    "manual_time_s": manual_time,
    "sklearn_time_s": sk_time,
    "cum_explained_2": cum_exp_2,
    "cum_explained_3": cum_exp_3,
    "loss_variance_2": loss_var_2,
    "loss_variance_3": loss_var_3,
    "mse_reconstruction_2": mse_rec_2,
    "mse_reconstruction_3": mse_rec_3,
    "report_path": report_path,
    "fig_2d_manual": "/lab1/data/pca_manual_2d.png",
    "fig_2d_sklearn": "/lab1/data/pca_sklearn_2d.png",
    "fig_3d_manual": "/lab1/data/pca_manual_3d.png",
    "fig_3d_sklearn": "/lab1/data/pca_sklearn_3d.png",
}

summary
