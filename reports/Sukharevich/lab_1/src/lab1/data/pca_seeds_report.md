# PCA Report — Seeds Dataset (Variant 6)

**Data**: Seeds (features: 7, classes: last column).  
**Preprocessing**: Standardization (z-score).

## Two independent PCA implementations
1. **Manual (NumPy)**: covariance → `numpy.linalg.eig` → sort → project.  
   Runtime: 0.006749 s.
2. **sklearn.decomposition.PCA**: direct fit/transform (2D & 3D).  
   Runtime: 0.002719 s.

## Explained variance (manual eig)
- Eigenvalues (descending): [5.0553e+00 1.2033e+00 6.8125e-01 6.8692e-02 1.8803e-02 5.3576e-03
 8.1628e-04]  
- Explained variance ratio: [7.1874e-01 1.7108e-01 9.6858e-02 9.7664e-03 2.6734e-03 7.6172e-04
 1.1606e-04]
- Cumulative (2 PCs): **0.8898** → variance *retained*  
  ⇒ Variance *lost* (2 PCs): **0.1102**
- Cumulative (3 PCs): **0.9867** → variance *retained*  
  ⇒ Variance *lost* (3 PCs): **0.0133**

## Reconstruction loss (MSE in standardized space)
- Using 2 PCs: **0.110175**
- Using 3 PCs: **0.013318**

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
