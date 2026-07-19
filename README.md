# PCA vs Kernel PCA

A study of Principal Component Analysis (PCA) versus Kernel PCA (KPCA, Gaussian kernel) for dimensionality reduction, evaluated on both synthetic data and face recognition tasks. KPCA is implemented from scratch (kernel matrix construction, centering, eigendecomposition, out-of-sample projection) and compared with PCA under identical experimental protocols.

## Core files

| File | Description |
|---|---|
| [pca_kpca.ipynb](pca_kpca.ipynb) | PCA vs KPCA on synthetic 2-D data (concentric circles, etc.): visualization of nonlinear projections and 1-NN accuracy vs number of components |
| [pca_kpca_face_recognition.ipynb](pca_kpca_face_recognition.ipynb) | PCA vs KPCA face recognition on Yale Faces and Extended Yale B: accuracy vs number of components, effect of training size, validation-based model selection |
| [KPCA_Report.md](KPCA_Report.md) | Full write-up: method derivation, experimental setup, results and analysis (figures in [figures/](figures/)) |

`archive/` contains earlier working drafts (PCA / LDA / Fisherfaces experiments) and an old report, kept for reference.

## Datasets

The face notebooks read from `Datasets/` using relative paths — keep the directory layout as-is:

```
Datasets/
├── att_faces/        # ATT (ORL) faces, 40 subjects x 10 images, 112x92
├── yalefaces/        # Yale Faces, 15 subjects, ~11 images each
└── yaleB_extended/   # Extended Yale B (cropped), ~38 subjects, varying illumination
```

## Requirements

Python 3.12 (see `.python-version` / `pyproject.toml`). Dependencies:

```
pip install numpy scipy pandas matplotlib pillow opencv-python altair scikit-learn
```

## How to run

1. Place the datasets under `Datasets/` as shown above.
2. Open either notebook in Jupyter / VS Code and run all cells top to bottom:
   - `pca_kpca.ipynb` needs no datasets (synthetic data only).
   - `pca_kpca_face_recognition.ipynb` requires `Datasets/yalefaces/` and `Datasets/yaleB_extended/`.
3. Read `KPCA_Report.md` for the consolidated results and analysis.
