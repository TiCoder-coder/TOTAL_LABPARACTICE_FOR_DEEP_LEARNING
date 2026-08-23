# Phase 4 — Exploratory Data Analysis Results

[Phase 3 results](phase_03_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 5 results](phase_05_results.md)

| Notebook section | Description | Artifact |
|---|---|---|
| Dataset Boundary and EDA Contract | Raw Train contract: 45,000 RGB images, `uint8`, shape `32 × 32 × 3`, range `[0, 255]` | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Exact Class Distribution | Exact Train class-count table and chart | [eda_train_class_distribution.png](../../reports/eda_train_class_distribution.png) |
| RGB Pixel-Intensity Distribution and Quantiles | Exact RGB statistics and quantiles | [eda_rgb_brightness_contrast.png](../../reports/eda_rgb_brightness_contrast.png) |
| Image Brightness and Contrast by Class | Per-class brightness and contrast distributions | [eda_per_class_brightness_contrast.png](../../reports/eda_per_class_brightness_contrast.png) |
| Data Quality and Exact-Duplicate Audit | Invalid labels, constant images, exact duplicates and conflicts | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| RGB Spatial-Feature Correlation | Deterministic RGB spatial-feature correlation | [eda_rgb_feature_correlation.png](../../reports/eda_rgb_feature_correlation.png) |
| Raw PCA in Three Dimensions | Raw RGB PCA in three dimensions | [eda_pca_3d.png](../../reports/eda_pca_3d.png) |
| PCA Explained Variance | PCA cumulative explained variance | [eda_pca_explained_variance.png](../../reports/eda_pca_explained_variance.png) |
| RGB Principal-Component Images | First three RGB principal-component images | [eda_rgb_principal_components.png](../../reports/eda_rgb_principal_components.png) |
| Standardized PCA and t-SNE | Standardized PCA and t-SNE | [eda_standardized_pca_tsne.png](../../reports/eda_standardized_pca_tsne.png) |
| Representative Samples and Class Means | Deterministic representative samples and class means | [eda_representative_samples.png](../../reports/eda_representative_samples.png) |
| Statistical Extremes | Brightness and contrast extremes | [eda_brightness_contrast_extremes.png](../../reports/eda_brightness_contrast_extremes.png) |

Train-only RGB means are `0.491655`, `0.482333`, and `0.446679`. The class table confirms approximate 90/10 class balance in Train/Validation and exactly 1,000 Test images per class. EDA remains descriptive and does not provide a model-performance or selection conclusion.

The current notebook result is produced through
`processing_own_phase/eda.py`: analysis is restricted by the saved Train
indices. The approved renderer then produces the complete Practice 1 method
adaptation: exact RGB statistics, per-class distributions, duplicate audit,
feature correlation, raw and standardized PCA, RGB component images, t-SNE,
representative samples, class means and statistical extremes.
