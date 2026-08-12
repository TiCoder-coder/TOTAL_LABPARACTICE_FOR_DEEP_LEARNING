# Phase 4 — Exploratory Data Analysis Results

[Phase 3 results](phase_03_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 5 results](phase_05_results.md)

| Output | Description | Artifact |
|---|---|---|
| Cell 8 | Raw Train contract: 45,000 RGB images, `uint8`, shape `32 × 32 × 3`, range `[0, 255]` | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 8 | Class-count table for Train, Validation, and Test | [class_distribution.png](../../reports/class_distribution.png) |
| Cell 8 | Train-only RGB intensity and brightness/contrast analysis | [eda_rgb_brightness_contrast.png](../../reports/eda_rgb_brightness_contrast.png) |
| Cell 8 | Quality audit: zero invalid labels, constant images, exact duplicates, and conflicting-label duplicates | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 9 | Representative CIFAR-10 image grid and examples from every class | [data_samples.png](../../reports/data_samples.png) |
| Cell 9 | Training-subset class-average images | [eda_class_mean_images.png](../../reports/eda_class_mean_images.png) |
| Cell 9 | Darkest and brightest Train examples | [eda_brightness_extremes.png](../../reports/eda_brightness_extremes.png) |
| Cell 9 | Seeded, class-balanced PCA/t-SNE projection | [eda_pca_tsne.png](../../reports/eda_pca_tsne.png) |

Train-only RGB means are `0.491655`, `0.482333`, and `0.446679`. The class table confirms approximate 90/10 class balance in Train/Validation and exactly 1,000 Test images per class. EDA remains descriptive and does not provide a model-performance or selection conclusion.
