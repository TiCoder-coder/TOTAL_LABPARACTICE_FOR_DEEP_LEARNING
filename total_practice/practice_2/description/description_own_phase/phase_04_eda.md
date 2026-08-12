# Phase 4 — Exploratory Data Analysis

## Notebook location

- Notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb)
- Main cells: **Cells 7–9**
- Visualization helper: [visualize.py](../../processing_own_phase/visualize.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. EDA objective

EDA follows the same presentation sequence as Practice 1, adapted to RGB CIFAR-10 and limited to checks needed before modeling:

- verify the raw `uint8` data contract and label range;
- confirm that all ten classes are present;
- inspect class distribution after the Train/Validation split;
- measure training-only RGB intensity, image brightness, and contrast;
- audit constant images, exact duplicates, and conflicting labels;
- view representative images from each class;
- inspect class-average images and brightness extremes;
- visualize a deterministic, class-balanced PCA/t-SNE sample;
- recognize low resolution, complex backgrounds, and within-class variation;
- visually verify that displayed tensors and transformations preserve meaningful content.

Pixel-level EDA reads only the 45,000-image Train subset. Validation and Test pixels are not inspected. The official Test class distribution may be shown as a fixed dataset-integrity property, but Test examples, errors, and probabilities are excluded from this phase.

## 2. Raw-data contract

Cell 8 reads raw CIFAR-10 arrays through the Train subset indices and asserts:

- shape `(45000, 32, 32, 3)`;
- dtype `uint8`;
- pixel range `[0, 255]`;
- label range `[0, 9]`;
- all ten classes present;
- Train and Validation indices are disjoint.

## 3. Class distribution

The class-distribution table reports counts for Train, Validation, and Test. Because Train and Validation are created through a seeded random split, their per-class counts are near, but not guaranteed to be exactly, 90% and 10% of each class.

The official Test set contains exactly 1,000 images per class. This balance is a property of CIFAR-10 rather than a sampling result produced by the notebook.

Artifact:

- [class_distribution.png](../../reports/class_distribution.png)

### Interpretation

- Train contains approximately 4,500 images per class.
- Validation contains approximately 500 images per class.
- Test contains exactly 1,000 images per class.
- No class is absent or severely underrepresented.

Conclusion: class weighting or resampling is not required solely because of imbalance.

## 4. RGB intensity, brightness, and contrast

The Train-only channel statistics are:

| Channel | Mean | Standard deviation |
|---|---:|---:|
| Red | 0.491655 | 0.247081 |
| Green | 0.482333 | 0.243509 |
| Blue | 0.446679 | 0.261581 |

The combined visualization uses a seeded 5,000-image sample for RGB histograms and 2,000 of those images for the brightness/contrast scatter. This preserves deterministic presentation without materializing an unnecessarily large plot.

Artifact:

- [eda_rgb_brightness_contrast.png](../../reports/eda_rgb_brightness_contrast.png)

## 5. Data-quality and duplicate audit

The 45,000-image Train subset contains:

- zero invalid labels or missing classes;
- zero constant images;
- zero exact-duplicate groups;
- zero conflicting-label duplicate groups.

Exact duplicates are detected from the raw image bytes. This audit does not use augmented or resized tensors, so random transforms cannot create false duplicate evidence.

## 6. Sample-image grids, class means, and extremes

The notebook displays CIFAR-10 examples to show that:

- original resolution is only `32 × 32`;
- some classes have similar shapes, especially `cat`/`dog` and `automobile`/`truck`;
- objects can be small, off-center, partially occluded, or surrounded by distracting backgrounds;
- pose, color, scale, and background vary substantially within one class.

Artifacts:

- [data_samples.png](../../reports/data_samples.png)
- [class_examples.png](../../reports/class_examples.png)
- [eda_class_mean_images.png](../../reports/eda_class_mean_images.png)
- [eda_brightness_extremes.png](../../reports/eda_brightness_extremes.png)

These are **input or sample grids**, which are different from the prediction grid in Phase 11. Input grids describe the dataset before modeling; prediction grids describe model behavior after final evaluation.

## 7. PCA and t-SNE projection

The projection uses exactly 100 Train images from each class, selected with seed 42. Images are downsampled to `16 × 16`, standardized, reduced to 50 dimensions with randomized PCA, and then embedded in two dimensions with t-SNE.

Artifact:

- [eda_pca_tsne.png](../../reports/eda_pca_tsne.png)

PCA/t-SNE overlap is descriptive evidence that raw pixels do not linearly separate CIFAR-10 semantics. It is not used as a model-selection score.

## 8. Transform inspection

When before/after examples are displayed, the purpose is to verify that:

- resizing preserves the three RGB channels;
- crop, horizontal flip, and color jitter retain class semantics;
- images are correctly denormalized before human-readable visualization;
- no transform creates obviously corrupted samples.

The transform inspection is displayed directly in the notebook and is not stored as a separate canonical PNG. Leakage prevention is established by code order, object identity, and tests rather than by a visualization alone.

## 9. Technical observations

1. CIFAR-10 is balanced, so both Accuracy and Macro F1 are appropriate.
2. Low resolution makes fine-grained animal discrimination difficult.
3. Visually similar classes are expected to create larger off-diagonal confusion values.
4. Mild augmentation is reasonable for Train because it increases input variation without changing labels.
5. Transfer learning requires resizing to the input resolution expected by ResNet18.

These observations motivate preprocessing and evaluation choices but do not authorize Test-based hyperparameter tuning.

## 10. What EDA does not prove

- Class balance does not prove that the model performs well.
- Dataset statistics do not replace model-evaluation metrics.
- A few visually acceptable images do not prove every transformation is safe.
- Inspecting Test examples must not be used to choose augmentation or fine-tuning strategy.

Model quality is established later through multi-epoch Validation histories and final Test artifacts.

## 11. Suggested presentation script

> EDA uses only the 45,000 Train images for pixel-level analysis. It confirms the raw RGB contract, balanced classes, healthy labels, and no exact duplicates. Channel intensity, brightness, class means, and PCA/t-SNE show substantial visual overlap at 32-by-32 resolution. These findings motivate augmentation and transfer learning, while Validation remains the only source for model selection.

## 12. Transition to the next phase

[Phase 5](phase_05_preprocessing.md) documents the exact Train and evaluation transforms and the evidence that preprocessing does not leak information across splits.
