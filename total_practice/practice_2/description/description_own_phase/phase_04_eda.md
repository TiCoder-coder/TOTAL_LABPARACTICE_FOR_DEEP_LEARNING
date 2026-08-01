# Phase 4 — Exploratory Data Analysis

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 7–9**
- Visualization helper: [visualize.py](../../processing_own_phase/visualize.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. EDA objective

EDA is deliberately limited to checks needed before modeling:

- confirm that all ten classes are present;
- inspect class distribution after the Train/Validation split;
- view representative images from each class;
- recognize low resolution, complex backgrounds, and within-class variation;
- visually verify that displayed tensors and transformations preserve meaningful content.

EDA must not become a route for Test-driven design. The official Test class distribution may be shown as a dataset-integrity property, but Test errors or labels are not used to choose the model in this phase.

## 2. Class distribution

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

## 3. Sample-image grids

The notebook displays CIFAR-10 examples to show that:

- original resolution is only `32 × 32`;
- some classes have similar shapes, especially `cat`/`dog` and `automobile`/`truck`;
- objects can be small, off-center, partially occluded, or surrounded by distracting backgrounds;
- pose, color, scale, and background vary substantially within one class.

Artifacts:

- [data_samples.png](../../reports/data_samples.png)
- [class_examples.png](../../reports/class_examples.png)

These are **input or sample grids**, which are different from the prediction grid in Phase 11. Input grids describe the dataset before modeling; prediction grids describe model behavior after final evaluation.

## 4. Transform inspection

When before/after examples are displayed, the purpose is to verify that:

- resizing preserves the three RGB channels;
- crop, horizontal flip, and color jitter retain class semantics;
- images are correctly denormalized before human-readable visualization;
- no transform creates obviously corrupted samples.

The transform inspection is displayed directly in the notebook and is not stored as a separate canonical PNG. Leakage prevention is established by code order, object identity, and tests rather than by a visualization alone.

## 5. Technical observations

1. CIFAR-10 is balanced, so both Accuracy and Macro F1 are appropriate.
2. Low resolution makes fine-grained animal discrimination difficult.
3. Visually similar classes are expected to create larger off-diagonal confusion values.
4. Mild augmentation is reasonable for Train because it increases input variation without changing labels.
5. Transfer learning requires resizing to the input resolution expected by ResNet18.

These observations motivate preprocessing and evaluation choices but do not authorize Test-based hyperparameter tuning.

## 6. What EDA does not prove

- Class balance does not prove that the model performs well.
- Dataset statistics do not replace model-evaluation metrics.
- A few visually acceptable images do not prove every transformation is safe.
- Inspecting Test examples must not be used to choose augmentation or fine-tuning strategy.

Model quality is established later through multi-epoch Validation histories and final Test artifacts.

## 7. Suggested presentation script

> EDA confirms that every split remains balanced and no class is missing. CIFAR-10 images are very small, and several class pairs such as cat and dog are visually similar. Therefore, I use mild Train augmentation and report macro metrics and confusion matrices in addition to overall Accuracy.

## 8. Transition to the next phase

[Phase 5](phase_05_preprocessing.md) documents the exact Train and evaluation transforms and the evidence that preprocessing does not leak information across splits.
