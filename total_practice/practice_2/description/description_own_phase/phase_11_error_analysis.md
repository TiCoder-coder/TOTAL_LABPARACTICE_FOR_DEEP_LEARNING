# Phase 11 — Error Analysis and Visualization

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 24–28**
- Source: [final_evaluate.py](../../processing_own_phase/final_evaluate.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

Final metrics describe how well the model performs; error analysis explains **where and how it fails**. This phase is descriptive and must not be used to reselect the current model after Test has been opened. Insights can guide a future experiment, but that new experiment must return to Validation-only development.

## 2. Confusion analysis

The notebook reads the canonical confusion matrix and identifies the largest off-diagonal entries. Prominent confusion directions include:

- `cat → dog`;
- `dog → cat`;
- `bird → cat`;
- `airplane → ship`;
- `horse → dog`.

These observations come from the official artifact rather than assumptions made before training. Cat/dog confusion is plausible because CIFAR-10 images are very small and can lack the detail needed to distinguish animal faces, fur, pose, and background.

Artifacts:

- [confusion_matrix_raw.png](../../reports/confusion_matrix_raw.png)
- [confusion_matrix_normalized.png](../../reports/confusion_matrix_normalized.png)

## 3. Confidence distribution

- [confidence_distribution.png](../../reports/confidence_distribution.png)

The confidence visualization separates or compares correct and incorrect predictions.

Interpretation:

- **Correct with high confidence:** the model is decisive and correct.
- **Correct with low confidence:** the sample is difficult, but the selected class is still correct.
- **Incorrect with low confidence:** uncertainty is understandable and visible.
- **Incorrect with high confidence:** the model is overconfident, which is the more serious failure mode.

High confidence does not prove calibration. A formal calibration study would require a reliability diagram, Expected Calibration Error, or a similar metric evaluated under a declared protocol.

## 4. Correct and incorrect galleries

- [prediction_gallery_correct.png](../../reports/prediction_gallery_correct.png)
- [prediction_gallery_incorrect.png](../../reports/prediction_gallery_incorrect.png)

Every tile should show:

- the denormalized image;
- true label;
- predicted label;
- model confidence.

The correct gallery confirms that the model handles representative examples. The incorrect gallery exposes class similarity, small objects, occlusion, unusual viewpoints, and possible background bias.

## 5. Mixed prediction grid

- [prediction_grid_mixed.png](../../reports/prediction_grid_mixed.png)

The mixed grid intentionally combines:

- **challenging correct predictions:** correct results with relatively low confidence;
- **confident mistakes:** incorrect results with high confidence.

This layout is more informative for presentation than a grid containing only correct or only incorrect predictions because it places uncertainty and overconfidence side by side. Every tile includes sufficient annotation to be understood without looking up the CSV manually.

The mixed prediction grid differs from the input grid in Phase 4:

- the input grid describes the dataset before modeling;
- the prediction grid describes model behavior after final evaluation.

## 6. Generalization comparison

- [validation_test_comparison_current.png](../../reports/validation_test_comparison_current.png)

This plot uses the selected checkpoint's verified Validation Accuracy and the current official Test Accuracy. It replaces older comparison images that may belong to a one-epoch quick run. The 0.77-percentage-point decrease is small and does not indicate severe generalization collapse.

## 7. Diagnostic hypotheses

Artifact evidence supports several hypotheses:

1. Animal classes are more difficult than several vehicle classes because shape and texture overlap.
2. The original `32 × 32` resolution limits fine-grained recognition.
3. Partial fine-tuning adapts high-level representations more effectively than head-only training.
4. High-confidence mistakes suggest that raw softmax scores are not necessarily calibrated probabilities.

These are diagnostic hypotheses rather than causal conclusions. Confirmation requires new controlled experiments chosen by Validation, not repeated optimization against the current Test set.

## 8. Valid directions for a new research cycle

- Run multiple seeds and report mean ± standard deviation.
- Increase the epoch budget under a fixed protocol.
- Evaluate augmentation factors one at a time.
- Study calibration using Validation.
- Add Grad-CAM to inspect whether the model attends to objects or backgrounds.
- Define an explicit BatchNorm policy for frozen blocks.

Because Test has already been opened for final reporting, repeatedly modifying the model based on these Test errors and reevaluating on the same Test set would gradually turn Test into implicit Validation.

## 9. Visualization coverage

The notebook already includes:

- class-distribution and sample-image grids;
- learning curves and learning-rate history;
- controlled experiment comparison;
- Validation–Test comparison;
- raw and normalized confusion matrices;
- confidence distribution;
- correct and incorrect galleries;
- a mixed prediction grid.

Grad-CAM remains an optional interpretability extension, not a missing requirement for the instructor's requested model metrics.

## 10. Suggested presentation script

> Beyond Accuracy, I analyze class-level errors and confidence. The model frequently confuses cat and dog and has several difficult animal-class pairs. The mixed grid shows both correct low-confidence predictions and incorrect high-confidence predictions, demonstrating that strong overall Accuracy does not guarantee trustworthy confidence for every sample.

## 11. Transition to the next phase

[Phase 12](phase_12_reproducibility_conclusion.md) consolidates reproducibility evidence, limitations, submission checks, and the final conclusion.
