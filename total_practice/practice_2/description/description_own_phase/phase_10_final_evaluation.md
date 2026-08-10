# Phase 10 — Final Test Evaluation

## Notebook location

- Notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb)
- Main cells: **Cells 22–23**
- Source: [final_evaluate.py](../../processing_own_phase/final_evaluate.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Entry conditions

Final Test is permitted only when:

- the experiment has been selected through Validation;
- the selected best checkpoint exists;
- a newly reconstructed model has loaded the checkpoint;
- Validation Verification reports PASS;
- the checkpoint and configuration are locked.

If verification fails, the pipeline raises a clear exception. It must not construct the official Test loader or generate new Test metrics.

## 2. Official Test workflow

```text
Validation Verification PASS
       ↓
Construct the official CIFAR-10 Test loader
       ↓
model.eval() with no-gradient/inference mode
       ↓
Process all 10,000 Test images exactly once
       ↓
Collect loss, predictions, probabilities, and labels
       ↓
Generate metrics and artifacts
       ↓
Run consistency checks
```

Test does not update weights, scheduler state, threshold, strategy, or checkpoint selection.

## 3. Official Final Test metrics

Values are read from [summary.json](../../outputs/summary.json):

| Metric | Value |
|---|---:|
| Test Accuracy | 0.8895 |
| Test Loss | 0.3415387766 |
| Macro Precision | 0.8928359014 |
| Macro Recall | 0.8895 |
| Macro F1 | 0.8899327014 |
| Test Samples | 10,000 |
| Test Evaluation Count | 1 |

Cell 23 reads these values into a DataFrame. It does not declare a `VERIFIED_RESULT` dictionary or any other manual metric structure.

## 4. Result interpretation

- Accuracy of 88.95% means 8,895 of 10,000 Test images are correct.
- Macro F1 is close to Accuracy, indicating relatively balanced performance across the ten classes.
- Macro Precision is slightly higher than Macro Recall, but the difference is small.
- Validation Accuracy is 89.72% and Test Accuracy is 88.95%, producing a generalization gap of approximately **0.77 percentage points**.

Current comparison artifact:

- [validation_test_comparison_current.png](../../reports/validation_test_comparison_current.png)

## 5. Classification report

- [classification_report.csv](../../outputs/classification_report.csv)

This file stores per-class Precision, Recall, F1-score, and support. It reveals strong and weak classes that a single overall Accuracy cannot identify.

## 6. Prediction artifact

- [predictions.csv](../../outputs/predictions.csv)

Each row represents one Test sample and includes the true label, predicted label, confidence, correctness, and ten `probability_<class>` columns. The file must contain exactly 10,000 data rows. The probability rows must be finite, remain within `[0, 1]`, sum to one within tolerance, and agree with both the predicted-label argmax and exported confidence.

## 7. Confusion matrices

Artifacts:

- [confusion_matrix_raw.png](../../reports/confusion_matrix_raw.png)
- [confusion_matrix_normalized.png](../../reports/confusion_matrix_normalized.png)
- [confusion_matrix.csv](../../outputs/confusion_matrix.csv)

### Raw confusion matrix

Each cell is an **image count**. The sum of all cells must equal 10,000. The diagonal contains correct predictions; its sum divided by 10,000 must equal Test Accuracy.

Use the raw matrix to answer count-based questions such as: “How many true cat images were predicted as dog?”

### Normalized confusion matrix

Each row is divided by the number of true samples in that class. For CIFAR-10 Test, each row is based on 1,000 samples. Every row should sum to approximately one, allowing for floating-point rounding.

Use the normalized matrix to compare error rates across classes, especially when supports are unequal in another dataset.

## 8. Artifact-consistency checks

Before accepting the result, the pipeline verifies:

- `confusion_matrix.sum() == test_samples`;
- `trace(confusion_matrix) / test_samples` matches `summary.test_accuracy` within tolerance;
- `len(predictions) == test_samples`;
- prediction-row Accuracy matches the summary;
- all ten probability columns exist and each row forms a valid class distribution;
- probability argmax matches `predicted_label_id` and its selected value matches `confidence`;
- classification-report support totals Test samples;
- `test_evaluation_count == 1`.

Any inconsistency must raise an exception rather than display conflicting metrics.

## 9. Additional final artifacts

- [metrics_bar.png](../../reports/metrics_bar.png)
- [confidence_distribution.png](../../reports/confidence_distribution.png)
- [prediction_gallery_correct.png](../../reports/prediction_gallery_correct.png)
- [prediction_gallery_incorrect.png](../../reports/prediction_gallery_incorrect.png)

## 10. ROC and Precision–Recall analysis

Cell 23 computes multiclass ROC and Precision–Recall curves directly from the exported per-class probabilities. Each CIFAR-10 class is treated with a declared **One-vs-Rest** strategy. The figure contains per-class curves plus micro and macro summaries. The notebook calculates these curves from `predictions.csv`; it does not substitute hardcoded AUC or Average Precision values.

ROC and Precision–Recall are supplementary to the primary Accuracy and Macro F1 results. They expose ranking quality across thresholds, while the confusion matrix and classification report describe behavior at the fixed argmax decision rule.

## 11. Suggested presentation script

> After checkpoint verification passes, the pipeline evaluates all 10,000 Test images exactly once. The final result is 88.95% Accuracy and 88.99% Macro F1. The confusion matrix sums to 10,000, and Accuracy calculated from its diagonal matches `summary.json`, proving that the final artifacts are internally consistent.

## 12. Transition to the next phase

[Phase 11](phase_11_error_analysis.md) examines confusion pairs, confidence, and prediction grids to understand how the model fails.
