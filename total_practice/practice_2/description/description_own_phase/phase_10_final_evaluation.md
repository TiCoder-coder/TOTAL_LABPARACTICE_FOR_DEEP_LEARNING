# Phase 10 — Locked Final Test Evaluation

## Single-pass rule

After Phase 9 passes, the official 10,000-image CIFAR-10 Test set is evaluated
exactly once. The SHA256-keyed
[`Final Test receipt`](../../outputs/final_test_receipt_a906600b717f.json)
records `test_evaluation_count = 1` and prevents a second run for this
checkpoint. Running the presentation notebook reads stored artifacts and does
not repeat Test inference.

## Final metrics

| Metric | Value |
|---|---:|
| Validation accuracy | 94.42% |
| Validation loss | 0.430360 |
| Test accuracy | **94.06%** |
| Test loss | **0.226486** |
| Macro precision | 0.940558 |
| Macro recall | 0.940600 |
| Macro F1 | **0.940458** |
| Test samples | 10,000 |

The 0.36-percentage-point Validation-to-Test accuracy decrease is small. The
loss values are not directly comparable as generalization-gap evidence because
Validation uses the training criterion with label smoothing while the final
Test report uses standard Cross Entropy.

## Per-class fixed-decision metrics

[`classification_report.csv`](../../outputs/classification_report.csv) contains
all ten CIFAR-10 classes and the following fields:

- Precision, Recall, F1 and support;
- TP, TN, FP and FN;
- `TPR = TP / (TP + FN)`;
- `FPR = FP / (FP + TN)`;
- `class_accuracy = (TP + TN) / N`.

The raw [confusion matrix](../../outputs/confusion_matrix.csv) sums to 10,000,
and its diagonal reproduces Test accuracy `0.9406` exactly.

## ROC and Precision–Recall

The exported probabilities in
[`predictions.csv`](../../outputs/predictions.csv) support One-vs-Rest curves
for every class. Canonical artifacts:

- [directly displayed ROC/PR image](../../reports/roc_pr_curves_notebook.png)
- [curve summary JSON](../../outputs/roc_pr_summary.json)
- [ROC points CSV](../../outputs/roc_curve_per_class.csv)
- [PR points CSV](../../outputs/pr_curve_per_class.csv)

Micro ROC-AUC is `0.996542`; micro Average Precision is `0.982046`. Per-class
AUC and AP values are displayed in the notebook before the locked curve image.

## Other presentation artifacts

- [raw confusion matrix](../../reports/confusion_matrix_raw.png)
- [normalized confusion matrix](../../reports/confusion_matrix_normalized.png)
- [per-class metric bars](../../reports/metrics_bar.png)
- [Validation–Test comparison](../../reports/validation_test_comparison_current.png)
