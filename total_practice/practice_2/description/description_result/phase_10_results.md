# Phase 10 — Final Test Evaluation Results

[Phase 9 results](phase_09_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 11 results](phase_11_results.md)

| Metric | Official value |
|---|---:|
| Test Accuracy | 0.8895 |
| Test Loss | 0.3415387766 |
| Macro Precision | 0.8928359014 |
| Macro Recall | 0.8895 |
| Macro F1 | 0.8899327014 |
| Test samples | 10,000 |
| Test evaluation count | 1 |

Artifacts displayed in Cell 23:

- [summary.json](../../outputs/summary.json)
- [classification_report.csv](../../outputs/classification_report.csv)
- [predictions.csv](../../outputs/predictions.csv)
- [metrics_bar.png](../../reports/metrics_bar.png)
- [confusion_matrix_raw.png](../../reports/confusion_matrix_raw.png)
- [confusion_matrix_normalized.png](../../reports/confusion_matrix_normalized.png)
- [validation_test_comparison_current.png](../../reports/validation_test_comparison_current.png)

Cell 23 also renders:

- a per-class report extended with TP, TN, FP, FN, TPR, and FPR;
- multiclass One-vs-Rest ROC and Precision–Recall curves from the ten exported probability columns;
- a ground-truth/prediction table combining confident errors with challenging correct cases.

The machine-readable [confusion_matrix.csv](../../outputs/confusion_matrix.csv) sums to 10,000, and diagonal Accuracy matches `summary.json`. The regenerated [predictions.csv](../../outputs/predictions.csv) contains all ten `probability_<class>` columns and passes probability, argmax, confidence, sample-count, and Accuracy consistency checks.
## Updated locked Final Test

The locked winner was evaluated on the official 10,000-image CIFAR-10 Test set
exactly once. Accuracy is `94.06%`, loss is `0.226486`, and macro F1 is
`0.940458`. The persistent receipt is keyed by checkpoint SHA256 and prevents a
second Final Test run for this checkpoint.

`classification_report.csv` contains per-class TP, TN, FP, FN, TPR, FPR, and
`class_accuracy = (TP + TN) / N`. One-vs-Rest ROC and Precision–Recall outputs
cover all ten classes; micro ROC-AUC is `0.996542` and micro average precision
is `0.982046`.
