# Phase 10 — Final Test Evaluation Results

[Phase 9 results](phase_09_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/demo_practice_2.ipynb) | [Phase 11 results](phase_11_results.md)

| Metric | Official value |
|---|---:|
| Test Accuracy | 0.8895 |
| Test Loss | 0.3415388059 |
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

The machine-readable [confusion_matrix.csv](../../outputs/confusion_matrix.csv) sums to 10,000, and diagonal Accuracy matches `summary.json`.
