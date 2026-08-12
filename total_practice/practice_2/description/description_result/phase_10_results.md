# Phase 10 Results — Final Test and Curves

Notebook cells 22–23 read the locked Final Test artifacts.

| Metric | Value |
|---|---:|
| Test Accuracy | 94.06% |
| Test Loss | 0.226486 |
| Macro Precision | 0.940558 |
| Macro Recall | 0.940600 |
| Macro F1 | 0.940458 |
| Micro ROC-AUC | 0.996542 |
| Micro Average Precision | 0.982046 |
| Test evaluations | 1 |

The [classification report](../../outputs/classification_report.csv) has TP,
TN, FP, FN, TPR, FPR and class accuracy for every class. The notebook displays
the per-class AUC/AP table and embeds the
[ROC/PR image](../../reports/roc_pr_curves_notebook.png) directly.

Machine-readable evidence:

- [summary](../../outputs/summary.json)
- [predictions with ten probability columns](../../outputs/predictions.csv)
- [confusion matrix](../../outputs/confusion_matrix.csv)
- [ROC/PR summary](../../outputs/roc_pr_summary.json)
- [Final Test receipt](../../outputs/final_test_receipt_a906600b717f.json)
