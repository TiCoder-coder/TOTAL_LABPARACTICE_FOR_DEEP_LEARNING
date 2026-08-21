# Phase 12 — Reproducibility and Conclusion

## Reproducibility contract

The official notebook is
[`practice_2_presentation.ipynb`](../../notebooks/practice_2_presentation.ipynb).
Its `Run All` path reads artifacts, rebuilds EDA, and repeats Validation
verification. It does not train and does not invoke Final Test inference.

The current run is reproducible from these records:

- [six-run ranking](../../outputs/hyperparameter_ranking.csv);
- [locked winner and selection policy](../../outputs/hyperparameter_selection_locked.json);
- [winner epoch log](../../runs/E2_resnet18_partial_2b5b94de/metrics.jsonl);
- [checkpoint manifest](../../runs/E2_resnet18_partial_2b5b94de/checkpoint_manifest.json);
- [Final Test receipt](../../outputs/final_test_receipt_a906600b717f.json);
- [final metrics](../../outputs/summary.json);
- [per-class report](../../outputs/classification_report.csv);
- [prediction probabilities](../../outputs/predictions.csv).

For presentation, these sources are aggregated without mutation into
[`visualization_data.json`](../../outputs/visualization_data.json). The
[HTML dashboard](../../reports/practice_2_training_dashboard.html) reads only
that file; the original run artifacts remain independently auditable.

The checkpoint SHA256 is
`a906600b717f94aa4cf40ca8504f1b82f6618b80cc6bf9b1a7a63813cfd3223b`.
SHA256 detects changed bytes; it is not an independently issued signature.

## Final result

| Item | Value |
|---|---:|
| Backbone | ImageNet-pretrained ResNet18 |
| Fine-tuning | `partial_finetune` |
| Head / backbone LR | `1e-3 / 1e-4` |
| Hidden layers | `[]` |
| Selected epoch | 24 |
| Validation loss | 0.430360 |
| Validation accuracy | 94.42% |
| Test accuracy | 94.06% |
| Test loss | 0.226486 |
| Test macro F1 | 0.940458 |
| Micro ROC-AUC | 0.996542 |
| Micro AP | 0.982046 |
| Final Test evaluations | 1 |

The linear head outperforms the deeper MLP heads by Validation loss. The small
Validation-to-Test accuracy difference indicates stable generalization for the
locked checkpoint.

## Presentation visuals

- [winner training dashboard](../../reports/winner_training_log_dashboard.png)
- [integrated artifact-only HTML dashboard](../../reports/practice_2_training_dashboard.html)
- [ROC and PR curves](../../reports/roc_pr_curves_notebook.png)
- [raw confusion matrix](../../reports/confusion_matrix_raw.png)
- [normalized confusion matrix](../../reports/confusion_matrix_normalized.png)
- [confidence distribution](../../reports/confidence_distribution.png)

## Limitations

- Search uses one fixed seed; repeated-seed mean and standard deviation remain
  future work.
- The search space is deliberately small and does not prove a global optimum.
- Test must not be reused to tune this checkpoint.
- Grad-CAM or another attribution method remains future interpretability work.

## Safe commands

View the notebook without training:

```bash
jupyter notebook notebooks/practice_2_presentation.ipynb
```

Only when a genuinely new search is required:

```bash
python -m processing_own_phase.hyperparameter_search --output-dir outputs
```
