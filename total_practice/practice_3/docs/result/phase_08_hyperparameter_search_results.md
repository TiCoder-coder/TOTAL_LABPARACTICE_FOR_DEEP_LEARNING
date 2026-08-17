# Phase 08 – Hyperparameter Search

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 18` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Final Validation Ranking` sorting all 6 controlled hyperparameter configurations, followed by the `Validation Loss Comparison` bar chart.

## Processing Source
- [`experiment_runner_v2_3.py`](../../processing_own_phase/experiment_runner_v2_3.py)
- [`experiment_registry_v2.py`](../../processing_own_phase/experiment_registry_v2.py)

## Artifact Sources
- [`final_validation_ranking.csv`](./practice_3_v2_3/final_validation_ranking.csv)
- [`final_validation_ranking.json`](./practice_3_v2_3/final_validation_ranking.json)
- [`final_validation_winner_lock.json`](./practice_3_v2_3/final_validation_winner_lock.json)
- [`experiment_best_val_loss_comparison.png`](./practice_3_v2_3/figures/experiment_best_val_loss_comparison.png)

## Results
- **Primary selection metric**: `Validation Loss (lower is better)`.
- **Selected Winner**: `E4_weight_decay_0.05`.

### Complete Experiment Validation Ranking
| Rank | Run ID | Learning Rate | Weight Decay | Dropout | Strategy | Best Epoch | Stop Epoch | Best Val Loss | Val Acc | Val F1 | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **E4_weight_decay_0.05** | **1e-5** | **0.05** | **0.2** | **full** | **2.0** | **6.0** | **0.4153** | **0.8677** | **0.8678** | **WINNER** |
| 2 | E1_lr_1e-5 | 1e-5 | 0.01 | 0.2 | full | 2.0 | 6.0 | 0.4156 | 0.8656 | 0.8660 | VALID |
| 3 | E5b_classifier_dropout_0.40 | 1e-5 | 0.01 | 0.4 | full | 2.0 | 6.0 | 0.4163 | 0.8656 | 0.8655 | VALID |
| 4 | E6c_staged_finetune | 1e-5 | 0.01 | 0.2 | staged | 3.0 | 7.0 | 0.4185 | 0.8656 | 0.8646 | VALID |
| 5 | E3_lr_3e-5 | 3e-5 | 0.01 | 0.2 | full | 1.0 | 5.0 | 0.4275 | 0.8552 | 0.8478 | VALID |
| 6 | E2_lr_2e-5 | 2e-5 | 0.01 | 0.2 | full | 2.0 | 6.0 | 0.4396 | 0.8583 | 0.8486 | VALID |

## Evidence
- `Final Validation Ranking` table rendered in Notebook [`Cell 18`](../../notebook_practice_3/practice_3.ipynb).
- Validation Loss Comparison figure [`experiment_best_val_loss_comparison.png`](./practice_3_v2_3/figures/experiment_best_val_loss_comparison.png) displayed in Cell 15.
- Locked selection metadata in [`final_validation_winner_lock.json`](./practice_3_v2_3/final_validation_winner_lock.json).

## Summary
Across 6 valid hyperparameter search runs evaluated on the Validation split, `E4_weight_decay_0.05` achieved the lowest Validation Loss (0.4153) and highest Macro-F1 (0.8678), successfully locking it as the winner model checkpoint.
