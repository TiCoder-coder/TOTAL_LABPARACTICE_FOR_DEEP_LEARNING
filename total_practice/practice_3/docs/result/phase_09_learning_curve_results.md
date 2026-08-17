# Phase 09 – Learning Curves & Overfitting

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cells:
- [`Cell 22` (Code)](../../notebook_practice_3/practice_3.ipynb): Winner E4 trajectories.
- [`Cell 24` (Code)](../../notebook_practice_3/practice_3.ipynb): Experiment comparison curves across all 6 configurations.

Cell output:
Rendered graphical figures displaying per-epoch training/validation loss, accuracy, F1 trajectories, learning rate schedule, and early stopping behavior.

## Processing Source
- [`experiment_runner_v2_3.py`](../../processing_own_phase/experiment_runner_v2_3.py)

## Artifact Sources
- [`winner_train_val_loss.png`](./practice_3_v2_3/figures/winner_train_val_loss.png)
- [`winner_validation_accuracy.png`](./practice_3_v2_3/figures/winner_validation_accuracy.png)
- [`winner_validation_f1.png`](./practice_3_v2_3/figures/winner_validation_f1.png)
- [`winner_learning_rate.png`](./practice_3_v2_3/figures/winner_learning_rate.png)
- [`experiment_best_vs_stop_epoch.png`](./practice_3_v2_3/figures/experiment_best_vs_stop_epoch.png)
- [`experiment_val_loss_curves.png`](./practice_3_v2_3/figures/experiment_val_loss_curves.png)
- [`final_validation_winner_lock.json`](./practice_3_v2_3/final_validation_winner_lock.json)

## Results
| Training Attribute | Value | Description |
|---|---:|---|
| **Maximum Epoch Budget** | 15 | Upper computational bound |
| **Selected Best Epoch** | **2.0** | Checkpoint with minimum Validation Loss (0.4153) |
| **Actual Stop Epoch** | **6.0** | Early Stopping triggered after patience = 4 epochs without improvement |
| **Best Val Accuracy** | 0.8677 | Peak classification accuracy at Epoch 2 |
| **Best Val Macro-F1** | 0.8678 | Peak balanced F1 score at Epoch 2 |

### Epoch Progression of Winner E4
- **Epoch 1**: Train Loss ~0.53, Val Loss ~0.43 (rapid convergence)
- **Epoch 2 (Best Epoch)**: Train Loss ~0.38, Val Loss **0.4153** (optimal generalization, checkpoint locked)
- **Epoch 3–6**: Train loss continues downward (~0.18) while Val Loss rises (~0.52), indicating the onset of fine-tuning overfitting. Early Stopping safely halts training at Epoch 6.

## Evidence
- Winner E4 loss and accuracy plots rendered in Notebook [`Cell 22`](../../notebook_practice_3/practice_3.ipynb) ([`winner_train_val_loss.png`](./practice_3_v2_3/figures/winner_train_val_loss.png)).
- Multi-experiment comparison curves rendered in Notebook [`Cell 24`](../../notebook_practice_3/practice_3.ipynb) ([`experiment_val_loss_curves.png`](./practice_3_v2_3/figures/experiment_val_loss_curves.png), [`experiment_best_vs_stop_epoch.png`](./practice_3_v2_3/figures/experiment_best_vs_stop_epoch.png)).

## Summary
The winning model `E4` reaches its optimal generalization state at Epoch 2 (Val Loss 0.4153). Early Stopping successfully bounded training at Epoch 6 (Best Epoch 2 + Patience 4), preventing overfitting degradation.
