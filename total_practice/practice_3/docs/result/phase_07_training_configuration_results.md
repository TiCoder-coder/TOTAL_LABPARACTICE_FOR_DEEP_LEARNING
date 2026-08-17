# Phase 07 – Training Configuration

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 16` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Training Configuration` listing optimization hyperparameters and their operational purposes.

## Processing Source
- [`experiment_protocol_v2.py`](../../processing_own_phase/experiment_protocol_v2.py)
- [`experiment_runner_v2_3.py`](../../processing_own_phase/experiment_runner_v2_3.py)

## Artifact Sources
- [`training_authorization.json`](./practice_3_v2_3/training_authorization.json)
- [`protocol_manifest.json`](./practice_3_v2_3/protocol_manifest.json)

## Results
| Parameter | Value | Purpose |
|---|---|---|
| **Max Epochs** | 15 | Maximum computational training budget |
| **Early Stopping Patience** | 4 | Halt execution if `eval_loss` fails to improve for 4 consecutive epochs |
| **Batch Size** | 16 | Training mini-batch size |
| **Warmup Steps** | 720 | Linear warmup steps (~1.5 epochs) to prevent early gradient destabilization |
| **Label Smoothing** | 0.1 | Regularization to soften cross-entropy targets and prevent overconfidence |
| **Selection Metric** | `eval_loss` | Primary checkpoint selection criterion (lower is better) |
| **Optimizer** | AdamW | Decoupled weight decay regularization |
| **Gradient Clipping** | 1.0 | Maximum gradient norm threshold |

## Evidence
- `Training Configuration` table in Notebook [`Cell 16`](../../notebook_practice_3/practice_3.ipynb).
- Protocol settings authorized in [`training_authorization.json`](./practice_3_v2_3/training_authorization.json).

## Summary
The training protocol establishes a 15-epoch budget governed by Early Stopping with patience 4 on `eval_loss`. Mini-batch size is set to 16 with 720 warmup steps and 0.1 label smoothing to ensure smooth and stable fine-tuning convergence.
