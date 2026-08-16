# Part 3.2 — Real Controlled Training Plan

## Objective

Perform real controlled training for Practice 3 v2 using the three frozen learning rates:

- `2e-5`
- `3e-5`
- `5e-5`

Each experiment must be trained independently using the same dataset, model, seed, and training configuration. Only the Learning Rate is allowed to differ between experiments.

## Frozen Protocol

- Protocol version: `practice_3_v2.0`
- Execution protocol hash: `fdfcbb87b20d0bc618890a51c70a9689d618682b5a7cdf8edf05b638fa347055`
- Train: 7,676 samples
- Validation: 960 samples
- Holdout: 960 samples — SEALED
- Model: DistilBERT
- Max length: 80
- Seed: 42
- Train batch size: 16
- Eval batch size: 32
- Max epochs: 10
- Optimizer: `ADAMW_TORCH_FUSED`
- Scheduler: Linear
- Weight decay: 0.01
- Gradient clipping: 1.0
- Early Stopping:
  - Monitor: Validation Loss
  - Patience: 2
  - Threshold: `1e-6`

## Experiment Order

Training must be executed in the following fixed order:

1. `p3v2_lr_2e-5`
2. `p3v2_lr_3e-5`
3. `p3v2_lr_5e-5`

Each run must start from a fresh pretrained DistilBERT model.

Do not reuse:

- model weights;
- optimizer state;
- scheduler state;
- Trainer state

from a previous experiment.

## Training Flow

Each run must follow:

```text
Verify execution protocol
        ↓
Verify run configuration
        ↓
PLANNED → RUNNING
        ↓
Reset seed
        ↓
Load v2 Train + Validation
        ↓
Create fresh pretrained DistilBERT
        ↓
Create Trainer
        ↓
Trainer.train()
        ↓
Validation after each epoch
        ↓
Apply Early Stopping if required
        ↓
Save best checkpoint
        ↓
Save training history
        ↓
Save run summary
        ↓
RUNNING → COMPLETED