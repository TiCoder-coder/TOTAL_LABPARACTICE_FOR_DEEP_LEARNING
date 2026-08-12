# Phase 8 — Validation-only Hyperparameter Search

## Purpose

The earlier E1/E2 comparison remains historical baseline evidence. The current
official workflow extends it with a two-stage, Validation-only hyperparameter
search on ResNet18 `partial_finetune`.

## Stage 1 — Learning rate

The following head/backbone pairs were compared while classifier head, seed,
split, augmentation, optimizer, scheduler, batch size and epoch budget remained
fixed:

| Candidate | Head LR | Backbone LR | Best epoch | Validation loss | Accuracy at selected checkpoint |
|---|---:|---:|---:|---:|---:|
| `lr_current` | 0.001 | 0.0001 | 24 | **0.430360** | 94.42% |
| `lr_mid` | 0.0006 | 0.00006 | 24 | 0.440741 | 94.56% |
| `lr_low` | 0.0003 | 0.00003 | 24 | 0.462627 | 93.98% |

The primary metric is minimum Validation loss, so `lr_current` wins despite
`lr_mid` having slightly higher accuracy at its own selected checkpoint.

## Stage 2 — Classifier head

The winning learning rate was fixed while head depth changed:

| Candidate | Hidden layers | Epochs trained | Best epoch | Validation loss | Accuracy at selected checkpoint |
|---|---|---:|---:|---:|---:|
| `linear` | `[]` | 25 | 24 | **0.430360** | 94.42% |
| `mlp_2` | `[256, 128]` | 25 | 24 | 0.467199 | 94.46% |
| `mlp_1` | `[256]` | 9 (Early Stop) | 5 | 0.489788 | 93.00% |

The simple linear head wins. Extra hidden layers do not improve Validation
loss for this pretrained representation.

## Canonical artifacts

- [complete ranking CSV](../../outputs/hyperparameter_ranking.csv)
- [locked selection JSON](../../outputs/hyperparameter_selection_locked.json)
- [winner dashboard](../../reports/winner_training_log_dashboard.png)
- [search implementation](../../processing_own_phase/hyperparameter_search.py)

No Test metric appears in the ranking. Test remains locked until the winner
passes Phase 9 verification.
