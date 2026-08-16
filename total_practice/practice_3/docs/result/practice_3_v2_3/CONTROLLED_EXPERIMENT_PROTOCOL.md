# Practice 3 v2.3 — Controlled Experiment Protocol

## Overview
This document defines the strict, multi-stage experimental protocol for the `practice_3_v2.3` namespace. The goal is to provide DistilBERT with an extended training opportunity while adhering strictly to controlled experimental methodologies.

## Protocol Versions & Justification
- **v2.1**: Suffered from extreme confidence overfitting early in training.
- **v2.2**: Introduced `label_smoothing_factor=0.1` and `warmup_steps=480`, proving stable, but stopped prematurely at Epoch 4 due to a restrictive `early_stopping_patience = 2`.
- **v2.3**: Increases `max_epochs` to 15, `early_stopping_patience` to 4, and `warmup_steps` to 720 (10% of 7200 total steps) to allow convergence.

## Data & Splits
- **Fingerprint**: `4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13` (carried over unchanged from v2.1)
- **Train**: 7676 samples
- **Validation**: 960 samples
- **Holdout**: 960 samples (SEALED, evaluation_count=0)
- **Official Test**: Excluded entirely from memory

## Stage A: Learning Rate Search
Three parallel configurations differing ONLY in their `learning_rate`:
- **E1**: `1e-5`
- **E2**: `2e-5`
- **E3**: `3e-5`

### Stage A Ranking
Following the completion of E1, E2, and E3, a Best LR is selected by:
1. Lowest Validation Loss (tolerance 1e-6)
2. Higher Validation F1
3. Higher Validation Accuracy
4. Earlier Best Epoch
5. Lexical experiment ID fallback

## Stage B: DistilBERT-Specific Tuning
These experiments are `BLOCKED_PENDING_STAGE_A` until Stage A officially completes and a Best LR is locked.
- **E4**: Best LR + `weight_decay = 0.05`
- **E5**: Best LR + `classifier_dropout = 0.20`
- **E6**: Best LR + `fine_tuning_strategy = staged` (DistilBERT-specific staged unfreezing)

## Controlled Experiment Rule
Each experiment isolates exactly **one** variable against its parent baseline. Multiple independent changes are strictly forbidden.
