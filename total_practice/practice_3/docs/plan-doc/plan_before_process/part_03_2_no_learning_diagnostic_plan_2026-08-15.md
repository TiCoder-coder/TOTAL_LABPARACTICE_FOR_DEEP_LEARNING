# PART 3.2 — No-Learning Diagnostic Plan

Date: 2026-08-15  
Scope: temporary diagnostic isolation only; no official experiment or protocol mutation.

## Objective

Determine whether the observed no-learning and class-0 collapse is primarily associated with MPS execution, learning-rate strength, full-model optimization versus classification-head optimization, or a deeper gradient/logit problem.

## Safety boundary

- Use only 64 deterministically selected, balanced records from locked v2 Train: 32 class 0 and 32 class 1.
- Never materialize Validation, v2 Holdout, or official Test for this diagnostic.
- Never call the official single-run runner or touch its output paths.
- Write runtime files only beneath an automatically removed `/private/tmp` directory.
- Do not read/write the real registry, alter frozen configs, select a winner, or reset `p3v2_lr_2e-5`.

## Locked diagnostic fixture

- Same 64 source IDs and labels for every case
- `distilbert/distilbert-base-uncased`, fresh initialization per case
- Seed/data seed 42
- Max length 80, truncation and dynamic padding
- Batch size 16
- 120 optimizer steps (equivalent to 30 passes over 64 records)
- AdamW, weight decay 0.01, linear decay, no warmup, max gradient norm 1.0
- No evaluation/save strategy and no TensorBoard reporting

The non-fused AdamW implementation is used across the MPS/CPU comparison so device is the only changed factor. The preceding audit already showed that fused and non-fused AdamW on MPS produced effectively identical tiny-fixture results.

## Controlled matrix

| Case | Device | LR | Trainable scope | Purpose |
|---|---|---:|---|---|
| `mps_full_lr2e5` | MPS | 2e-5 | full model | current-LR reference |
| `cpu_full_lr2e5` | CPU | 2e-5 | full model | device-only control |
| `mps_full_lr5e5` | MPS | 5e-5 | full model | LR-only control |
| `mps_full_lr1e4` | MPS | 1e-4 | full model | stronger LR-only control |
| `mps_head_lr1e4` | MPS | 1e-4 | pre-classifier + classifier | scope-only control against full `1e-4` |

If MPS is unavailable, record the MPS cases as blocked rather than substituting another backend.

## Measurements

For every case capture:

- initial/final Train loss and accuracy;
- predicted class counts;
- classifier, pre-classifier, and representative backbone gradient norms from a deterministic probe batch;
- classifier and representative backbone parameter deltas;
- initial/final logits mean, standard deviation, minimum, maximum and mean absolute class margin;
- runtime, steps, device, LR, trainable scope and optimizer.

Gradient probes use the same Train-only fixture, perform no optimizer step, and are cleared before/after measurement. Seeds are reset before actual diagnostic training.

## Interpretation

- MPS materially worse than CPU under the same case indicates backend sensitivity.
- Higher LR reaching approximately 90%+ Train accuracy indicates insufficient effective optimization strength at the locked LR.
- Head-only materially outperforming matched full fine-tuning indicates optimization interference/dilution from full-backbone updates.
- No case reaching high Train accuracy indicates a deeper runtime/model/data implementation issue.

These are diagnostic findings only. They cannot update official metrics or authorize a protocol change.

## Outputs

- Temporary machine-readable measurements beneath `/private/tmp` during execution only.
- Persistent analysis: `docs/plan-doc/analysis_error/part_03_2_no_learning_diagnostic_result_2026-08-15.md`.

## Completion criteria

All available controlled cases finish with comparable measurements; official artifacts and registry remain unchanged; Holdout/Test access remains false; findings identify or narrow the likely cause; no final fix is implemented.
