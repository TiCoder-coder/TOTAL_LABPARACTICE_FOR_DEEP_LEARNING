# Phase 36 S14 FFN Sweep

## Objective

Compare absolute FFN widths 64, 128 and 256 while freezing data, attention, depth and training policy.

## Validation evidence

- F64: RMSE 58.39803077736952 Wh; parameters 52673
- F128: RMSE 58.08190056355405 Wh; parameters 69185
- F256: RMSE 57.69679988114431 Wh; parameters 102209

## Winner

F256 (`ffn_dim=256`) wins by full-precision Validation RMSE. It is an upper-boundary winner; no F512 condition was added.

## Verification and limitations

F64 and F256 BEST checkpoints were strict-loaded and recomputed on full ordered Validation. F128 was reused without retraining and retains the inherited incomplete-artifact warning. This is a single-seed Validation-only result. Test remained forbidden.

## Phase 37 handoff

Reuse the selected MSE winner; train only the Huber condition.
