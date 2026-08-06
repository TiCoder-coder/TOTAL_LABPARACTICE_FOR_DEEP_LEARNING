# R8 Staged Transfer Learning

## Scope

R8 implements the staged ResNet18 transfer-learning protocol and its controlled experiment registry. It does not claim model-performance evidence because R7 remains blocked, R6 has not selected a transform, the Train manifest is not authorized, and the approved pretrained checkpoint is unavailable locally.

No dataset image, Validation content, or Test content is read. No real optimization step is executed. R9 is not executed.

## Controlled Experiments

`M0` is the immediate layer4 fine-tuning baseline. `M1` adds four classifier-head warmup epochs. `M2` changes only dropout from `0.20` to `0.35` relative to `M1`. `M3` changes only label smoothing from `0.00` to `0.05` relative to `M1`.

Every experiment must start from one shared `IMAGENET1K_V1` ResNet18 backbone state and the same deterministic classifier-head initialization. The declared experiment seeds are `42`, `123`, and `2026`.

## Stage Contract

Staged experiments begin with only the classifier head trainable. The warmup duration is fixed at four epochs, within the approved three-to-five epoch range. The best warmup checkpoint is selected by Validation Macro F1 with corrected Validation loss as the tie-breaker.

Layer4 cannot be unfrozen before all warmup epochs finish. The session restores the strict offline best warmup checkpoint before changing trainability. The fine-tuning stage exposes only `layer4` and `fc` parameters.

The AdamW head learning rate is `5e-4`; the backbone learning rate is `2e-5`. Both fall inside the approved ranges. ReduceLROnPlateau and early stopping receive only `exact_global_cross_entropy_mean` Validation loss from the R7 contract.

## Repeated-Seed Selection

The selector requires one row for every M0-M3 and seed combination, exactly one shared initial-state hash, and only the declared Train/Validation fields. Any Test-derived field, incomplete seed coverage, duplicate run, invalid metric, or mixed initialization hash is rejected.

A staged candidate can advance only when mean Validation Macro F1 improves, mean Validation Accuracy does not materially decrease, and the mean Train/Validation generalization gap decreases relative to M0. Synthetic values used by unit verification are explicitly excluded from performance evidence and cannot authorize a candidate.

## Verification Result

All protocol checks pass using an offline synthetic tensor fixture. The fixture verifies shared initialization, head-only warmup, best-checkpoint restoration, layer4-only unfreezing, named learning rates, corrected Validation controls, repeated-seed aggregation, and Test-field rejection.

The overall R8 gate remains blocked. No pretrained state, repeated-seed Validation result, or candidate configuration is authorized. The canonical notebook and source images remain unchanged.
