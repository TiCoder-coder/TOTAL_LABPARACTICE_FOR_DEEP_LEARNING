# R7 Training Mathematics and Offline Checkpoint Correctness

## Scope

R7 establishes the reusable correctness contract for later training phases. It does not train on the project dataset, inspect Validation or Test content, construct a Test loader, select a transform recipe, or execute R8.

R6 remains blocked because no transform recipe or Train manifest has been authorized. R7 therefore verifies implementation behavior with deterministic synthetic tensors and an ephemeral checkpoint while preserving the predecessor gate.

## Epoch Loss Contract

Every batch computes unreduced cross-entropy values. The batch numerator is the sum of those values. The denominator is the number of targets for unweighted cross-entropy and the sum of target-class weights for weighted cross-entropy. Epoch loss is computed once from the global numerator divided by the global denominator.

This contract is shared by the `Train`, `Validation`, and future `Test` metric scopes. It is invariant to batch partitioning and matches PyTorch `CrossEntropyLoss(reduction="mean")` for unweighted, weighted, smoothed, and weighted-smoothed cases.

The baseline is unweighted cross-entropy with label smoothing set to `0.0`. Class weighting and label smoothing remain opt-in, separately identified experiments.

## Validation Controls

The scheduler and early-stopping router accepts only a metric whose split is `Validation` and whose contract is `exact_global_cross_entropy_mean`. Metrics from another split or a legacy batch-mean contract are rejected before either control is updated.

## Model and Optimizer Contract

ResNet18 is constructed with `weights=None`. The classifier head and trainable backbone parameters are assigned to non-overlapping optimizer groups named `head` and `backbone`. Their learning rates are exposed as separate values in the verification artifact.

After `model.train()`, every frozen BatchNorm layer is returned to evaluation mode so its running statistics remain unchanged. BatchNorm layers in a trainable block remain in training mode.

## Checkpoint Contract

The complete model state, architecture metadata, classifier dimensions, dropout value, and deterministic state hash are saved together. Reload creates the architecture with `weights=None`, strict-loads the complete state dict, verifies its hash, and enters evaluation mode.

Offline reload is tested while torchvision download functions are forced to fail. The original and reloaded models produce byte-identical output tensors on the same deterministic synthetic input.

## Gate Result

All R7 correctness exit checks pass. The overall R7 gate remains blocked by `R6_GATE_NOT_PASSED`, `TRANSFORM_RECIPE_NOT_SELECTED`, and `TRAIN_MANIFEST_NOT_AUTHORIZED`.

No real training was performed. Validation and Test content access counts are zero. The canonical notebook and source images were not modified. R8 was not executed.
