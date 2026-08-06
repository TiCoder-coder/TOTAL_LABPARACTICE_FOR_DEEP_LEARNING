# R10 Robust Model Selection

## Scope

R10 implements repeated-seed Validation model selection and final-checkpoint freezing. It does not select a real finalist because R9 has not authorized an architecture and no repeated-seed finalist results exist.

No dataset image, Validation content, or Test content is read. No optimization step is executed. R11 is not executed.

## Metric Contract

The required seeds are `42`, `123`, and `2026`. Primary selection uses mean Validation Macro F1. Secondary selection uses mean Validation Accuracy, followed by the lower mean Train/Validation generalization gap.

Accuracy, precision, recall, F1, and class support are derived directly from each run's ordered ten-class confusion matrix. Result rows cannot supply alternative scalar Validation metrics.

Each finalist summary reports Validation Accuracy and Macro F1 mean and sample standard deviation, per-seed Wilson intervals, a descriptive pooled Wilson interval, per-class recall mean and standard deviation, and row-normalized confusion stability.

## Consistent Winner Rule

A finalist must win at least two of the three seeds and exceed the runner-up mean Validation Macro F1 by at least `0.003`. A leader that has a higher mean but wins only one seed is not selected.

Every finalist must cover exactly the three declared seeds, use one configuration hash and one initialization hash across those seeds, and provide valid model-state and checkpoint hashes. Any Test field, missing seed, duplicate seed, invalid confusion matrix, or mixed configuration fails closed.

## Final Checkpoint Rule

After a configuration wins, the final checkpoint is the selected configuration's seed-42 checkpoint chosen by best Validation Macro F1. Before Test authorization, the freeze record verifies and stores the selection-record hash, configuration hash, model-state hash, and checkpoint-file hash.

Selection-record tampering, configuration mismatch, or checkpoint-file mutation invalidates the freeze operation.

Synthetic fixtures verify the statistical and freeze logic but cannot authorize a finalist or checkpoint.

## Gate Result

All R10 protocol checks pass. The overall gate remains blocked because R9 is blocked, no architecture is authorized, repeated-seed finalist evidence is unavailable, no robust finalist is selected, and no real final checkpoint is frozen.

The canonical notebook and source images remain unchanged.
