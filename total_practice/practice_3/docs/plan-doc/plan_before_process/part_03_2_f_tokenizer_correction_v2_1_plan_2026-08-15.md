# PART 3.2.F — Tokenizer Correction and Practice 3 v2.1 Preparation Plan

## Objective

Create an isolated `practice_3_v2.1` protocol that changes only tokenizer artifact resolution, adds fail-closed tokenizer integrity checks, reuses the locked v2.0 dataset split, passes a Train-only learning smoke test and zero-training integration, and authorizes the user to manually start one v2.1 run.

## Root cause and correction

v2.0 mapped both model and tokenizer to the namespaced cache. That cache has valid model weights but no canonical tokenizer files, yielding a five-token all-`[UNK]` tokenizer. v2.1 separates resolution:

- model: `distilbert/distilbert-base-uncased`;
- tokenizer: `distilbert-base-uncased` with verified vocabulary size 30,522.

No model, dataset, split, metric, LR, epoch, Early Stopping or ranking methodology changes.

## v2.0 preservation

Treat all v2.0 manifests, configs, registry states, completed run evidence and output directories as read-only historical evidence. Capture/verify their state during preparation. Never reset or execute a v2.0 run.

## v2.1 namespace and protocol

Create `docs/result/practice_3_v2_1` and `runs/practice_3_v2_1`. Copy the validated split manifest payload unchanged so its fingerprint remains `4d22ccf...bb13`. Create three isolated run configs and PLANNED registry records for `p3v21_lr_2e-5`, `p3v21_lr_3e-5`, and `p3v21_lr_5e-5`. Create a separate sealed Holdout state with count zero.

## Tokenizer validation

Fail closed unless vocab size is exactly 30,522, required special tokens/IDs exist and are in range, canonical known words do not map to `[UNK]`, deterministic v2.1 Train samples have a low `[UNK]` ratio, and distinct texts produce multiple distinct encodings. Store counts/hashes only; do not persist raw sample text.

## Train-only smoke test

Use a temporary balanced known-easy synthetic fixture, CPU, fresh DistilBERT, frozen backbone and a manual PyTorch optimizer loop. Do not call `Trainer.train()`. Require at least 90% Train accuracy. Store only diagnostic summary evidence in v2.1 results.

## Zero-training integration

Materialize locked Train/Validation only, tokenize with the corrected tokenizer, validate counts/labels/lengths, construct fresh model/Trainer/fused optimizer/linear scheduler/Early Stopping for each run, inspect dynamic batches and run inference-only forward sanity. Do not train/evaluate or create TensorBoard events/checkpoints.

## Freeze and authorization

Hash the v2.1 protocol manifest, unchanged split manifest, tokenizer validation artifact, three config hashes and ranking rule into a new execution protocol hash. Authorize only when all integrity, smoke, integration, output-cleanliness, registry and Holdout guards pass.

## Runner

Provide a v2.1-only CLI requiring one explicit run ID. Its real path verifies authorization/hash/tokenizer/config/registry/order/Holdout/output cleanliness before constructing fresh Train/Validation state and transitioning PLANNED to RUNNING immediately before `Trainer.train()`. This task validates CLI/preflight only and never invokes the real path.

## Validation and completion

Run syntax/import, tokenizer pass/fail fixtures, hash/registry/authorization checks, CLI rejection and zero-training integration. Completion requires authorization true, all v2.1 runs PLANNED, no events/checkpoints/metrics, Holdout count zero, Test excluded and v2.0 unchanged.
