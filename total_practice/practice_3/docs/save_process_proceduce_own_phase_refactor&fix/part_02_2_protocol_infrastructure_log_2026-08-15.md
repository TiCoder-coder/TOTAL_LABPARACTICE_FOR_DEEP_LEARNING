# Part 2.2 — Protocol Infrastructure Log

Date: 2026-08-15  
Protocol: `practice_3_v2.0`

## Files audited

- Frozen Part 2.1 protocol plan and its SHA-256.
- Practice 3 codebase audit and Part 1 process log.
- Current config, Phase 8 metrics/configuration, Phase 9 training implementation/configuration.
- Phase 9, Phase 11, and Phase 14 v1 manifests and weight/package fingerprints.
- Cached Rotten Tomatoes official Train and Validation splits only.

## Files added

- `processing_own_phase/experiment_protocol_v2.py`
- `processing_own_phase/dataset_protocol_v2.py`
- `processing_own_phase/experiment_registry_v2.py`
- `processing_own_phase/experiment_runner_v2.py`
- `processing_own_phase/holdout_guard_v2.py`
- `processing_own_phase/initialize_protocol_v2.py`
- `tests/test_protocol_v2.py`
- Part 2.2 refactor plan.
- Real protocol metadata under `docs/result/practice_3_v2/`.
- This process log.

## Files modified

No v1 source, notebook, historical artifact, checkpoint, Test result, package, or Part 2.1 protocol file was modified by Part 2.2.

## Dataset manifest

- Official source requested: Train 8,530 and Validation 1,066.
- Official Test requested: false.
- Development pool: 9,596.
- v2 Train/Validation/Holdout: 7,676/960/960.
- Per-class counts: 3,838/480/480.
- Exact duplicates: 0.
- Source-ID and exact-text cross-split overlaps: 0.
- Holdout raw content in manifest: false.
- Development pool SHA-256: `08fad26ecaa3a68a90257f1f3c330eb5fb060c8d3f3e9c2c705b3c30fccb3621`.
- Split manifest SHA-256: `4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13`.

## Config and registry status

- Three deterministic run configs created for `2e-5`, `3e-5`, and `5e-5`.
- Identity/log-path/hash fields are run-specific; every experimental fixed variable is identical and only LR varies.
- Config hashes verified by canonical sorted UTF-8 JSON.
- Registry contains exactly three unique `PLANNED` records.
- All metric/checkpoint/runtime result fields remain `null`.
- No synthetic result appears in the real namespace.
- Completed-run cache rule is `LOAD` only when config hash and artifacts validate; otherwise `INVALID`, never silent overwrite.

## Optimizer preflight

- Optimizer: `ADAMW_TORCH_FUSED`.
- Device seen in authorized environment: MPS.
- Result: `SUPPORTED`.
- Check performed: zero-step optimizer construction only.
- Backward called: false.
- Optimizer step called: false.
- No fallback or protocol change occurred.

## TensorBoard status

- Unique directories prepared under `runs/practice_3_v2/<run_id>/`.
- Future required scalars are recorded in the protocol manifest.
- TensorBoard dependency is installed.
- Event files created: false.

## Guard and winner status

- Winner manifest exists: false.
- Winner status: `NOT_SELECTED`.
- Holdout status: `SEALED`.
- `holdout_access_allowed = false`.
- `holdout_evaluation_count = 0`.
- Synthetic winner validation and one-time 0→1 attempt claim passed only in a temporary directory.
- A second synthetic claim was rejected.
- Real Holdout was not materialized, inspected, predicted, or evaluated.

## Synthetic/static tests

Eleven tests passed:

- deterministic balanced split and hashes;
- exact conflicting-label duplicate rejection;
- three run configs and deterministic hashes;
- invalid LR and duplicate run-ID rejection;
- registry lifecycle and completed-run cache protection;
- optimizer construction preflight;
- Transformers TrainingArguments and Early Stopping compatibility without Trainer;
- Validation-only ranking across loss/F1/Accuracy/epoch ties;
- Test/Holdout ranking-field rejection;
- synthetic winner and one-time Holdout claim;
- non-COMPLETED winner rejection.

All v2 Python/test files parsed successfully. Real metadata schema/hash/read-back validation passed.

## No-training and v1 preservation

- Trainer created: false.
- `Trainer.train()`: not present/called.
- Model loaded: false.
- Training/backward/optimizer step: false.
- v1 Test loaded/evaluated: false.
- v2 Holdout evaluated: false.
- Real experiment executed: false.
- v1 `test_evaluation_count`: still 1.
- v1 checkpoint and Phase 14 package SHA-256 remain `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.

## Blockers

None. The optimizer construction preflight is supported. This does not claim bitwise determinism or prove future kernel execution beyond the non-training preflight permitted in Part 2.2.

## Readiness

PART 2.2 infrastructure is ready for Part 2.3 review/execution planning. No experiment may start until the next part is explicitly approved.

