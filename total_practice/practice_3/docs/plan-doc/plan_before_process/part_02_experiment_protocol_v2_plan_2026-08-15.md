# Part 2 — Experiment Protocol v2 Plan

Status: **READY FOR REVIEW — implementation and execution not started**  
Protocol version: `practice_3_v2.0`  
Design date: 2026-08-15

## 1. Objective

Pre-register a controlled three-learning-rate experiment before building its infrastructure. Practice 3 v1 remains immutable historical evidence. Part 2.1 performs documentation-only audit and protocol design: no training, v1 Test access, or v2 holdout evaluation.

## 2. Current v1 state — audited

| Item | Evidence-backed v1 value |
|---|---|
| Dataset | `cornell-movie-review-data/rotten_tomatoes` |
| Splits | Train 8,530; Validation 1,066; Test 1,066; each exactly class-balanced |
| Base model/tokenizer | `distilbert-base-uncased` |
| Maximum token length | 80 |
| Seed/data seed | 42/42 |
| Baseline learning rate | `2e-5` |
| Epochs | 3 |
| Train/eval batch | 16/32 |
| Authoritative model | `checkpoint-1068`, epoch 2, step 1068 |
| Selection evidence | minimum Validation loss; Test not used for selection |
| v1 Test status | Final Test already evaluated |
| v1 Test count | `test_evaluation_count = 1` |
| Weight/package SHA-256 | `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660` |

Evidence: Phase 4/5 dataset artifacts, Phase 8 configuration source/artifact, Phase 9 training manifest/configuration/selected checkpoint, Phase 11 evaluation manifest, Phase 14 save/reload verification, and the post-Test methodology-conflict analysis.

## 3. Methodology problem

The v1 Test result was visible before a proposed LR sweep. Therefore the official v1 Test is historical and is no longer an untouched evaluation set for v2. New LR candidates cannot legitimately replace the v1 winner and reuse its Test score. v2 must be a separate protocol with a newly locked holdout and separate access counter.

## 4. v1/v2 separation

### Practice 3 v1 — immutable historical protocol

- completed Phase 0–15;
- authoritative `checkpoint-1068`, epoch 2;
- historical Test result retained;
- `test_evaluation_count = 1`;
- existing Phase 9/11/14 result directories are never overwritten by v2.

### Practice 3 v2 — new controlled protocol

- LR candidates: `2e-5`, `3e-5`, `5e-5`;
- winner: `TBD`;
- new protocol-relative untouched holdout;
- initial `holdout_evaluation_count = 0`;
- new artifact and TensorBoard namespaces containing `practice_3_v2`.

## 5. Dataset protocol v2

```text
Original official Train + Validation (development source only)
  → deterministic label-stratified repartition
  → v2 Train + v2 Validation + locked v2 Holdout
  → controlled LR runs on Train/Validation
  → Validation ranking
  → winner manifest locked
  → Holdout evaluation once
```

The original official Test is excluded from every v2 split and must not be loaded by v2 infrastructure.

- **Train:** the only split used to update weights.
- **Validation:** Early Stopping, best epoch/checkpoint, LR comparison, run ranking, and winner selection.
- **Holdout:** final evaluation once, only after a valid winner lock.

Holdout is prohibited from EDA/config decisions, qualitative inspection, preprocessing changes, LR/epoch/model selection, threshold tuning, and error inspection before final evaluation.

## 6. New holdout design

### Source and counts

Use only original official Train (8,530) plus Validation (1,066): 9,596 records, 4,798 per label. Exclude the 1,066-record official Test entirely.

For each label independently, deterministically permute stable source identities using seed 42, then allocate:

| v2 split | Per class | Total |
|---|---:|---:|
| Train | 3,838 | 7,676 |
| Validation | 480 | 960 |
| Holdout | 480 | 960 |

This gives exact 50/50 balance. Stable identity is `<original_split>:<original_index>`; record order after allocation is canonicalized by stable identity.

### Duplicate and leakage contract

Before finalizing the split manifest, verify exact `(text, label)` duplicates and exact text overlaps. Current Phase 5 evidence reports zero duplicates/overlaps in the original splits. Part 2.2 must re-verify this from the development source. If duplicates are discovered, all identical-text records must be assigned as one group; conflicting labels or inability to preserve locked counts must stop implementation for protocol review. No duplicate may cross v2 splits.

### Fingerprints

Persist without exposing holdout examples:

- upstream Hugging Face dataset fingerprint and library version;
- SHA-256 of canonical `source_id\tlabel\tUTF-8 text` records for the development pool;
- SHA-256 of ordered source-ID lists for each split;
- combined split-manifest SHA-256;
- counts, label counts, duplicate/overlap checks, seed, and algorithm identifier.

After creation, holdout text/labels are accessible only inside the guarded final-evaluation path. Before winner lock, registry/notebook output may expose count and fingerprint, never examples or metrics.

### Important limitation

The proposed holdout is untouched **within v2 after protocol lock**, but its records came from v1 Train/Validation and therefore are not globally unseen in project history. A globally novel holdout requires a separately acquired, provenance-compatible external dataset and a new approved protocol. This limitation must remain visible in v2 reports.

## 7. Controlled experiment design

Only `learning_rate` varies:

| Run ID | Learning rate |
|---|---:|
| `p3v2_lr_2e-5` | `2e-5` |
| `p3v2_lr_3e-5` | `3e-5` |
| `p3v2_lr_5e-5` | `5e-5` |

No candidate may be added, removed, or modified after any result is observed. Every run starts from a fresh generic pretrained checkpoint; no model, optimizer, scheduler, or global training state is reused. Execution follows table order, but ranking is order-independent.

## 8. Fixed variables

| Variable | Locked v2 value | Evidence/status |
|---|---|---|
| Model | `distilbert-base-uncased` + sequence-classification head, 2 labels | v1 Phase 7 |
| Tokenizer | `distilbert-base-uncased` | v1 Phase 6/7 |
| Labels | 0 NEGATIVE, 1 POSITIVE | v1 contract |
| Max length | 80, truncation enabled, dynamic padding | v1 Phase 5/6 |
| Seed/data seed/split seed | 42/42/42 | v1 config; split seed newly locked |
| Train/eval batch | 16/32 | Phase 8 and checkpoint args |
| Optimizer | Transformers `ADAMW_TORCH_FUSED` | v1 `training_args.bin` |
| Scheduler | linear, zero warmup steps | v1 `training_args.bin` |
| Weight decay | 0.01 | Phase 8/artifact |
| Gradient clipping | max norm 1.0 | v1 `training_args.bin` |
| Metrics | binary Accuracy, Precision, Recall, F1; positive class 1; `zero_division=0` | Phase 8 implementation |
| Evaluation/log/save | each epoch | Phase 8/checkpoint args |
| Primary selection metric | Validation loss, lower is better | v1 and v2 protocol |
| `load_best_model_at_end` | true | Phase 8/checkpoint args |
| Checkpoint retention | save every epoch; best checkpoint must remain available | separate v2 directory |
| Reporting | TensorBoard | v2 monitoring requirement; v1 used no reporter |

If `ADAMW_TORCH_FUSED` is unavailable on the execution device, Part 2.2 must record a compatibility blocker or seek approval for a predeclared cross-device AdamW implementation. It must not silently give candidates different optimizers.

## 9. Epoch and Early Stopping protocol

- `max_epochs = 10`
- `early_stopping_patience = 2`
- `early_stopping_threshold = 1e-6`
- monitored metric: Validation loss
- direction: lower is better
- evaluation/save interval: epoch
- best epoch: epoch with minimum Validation loss, not the final epoch by default

Patience 2 permits two consecutive non-improving epoch evaluations while avoiding unnecessary execution to epoch 10. The threshold aligns improvement and tie handling. Every run records `stopped_epoch`, `best_epoch`, `best_val_loss`, and `best_checkpoint`.

## 10. Deterministic ranking rule

Use absolute tolerance `1e-6` for loss, F1, and Accuracy comparisons:

1. Lowest best Validation loss.
2. If loss differs by at most `1e-6`, higher Validation F1.
3. If F1 differs by at most `1e-6`, higher Validation Accuracy.
4. If Accuracy differs by at most `1e-6`, earlier best epoch.
5. If still tied, lexicographically smaller deterministic `run_id` only for deterministic serialization.

The same hierarchy resolves tied epochs within a run. Test/holdout fields are prohibited in candidate records. The rule and tolerance are immutable after any v2 run begins.

## 11. TensorBoard contract

Required scalar series: Train loss, Validation loss, Validation Accuracy, Validation F1, learning rate, epoch, and global step.

```text
runs/practice_3_v2/
├── p3v2_lr_2e-5/
├── p3v2_lr_3e-5/
└── p3v2_lr_5e-5/
```

Each run writes only to its directory. No synthetic/fabricated event file is allowed. Registry entries use relative log paths.

## 12. Experiment registry schema

Required fields:

```text
run_id
protocol_version
learning_rate
seed
max_epochs
early_stopping_patience
early_stopping_threshold
stopped_epoch
best_epoch
best_val_loss
best_val_accuracy
best_val_precision
best_val_recall
best_val_f1
best_checkpoint
runtime_seconds
tensorboard_log_dir
config_snapshot
config_hash
dataset_fingerprint
split_fingerprint
status
```

Allowed lifecycle: `PLANNED → RUNNING → COMPLETED` or `FAILED`. A completed run with matching config/artifact hashes is loaded, not retrained. Duplicate `run_id` with another config hash is a hard failure.

## 13. Per-run config snapshot

Each run stores immutable `run_config.json` before training:

```text
run_id
protocol_version
model
tokenizer
learning_rate
max_epochs
early_stopping_patience
early_stopping_threshold
batch_size
eval_batch_size
max_length
seed
data_seed
optimizer
scheduler
weight_decay
gradient_clipping
evaluation_strategy
logging_strategy
save_strategy
selection_metric
greater_is_better
ranking_tolerance
dataset_fingerprint
split_fingerprint
library_versions
```

Serialize canonical UTF-8 JSON with sorted keys and compute SHA-256. The snapshot becomes immutable when a run enters `RUNNING`.

## 14. Winner-lock contract

Part 2.1 creates no winner. After all three runs complete, Part 3 may atomically create `winner_manifest.json` containing:

```text
protocol_version
run_id
learning_rate
best_epoch
checkpoint
validation_metrics
config_hash
checkpoint_hash
dataset_fingerprint
split_fingerprint
selection_rule
ranking_tolerance
locked_at
```

The manifest is valid only if its run is the deterministic registry winner, all hashes/files verify, and no holdout access has occurred. Once locked, checkpoint/config/ranking cannot change. A post-lock modification invalidates holdout permission rather than permitting reranking.

## 15. Holdout access contract

Initial state:

```text
winner_manifest = absent
holdout_access_allowed = false
holdout_evaluation_count = 0
```

Access is allowed only when the winner manifest and referenced hashes validate. The final evaluator atomically claims a one-time attempt before loading holdout records. Successful or failed model evaluation consumes the attempt and cannot be reset. Notebook Run All loads frozen v2 artifacts and never reevaluates holdout.

The v2 counter is independent of v1 `test_evaluation_count`. v1 Test artifacts are never read by v2 ranking or final evaluation.

## 16. Artifact namespace and flow

```text
docs/result/practice_3_v2/
├── protocol_manifest.json
├── dataset_split_manifest.json
├── holdout_access_state.json
├── experiment_registry.json
├── experiments/<run_id>/run_config.json
├── experiments/<run_id>/...training artifacts...
└── winner_manifest.json                 # absent until winner lock

runs/practice_3_v2/<run_id>/             # real TensorBoard events only
```

Historical v1 locations are read-only and never reused for v2 writes.

## 17. Part 2.2 expected files

Expected new implementation files:

- `processing_own_phase/experiment_protocol_v2.py`
- `processing_own_phase/dataset_protocol_v2.py`
- `processing_own_phase/experiment_registry_v2.py`
- `processing_own_phase/holdout_guard_v2.py`
- synthetic/static verification artifacts under `docs/result/practice_3_v2/`;
- a Part 2.2 plan/log and compact notebook protocol presentation only if explicitly approved.

Part 2.2 builds and synthetically verifies infrastructure only. Trainer construction/training remains Part 3.

## 18. Files not to change

- all v1 artifacts, checkpoints, figures, Test predictions/evaluation, saved package, and summaries;
- Phase 0–15 methodology/source unless a separate approved compatibility fix is required;
- v1 notebook outputs in Part 2.1;
- `docs/current_flow/` finalization;
- Practice 1, Practice 2, `COURSE_WORK/`, and `practice_3 copy.ipynb`.

## 19. Validation strategy

### Part 2.1

- read-only cross-check of v1 manifests/configuration/checkpoint/package hashes;
- document existence/schema checks;
- Git scope verification;
- no dataset/model loading or evaluation.

### Part 2.2 infrastructure

- synthetic ranking/tolerance/tie records;
- synthetic registry, winner-lock, and one-time holdout state transitions;
- deterministic split tests on synthetic balanced IDs, including duplicate/leakage failure fixtures;
- config canonicalization/hash/read-back tests;
- namespace and overwrite-protection tests;
- static import/compile checks;
- verify `holdout_evaluation_count = 0`, no Trainer/training, and no Test/holdout provider call.

## 20. Technical risk review

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| Reusing viewed v1 Test | Critical | Phase 11 count is 1 | Exclude official Test from v2 |
| Accidental holdout access | Critical | Holdout must remain unseen until lock | State machine, atomic attempt receipt, provider behind guard |
| Holdout not globally novel | Major | Source records were v1 Train/Validation | State limitation; external data required for globally novel evidence |
| Cross-split leakage | Critical | New repartition required | Stable IDs, duplicate grouping, overlap assertions, split hashes |
| Global-state contamination | Major | Three fresh sequential runs | reset seeds, fresh model/Trainer/optimizer/scheduler, release device state |
| Artifact overwrite | Critical | v1 evidence is authoritative | separate `practice_3_v2` namespace and exclusive creation |
| Duplicate run ID | Major | deterministic IDs predeclared | uniqueness plus config-hash guard |
| Inconsistent config | Critical | only LR may vary | canonical snapshots and cross-run fixed-field comparison |
| Retraining completed run | Major | notebook may Run All | completed-run artifact guard |
| Ranking-rule drift | Critical | post-result change biases selection | hash protocol/rule before runs |
| v1/v2 artifact confusion | Major | both remain in repository | version fields, separate paths/counters, historical labels |
| Device-dependent optimizer | Major | v1 used fused AdamW | preflight compatibility; no silent substitution |

## 21. Definition of done

Part 2.1 is complete when the v1 state is evidence-backed; v1 Test is historical; source, counts, deterministic strategy, and limitations of the new holdout are fixed; LR candidates/fixed variables/Early Stopping/ranking are pre-registered; TensorBoard, registry, config, winner-lock, and holdout-guard contracts are specified; and no training/evaluation or historical artifact mutation occurred.

Part 2.2 may begin only after review and approval. Part 2.1 creates no infrastructure, split, registry, winner, or holdout artifact.
