# Phase 09 Fine-Tuning Plan — 2026-08-12

## 1. Objective

Fine-tune the verified generic DistilBERT binary classifier on the official
Rotten Tomatoes Train split, monitor only the official Validation split after
each epoch, save auditable epoch checkpoints, and deterministically select the
best checkpoint for downstream phases.

Phase 9 will first execute a small development/debug run. Only after that run
passes will it start one fresh, full baseline run over all 8,530 Train and
1,066 Validation samples for three epochs. Test remains completely isolated.

This document is planning only. It does not construct a `Trainer` or run
training.

## 2. Verified Inputs and Contracts

Phase 9 must consume the existing verified objects rather than recreate the
pipeline:

| Source | Required input |
|---|---|
| Phase 6 | `tokenized_dataset["train"]`, 8,530 samples |
| Phase 6 | `tokenized_dataset["validation"]`, 1,066 samples |
| Phase 6 | `DataCollatorWithPadding` from the existing tokenizer |
| Phase 6 | `MAX_TOKEN_LENGTH=80`, existing tokenization and labels `{0,1}` |
| Phase 7 | fresh `DistilBertForSequenceClassification` from `build_model()` |
| Phase 7 | `num_labels=2`, all 66,955,010 parameters trainable |
| Phase 8 | `compute_metrics` for Accuracy, Precision, Recall and F1 |
| Phase 8 | approved baseline `TrainingArguments` |
| Phase 8 | deterministic validation-checkpoint ranking policy |

The Test split must not be passed into any Phase 9 function, `Trainer`, debug
run, checkpoint selector or artifact builder.

## 3. Training Flow

```text
Verified Phase 6/7/8 contracts
        ↓
Create isolated debug model + debug Trainer
        ↓
Deterministic Train/Validation subsets
        ↓
One debug epoch
        ↓
Debug pipeline verification
        ↓ PASS only
Discard debug model as a final-run input
        ↓
Reset seed 42 + construct fresh Phase 7 model
        ↓
Create full Trainer with full Train/Validation
        ↓
Train for 3 epochs
        ↓
Validation metrics + checkpoint at every epoch
        ↓
Save Trainer state and normalized training history
        ↓
Build checkpoint records from real Validation logs
        ↓
Rank by eval_loss → eval_f1 → earlier epoch
        ↓
Load and verify selected checkpoint
        ↓
Save Phase 9 manifest/artifacts for Phase 10–12
```

Phase 9 must never silently resume a previous debug or full run. Whether a run
is fresh or resumed must be explicit in its manifest. The baseline described
here is a fresh run.

## 4. Trainer Construction Design

Core implementation will live in:

```text
processing_own_phase/phase_09_fine_tuning.py
```

The planned Trainer factory receives explicit dependencies:

- `model`: a fresh model returned by Phase 7 `build_model()`;
- `train_dataset`: tokenized Train or deterministic Train debug subset;
- `eval_dataset`: tokenized Validation or deterministic Validation debug
  subset;
- `data_collator`: existing Phase 6 `DataCollatorWithPadding`;
- `compute_metrics`: Phase 8 callable;
- `training_args`: Phase 8 baseline or clearly identified debug copy;
- tokenizer/processing interface required by the installed Transformers
  5.14.1 `Trainer` API.

The implementation must inspect the installed `Trainer` signature and use the
supported tokenizer/`processing_class` argument without changing Phase 6
tokenization. No Test dataset parameter will exist in the Phase 9 public
training entry point.

Pre-construction assertions:

- dataset identities are explicitly `train` and `validation`;
- sample counts match the selected mode;
- required columns contain `input_ids`, `attention_mask`, and `labels`;
- labels remain subsets of `{0,1}`;
- model has two labels and all intended parameters remain trainable;
- data collator and Phase 8 metric/config verification are PASS;
- output directories for debug and full runs differ;
- no pre-existing checkpoint is resumed unless a later, separately approved
  recovery plan explicitly requests it.

## 5. Baseline Full-Run Configuration

The final baseline reuses the effective Phase 8 `TrainingArguments` exactly:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Per-device Train batch size | 16 |
| Per-device Validation batch size | 32 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Seed | 42 |
| Data seed | 42 |
| Evaluation strategy | epoch |
| Logging strategy | epoch |
| Save strategy | epoch |
| Load best model at end | `True` |
| Trainer primary best metric | `eval_loss` |
| Greater is better | `False` |
| External reporting | disabled |

The official full datasets are fixed:

- Train: 8,530 samples;
- Validation: 1,066 samples.

This is a controlled baseline, not a claim of optimal hyperparameters.

## 6. Debug Run Before Full Run

The debug run is a pipeline test, not a final experiment and not a source of
final metrics or checkpoint selection.

Planned deterministic subsets follow the master plan:

- first 1,000 indexed Train samples;
- first 200 indexed Validation samples;
- one debug epoch;
- seed and data seed 42;
- same batch sizes, learning rate, weight decay, collator and metrics as the
  baseline;
- separate output directory under
  `docs/result/phase_09_training/debug_checkpoints/`.

Debug PASS requires:

- Trainer construction succeeds;
- at least one real optimizer update occurs;
- training loss is present and finite;
- Validation evaluation completes with finite `eval_loss`, Accuracy,
  Precision, Recall and F1;
- a debug checkpoint and Trainer log history are produced;
- Train/Validation sample counts equal 1,000/200;
- Test access remains false.

After debug PASS:

1. retain only the debug verification summary and paths needed for audit;
2. do not use debug metrics as final metrics;
3. do not continue training the debug model;
4. call the Phase 1 seed setup again with seed 42;
5. construct a new model from the generic Phase 7 checkpoint;
6. create new full-run `TrainingArguments` with a distinct full checkpoint
   directory;
7. start the full baseline from pretrained weights, not from a debug
   checkpoint.

If debug fails, stop before full training, document the error under the
workflow rules, and do not fabricate full-run artifacts.

## 7. Validation Monitoring and Training History

The full Trainer will evaluate and save at each epoch. Phase 9 must preserve
all real log-history records rather than only the final values.

Normalized history should distinguish:

- training records: epoch, step, loss, learning rate;
- Validation records: epoch, step, `eval_loss`, `eval_accuracy`,
  `eval_precision`, `eval_recall`, `eval_f1`, and evaluation runtime fields;
- run-summary records: total training runtime, samples/steps per second and
  final training loss where supplied by Trainer.

All numeric fields must be JSON serializable and finite where applicable.
There must be one Validation record per completed baseline epoch. Records must
be ordered by epoch/step and associated with the checkpoint saved at that
epoch.

## 8. Deterministic Best-Checkpoint Selection

The authoritative Phase 9 policy is:

```text
1. minimum eval_loss
2. if eval_loss ties, maximum eval_f1
3. if both tie, earlier epoch/checkpoint
```

`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, and
`greater_is_better=False` make Trainer load its own minimum-loss checkpoint,
but Trainer does not implement the required F1 tie-break. Therefore Phase 9
must not assume `trainer.state.best_model_checkpoint` is automatically the
authoritative winner.

After training, the implementation will:

1. extract real epoch-level Validation records from Trainer log history;
2. map each record's global step to the actual `checkpoint-<step>` directory;
3. assert the checkpoint directory and required model/state files exist;
4. build records containing only checkpoint, epoch, step, `eval_loss`, and
   `eval_f1`;
5. apply Phase 8 `select_best_checkpoint()` to those records;
6. record both Trainer's primary-loss choice and the policy-selected choice;
7. if they differ because of an exact loss tie, explicitly load the
   policy-selected checkpoint into the Trainer/model before handoff;
8. verify the loaded model path/config and perform a Train- or Validation-only
   finite forward/evaluation sanity check;
9. write the authoritative selected-checkpoint record and selection reason.

Loss ties must be compared deterministically using the executed numeric values
stored in the logs. No tolerance or rounding may be introduced silently; if a
tolerance is later desired it requires an approved methodology change.

Test metrics, predictions and labels are prohibited from checkpoint records
and ranking.

## 9. Artifact Flow

All Phase 9 outputs will be scoped under:

```text
docs/result/phase_09_training/
```

Planned artifacts:

| Artifact | Producer | Consumer/purpose |
|---|---|---|
| `debug_verification.json` | Debug run verifier | Proves pipeline passed before full training |
| `training_configuration.json` | Full-run setup | Records effective baseline, versions, seed and split counts |
| `training_history.json` | Normalized Trainer log history | Phase 10 learning curves; audit |
| `validation_metrics_by_epoch.json` | Validation log extraction | Phase 10 curves and Phase 11 Validation reporting |
| `checkpoint_records.json` | Checkpoint/log mapper | Auditable selection candidates |
| `selected_checkpoint.json` | Deterministic selector | Authoritative model input to Phase 10–14 |
| `trainer_state.json` | Trainer/full-run state copy or canonical path | Recovery and log-history provenance |
| `phase_09_training_manifest.json` | Phase 9 final verifier | Single PASS/FAIL index over configuration, artifacts and prohibited actions |
| `checkpoints/checkpoint-<step>/` | Trainer epoch saves | Reloadable candidate model states |

The manifest must store relative and resolved paths, file existence/read-back
checks, run mode, Python/Transformers/Torch versions, device, start/end time,
runtime, Train/Validation sample counts, completed epochs/steps, selected
checkpoint and the explicit `test_accessed=false` statement.

Artifacts must contain real Trainer outputs. Debug values must be clearly
namespaced and never mixed into full-run history. No Phase 10–12 consumer
should need to call `Trainer.train()` again.

## 10. Notebook Call and Presentation Design

Append Phase 9 after the verified Phase 8 section. Core training, extraction,
ranking, checkpoint loading and serialization remain in
`phase_09_fine_tuning.py`.

The notebook will only:

1. import Phase 9 orchestration/report functions;
2. pass existing Phase 6–8 objects;
3. call the explicit debug-run entry point;
4. assert debug PASS;
5. call the explicit fresh full-run entry point once;
6. assert full-run and checkpoint-selection PASS;
7. display configuration, per-epoch Train/Validation metrics, checkpoint
   candidates, selected checkpoint, runtime and artifact paths;
8. state that Test was not accessed.

The notebook must not contain a manual training loop or duplicate Trainer,
history-processing or checkpoint-selection logic.

Because Run All after Phase 9 would retrain, the implementation must design a
clear execution guard/mode for later presentation runs: reuse verified Phase 9
artifacts and selected checkpoint by default, while an explicit fresh-training
command performs the single authorized training run. The guard must validate
artifact completeness and checkpoint provenance; it must never present stale
or partial artifacts as a new PASS run.

## 11. Test Isolation

Phase 9 uses only:

```text
Train → parameter updates
Validation → epoch monitoring and checkpoint selection
```

Required enforcement:

- public Phase 9 trainer/orchestration functions accept explicit Train and
  Validation datasets rather than an entire `DatasetDict` where practical;
- no `test` argument or Test dataset is supplied to Trainer;
- artifact builders reject keys/records containing Test metrics;
- checkpoint ranking rejects Test fields through the Phase 8 selector;
- no `trainer.predict(test)` or `trainer.evaluate(test)` call exists;
- manifest records `test_accessed=false`;
- Test remains untouched until the separately planned final evaluation phase.

Phase 9 Validation metrics are development metrics, not final Test results.

## 12. Verification and Failure Handling

Before declaring Phase 9 PASS, verify:

- debug run PASS precedes the full run;
- full model is freshly initialized after debug;
- full sample counts are exactly 8,530 Train and 1,066 Validation;
- three full epochs completed;
- effective hyperparameters match Phase 8;
- optimizer and scheduler steps occurred only inside the authorized Trainer
  training runs;
- Train and Validation losses are finite;
- Accuracy, Precision, Recall and F1 are finite and within `[0,1]`;
- exactly three epoch-level Validation records exist for the full baseline;
- expected epoch checkpoints and Trainer state exist;
- checkpoint records map to real directories;
- policy selection matches loss/F1/earlier-epoch ordering;
- the authoritative selected checkpoint is actually loaded and reloadable;
- all JSON artifacts pass read-back validation;
- Test was not accessed;
- notebook presentation matches the saved artifacts;
- no Phase 10 implementation was started.

If an error occurs, follow the mandated workflow:

```text
ERROR → analysis_error document → fix plan → approved fix → rerun → verify
```

Do not silently restart, resume, delete or overwrite a material training run.

## 13. Completion Criteria

Phase 9 is complete only when:

1. the approved plan is implemented in
   `processing_own_phase/phase_09_fine_tuning.py`;
2. the deterministic 1,000/200 debug run passes;
3. a freshly initialized model completes one full 8,530/1,066, three-epoch
   baseline run;
4. Validation is evaluated and a checkpoint is saved at each epoch;
5. real training and Validation history is normalized and saved;
6. every selection candidate maps to a real checkpoint;
7. best checkpoint selection follows minimum loss, maximum F1, then earlier
   epoch exactly;
8. the policy-selected checkpoint is loaded, verified and designated as the
   sole authoritative downstream checkpoint;
9. all planned artifacts and the final manifest exist and pass read-back;
10. notebook calls/reports Phase 9 without embedding core logic;
11. Phase 0–9 evidence has no unresolved regression;
12. Test access is demonstrably false;
13. no Phase 10 work has begun;
14. the Phase 9 process log is complete.

Passing Phase 9 authorizes Phase 10 planning only, not automatic Phase 10
implementation or Test evaluation.

## 14. Out of Scope

This plan and Phase 9 must not:

- access, evaluate or predict the Test split;
- use Test results for model, epoch or checkpoint selection;
- perform hyperparameter search or claim the baseline is optimal;
- alter Phase 6 preprocessing, tokenizer, dataset splits or
  `MAX_TOKEN_LENGTH`;
- freeze layers or change the Phase 7 architecture;
- change the Phase 8 metric definitions or baseline configuration;
- implement learning-curve figures (Phase 10);
- perform final Validation/Test reporting or confusion matrix work
  (Phase 11);
- implement error analysis (Phase 12);
- start any Phase 10–15 implementation;
- train during this planning turn.

## 15. Expected Files

This planning turn creates only:

```text
docs/plan-doc/plan_before_process/phase_09_fine_tuning_plan_2026-08-12.md
```

After separate implementation approval, expected additions/modifications are:

```text
processing_own_phase/phase_09_fine_tuning.py
notebook_practice_3/practice_3.ipynb
docs/result/phase_09_training/debug_verification.json
docs/result/phase_09_training/training_configuration.json
docs/result/phase_09_training/training_history.json
docs/result/phase_09_training/validation_metrics_by_epoch.json
docs/result/phase_09_training/checkpoint_records.json
docs/result/phase_09_training/selected_checkpoint.json
docs/result/phase_09_training/trainer_state.json
docs/result/phase_09_training/phase_09_training_manifest.json
docs/result/phase_09_training/checkpoints/checkpoint-<step>/...
docs/save_process_proceduce_own_phase_refactor&fix/
phase_09_fine_tuning_log_2026-08-12.md
```
