# Phase 11 Validation & Final Test Evaluation Process Log — 2026-08-14

## Scope

Implemented only Phase 11 under the approved plan. The fixed Phase 9
checkpoint was reloaded, verified on Validation, and then evaluated exactly
once on the official Test split. No training, backward pass, optimizer or
scheduler step, checkpoint ranking/change, confusion matrix, error analysis or
Phase 12 work occurred.

## Authoritative Checkpoint

- Path: `docs/result/phase_09_training/checkpoints/checkpoint-1068`
- Selected epoch: 2
- Selected step: 1068
- Model class: `DistilBertForSequenceClassification`
- Model type: `distilbert`
- Labels: `0 -> NEGATIVE`, `1 -> POSITIVE`
- Weight file size: 267,832,560 bytes
- Weight SHA-256:
  `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`
- Selection-record SHA-256:
  `1ed11b3b832c765dd522d0b9363aa981930290e9d2d904b962cefc1ba0abb52a`

The selected-checkpoint record, Phase 9 manifest, Trainer state and epoch-2
checkpoint record resolved to the same path. Phase 11 did not rerank any
checkpoint.

## Validation Gate

The reloaded checkpoint was evaluated on all 1,066 locked Validation samples
before the Test provider was invoked.

| Metric | Phase 9 epoch 2 | Reloaded | Delta | Tolerance | Status |
|---|---:|---:|---:|---:|---|
| Loss | 0.3904653192 | 0.3904653192 | 0 | 1e-5 | PASS |
| Accuracy | 0.8564727955 | 0.8564727955 | 0 | 1e-6 | PASS |
| Precision | 0.8492647059 | 0.8492647059 | 0 | 1e-6 | PASS |
| Recall | 0.8667917448 | 0.8667917448 | 0 | 1e-6 | PASS |
| F1 | 0.8579387187 | 0.8579387187 | 0 | 1e-6 | PASS |

Validation verification status was `PASS`; only then was the one-time Test
attempt receipt created.

## Final Test Evaluation

One `Trainer.predict` pass evaluated all 1,066 official Test samples and
returned loss, metrics, logits and labels together. `evaluate(Test)` was not
called separately.

| Metric | Final Test |
|---|---:|
| Loss | 0.4485123158 |
| Accuracy | 0.8452157598 |
| Precision | 0.8458646617 |
| Recall | 0.8442776735 |
| F1 | 0.8450704225 |

The logits shape was `[1066,2]`; all logits and metrics were finite. Test was
not used for model/checkpoint selection, and the weight fingerprint was
unchanged after evaluation.

## Prediction Evidence

`test_predictions.json` contains exactly 1,066 ordered records, indexed 0 to
1065 with unique IDs `rotten_tomatoes:test:<index>`. Each record includes raw
text, true/predicted numeric and named labels, negative/positive probability,
predicted-class confidence and correctness. Labels are binary, probabilities
are finite/in range, and each probability pair sums to 1 within `1e-6`.

The artifact is intended as the frozen Phase 12 input; Phase 12 should not
reevaluate Test.

## One-Time Guard Verification

The transactional receipt completed with:

```text
status = FINAL_TEST_COMPLETE
test_evaluation_count = 1
```

After the completed artifact set existed, the guarded entry point was called
with a Test provider that deliberately raises if invoked. The call returned:

```text
guard_action = loaded_verified_phase_11_artifacts_no_test_reevaluation
test_evaluation_count = 1
prediction_count = 1066
```

The raising provider was not invoked. The guard also revalidates artifact
hashes, the current model-weight SHA-256 and selection-record SHA-256.

## Notebook Run All

The notebook ran successfully from Phase 0 through Phase 11 using the
repository `.venv` Python 3.11.14 and the repository kernelspec. Code-cell
execution counts are continuous from 1 through 12; no cell has an error
output. Phase 9 reported its existing no-retraining guard, and the checkpoint,
Phase 9 manifest and selected-record timestamps/sizes remained unchanged.

The Phase 11 cell displays:

- checkpoint identity and fingerprint;
- Validation reproducibility table;
- Final Test metrics;
- Validation-versus-Test comparison;
- prediction artifact/count and one-time policy flags.

The MPS DataLoader emitted a benign warning that pinned memory is unsupported;
evaluation completed normally and all required checks passed.

## Artifacts

- `docs/result/phase_11_evaluation/checkpoint_verification.json`
- `docs/result/phase_11_evaluation/validation_evaluation.json`
- `docs/result/phase_11_evaluation/test_evaluation.json`
- `docs/result/phase_11_evaluation/test_predictions.json`
- `docs/result/phase_11_evaluation/final_test_attempt_receipt.json`
- `docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json`

## Isolation Flags

- `test_evaluation_count=1`
- `test_used_for_selection=false`
- `training_performed=false`
- `backward_called=false`
- `optimizer_step_performed=false`
- `scheduler_step_performed=false`
- `checkpoint_changed_after_test=false`
- `phase_12_started=false`

## Result

Phase 11 implementation, Validation gate, one-time Final Test evaluation,
prediction artifacts, guard and notebook Run All verification PASS. Phase 11
is ready for Phase 12 planning only; Phase 12 was not started.
