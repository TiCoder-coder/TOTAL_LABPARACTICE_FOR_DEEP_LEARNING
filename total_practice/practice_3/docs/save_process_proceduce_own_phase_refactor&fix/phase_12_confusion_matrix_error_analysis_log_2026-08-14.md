# Phase 12 Confusion Matrix & Error Analysis Process Log — 2026-08-14

## Scope

Implemented only Phase 12 using the frozen Phase 11 JSON artifacts. No model,
checkpoint, dataset/Test provider, Trainer, prediction/evaluation path,
training/backward/update operation or Phase 13 implementation was used.

## Frozen Inputs

- `docs/result/phase_11_evaluation/test_predictions.json`
- `docs/result/phase_11_evaluation/test_evaluation.json`
- `docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json`

The manifest was `FINAL_TEST_COMPLETE`; prediction size/SHA-256 matched the
manifest, prediction/sample count was 1,066, IDs/indices were complete and
unique, and `test_evaluation_count=1`.

## Confusion Matrix

| | Predicted NEGATIVE | Predicted POSITIVE |
|---|---:|---:|
| Actual NEGATIVE | TN = 451 | FP = 82 |
| Actual POSITIVE | FN = 83 | TP = 450 |

- Total: 1,066
- Correct: 901
- Incorrect: 165
- Actual NEGATIVE conditional accuracy: 451/533 = 0.8461538462
- Actual NEGATIVE error rate: 82/533 = 0.1538461538
- Actual POSITIVE conditional accuracy: 450/533 = 0.8442776735
- Actual POSITIVE error rate: 83/533 = 0.1557223265

Every count/rate/row-total check passed. The PNG reopened successfully with
pixel shape `[1050,1309,4]`.

## Derived Metric Verification

| Metric | Phase 11 | Confusion-derived | Delta | Tolerance |
|---|---:|---:|---:|---:|
| Accuracy | 0.8452157598 | 0.8452157598 | 0 | 1e-12 |
| Precision | 0.8458646617 | 0.8458646617 | 0 | 1e-12 |
| Recall | 0.8442776735 | 0.8442776735 | 0 | 1e-12 |
| F1 | 0.8450704225 | 0.8450704225 | 0 | 1e-12 |

Loss was not derived from labels and was not recomputed.

## Error Analysis

All 165 incorrect predictions were preserved and sorted by confidence
descending, then sample index ascending:

- `NEGATIVE_TO_POSITIVE`: 82
- `POSITIVE_TO_NEGATIVE`: 83

Representative selection used the first five ranked records from each
direction, producing 10 records without manual cherry-picking. All selection,
direction-count, order and ID-uniqueness assertions passed.

For all predictions and analysis groups, the module derived confidence,
character length, whitespace-token word length and fixed lexical flags for
negation, contrast, question marks and exclamation marks. The interpretation
states these are descriptive co-occurrences and not causal explanations.

## Headless Plot Fix

The first standalone dry run aborted because the default macOS GUI Matplotlib
backend was unsuitable for a headless process. No Phase 12 artifact was
created and no Phase 11 file was changed. The error and fix plan were saved
under the workflow directories. Setting the module backend to `Agg` resolved
the issue; artifact-only verification and notebook Run All then passed.

## Artifacts

- `docs/result/phase_12_confusion_matrix.png`
- `docs/result/phase_12_confusion_matrix.json`
- `docs/result/phase_12_error_samples.json`
- `docs/result/phase_12_error_analysis.json`

## Notebook Run All

The notebook ran from Phase 0 through Phase 12 using repository `.venv`
Python 3.11.14. All 13 code cells have continuous execution counts `1..13`
and no error output.

- Phase 9 guard loaded verified training artifacts without retraining.
- Phase 11 guard action was
  `loaded_verified_phase_11_artifacts_no_test_reevaluation`.
- Phase 11 JSON timestamps/sizes remained unchanged.
- Test evaluation count remained exactly one.
- Phase 12 displayed the confusion figure, cells/rates, metric comparisons,
  class-level analysis, error directions, representative errors and cautious
  descriptive evidence.

## Isolation Flags

- `model_loaded=false`
- `checkpoint_loaded=false`
- `dataset_loaded=false`
- `test_provider_called=false`
- `test_evaluated=false`
- `prediction_performed=false`
- `training_performed=false`
- `checkpoint_changed=false`
- `phase_13_started=false`
- `test_evaluation_count=1`

## Result

Phase 12 implementation, frozen-source verification, confusion matrix,
derived metrics, deterministic error analysis, artifacts and notebook Run All
PASS. Phase 12 is ready for Phase 13 planning only; Phase 13 was not started.
