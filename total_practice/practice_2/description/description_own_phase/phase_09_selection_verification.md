# Phase 9 — Model Selection and Checkpoint Verification

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 18–21**
- Selection source: [experiment.py](../../processing_own_phase/experiment.py)
- Verification source: [final_evaluate.py](../../processing_own_phase/final_evaluate.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

This phase converts E1/E2 results into an auditable model decision:

1. read genuine histories and run summaries;
2. create a comparison table using Validation metrics only;
3. select the best experiment according to the declared rule;
4. resolve the best checkpoint of the selected experiment;
5. reconstruct a new model and reload the checkpoint;
6. reevaluate the complete Validation set;
7. allow progression to Final Test only if verification passes.

## 2. E1/E2 comparison table

Cell 19 reports:

- `experiment`;
- `strategy`;
- `epochs_trained`;
- `best_epoch`;
- `best_val_accuracy`;
- `best_val_loss`;
- Validation `macro_f1`;
- `training_time`.

The table must not contain Test Accuracy or Test Loss.

Sources:

- [controlled_experiment_comparison.csv](../../outputs/controlled_experiment_comparison.csv)
- [controlled_experiment_comparison.png](../../reports/controlled_experiment_comparison.png)

## 3. Selection rule and result

`selected_experiment` is computed from the best Validation Accuracy and stored in a machine-readable artifact. It is not entered manually in the notebook.

Current controlled result:

| Item | Value |
|---|---|
| E1 strategy | `head_only` |
| E2 strategy | `partial_finetune` |
| Selected experiment | `E2_resnet18_partial` |
| Best epoch | 4 |
| Recorded best Validation Accuracy | 0.8972 |
| Selection source | `validation_only` |
| Test data used in selection | `false` |

Source:

- [controlled_experiment_selection.json](../../outputs/controlled_experiment_selection.json)

These values describe the current artifact. Notebook logic reads the file and does not hardcode them.

## 4. Selected checkpoint

Official checkpoint:

- [E2 best.pt](../../runs/E2_resnet18_partial_6c5d4ec5/best.pt)

The workflow must not use:

- the E1 checkpoint;
- the final in-memory model state from epoch 5;
- a checkpoint selected after inspecting Test;
- another checkpoint merely because it appears to have a higher Test score.

## 5. Validation Verification protocol

```text
Read the selection record
       ↓
Rebuild the model with the recorded architecture and strategy
       ↓
Load selected best.pt
       ↓
Evaluate the deterministic Validation loader
       ↓
Compare reloaded and recorded Validation Accuracy
       ↓
PASS → Test evaluation is allowed
FAIL → raise an exception and stop
```

Current official verification:

- recorded Validation Accuracy: `0.8972`;
- reloaded Validation Accuracy: `0.8972`;
- absolute delta: `0.0`;
- status: **PASS**.

Source:

- [summary.json](../../outputs/summary.json)

The verification fields are stored directly in `summary.json` as `validation_verification`, `recorded_val_accuracy`, `verified_val_accuracy`, and `validation_accuracy_delta`.

## 6. Why verification is necessary

A correct selection CSV does not prove that the checkpoint file is correct. Failures can result from:

- a wrong or stale checkpoint path;
- checkpoint overwrite;
- architecture or strategy mismatch during reload;
- a different Validation split or transform;
- metadata that no longer matches the model weights.

Validation reevaluation turns the checkpoint into a verifiable artifact. A mismatch beyond the declared tolerance must stop the pipeline before Test is accessed.

## 7. Cell 21 responsibilities

Cell 21 may:

- read selection and checkpoint metadata;
- run or display Validation Verification;
- show PASS/FAIL, expected accuracy, reloaded accuracy, and delta;
- raise an exception when verification fails.

Cell 21 must not:

- evaluate Test;
- enter metrics manually;
- replace the selected checkpoint;
- reselect the experiment using Test performance.

## 8. Audit checks

- The selected experiment originates from Validation-only comparison.
- The selection artifact contains no Test metrics.
- `best_epoch` lies within the trained epoch range.
- The checkpoint path exists.
- Reloaded Validation support equals 5,000.
- Accuracy delta lies within tolerance.
- The checkpoint training log records `Using device: mps`.

## 9. Suggested presentation script

> E2 wins the Validation-only comparison, with its best result at epoch 4. I do not trust the metadata alone: the pipeline rebuilds the model, reloads `best.pt`, and evaluates all 5,000 Validation samples again. It reproduces 89.72% exactly with zero delta, so verification passes and Test access can proceed.

## 10. Transition to the next phase

[Phase 10](phase_10_final_evaluation.md) evaluates the locked checkpoint on the official Test set exactly once and validates all final artifacts.
