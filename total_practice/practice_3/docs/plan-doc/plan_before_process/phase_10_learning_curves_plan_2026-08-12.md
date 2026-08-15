# Phase 10 Learning Curves & Training Analysis Plan — 2026-08-12

## A. Objective

Read and verify the completed Phase 9 artifacts, reconstruct the three-epoch
training trajectory, create presentation-ready learning curves, and provide a
careful evidence-based generalization analysis.

Phase 10 is artifact-only analysis. It must not create a Trainer, train again,
evaluate a model, access any dataset split, or change the selected checkpoint.

Planned architecture:

```text
Phase 9 JSON artifacts
        ↓
processing_own_phase/phase_10_learning_curves.py
        ↓
validate + join epoch records + plot + analyze
        ↓
Phase 10 PNG/JSON artifacts
        ↓
practice_3.ipynb call + display + interpretation
```

## B. Input Artifacts

Phase 10 will read these existing Phase 9 files directly:

```text
docs/result/phase_09_training/training_history.json
docs/result/phase_09_training/validation_metrics_by_epoch.json
docs/result/phase_09_training/checkpoint_records.json
docs/result/phase_09_training/selected_checkpoint.json
docs/result/phase_09_training/phase_09_training_manifest.json
```

Required input contract:

- manifest status is `PASS`;
- completed epochs equal 3;
- Test access is false;
- training history contains exactly one Train-loss record and one Validation
  record for each epoch/step pair;
- Validation artifact contains exactly epochs 1, 2 and 3;
- checkpoint records map those epochs to steps 534, 1068 and 1602;
- selected checkpoint record identifies epoch 2, step 1068 and
  `checkpoint-1068`;
- the selected checkpoint remains unchanged and is never reloaded or
  reevaluated in Phase 10.

The implementation must load JSON from disk. The notebook must not pass
hard-coded metric arrays.

Executed Phase 9 reference values to verify against the artifacts, not to use
as the plot's data source:

| Epoch | Validation loss | Validation F1 |
|---:|---:|---:|
| 1 | 0.4152307808 | 0.8379513014 |
| 2 | 0.3904653192 | 0.8579387187 |
| 3 | 0.4926925004 | 0.8576675849 |

Reference comparisons should use an explicit small floating-point tolerance
only for validation, while plots retain the original artifact values.

## C. Visualization Design

Core plotting and record transformation will live in the planned module:

```text
processing_own_phase/phase_10_learning_curves.py
```

### Figure 1 — Loss Curves

Output:

```text
docs/result/phase_10_loss_curves.png
```

Design:

- x-axis: epoch 1, 2, 3;
- line 1: epoch-level Train loss from `training_history.json`;
- line 2: Validation loss from
  `validation_metrics_by_epoch.json`;
- markers at all observed epochs;
- vertical line and emphasized point at selected epoch 2;
- annotation containing `Selected: epoch 2 / checkpoint-1068`;
- readable legend, title, axes, integer epoch ticks and light grid;
- no interpolation beyond connecting the three real observed points;
- save with fixed figure size and sufficient DPI for notebook presentation.

Train and Validation losses are different aggregates. The analysis may compare
their direction across epochs but must not claim their absolute gap alone is a
calibrated generalization measure.

### Figure 2 — Validation Metrics

Output:

```text
docs/result/phase_10_validation_metrics.png
```

Design:

- x-axis: epoch 1, 2, 3;
- line 1: Validation Accuracy;
- line 2: Validation F1;
- y-axis constrained to `[0,1]` or a clearly labeled readable subrange that
  does not exaggerate small differences;
- selected epoch 2 emphasized and annotated with `checkpoint-1068`;
- real artifact values shown with compact numeric labels or a companion table;
- consistent selected-epoch color/style with the loss figure.

The required minimum visual evidence is therefore present without combining
loss and score metrics onto a misleading shared y-axis.

## D. Selected Checkpoint Annotation

The annotation must be derived from `selected_checkpoint.json`, not written as
an independent plotting constant.

Before plotting, verify:

```text
selected epoch == 2
selected step == 1068
basename(selected checkpoint) == checkpoint-1068
selected eval_loss matches epoch-2 Validation record
selected eval_f1 matches epoch-2 Validation record
```

The same selected epoch/checkpoint will be marked in both figures and recorded
in the Phase 10 analysis JSON. Phase 10 does not rerank checkpoints or change
the Phase 9 decision.

## E. Generalization and Overfitting Analysis

The planned analysis function will calculate changes directly from artifact
values:

```text
Epoch 1 → 2:
  Train loss decreases
  Validation loss decreases
  Validation Accuracy/F1 improve

Epoch 2 → 3:
  Train loss continues decreasing
  Validation loss increases
  Validation Accuracy and F1 are approximately flat/slightly lower
```

Required interpretation:

- Epoch 1 to 2 indicates improved fit and improved Validation behavior.
- Epoch 2 to 3 shows divergence: Train loss continues down while Validation
  loss rises from about 0.390465 to 0.492693.
- This is evidence that generalization begins to worsen after epoch 2 and is a
  pattern consistent with the onset of overfitting.
- Epoch 2 is therefore consistent with the selected minimum-Validation-loss
  checkpoint.
- The report must not claim severe, definitive or broadly generalizable
  overfitting from only three epochs and one run. It must state that the short
  trajectory is limited evidence.

The analysis JSON should include absolute deltas for Train loss, Validation
loss, Accuracy and F1 across `1→2` and `2→3`, plus boolean conditions used to
support each conclusion. Interpretive text must be generated from those
conditions rather than printed unconditionally.

## F. Artifact Design

Phase 10 will create exactly these main results:

```text
docs/result/phase_10_loss_curves.png
docs/result/phase_10_validation_metrics.png
docs/result/phase_10_learning_curve_analysis.json
```

Minimum JSON contents:

- overall status;
- source artifact paths and successful read-back status;
- three joined epoch records containing epoch, step, Train loss, Validation
  loss, Accuracy and F1;
- artifact-derived selected epoch/checkpoint/metrics;
- expected-reference verification status;
- all finite/range/order checks;
- calculated epoch-to-epoch deltas;
- generalization conditions and cautious interpretation;
- plot artifact paths and existence/non-empty checks;
- explicit `training_performed=false`, `model_loaded=false`,
  `model_evaluated=false`, `dataset_loaded=false`, `test_accessed=false`,
  `checkpoint_changed=false`, and `phase_11_started=false`.

JSON and PNG files must be saved under `docs/result/`, read back or reopened,
and verified. PNG validation should confirm that each image is non-empty and
decodable with the expected dimensions; no visual values may be fabricated.

## G. Notebook Presentation

Append a Phase 10 section after Phase 9. The notebook will only:

1. import Phase 10 public artifact-analysis functions;
2. call the loader/verifier/plot orchestrator;
3. assert Phase 10 status is PASS;
4. display a three-row epoch metrics table;
5. display both generated PNG figures inline;
6. display the selected epoch/checkpoint and calculated deltas;
7. present the cautious artifact-derived generalization interpretation;
8. display the JSON/PNG artifact paths and prohibited-action flags.

No data extraction, plotting algorithm, checkpoint selection or analysis
condition should be duplicated as large notebook logic. The notebook must not
call Phase 9 training functions except its existing guarded Phase 9 cell,
which continues to reuse verified artifacts without retraining.

## H. Validation Checks

Phase 10 implementation must verify:

- every required Phase 9 input artifact exists and parses as JSON;
- Phase 9 manifest is PASS and `test_accessed=false`;
- only full baseline records are analyzed; debug metrics are excluded;
- epochs are exactly `[1,2,3]`, strictly ordered and unique;
- steps are exactly `[534,1068,1602]`, ordered and match checkpoints;
- one Train-loss and one Validation record join for each epoch/step;
- Train loss, Validation loss, Accuracy and F1 are finite;
- Accuracy and F1 are within `[0,1]`;
- artifact values match the supplied reference Validation loss/F1 values
  within the declared verification tolerance;
- selected record is epoch 2 / step 1068 / `checkpoint-1068`;
- selected metrics match the epoch-2 record;
- Train loss decreases from epoch 1→2 and 2→3;
- Validation loss decreases from epoch 1→2 and increases from epoch 2→3;
- two figures exist, are non-empty and can be reopened;
- analysis JSON save/read-back matches the in-memory result;
- no model, Trainer, dataset split or checkpoint is loaded/changed;
- Test is not accessed;
- notebook Run All through Phase 10 uses the Phase 9 no-retraining guard and
  completes without regression;
- Phase 11 is not started.

## I. Completion Criteria

Phase 10 is complete only when:

1. the approved plan is implemented in
   `processing_own_phase/phase_10_learning_curves.py`;
2. all three Phase 9 epochs are reconstructed from real artifacts;
3. the loss figure shows Train and Validation loss with selected epoch 2;
4. the Validation figure shows Accuracy and F1 with selected epoch 2;
5. `checkpoint-1068` is consistently annotated from the selected artifact;
6. the generalization analysis is derived from calculated deltas and uses
   appropriately cautious language;
7. both PNGs and the analysis JSON exist and pass integrity/read-back checks;
8. notebook calls, displays and interprets the saved results without embedding
   core processing logic;
9. notebook runs continuously through Phase 10 without retraining Phase 9;
10. no model evaluation, dataset loading in Phase 10, Test access, checkpoint
    change or Phase 11 implementation occurs;
11. the Phase 10 workflow process log is created.

Passing Phase 10 permits Phase 11 planning only. It does not authorize Test
evaluation or Phase 11 implementation.

## J. Out of Scope

Phase 10 must not:

- call `Trainer.train()` or rerun Phase 9 training;
- construct a Trainer, optimizer or scheduler;
- load or evaluate any model/checkpoint;
- access Train, Validation or Test datasets directly;
- evaluate or predict Test;
- recompute Validation metrics from model predictions;
- rerank, replace, copy or delete Phase 9 checkpoints;
- change the selected epoch/checkpoint;
- change dataset, tokenizer, model, preprocessing or methodology;
- create confusion matrices or class-level evaluation;
- implement final Validation/Test evaluation;
- begin Phase 11 or any later phase;
- hard-code chart values when the Phase 9 artifacts provide them;
- claim definitive overfitting from a three-epoch single run.

This planning turn creates only:

```text
docs/plan-doc/plan_before_process/phase_10_learning_curves_plan_2026-08-12.md
```
