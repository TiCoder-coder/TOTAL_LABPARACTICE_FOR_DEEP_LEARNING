# Phase 12 Confusion Matrix & Error Analysis Plan — 2026-08-14

## A. Analysis Flow

### Objective

Derive the complete binary confusion matrix, class-level behavior and a
deterministic error analysis exclusively from the frozen Phase 11 prediction
artifact. Phase 12 is post-evaluation artifact analysis; it must not load a
model/checkpoint, dataset pipeline or Test provider and must not perform any
new prediction/evaluation.

Planned architecture:

```text
Phase 11 FINAL_TEST_COMPLETE manifest
        +
frozen test_predictions.json
        +
frozen test_evaluation.json
        ↓
validate hashes/schema/counts
        ↓
processing_own_phase/phase_12_error_analysis.py
        ↓
confusion counts/rates + derived metrics
        ↓
all deterministic errors + representative subset
        ↓
figures/JSON artifacts
        ↓
practice_3.ipynb call + display + cautious interpretation
```

### Verified frozen inputs

Phase 12 will read only:

```text
docs/result/phase_11_evaluation/test_predictions.json
docs/result/phase_11_evaluation/test_evaluation.json
docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json
```

Before analysis, require:

- manifest status `FINAL_TEST_COMPLETE`;
- `test_evaluation_count == 1`;
- `test_used_for_selection == false`;
- `training_performed == false`;
- `checkpoint_changed_after_test == false`;
- `phase_12_started == false` in the Phase 11 handoff state;
- prediction artifact size and SHA-256 match the frozen Phase 11 manifest;
- exactly 1,066 prediction records with the expected schema;
- unique and ordered `sample_index` values `0..1065`;
- 1,066 unique `sample_id` values;
- binary true/predicted labels and matching label names;
- finite probabilities/confidence in `[0,1]` and internally consistent
  `correct` flags.

Executed Phase 11 reference evidence, to verify rather than hard-code as the
calculation source:

```text
correct = 901
incorrect = 165
TN = 451, FP = 82, FN = 83, TP = 450
```

The module must compute these values anew from `test_predictions.json`. It
must not read raw Test data separately or call any Phase 11 evaluation entry
point.

## B. Confusion Matrix Design

Use the conventional binary layout:

| | Predicted NEGATIVE | Predicted POSITIVE |
|---|---:|---:|
| Actual NEGATIVE | TN | FP |
| Actual POSITIVE | FN | TP |

For each cell, calculate and store:

- absolute count;
- rate over all 1,066 Test records;
- row-conditional rate within its actual class.

Required totals:

```text
TN + FP + FN + TP = 1066
TN + TP = correct count
FP + FN = incorrect count
correct + incorrect = 1066
```

### Figure

Create:

```text
docs/result/phase_12_confusion_matrix.png
```

The figure should be a readable 2×2 heatmap with:

- explicit Actual/Predicted axes;
- full class names `NEGATIVE` and `POSITIVE`;
- each cell annotated with count and row-normalized percentage;
- a color scale that does not obscure text;
- title stating this is frozen final-Test prediction analysis;
- deterministic layout/size/DPI suitable for notebook display.

The JSON should additionally retain overall percentages even if the figure
shows row-normalized percentages. No model metric or prediction may be
recomputed by model inference.

### Derived metrics

Derive from confusion counts:

```text
Accuracy  = (TP + TN) / N
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Use safe zero-denominator handling. Compare Accuracy, Precision, Recall and F1
against frozen `test_evaluation.json` with absolute tolerance `1e-12`. Loss is
not derivable from label predictions and must not be recreated or compared as
a confusion-derived metric.

## C. Error Selection Strategy

### Complete error set

Filter every record where:

```text
true_label != predicted_label
```

Verify this is identical to `correct == false`. Preserve all incorrect records
in the analysis artifact, enriched only with deterministic derived fields.

Two error directions must be handled independently:

- `NEGATIVE_TO_POSITIVE` (`true=0`, `predicted=1`, false positives);
- `POSITIVE_TO_NEGATIVE` (`true=1`, `predicted=0`, false negatives).

### Deterministic ranking

For the global error list and within each direction, sort by:

```text
1. confidence descending
2. sample_index ascending
```

No manual reordering or subjective sample replacement is allowed.

### Representative errors

Select exactly the first five ranked errors from each direction, producing ten
representative high-confidence errors when both directions have at least five
records. If a direction has fewer than five, include all available records and
record the shortfall rather than substituting manually chosen examples.

Each representative record should contain:

- deterministic rank and direction rank;
- sample index/ID;
- full text;
- true/predicted numeric and named labels;
- confidence and both class probabilities;
- correctness flag;
- character count and whitespace-token word count;
- transparent lexical-evidence flags described below;
- `selection_reason = top_confidence_within_error_direction`.

This policy provides high-confidence mistakes from both directions and avoids
cherry-picking.

## D. Artifact Design

Core implementation is planned for:

```text
processing_own_phase/phase_12_error_analysis.py
```

Planned result artifacts:

| Artifact | Contents / purpose |
|---|---|
| `docs/result/phase_12_confusion_matrix.png` | Presentation-ready count/row-rate heatmap |
| `docs/result/phase_12_confusion_matrix.json` | TN/FP/FN/TP, all rates, totals, derived metrics and Phase 11 comparisons |
| `docs/result/phase_12_error_samples.json` | All ranked errors plus the deterministic 5+5 representative subset and selection policy |
| `docs/result/phase_12_error_analysis.json` | Error-direction/class summaries, length/confidence evidence, lexical flags, validation checks and isolation flags |

All JSON files must be written under `docs/result/`, parsed back, and checked
against the in-memory result. The PNG must exist, be non-empty and decodable.

### Error characteristic analysis

For every frozen prediction, calculate from its existing text only:

- character length;
- whitespace-token word count;
- confidence;
- true/predicted label and error direction.

Summarize count, mean, median, minimum, maximum, P25, P75 and P90 for length
and confidence across:

- correct vs incorrect predictions;
- `NEGATIVE_TO_POSITIVE` vs `POSITIVE_TO_NEGATIVE` errors;
- actual NEGATIVE vs actual POSITIVE classes.

Optional text flags must use fixed, reviewable rules rather than subjective
manual labels. Planned flags:

- negation cue: case-insensitive tokens/patterns such as `not`, `no`, `never`,
  `n't`;
- contrast cue: `but`, `however`, `although`, `though`, `yet`, `while`;
- question mark present;
- exclamation mark present.

For each flag, report counts/rates among correct and incorrect predictions and
by error direction. These are descriptive co-occurrences only. The analysis
must not claim a cue caused an error, automatically label sarcasm/ambiguity, or
generalize beyond the frozen 1,066-sample Test artifact.

The Phase 12 analysis artifact should explicitly record:

```text
source = frozen_phase_11_predictions_only
model_loaded = false
dataset_loaded = false
test_provider_called = false
test_evaluated = false
training_performed = false
checkpoint_changed = false
phase_13_started = false
```

## E. Notebook Presentation

Append Phase 12 after the completed Phase 11 section. The notebook will only:

1. import the Phase 12 artifact-analysis function;
2. call it without passing a model, tokenizer, dataset or Test split;
3. assert the Phase 12 status and isolation flags;
4. display the confusion matrix figure;
5. display a table with TN, FP, FN, TP, overall rates and row rates;
6. display Phase 11 metrics beside confusion-derived Accuracy, Precision,
   Recall and F1 with deltas/pass status;
7. display class-level conditional accuracy/error-rate tables;
8. display counts and summaries for both error directions;
9. display the deterministic representative 5+5 error table, including text,
   confidence, labels, length and lexical flags;
10. present cautious, artifact-derived interpretation and artifact paths;
11. state explicitly that Test was not reevaluated and Phase 13 was not
    started.

The notebook must not implement confusion calculations, ranking logic,
feature rules or metric derivation itself. It must not call
`run_or_load_phase_11`, model `predict`, `evaluate`, or any Test provider from
the Phase 12 cell.

## F. Verification Checks

Phase 12 must verify all of the following before PASS:

### Frozen-source integrity

- Phase 11 manifest is `FINAL_TEST_COMPLETE`;
- Test evaluation count remains exactly 1;
- prediction artifact hash/size matches the Phase 11 manifest;
- prediction count is exactly 1,066;
- sample indices are complete, ordered and unique;
- sample IDs are unique;
- required fields exist on every record;
- labels/predictions are subsets of `{0,1}`;
- label names match the numeric mapping;
- probabilities/confidence are finite, in range and consistent;
- stored `correct` equals `(true_label == predicted_label)` for every record.

### Confusion and metrics

- confusion sum equals 1,066;
- correct plus incorrect equals 1,066;
- matrix diagonal equals correct count;
- off-diagonal equals incorrect count;
- actual-class row totals equal artifact true-label counts;
- predicted-class column totals equal artifact predicted-label counts;
- all counts are non-negative integers;
- all rates are finite and within `[0,1]`;
- row-normalized rates sum to 1 within `1e-12` for non-empty rows;
- derived Accuracy, Precision, Recall and F1 match Phase 11 within `1e-12`.

### Error analysis

- all and only incorrect records appear in the error list;
- error-direction counts equal FP and FN respectively;
- sorted order follows confidence descending/sample index ascending;
- representative subset follows the documented first-five-per-direction rule;
- no representative ID is duplicated;
- length/confidence summaries are finite and computed from frozen fields/text;
- lexical flags follow fixed rules and do not mutate text.

### Prohibited actions

- no model/checkpoint is loaded;
- no dataset/Test provider is loaded or called;
- no `Trainer`, prediction or evaluation function is created/called;
- no training/backward/optimizer/scheduler step occurs;
- selected checkpoint/fingerprint is not changed;
- Phase 11 artifact hashes and `test_evaluation_count=1` remain unchanged;
- Phase 13 is not started;
- notebook runs through Phase 12 with Phase 9/11 guards and no regression.

## G. Completion Criteria

Phase 12 is complete only when:

1. the approved plan is implemented in
   `processing_own_phase/phase_12_error_analysis.py`;
2. only frozen Phase 11 JSON artifacts are read;
3. all 1,066 predictions pass schema, identity and integrity checks;
4. TN/FP/FN/TP counts and overall/row-normalized rates are derived and sum
   correctly;
5. confusion-derived Accuracy, Precision, Recall and F1 reproduce Phase 11
   within `1e-12`;
6. the confusion-matrix PNG and JSON pass read-back/decoding checks;
7. every incorrect prediction is preserved in a deterministic ranked list;
8. the representative subset follows the fixed top-five-per-direction rule;
9. class/error direction, confidence, text-length and transparent lexical-cue
   summaries are derived without causal overclaiming;
10. all four planned artifacts exist and pass integrity/read-back checks;
11. notebook calls/displays Phase 12 without embedded core logic;
12. notebook runs continuously through Phase 12 without Test reevaluation or
    regression;
13. Phase 11 Test count remains exactly one and its frozen artifacts are not
    modified;
14. no model/dataset loading, training, prediction, evaluation, checkpoint
    change or Phase 13 work occurs;
15. the Phase 12 process log is created.

Passing Phase 12 authorizes Phase 13 planning only. It does not authorize
automatic Phase 13 implementation.

## H. Out of Scope

Phase 12 must not:

- load the DistilBERT model or any checkpoint;
- load the Rotten Tomatoes dataset or tokenized splits;
- call Phase 11 evaluation, Test provider, `Trainer`, `predict` or `evaluate`;
- rerun or alter the one-time Final Test evaluation;
- train, fine-tune, run backward or update optimizer/scheduler state;
- select, rerank, replace, copy or delete checkpoints;
- change tokenizer, preprocessing, split, label mapping or methodology;
- manually cherry-pick, reorder or replace representative errors;
- infer sarcasm, ambiguity, sentiment cause or linguistic failure category
  without explicit artifact evidence;
- implement new-sentence/custom inference (Phase 13);
- save/reload a reusable model package (Phase 14);
- write the final summary (Phase 15);
- implement or begin Phase 13 or any later phase;
- perform any new model evaluation.

This planning turn creates only:

```text
docs/plan-doc/plan_before_process/
phase_12_confusion_matrix_error_analysis_plan_2026-08-14.md
```

After separate implementation approval, expected additions/modifications are:

```text
processing_own_phase/phase_12_error_analysis.py
notebook_practice_3/practice_3.ipynb
docs/result/phase_12_confusion_matrix.png
docs/result/phase_12_confusion_matrix.json
docs/result/phase_12_error_samples.json
docs/result/phase_12_error_analysis.json
docs/save_process_proceduce_own_phase_refactor&fix/
phase_12_confusion_matrix_error_analysis_log_2026-08-14.md
```
