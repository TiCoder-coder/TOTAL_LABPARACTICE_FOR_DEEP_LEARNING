# Phase 11 Validation & Final Test Evaluation Plan — 2026-08-14

## A. Objective

Reload the single authoritative Phase 9 checkpoint, verify that its Validation
evaluation is reproducible, and only after that gate passes perform exactly one
final evaluation pass over the official Test split.

The authoritative model is fixed before Phase 11 starts:

```text
Phase 9 selected_checkpoint.json
        ↓
epoch 2 / step 1068 / checkpoint-1068
        ↓
checkpoint identity + architecture verification
        ↓ PASS only
Validation reload evaluation
        ↓ reproducibility PASS only
create one-time Test-attempt receipt
        ↓
one Test prediction/evaluation pass
        ↓
metrics + prediction-level artifacts + immutable completion manifest
```

Phase 11 is evaluation only. It must not rank checkpoints, train, fine-tune,
run backward, update an optimizer/scheduler, or change the model after Test is
observed. This document is planning only: no model is loaded and Test is not
accessed in this turn.

## B. Verified Inputs

Phase 11 will consume the existing verified contracts rather than reconstruct
or alter them:

| Source | Verified input |
|---|---|
| Phase 6 | Rotten Tomatoes official splits: Train 8,530, Validation 1,066, Test 1,066 |
| Phase 6 | Labels `{0,1}`, `MAX_TOKEN_LENGTH=80`, `input_ids`, `attention_mask`, `labels` |
| Phase 6 | DistilBERT tokenizer and `DataCollatorWithPadding` |
| Phase 7 | `DistilBertForSequenceClassification`, binary label mapping |
| Phase 8 | Accuracy, Precision, Recall and F1 with positive class `1`, `zero_division=0` |
| Phase 9 | PASS manifest, three Validation records, authoritative checkpoint selection |
| Phase 10 | PASS artifact-only learning-curve analysis; selected checkpoint unchanged |

Authoritative selection evidence:

```text
selected epoch:       2
selected step:        1068
selected checkpoint:  docs/result/phase_09_training/checkpoints/checkpoint-1068
selection primary:    minimum eval_loss
tie-break:            maximum eval_f1
Test used to select:  false
```

The Phase 9 epoch-2 Validation reference is:

| Metric | Phase 9 value |
|---|---:|
| Loss | `0.39046531915664673` |
| Accuracy | `0.8564727954971857` |
| Precision | `0.8492647058823529` |
| Recall | `0.8667917448405253` |
| F1 | `0.8579387186629527` |

The implementation must read these values from
`validation_metrics_by_epoch.json`; they are documented here for review, not
hard-coded as evaluation output.

## C. Checkpoint Verification

Core implementation is planned for:

```text
processing_own_phase/phase_11_final_evaluation.py
```

Before any Test provider is invoked, the module must:

1. parse `selected_checkpoint.json`, `phase_09_training_manifest.json`,
   `checkpoint_records.json`, and the Phase 7 verification artifact;
2. require all relevant manifests/status fields to be `PASS` and Phase 9
   `test_accessed=false`;
3. require selected epoch `2.0`, step `1068`, and checkpoint basename
   `checkpoint-1068`;
4. require the same checkpoint path in the selected record, Phase 9 manifest,
   Trainer state, and epoch-2 checkpoint record;
5. require that the checkpoint directory and `model.safetensors` (or the
   supported equivalent weight file) exist and are non-empty;
6. compute and record a SHA-256 fingerprint of the authoritative model weights
   and a fingerprint of the selection record before evaluation;
7. reload only that path with `AutoModelForSequenceClassification` or the
   existing compatible Phase 7 loader, without invoking the generic pretrained
   checkpoint;
8. verify the loaded class/config identifies DistilBERT sequence
   classification, `model_type == "distilbert"`, and
   `model.config.num_labels == 2`;
9. verify `id2label == {0: "NEGATIVE", 1: "POSITIVE"}` and
   `label2id == {"NEGATIVE": 0, "POSITIVE": 1}`;
10. put the model in evaluation mode and confirm no parameter update hooks or
    training path are active.

The serialized checkpoint `config.json` currently derives `num_labels` from
its two-entry label map rather than storing a literal `num_labels` key. The
authoritative check is therefore the reloaded `model.config.num_labels == 2`
plus the exact mappings; absence of a raw JSON key alone is not a failure.

No checkpoint list will be sorted or reranked in Phase 11. A mismatch among
selection sources, fingerprint, epoch or step is a hard failure before
Validation and Test.

Planned evidence:

```text
docs/result/phase_11_evaluation/checkpoint_verification.json
```

## D. Validation Verification

Validation is a mandatory gate before final Test evaluation.

The evaluation function will receive only the locked tokenized Validation
split, the Phase 6 collator/tokenizer contract, and the already verified
authoritative model. It will:

- assert exactly 1,066 samples;
- assert required fields `input_ids`, `attention_mask`, and `labels`;
- assert the complete label set is a subset of `{0,1}`;
- run one deterministic evaluation/prediction pass in `model.eval()` under an
  inference/no-gradient context;
- use evaluation batch size 32 and the same Phase 8 `compute_metrics` contract;
- produce loss, Accuracy, Precision, Recall and F1;
- compare those metrics with the Phase 9 epoch-2 Validation record read from
  disk.

Explicit reproducibility tolerances:

```text
Validation loss:                  absolute tolerance <= 1e-5
Accuracy/Precision/Recall/F1:     absolute tolerance <= 1e-6
relative tolerance:               0 for the pass/fail comparison
```

The artifact must record the expected value, reloaded value, absolute delta,
tolerance and boolean result for every metric. All values must be finite; score
metrics must remain in `[0,1]`.

If any checkpoint, split, numerical or tolerance check fails:

```text
Validation verification FAIL
        ↓
save failure evidence
        ↓
do not invoke the Test provider
        ↓
stop Phase 11
```

Planned evidence:

```text
docs/result/phase_11_evaluation/validation_evaluation.json
```

## E. Final Test Evaluation

Only a successful Validation verification object from the same process and
checkpoint fingerprint can unlock Test evaluation.

To make the ordering explicit, the public orchestrator should accept a lazy
Test provider/callback rather than accessing the Test split before the
Validation gate. The provider will expose the already locked official Phase 6
tokenized Test split and, after the gate, the aligned raw Test text needed for
prediction artifacts. Earlier Phase 4–6 schema/preprocessing checks do not
count as model evaluation; Phase 11 must still record that no Test model
evaluation occurred before this gate.

Before inference, verify:

- official Test count is exactly 1,066;
- raw and tokenized Test counts are identical;
- fields include `input_ids`, `attention_mask`, and `labels`;
- every label belongs to `{0,1}`;
- raw and tokenized rows retain deterministic index alignment.

Perform exactly one model pass over Test. Prefer one evaluation/prediction API
call (for example, `Trainer.predict(..., metric_key_prefix="test")`) that
returns logits, labels, loss and Phase 8 metrics together. Do not call
`evaluate(Test)` and then `predict(Test)`, because that would evaluate Test
twice.

The single pass must produce:

- Test loss;
- Accuracy;
- Precision;
- Recall;
- F1;
- logits/predicted labels needed to serialize prediction-level evidence.

After Test results exist, the model/checkpoint fingerprint must be checked
again and remain identical. No code path may train, tune, select another
checkpoint, alter preprocessing, or rerun Test to improve the report.

## F. Metrics Contract

Phase 11 reuses Phase 8 definitions unchanged:

```text
Accuracy
Precision (binary, positive class = 1)
Recall    (binary, positive class = 1)
F1        (binary, positive class = 1)
zero_division = 0
```

Loss comes from the sequence-classification model with valid integer labels.
For both Validation and Test:

- loss must be finite and non-negative;
- Accuracy, Precision, Recall and F1 must be finite and within `[0,1]`;
- prediction and label counts must equal the evaluated split size;
- predicted labels must be subsets of `{0,1}`.

Phase 11 will not add a confusion matrix or qualitative error categories;
those are Phase 12 responsibilities.

## G. Prediction Artifact Design

Planned primary prediction artifact:

```text
docs/result/phase_11_evaluation/test_predictions.json
```

Each of exactly 1,066 ordered records should contain:

```json
{
  "sample_index": 0,
  "sample_id": "rotten_tomatoes:test:0",
  "text": "raw official Test text",
  "true_label": 0,
  "true_label_name": "NEGATIVE",
  "predicted_label": 1,
  "predicted_label_name": "POSITIVE",
  "negative_probability": 0.0,
  "positive_probability": 0.0,
  "confidence": 0.0,
  "correct": false
}
```

Probabilities must be computed from the single-pass logits using a numerically
stable softmax. Verification must assert both probabilities are finite and in
`[0,1]`, their sum is within `1e-6` of 1, confidence equals the predicted-class
probability, and no sample/index is missing or duplicated.

Including raw text is appropriate because Phase 12 needs deterministic error
analysis without reevaluating the model or touching Test again. Phase 12 must
read this frozen artifact.

## H. Test One-Time / Guard Policy

Phase 11 needs a guard analogous to Phase 9, but stricter because Test is a
one-time final evaluation.

### First authorized execution

1. verify no completed or partial Phase 11 attempt exists;
2. complete checkpoint and Validation verification;
3. atomically write an attempt receipt with status `STARTED`, unique attempt
   ID, checkpoint/selection fingerprints, Validation PASS evidence, intended
   Test split count and `test_evaluation_count=0`;
4. invoke the lazy Test provider and perform exactly one prediction pass;
5. save predictions and metric artifacts with read-back checks;
6. atomically finalize the manifest/receipt as `FINAL_TEST_COMPLETE` with
   `test_evaluation_count=1`.

### Later notebook Run All

If a complete manifest and all fingerprints/artifacts validate, the public
entry point must return:

```text
guard_action = loaded_verified_phase_11_artifacts_no_test_reevaluation
```

It must not reload/evaluate the model or invoke the Test provider.

If a `STARTED`/partial attempt exists without a complete manifest, the guard
must stop and require a documented audit. It must not silently rerun Test,
delete partial evidence, or claim completion. If complete artifacts fail
integrity/fingerprint checks, stop rather than reevaluate.

The manifest must state:

```text
checkpoint_selected_before_test = true
validation_verified_before_test = true
test_used_for_selection = false
test_evaluation_count = 1
training_performed = false
backward_called = false
optimizer_step_performed = false
scheduler_step_performed = false
checkpoint_changed_after_test = false
phase_12_started = false
```

## I. Artifact Flow

All Phase 11 results are planned under:

```text
docs/result/phase_11_evaluation/
```

Minimum artifacts:

| Artifact | Purpose / downstream consumer |
|---|---|
| `checkpoint_verification.json` | Provenance, architecture, mapping, epoch/step and fingerprints |
| `validation_evaluation.json` | Reloaded Validation metrics, Phase 9 references, deltas and tolerance checks |
| `test_evaluation.json` | Single final Test loss and aggregate metrics |
| `test_predictions.json` | Frozen ordered prediction records for Phase 12 |
| `phase_11_evaluation_manifest.json` | Authoritative status, guard state, action counters, paths and integrity checks |

Recommended additional transactional evidence:

```text
final_test_attempt_receipt.json
```

Every JSON write should be atomic where practical and followed by parse/read-
back verification. The manifest should record artifact size and SHA-256 hashes,
environment/package versions, device, seed, split counts, checkpoint identity,
timestamps and the exact one-time policy fields.

Artifact dependency flow:

```text
Phase 9 selection + epoch-2 metrics
        ↓
checkpoint_verification.json
        ↓
validation_evaluation.json (PASS gate)
        ↓
one Test pass
        ├── test_evaluation.json
        └── test_predictions.json
                ↓
phase_11_evaluation_manifest.json
                ↓
Phase 12 reads artifacts; no Test reevaluation
```

## J. Notebook Presentation

Append Phase 11 after the verified Phase 10 section. The notebook will contain
only orchestration and presentation:

1. import the Phase 11 guarded entry point/report loader;
2. pass the existing Validation inputs and a lazy provider for locked Test/raw
   text inputs;
3. call the entry point once;
4. assert checkpoint verification and Validation reproducibility PASS;
5. assert final manifest status is `FINAL_TEST_COMPLETE`;
6. display the fixed checkpoint identity/fingerprint;
7. display a Validation verification table with reference, reloaded value,
   delta, tolerance and PASS status;
8. display the final Test metrics table;
9. display a Validation-versus-Test comparison table;
10. display prediction/artifact paths and row counts, not Phase 12 error
    analysis;
11. state explicitly that Test was evaluated once only after Validation PASS,
    was never used for selection, and later Run All loads artifacts without
    reevaluating Test.

Checkpoint/model loading, evaluation, probability generation, serialization,
guard logic and integrity validation remain in the module. The notebook must
not duplicate a manual evaluation loop.

## K. Validation and Failure Handling

Required checks before declaring Phase 11 complete:

### Checkpoint

- selection/manifest/trainer-state paths agree exactly;
- epoch is 2 and step is 1068;
- checkpoint directory and weight file exist and are non-empty;
- weight/selection fingerprints are recorded and unchanged;
- reloaded model is DistilBERT binary sequence classification;
- label mappings are exact;
- no reranking or checkpoint replacement occurs.

### Validation

- count is exactly 1,066;
- required tokenized fields exist;
- labels are subsets of `{0,1}`;
- loss and all four metrics are finite;
- score metrics are in `[0,1]`;
- every metric matches Phase 9 epoch 2 within its explicit tolerance;
- failure prevents creation/invocation of the Test evaluation path.

### Test

- Test model evaluation begins only after Validation PASS;
- raw/tokenized counts are exactly 1,066 and aligned;
- label set is a subset of `{0,1}`;
- exactly one Test prediction call/pass is recorded;
- loss is finite and score metrics are finite/in `[0,1]`;
- prediction count equals 1,066;
- no missing or duplicate sample indices/IDs;
- every prediction/probability record passes its schema/range checks;
- selected checkpoint fingerprint remains unchanged after Test;
- no training, backward, optimizer or scheduler step occurs;
- Test is not used for selection and Phase 12 is not started.

### Failure workflow

Any failure follows:

```text
ERROR
  ↓
save exact evidence without fabricating completion
  ↓
docs/plan-doc/analysis_error/phase_11_<error>_YYYY-MM-DD.md
  ↓
approved fix/recovery plan
  ↓
no automatic Test rerun
```

Validation failure is safe to rerun only after an approved fix because Test
was never invoked. A partial or failed Test attempt requires explicit audit and
user approval; it must not be hidden behind an automatic retry.

## L. Completion Criteria

Phase 11 is complete only when:

1. the approved plan is implemented in
   `processing_own_phase/phase_11_final_evaluation.py`;
2. the authoritative Phase 9 epoch-2/step-1068 checkpoint is verified and
   fingerprinted without reranking;
3. its model family, binary configuration and label mapping pass;
4. Validation is evaluated first and all five metrics reproduce Phase 9 within
   declared tolerances;
5. the Validation gate demonstrably passes before any Test model evaluation;
6. the official 1,066-sample Test split is evaluated exactly once;
7. real finite Test loss, Accuracy, Precision, Recall and F1 are saved;
8. exactly 1,066 aligned prediction records with labels/probabilities/text are
   saved and verified;
9. checkpoint fingerprints are unchanged after Test;
10. all minimum artifacts and the one-time receipt/manifest pass integrity and
    read-back checks;
11. subsequent guarded calls load verified artifacts and do not reevaluate
    Test;
12. notebook calls the module and presents checkpoint, Validation and Test
    evidence without core processing logic;
13. notebook runs through Phase 11 without retraining Phase 9 or reevaluating
    Test after the first successful Phase 11 execution;
14. no training/backward/update, Test-based selection, checkpoint change,
    confusion matrix/error analysis or Phase 12 work occurs;
15. the workflow process log is created and Phase 0–10 show no regression.

Passing Phase 11 authorizes Phase 12 planning only. It does not authorize
automatic Phase 12 implementation.

## M. Out of Scope

Phase 11 must not:

- train, fine-tune, resume training or call `Trainer.train()`;
- run backward or any optimizer/scheduler update;
- tune hyperparameters or select/rerank checkpoints;
- use Test loss/metrics/predictions to change model, checkpoint,
  preprocessing, epoch count or methodology;
- evaluate Test more than once or automatically retry a Test attempt;
- create a confusion matrix, select representative errors, categorize errors
  or perform qualitative error analysis (Phase 12);
- run new-sentence/custom inference (Phase 13);
- save/reload the final reusable model package (Phase 14);
- modify Phase 0–10 unless a separately documented critical regression is
  found and approved;
- implement or start Phase 12 or later phases;
- access Test during this planning turn.

This planning turn creates only:

```text
docs/plan-doc/plan_before_process/
phase_11_validation_final_test_evaluation_plan_2026-08-14.md
```

After separate implementation approval, expected additions/modifications are:

```text
processing_own_phase/phase_11_final_evaluation.py
notebook_practice_3/practice_3.ipynb
docs/result/phase_11_evaluation/checkpoint_verification.json
docs/result/phase_11_evaluation/validation_evaluation.json
docs/result/phase_11_evaluation/test_evaluation.json
docs/result/phase_11_evaluation/test_predictions.json
docs/result/phase_11_evaluation/final_test_attempt_receipt.json
docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json
docs/save_process_proceduce_own_phase_refactor&fix/
phase_11_final_evaluation_log_2026-08-14.md
```
