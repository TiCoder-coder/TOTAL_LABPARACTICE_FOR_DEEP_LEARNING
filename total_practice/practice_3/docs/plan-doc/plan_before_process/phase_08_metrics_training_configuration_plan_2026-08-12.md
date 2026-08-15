# Phase 08 Metrics & Training Configuration Plan — 2026-08-12

## 1. Objective

Define and verify the binary-classification metric contract and the immutable
baseline Hugging Face `TrainingArguments` configuration that Phase 9 will use.
Phase 8 prepares configuration only: it does not create a `Trainer`, access
Test data, or perform any training operation.

The implementation architecture will remain:

```text
processing_own_phase/phase_08_metrics_training_configuration.py
        ↓
metric + configuration + policy verification
        ↓
docs/result/phase_08_metrics_training_configuration_verification.json
        ↓
notebook_practice_3/practice_3.ipynb
        ↓
call + tables + explanation
```

## 2. Inputs from Phase 7

Phase 8 accepts the verified Phase 7 contract without reconstructing or
modifying the model:

| Input | Verified value |
|---|---|
| Checkpoint | `distilbert/distilbert-base-uncased` |
| Model class | `DistilBertForSequenceClassification` |
| Model type | `distilbert` |
| Labels | `{0, 1}` |
| Label mapping | `0 -> NEGATIVE`, `1 -> POSITIVE` |
| Number of labels | 2 |
| Phase 7 logits shape | `[2, 2]` |
| Finite logits/loss | PASS |
| Parameters | 66,955,010 total; 66,955,010 trainable |
| Training performed through Phase 7 | No |
| Validation/Test accessed by Phase 7 | No |

The Phase 6 input contract also remains unchanged: Rotten Tomatoes official
Train/Validation/Test splits, `MAX_TOKEN_LENGTH=80`, existing tokenization,
`input_ids`, `attention_mask`, `labels`, and dynamic padding.

## 3. Metrics Design

The planned public `compute_metrics(eval_prediction)` function will:

1. accept the Hugging Face `EvalPrediction` interface (or its compatible
   predictions/label IDs contract);
2. unwrap logits if the predictions object is a tuple;
3. compute predicted class IDs with `argmax(axis=-1)`;
4. validate prediction and label lengths;
5. validate that observed labels and predictions are subsets of `{0, 1}`;
6. return plain Python floats under exactly these keys:
   `accuracy`, `precision`, `recall`, and `f1`.

Metric definitions are binary classification with class `1` (`POSITIVE`) as
the positive class:

- Accuracy: fraction of all predictions that match their labels.
- Precision: positive true predictions divided by all positive predictions.
- Recall: positive true predictions divided by all actual positives.
- F1: harmonic mean of binary precision and recall.

Precision, Recall, and F1 will use `average="binary"`, `pos_label=1`, and
`zero_division=0` so edge-case batches return deterministic finite values.
Loss is produced by the model/Trainer and is not recomputed by
`compute_metrics`.

Phase 8 verification will use a small synthetic logits/labels fixture with
manually known outcomes. This verifies metric wiring without touching Train,
Validation, or Test datasets and without running a model.

## 4. TrainingArguments Design

The baseline configuration is fixed as follows:

| Argument | Baseline value |
|---|---:|
| `num_train_epochs` | `3` |
| `per_device_train_batch_size` | `16` |
| `per_device_eval_batch_size` | `32` |
| `learning_rate` | `2e-5` |
| `weight_decay` | `0.01` |
| `seed` | `42` |
| `data_seed` | `42` |
| evaluation strategy | every epoch |
| save strategy | every epoch |
| logging strategy | every epoch |
| `load_best_model_at_end` | `True` |
| `metric_for_best_model` | `eval_loss` |
| `greater_is_better` | `False` |
| external reporting | disabled (`report_to="none"`) |

The implementation must inspect the installed Transformers 5.14.1
`TrainingArguments` signature and use its supported spelling for the epoch
evaluation parameter (for example `eval_strategy="epoch"` if that is the
installed API). It must not silently omit evaluation because of a renamed
argument.

The planned checkpoint output directory is scoped inside Practice 3, for
example:

```text
docs/result/phase_09_training/checkpoints/
```

Phase 8 may represent this path in the configuration but must not create a
checkpoint. Phase 9 owns training and checkpoint creation.

## 5. Best Checkpoint Policy

The selection policy is validation-only and lexicographically ordered:

```text
1. minimum validation loss
2. if validation loss is tied, maximum validation F1
```

The corresponding Trainer baseline is:

```text
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
greater_is_better = False
evaluation strategy = "epoch"
save strategy = "epoch"
```

Because the standard Trainer primary metric alone does not guarantee an F1
tie-break, Phase 8 will define a deterministic validation-checkpoint ranking
contract for Phase 9. Given executed checkpoint records, rank by
`(eval_loss ascending, eval_f1 descending)`. Exact remaining ties must be
resolved deterministically by earlier epoch/checkpoint order and documented;
they must never be resolved using Test results.

Phase 9 must record the winning checkpoint, its validation loss/F1 and the
selection reason. The model loaded for subsequent evaluation must correspond
to this policy. Test data is prohibited from metric configuration, checkpoint
ranking, epoch selection and hyperparameter selection.

## 6. Logging and Save Strategy

- Evaluate once at the end of every epoch.
- Log once at the end of every epoch.
- Save a checkpoint once at the end of every epoch.
- Keep Trainer logging local; do not send runs to external tracking services.
- Log at minimum epoch, training loss, validation loss, validation Accuracy,
  Precision, Recall, F1, and learning rate when Phase 9 executes training.
- Phase 9 must preserve Trainer history/state sufficiently for Phase 10
  learning curves and checkpoint-policy auditing.
- Phase 8 saves only a JSON configuration-verification artifact. It does not
  create a model checkpoint, optimizer state, scheduler state or training log.

The equality of evaluation and save strategies is a required precondition for
`load_best_model_at_end=True`.

## 7. Reproducibility

- Reuse the project seed `42`.
- Set both `seed=42` and `data_seed=42` in `TrainingArguments`.
- Reuse the Phase 1 seed setup before any later Trainer/model initialization.
- Do not shuffle, resplit, retokenize or remap the Phase 6 datasets in Phase 8.
- Record installed Transformers version and the serialized effective argument
  values in the Phase 8 artifact.
- Device placement remains portable; Phase 8 must not hard-code CUDA, MPS or
  CPU behavior. Trainer/device execution belongs to Phase 9.

These controls improve repeatability but do not claim bitwise-identical
training across different hardware backends.

## 8. Result Artifact

Planned executed evidence:

```text
docs/result/phase_08_metrics_training_configuration_verification.json
```

Minimum contents:

- overall `status`;
- metrics names and positive class;
- synthetic metric fixture predictions/labels;
- executed Accuracy, Precision, Recall, F1;
- finite/range checks for every metric;
- epochs, batch sizes, learning rate, weight decay, seed and data seed;
- effective evaluation, save and logging strategies;
- `load_best_model_at_end`, `metric_for_best_model`,
  `greater_is_better`;
- checkpoint output directory;
- primary and tie-break criteria;
- validation-only selection status;
- Transformers version and argument-construction status;
- explicit `trainer_created=false`, `training_performed=false`,
  `backward_called=false`, `optimizer_step_performed=false`,
  `test_accessed=false`, and `checkpoint_saved=false`.

The artifact must be generated from the constructed metric/configuration
objects, saved as JSON, read back and compared before Phase 8 can PASS.

## 9. Notebook Call Design

Append a concise Phase 8 section after the executed Phase 7 cell. The notebook
will only:

1. import public Phase 8 functions/constants;
2. run the synthetic `compute_metrics` verification;
3. construct and inspect `TrainingArguments` without a Trainer;
4. verify the primary/tie-break checkpoint policy using synthetic validation
   records only;
5. assert the returned Phase 8 status;
6. display metric, baseline configuration, logging/save and selection-policy
   tables;
7. save and display the Phase 8 JSON artifact path;
8. state explicitly that no Trainer/training/Test access occurred.

Metric calculation, argument construction, selection-policy validation and
artifact serialization stay in
`processing_own_phase/phase_08_metrics_training_configuration.py`; the
notebook must not duplicate large processing logic.

## 10. Validation Checks

Phase 8 implementation must verify all of the following:

- metric keys are exactly Accuracy, Precision, Recall and F1 in their planned
  lowercase return form;
- every metric is finite and within `[0, 1]`;
- binary positive label is `1`;
- synthetic known-answer metric fixture passes;
- epochs equal 3;
- Train batch size equals 16;
- evaluation batch size equals 32;
- learning rate equals `2e-5`;
- weight decay equals `0.01`;
- seed and data seed equal 42;
- evaluation, logging and save strategies are all epoch-based;
- `load_best_model_at_end is True`;
- primary selection is minimum `eval_loss`;
- higher `eval_f1` wins a validation-loss tie;
- Test is absent from the selection-policy inputs;
- effective `TrainingArguments` construction succeeds under the installed
  Transformers version;
- no Trainer, backward, optimizer step, training or checkpoint saving occurs;
- JSON save/read-back matches the in-memory verification;
- notebook still runs continuously through Phase 8 without regressing the
  Phase 0–7 evidence.

## 11. Completion Criteria

Phase 8 is complete only when:

1. the approved plan has been implemented in a dedicated Phase 8 module;
2. `compute_metrics` returns correct verified binary Accuracy, Precision,
   Recall and F1;
3. the effective baseline `TrainingArguments` contains every fixed value and
   strategy in this plan;
4. the validation-loss/F1 checkpoint policy is explicit and verified;
5. the notebook calls and presents the module after Phase 7;
6. the notebook runs from Phase 0 through Phase 8 without unresolved errors;
7. the Phase 8 artifact exists, passes read-back verification and contains no
   fabricated training results;
8. no Test access, Trainer construction or training action occurred;
9. the workflow process log for Phase 8 is updated.

## 12. Explicit Out of Scope

Phase 8 must not:

- create a Hugging Face `Trainer`;
- call `train()` or execute an epoch;
- run `backward()`;
- create or step an optimizer;
- create or step a scheduler;
- fine-tune or otherwise update model parameters;
- access or evaluate the Test split;
- use Test results for any decision;
- save a model/training checkpoint;
- change dataset splits, tokenizer, preprocessing or `MAX_TOKEN_LENGTH`;
- claim that the baseline hyperparameters are optimal;
- implement or begin Phase 9.

## 13. Output and Handoff to Phase 9

If Phase 8 passes, its handoff to Phase 9 consists of:

- the verified Phase 7 `model` contract;
- existing Phase 6 `tokenized_dataset` and dynamic data collator;
- public `compute_metrics` callable;
- constructed baseline `TrainingArguments`;
- fixed baseline hyperparameters;
- epoch evaluation/logging/save contract;
- validation-only checkpoint ranking policy: minimum validation loss, then
  maximum validation F1;
- Phase 8 verification JSON and notebook output;
- explicit proof that training and Test access have not yet occurred.

Phase 9 may then plan Trainer construction and fine-tuning. Passing Phase 8
does not authorize Phase 9 implementation automatically.
