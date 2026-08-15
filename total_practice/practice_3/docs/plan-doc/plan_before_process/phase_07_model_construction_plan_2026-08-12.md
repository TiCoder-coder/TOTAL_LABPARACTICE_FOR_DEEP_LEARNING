# Phase 07 Model Construction Plan — 2026-08-12

## 1. Objective

Construct the generic pretrained DistilBERT binary sequence-classification
model required by Exercise 2 and verify, with one small Train-only processed
batch, that the Phase 6 input contract is accepted and produces finite binary
classification logits and loss.

Phase 7 is complete only as **model construction + forward-pass sanity check**.
It must not train, tune, select, evaluate or save a model checkpoint.

## 2. Inputs

The verified Phase 6 contract is immutable input to this phase:

| Input | Required value/status |
|---|---|
| Dataset | `cornell-movie-review-data/rotten_tomatoes` |
| Train samples | 8,530 |
| Validation samples | 1,066 |
| Test samples | 1,066 |
| Labels | `{0, 1}` |
| Tokenizer | generic DistilBERT tokenizer; Phase 6 PASS |
| `MAX_TOKEN_LENGTH` | 80 |
| Tokenized fields | `input_ids`, `attention_mask`, `labels` |
| Padding | `DataCollatorWithPadding` at batch time |
| Sample preservation | PASS for all three splits |
| Decode sanity | PASS |
| Dynamic padding | PASS |

Phase 7 will consume the existing notebook objects `tokenizer`,
`tokenized_dataset` and the collator returned by
`get_data_collator(tokenizer)`. It will not remap, resplit or retokenize the
dataset.

Only Train samples may be used to construct the Phase 7 sanity batch.
Validation is not needed for model construction, and Test must not be accessed.

## 3. Dependencies

- Python 3.11 environment already verified by Phase 1.
- `torch==2.13.0`.
- `transformers==5.14.1`.
- Existing `processing_own_phase.phase_01_environment.get_device()`.
- Existing `processing_own_phase.phase_06_preprocessing.get_data_collator()`.
- Existing Phase 6 tokenized `DatasetDict`.
- Existing `RESULT_DIR` from `processing_own_phase.config`.

No new package is planned.

## 4. Model Checkpoint

Use the generic pretrained checkpoint:

```text
distilbert/distilbert-base-uncased
```

Load with:

```python
AutoModelForSequenceClassification.from_pretrained(...)
```

This is the generic language-pretrained model for Exercise 2, not the
sentiment-fine-tuned SST-2 checkpoint used by Exercise 1.

The current Phase 6 config uses the equivalent short Hub identifier
`distilbert-base-uncased`. Phase 7 will record the canonical checkpoint above
for unambiguous provenance without changing Phase 6 config or preprocessing.
Implementation must verify that the constructed model and Phase 6 tokenizer
are the intended DistilBERT-base-uncased family before declaring PASS.

## 5. Model Architecture

Planned construction flow:

```text
Phase 6 processed Train batch
        ↓
Generic pretrained DistilBERT backbone
        ↓
DistilBERT sequence-classification head
        ↓
Two output logits per sample
        ↓
NEGATIVE / POSITIVE task interface
```

Required constructor configuration:

```python
num_labels=2
id2label={0: "NEGATIVE", 1: "POSITIVE"}
label2id={"NEGATIVE": 0, "POSITIVE": 1}
```

No layer is to be frozen in Phase 7. Parameter flags should remain at the
pretrained sequence-classification model defaults so Phase 9 can perform full
fine-tuning. Phase 7 itself performs no parameter update.

## 6. Label Mapping

The mapping is explicit and bidirectionally consistent:

| Label ID | Label name |
|---:|---|
| 0 | `NEGATIVE` |
| 1 | `POSITIVE` |

Required checks:

```python
model.config.id2label == {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id == {"NEGATIVE": 0, "POSITIVE": 1}
model.config.num_labels == 2
```

The verifier must tolerate JSON converting integer dictionary keys to strings
only when reading the saved result artifact; the in-memory model configuration
must use integer `id2label` keys.

## 7. Device Strategy

Use the existing device detector with priority:

```text
CUDA → MPS → CPU
```

Planned sanity behavior:

1. Resolve a `torch.device` from the existing detector.
2. Move the model and the small collated Train batch to that device.
3. Run only the forward sanity check.
4. Return/store the actual device string.
5. Leave Phase 8/9 Trainer device placement decisions to their own phases.

CUDA must not be hard-coded. The implementation must work on MPS, CUDA and
CPU, and must not introduce CUDA-only AMP behavior in Phase 7.

## 8. Forward-Pass Sanity Check

Planned implementation sequence:

1. Confirm the provided tokenized object contains a non-empty Train split.
2. Select a small deterministic Train-only sample set, proposed batch size
   `B=2`.
3. Validate each selected label belongs to `{0, 1}`.
4. Use the existing `DataCollatorWithPadding` to produce aligned tensors.
5. Confirm batch keys include `input_ids`, `attention_mask` and `labels`.
6. Confirm batch tensor first dimensions all equal `B`.
7. Put the model in `eval()` mode for deterministic dropout behavior.
8. Run one `torch.no_grad()` forward pass with labels included.
9. Read `outputs.logits` and the model-provided classification `outputs.loss`.
10. Perform assertions and produce a serializable verification dictionary.

Supplying integer labels lets `AutoModelForSequenceClassification` compute its
standard sequence-classification loss. No separate optimizer or training loss
loop is needed.

## 9. Expected Logits Shape

For sanity batch size `B`:

```text
logits.shape == [B, 2]
```

For the proposed deterministic two-sample batch:

```text
logits.shape == [2, 2]
```

The verifier should derive `B` from the collated batch rather than accepting a
hard-coded expected first dimension. The final dimension must equal
`model.config.num_labels == 2`.

## 10. Finite-Loss and Finite-Logits Verification

Required numerical assertions:

```python
torch.isfinite(outputs.logits).all()
outputs.loss is not None
torch.isfinite(outputs.loss).all()
```

Record only the scalar executed sanity loss and PASS/FAIL statuses. The loss is
not a training metric, Validation metric, Test metric or estimate of model
quality. It is evidence that the model/input/loss contract is numerically
valid before training.

The sanity function must not call `backward()`, create gradients intentionally
or call an optimizer step.

## 11. Parameter Information

Record:

- total parameter count;
- trainable parameter count;
- frozen parameter count;
- trainable percentage;
- model class;
- backbone/model type;
- configured number of labels.

Expected invariant for the planned full-fine-tuning workflow:

```text
trainable parameters == total parameters
```

If the library checkpoint configuration produces a different result, report
the executed values and fail the intended full-fine-tuning readiness check
instead of silently changing parameter flags.

## 12. Notebook Call Design

After implementation approval, append a concise Phase 7 section after the
current Phase 6 cell in:

```text
notebook_practice_3/practice_3.ipynb
```

The notebook should only:

1. import Phase 7 public functions;
2. pass the existing `tokenized_dataset`, `tokenizer` and data collator;
3. call model construction;
4. call the Train-only forward verifier;
5. assert the returned overall status;
6. display model/checkpoint/label mapping and parameter tables;
7. display batch shape, logits shape, finite statuses, device and explicit
   `training_performed=False` / `test_accessed=False` evidence;
8. save/display the Phase 7 JSON artifact path.

Model loading, parameter counting, device transfer and forward-pass logic must
remain in the Phase 7 Python module, not be duplicated in the notebook.

## 13. Result Artifact

Planned result:

```text
docs/result/phase_07_model_verification.json
```

Proposed fields:

```text
status
checkpoint
model_class
model_type
num_labels
id2label
label2id
device
max_token_length
sanity_split
sanity_batch_size
batch_input_ids_shape
batch_attention_mask_shape
batch_labels_shape
observed_label_set
labels_valid
logits_shape
expected_logits_shape
logits_finite
loss_value
loss_finite
total_parameters
trainable_parameters
frozen_parameters
trainable_percentage
model_in_eval_mode_during_sanity
backward_called
optimizer_created
training_performed
validation_accessed
test_accessed
all_pass
```

All values must come from the executed implementation. Do not fabricate a
parameter count, loss value or device. No model weights/checkpoint are saved in
Phase 7.

## 14. Files Expected to Create or Modify

After separate implementation approval, expected files are:

- Create `processing_own_phase/phase_07_model_construction.py`.
- Modify `notebook_practice_3/practice_3.ipynb` only to add Phase 7 call and
  presentation cells.
- Create `docs/result/phase_07_model_verification.json` from executed evidence.
- Create the appropriate Phase 7 implementation/process log required by the
  workflow.
- Create an error-analysis document and fix plan only if an implementation or
  verification error actually occurs.

This planning step creates only this plan file. It must not create any of the
implementation/result files listed above.

## 15. Validation Checks

Phase 7 implementation will require all of the following:

- [ ] Generic `distilbert/distilbert-base-uncased` model loads successfully.
- [ ] Model is `AutoModelForSequenceClassification`-compatible.
- [ ] `num_labels == 2`.
- [ ] `id2label == {0: "NEGATIVE", 1: "POSITIVE"}`.
- [ ] `label2id == {"NEGATIVE": 0, "POSITIVE": 1}`.
- [ ] Existing Phase 6 tokenizer/preprocessing objects are consumed unchanged.
- [ ] Sanity data comes only from Train.
- [ ] Dynamic collator creates compatible batch tensors.
- [ ] Batch contains `input_ids`, `attention_mask`, `labels`.
- [ ] Labels are a subset of `{0, 1}`.
- [ ] Model accepts the batch on the detected MPS/CUDA/CPU device.
- [ ] `logits.shape == [B, 2]`.
- [ ] Every logit is finite.
- [ ] Model-provided loss exists and is finite.
- [ ] Total/trainable/frozen parameter counts are internally consistent.
- [ ] All parameters remain trainable for the later full-fine-tuning phase.
- [ ] Model is in evaluation mode only for the sanity forward.
- [ ] No backward pass, optimizer, scheduler, Trainer or epoch is used.
- [ ] `training_performed is False`.
- [ ] Validation and Test are not accessed.
- [ ] JSON artifact is serializable, exists and reflects executed evidence.
- [ ] Notebook call succeeds and presents the artifact without core logic
  duplication.
- [ ] Phase 0–6 source, preprocessing methodology and artifacts are unchanged.

## 16. Completion Criteria

Phase 7 will be considered complete only when:

1. This approved plan exists before implementation.
2. Core implementation resides in
   `processing_own_phase/phase_07_model_construction.py`.
3. The generic pretrained checkpoint and two-label mappings are explicit.
4. A deterministic Train-only dynamically padded batch completes one forward
   pass on the detected device.
5. Labels, batch shapes, `[B,2]` logits, finite logits and finite loss all PASS.
6. Parameter counts are real, consistent and presented.
7. No training, tuning, Validation evaluation or Test access occurs.
8. The executed verification is saved in
   `docs/result/phase_07_model_verification.json`.
9. Notebook calls the module and shows real output without embedding large
   construction logic.
10. No unresolved runtime error or Phase 0–6 regression remains.
11. Workflow log/result requirements are satisfied.
12. Phase 7 output contract is documented for Phase 8.

## 17. Explicit Out-of-Scope Items

Phase 7 must not:

- create `Trainer`;
- create `TrainingArguments`;
- define evaluation metrics;
- create an optimizer;
- create a scheduler;
- call `backward()`;
- run a training step or epoch;
- fine-tune or freeze/unfreeze as an experiment;
- search hyperparameters;
- select an epoch or checkpoint;
- evaluate Validation metrics;
- access or evaluate Test;
- save a model checkpoint or tokenizer artifact;
- modify the dataset, official splits, tokenizer, tokenized fields, padding or
  `MAX_TOKEN_LENGTH`;
- modify Phase 0–6;
- implement or modify Phase 8 onward.

## 18. Handoff / Output for Phase 8

Phase 8 may begin only after Phase 7 verification PASS. Its input contract will
be:

- constructed generic DistilBERT sequence-classification model;
- explicit two-label configuration;
- all parameters trainable for full fine-tuning;
- verified Phase 6 tokenizer, tokenized Train/Validation datasets and dynamic
  collator;
- verified batch-to-model interface;
- expected logits contract `[B,2]`;
- finite pre-training sanity loss/logits;
- actual device compatibility evidence;
- `phase_07_model_verification.json` with
  `training_performed=False`, `validation_accessed=False` and
  `test_accessed=False`.

Phase 8 will separately own metrics and training configuration. Phase 7 must
not pre-empt those decisions.

## Critical-Issue Review

No critical technical defect was found in the accepted Phase 6 input contract.
The existing handoff document contains an administrative note that its earlier
GitHub push was pending, but the current task explicitly declares Phase 0–6
PASS and accepted as Viên's input contract. This does not alter the Phase 7
technical plan and no Git action is part of this planning step.

