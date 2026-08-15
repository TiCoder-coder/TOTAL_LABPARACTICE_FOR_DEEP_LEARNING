# Phase 13 — New-Sentence Inference Plan

**Date:** 2026-08-14  
**Scope:** Planning only; no Phase 13 implementation or inference is performed by this document.  
**Prerequisite:** Phase 0–12 PASS.

## A. Objective

Build a reusable inference-only path for one new sentence or an ordered batch of new sentences. The path must reload the authoritative Phase 9 checkpoint selected and fingerprinted in Phase 11, apply the locked Phase 6 tokenizer contract, and return binary sentiment probabilities and labels.

The fixed flow is:

```text
New custom text(s)
        ↓
Validate and normalize input
        ↓
Verify checkpoint-1068 provenance and SHA-256
        ↓
Phase 6 tokenizer (max_length=80, truncation)
        ↓
DataCollatorWithPadding (dynamic batch padding)
        ↓
DistilBERT checkpoint in eval/inference mode
        ↓
logits [B, 2] → softmax probabilities
        ↓
NEGATIVE / POSITIVE + confidence
        ↓
Verification and result artifact
```

Phase 13 does not perform training, checkpoint selection, dataset evaluation, or Test access.

## B. Input / Output Contract

### Inputs

The public inference function will accept either:

- one Python `str`; or
- an ordered sequence/list of Python strings.

A single string is normalized to a one-item list. The implementation must reject an empty list, `None`, non-string items, empty strings, and whitespace-only strings with a clear error. Input order and original text must be preserved. Default notebook examples must be newly authored custom sentences and must not be copied from the Test split or Phase 11 predictions.

Suggested custom demo coverage:

- clearly positive;
- clearly negative;
- negation;
- mixed/contrast sentiment.

### Outputs

Return one ordered record for every input, containing at least:

- `input_index`;
- `text`;
- `predicted_label` as integer `0` or `1`;
- `predicted_label_name` as `NEGATIVE` or `POSITIVE`;
- `negative_probability`;
- `positive_probability`;
- `confidence`, equal to the probability of the predicted class.

The output count must equal the normalized input count. The label contract is fixed:

```text
0 -> NEGATIVE
1 -> POSITIVE
```

## C. Preprocessing

Phase 13 must reuse the Phase 6 preprocessing contract rather than introduce a new tokenizer policy:

- tokenizer checkpoint: `distilbert/distilbert-base-uncased`;
- `max_length = 80`;
- truncation enabled;
- no fixed-length padding during per-example tokenization;
- batch padding performed dynamically with `DataCollatorWithPadding`;
- produce `input_ids` and `attention_mask` tensors accepted by the model.

The implementation should reuse the existing Phase 6 tokenizer/config helpers where practical. It must verify that batch dimensions match the input count, `input_ids` and `attention_mask` have the same shape, and the padded sequence length does not exceed 80. Phase 6 methodology and artifacts must not be modified.

## D. Checkpoint Verification

The only authoritative model is:

- selected epoch: `2`;
- selected step: `1068`;
- selected checkpoint: `checkpoint-1068`;
- checkpoint path source: `docs/result/phase_09_training/selected_checkpoint.json`;
- authoritative weight SHA-256 source: `docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json`;
- expected weight SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.

Before inference, the module must:

1. read the Phase 9 selection record and Phase 11 evaluation manifest;
2. confirm both identify epoch 2, step 1068, and `checkpoint-1068`;
3. resolve and confirm the checkpoint directory and required model/config files exist;
4. recompute the checkpoint weight SHA-256 and require an exact match with Phase 11;
5. load the checkpoint without reranking or considering another checkpoint;
6. verify DistilBERT configuration, `num_labels == 2`, and the exact `id2label`/`label2id` mapping;
7. optionally recheck the fingerprint after inference to prove the checkpoint was not modified.

Any path, metadata, label-map, or fingerprint mismatch is a hard failure. The module must stop before producing predictions and must not fall back to another checkpoint or download a replacement model.

## E. Inference and Artifact Design

### Core implementation planned

Create in the implementation turn:

`processing_own_phase/phase_13_new_sentence_inference.py`

Core responsibilities:

- input validation and normalization;
- checkpoint provenance/fingerprint verification;
- tokenizer and dynamic-collator construction under the locked contract;
- device selection compatible with MPS, CUDA, and CPU without hard-coding CUDA;
- `model.eval()` plus `torch.inference_mode()` (or `torch.no_grad()`);
- logits-to-probability conversion with `softmax(dim=-1)`;
- ordered output construction;
- deterministic and numerical sanity checks;
- artifact writing and safe artifact loading.

No `Trainer`, optimizer, scheduler, backward pass, or dataset evaluator belongs in this module.

### Result artifact planned

Create only after real implementation and successful execution:

`docs/result/phase_13_inference_examples.json`

The artifact should include:

- overall status and timestamp;
- checkpoint path, epoch, step, and verified SHA-256;
- tokenizer checkpoint, `max_length`, truncation, and dynamic-padding policy;
- device used;
- explicit custom-input provenance (`custom_authored_not_test`);
- ordered input/output records;
- tensor/logits shape summary;
- probability and determinism verification results;
- isolation flags, including `training_performed=false`, `backward_called=false`, `optimizer_step_performed=false`, `scheduler_step_performed=false`, `test_accessed=false`, `test_evaluated=false`, `checkpoint_changed=false`, and `phase_14_started=false`.

A separate verification JSON is optional and should only be created if separating machine checks from presentation materially improves clarity. One complete, schema-validated artifact is preferred over duplicated evidence.

### Artifact guard

For the fixed notebook demo, compute an input signature from the ordered custom texts and store it with the checkpoint fingerprint and preprocessing contract. On later Run All executions, the notebook-facing orchestrator may load the existing artifact only when all of these values match and the artifact verification status is PASS. Otherwise it must run normal new-text inference and replace the Phase 13 artifact with real results.

This guard is a convenience and provenance control; it must never read Phase 11 Test predictions or trigger Test evaluation.

## F. Notebook Presentation

Append a Phase 13 section to `notebook_practice_3/practice_3.ipynb` during implementation. The notebook must only import/call the Phase 13 module and present returned evidence. It must not duplicate tokenization, model loading, softmax, fingerprinting, or verification logic.

The Phase 13 presentation should show:

1. authoritative checkpoint name, epoch/step, and abbreviated SHA-256;
2. tokenizer checkpoint, `max_length=80`, and dynamic-padding statement;
3. a table with text, predicted label, NEGATIVE probability, POSITIVE probability, and confidence;
4. verification status for probability validity, count preservation, determinism, and checkpoint provenance;
5. explicit statements that examples are custom, Test was not accessed/evaluated, no training occurred, and the checkpoint was unchanged;
6. the saved artifact path.

Notebook examples must be maintained as presentation inputs only. All processing remains in the `.py` module.

## G. Verification Checks and Failure Handling

The implementation must verify all of the following:

- input is one valid sentence or a non-empty ordered list of valid sentences;
- output count equals input count and input order is preserved;
- checkpoint path, epoch 2, step 1068, and SHA-256 match Phase 9/11 evidence;
- model config is DistilBERT with two labels and the exact label mapping;
- `input_ids` and `attention_mask` exist and have compatible batch shapes;
- sequence length after collation is no greater than 80;
- logits shape is exactly `[B, 2]`;
- every logit and probability is finite;
- every class probability lies in `[0, 1]`;
- each probability row sums to 1 within absolute tolerance `1e-6`;
- predicted label equals `argmax(probabilities)`;
- confidence equals the probability of the predicted class within absolute tolerance `1e-8`;
- two inference passes over the same inputs, checkpoint, device, and preprocessing contract yield identical predicted labels and probabilities equal within absolute tolerance `1e-7`;
- checkpoint fingerprint is unchanged after inference;
- no Phase 11 Test artifact, Test dataset/provider, training, backward, optimizer, or scheduler path is accessed.

Determinism is verified within the same execution environment/device. The plan does not claim bit-for-bit equality across different hardware backends.

On any failed assertion, Phase 13 status is FAIL, no PASS artifact may be fabricated, and the notebook must report the precise failed check. If the failure needs analysis, follow `PRACTICE3_WORKFLOW_HANDOFF_RULE.md` and create a dated analysis-error document before fixing.

## H. Completion Criteria

Phase 13 implementation will be complete only when:

- `processing_own_phase/phase_13_new_sentence_inference.py` exists and owns all core logic;
- only checkpoint-1068 is loaded and its SHA-256 matches Phase 11;
- one sentence and a multi-sentence batch both satisfy the input/output contract;
- locked Phase 6 tokenization, truncation, maximum length, and dynamic padding are used;
- eval/inference mode produces valid `[B, 2]` logits, probabilities, labels, and confidence;
- all numerical, count, label-map, provenance, and determinism checks PASS;
- `docs/result/phase_13_inference_examples.json` contains real verified evidence;
- the notebook calls the module and displays the required table and verification evidence;
- a notebook run through Phase 13 shows no regression in Phase 0–12 guards;
- Phase 9 does not retrain, Phase 11 does not reevaluate Test, and Test evaluation count remains 1;
- the workflow process log for implementation is saved in the required log directory;
- no Phase 14 implementation has started.

The verified Phase 13 output provides Phase 14 with an inference API and authoritative checkpoint/tokenizer provenance to use when designing save-and-reload equivalence checks. It does not itself create a final packaged model.

## I. Explicit Out of Scope

Phase 13 must not:

- train or fine-tune the model;
- create a Trainer, optimizer, scheduler, backward pass, or epoch loop;
- access, load, predict, or evaluate the Test split again;
- use Test examples as notebook demo inputs;
- change, select, rerank, overwrite, or save a new checkpoint;
- change the tokenizer, `MAX_TOKEN_LENGTH=80`, truncation, dynamic-padding policy, dataset, labels, or methodology from earlier phases;
- perform confusion-matrix or error analysis again;
- implement model packaging, save/reload equivalence, or any Phase 14 work;
- begin Phase 15.

## Files Expected in the Implementation Turn

| File | Planned action |
|---|---|
| `processing_own_phase/phase_13_new_sentence_inference.py` | Create core inference and verification module |
| `notebook_practice_3/practice_3.ipynb` | Add thin Phase 13 call and presentation cells |
| `docs/result/phase_13_inference_examples.json` | Create from real successful inference evidence |
| `docs/save_process_proceduce_own_phase_refactor&fix/<dated-phase-13-log>.md` | Create implementation/verification process log |

No implementation file, result artifact, notebook output, checkpoint, or Phase 14 file is created or modified during this planning step.
