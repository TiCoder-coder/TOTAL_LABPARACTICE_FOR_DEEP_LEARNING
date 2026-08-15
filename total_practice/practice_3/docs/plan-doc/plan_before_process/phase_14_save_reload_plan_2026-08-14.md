# Phase 14 — Save & Reload Plan

**Date:** 2026-08-14  
**Scope:** Planning only; no package is saved or reloaded in this step.  
**Prerequisite:** Phase 0–13 PASS.

## A. Save Flow

Phase 14 packages the already selected and evaluated model for reusable local inference. It must not train, modify, select, or rerank weights.

```text
Phase 9 selected_checkpoint.json
        +
Phase 11 authoritative fingerprint
        +
Phase 13 custom inputs and preprocessing contract
        ↓
Verify checkpoint-1068 / epoch 2 / step 1068 / SHA-256
        ↓
Load authoritative model and tokenizer locally
        ↓
Verify model, labels, tokenizer and MAX_TOKEN_LENGTH=80 contract
        ↓
save_pretrained() into a temporary package directory
        ↓
Verify required package files and hashes
        ↓
Atomically publish phase_14_saved_model/
        ↓
Reload only from the saved package
        ↓
Original-versus-reloaded equivalence checks
        ↓
Verification artifact + package manifest
```

Before saving, implementation must read and cross-check:

- `docs/result/phase_09_training/selected_checkpoint.json`;
- `docs/result/phase_11_evaluation/phase_11_evaluation_manifest.json`;
- `docs/result/phase_13_inference_examples.json`.

The source identity is fixed:

- checkpoint: `checkpoint-1068`;
- epoch: `2`;
- step: `1068`;
- authoritative checkpoint weight SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`.

The implementation must recompute the source checkpoint weight fingerprint and require an exact Phase 11 match before loading or saving. It may reuse Phase 13 provenance helpers but must not change Phase 13 behavior.

Use Hugging Face `save_pretrained()` for both model and tokenizer. Save first to a temporary sibling directory under `docs/result/`; only publish to the final package path after all required files are present and non-empty. If the final path already exists but is incomplete or invalid, stop and report the integrity failure rather than deleting or overwriting it automatically.

The SHA-256 of the newly serialized package weight file is recorded separately from the authoritative source fingerprint. It is not assumed to be byte-identical merely because the underlying tensors are equivalent.

## B. Package Structure

Final package directory:

`docs/result/phase_14_saved_model/`

Expected structure after real implementation:

```text
phase_14_saved_model/
├── config.json
├── model.safetensors              # or the actual non-empty model file emitted
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json        # when emitted by this tokenizer/version
└── package_manifest.json
```

The implementation must use and record the actual filenames returned/emitted by `save_pretrained()`. Minimum functional requirements are:

- one non-empty model weight file;
- `config.json`;
- tokenizer vocabulary/serialization required by `AutoTokenizer.from_pretrained()`;
- tokenizer configuration;
- all special-token information required to reproduce the source tokenizer.

`special_tokens_map.json` is required when emitted. If the installed Transformers version stores equivalent special-token definitions in another tokenizer artifact, the implementation must verify the IDs and mappings after reload and record the actual source; it must not fabricate a missing file.

`package_manifest.json` should include:

- manifest schema/version and creation time;
- source checkpoint path, epoch and step;
- source checkpoint weight SHA-256 and Phase 11 selection-record SHA-256;
- model class/type, `num_labels`, `id2label`, and `label2id`;
- tokenizer class, vocabulary size, special tokens and special-token IDs;
- locked preprocessing contract: maximum length 80, truncation enabled, dynamic padding;
- relative package file list with byte size and SHA-256;
- Transformers, Torch and Python versions;
- save method (`model.save_pretrained`, `tokenizer.save_pretrained`);
- isolation flags.

The manifest must not include its own hash in its internal file-hash table, avoiding a recursive self-hash. The external Phase 14 verification artifact may record the final manifest hash.

Training-only files such as optimizer state, scheduler state, RNG state, Trainer state, and training arguments must not be copied into the final reusable inference package.

## C. Reload Flow

After the package has been published successfully:

1. resolve the exact local `phase_14_saved_model/` directory;
2. verify the package manifest and every recorded package file hash;
3. reload with `AutoModelForSequenceClassification.from_pretrained(saved_path, local_files_only=True)`;
4. reload with `AutoTokenizer.from_pretrained(saved_path, local_files_only=True)`;
5. never fall back to the Hub, another checkpoint, or a base model;
6. set the reloaded model to `eval()`;
7. move original and reloaded models to the same clean MPS/CUDA/CPU device strategy;
8. tokenize the same ordered Phase 13 custom inputs under their respective source/reloaded tokenizers;
9. use truncation, `max_length=80`, no per-sample fixed padding and `DataCollatorWithPadding`;
10. run both models with `torch.inference_mode()`.

The implementation must verify that the reloaded model/config/tokenizer came from the saved package path. Network access is unnecessary and should be disabled or prevented by `local_files_only=True`.

## D. Equivalence Verification

### Fixed inputs

Use the exact ordered custom inputs stored in `docs/result/phase_13_inference_examples.json`. Verify:

- Phase 13 status is PASS;
- `input_provenance == "custom_authored_not_test"`;
- input count is 4;
- recomputed ordered-input signature equals the Phase 13 signature;
- none of the inputs comes from a Test-loading path in Phase 14.

### Tokenizer equivalence

Source-checkpoint and reloaded-package tokenization must have:

- identical ordered `input_ids`;
- identical `attention_mask`;
- identical padded batch shapes;
- identical special tokens and special-token IDs;
- sequence length no greater than 80;
- dynamic padding behavior preserved.

### Model equivalence

For the same batch and same device, compare authoritative-original versus package-reloaded outputs:

- both logits shapes exactly `[4, 2]`;
- all logits, probabilities and confidence values finite;
- predicted integer labels and label names exactly equal;
- output count equals 4 and original input order is preserved;
- maximum absolute logits difference `<= 1e-6`;
- maximum absolute probability difference `<= 1e-7`;
- maximum absolute confidence difference `<= 1e-7`;
- each probability row remains in `[0,1]` and sums to 1 within `1e-6`.

Also compare the loaded state dictionaries:

- parameter/buffer key sets, shapes and dtypes match;
- every corresponding tensor is exactly equal with `torch.equal` after moving to CPU.

Exact tensor equality is the primary proof that save/reload did not change weights; output tolerances accommodate backend-level floating-point behavior. Any predicted-label difference, tensor mismatch, non-finite value, tokenizer mismatch, or tolerance failure makes Phase 14 FAIL.

Optionally compare the original Phase 14 predictions against stored Phase 13 probabilities with absolute tolerance `1e-6` as a continuity check. This check must not replace the direct original-versus-reloaded comparison and must account for the recorded execution device.

## E. Artifact Design

Create after successful implementation and real verification:

### `docs/result/phase_14_saved_model/package_manifest.json`

Contains package provenance, semantic configuration, file inventory, per-file sizes and SHA-256 values, excluding its own recursive hash.

### `docs/result/phase_14_save_reload_verification.json`

Contains at least:

- overall PASS/FAIL status and timestamp;
- saved package path;
- source checkpoint path/name, epoch, step and authoritative SHA-256;
- package weight filename, size and its own SHA-256;
- package manifest path and SHA-256;
- source/reloaded model and tokenizer configuration summaries;
- Phase 13 input signature and count;
- per-input original/reloaded labels, probabilities and confidence;
- logits, probability and confidence maximum absolute differences;
- state-dictionary equivalence summary;
- tokenizer equivalence checks;
- package-file integrity checks;
- all completion checks and tolerances;
- guard action;
- `training_performed=false`;
- `backward_called=false`;
- `optimizer_step_performed=false`;
- `scheduler_step_performed=false`;
- `test_accessed=false`;
- `test_evaluated=false`;
- `test_evaluation_count=1`;
- `checkpoint_selected_or_changed=false`;
- `phase_15_started=false`.

Artifacts must be written atomically and read back for schema/integrity verification. No PASS artifact may be created if an assertion fails.

## F. Notebook Presentation

During implementation, append a Phase 14 section to `notebook_practice_3/practice_3.ipynb`. The notebook must only import/call the Phase 14 module and present returned evidence; it must not duplicate save, reload, hashing, tokenization, forward-pass or comparison logic.

Present:

1. authoritative source checkpoint, epoch/step and abbreviated source SHA-256;
2. saved package path;
3. package file table with relative filename, size and SHA-256;
4. model/config/tokenizer contract summary;
5. original-versus-reloaded prediction table containing text, both labels, both class probabilities, both confidence values and deltas;
6. equivalence-check table and maximum numerical differences;
7. guard action and isolation flags;
8. explicit statements that no training or Test reevaluation occurred and `test_evaluation_count` remains 1;
9. verification artifact and package-manifest paths.

## G. Guard and Integrity Checks

Planned public orchestration:

`run_or_load_phase_14()`

### First valid run

- verify authoritative provenance;
- refuse to use partial/unverified existing package content;
- save into a temporary package directory;
- verify and atomically publish package;
- reload only from the published package;
- perform full equivalence verification;
- write manifest and verification artifact.

### Later Run All executions

- do not call `save_pretrained()` again;
- do not overwrite the valid package;
- verify the package-manifest hash and all recorded file size/hash values;
- verify the current Phase 9/11 provenance still matches the manifest;
- verify the Phase 13 input signature still matches;
- load only the saved package locally;
- load the verified Phase 14 result evidence for presentation;
- optionally run a package-only lightweight inference integrity check on the same custom inputs;
- do not load/evaluate Test;
- return a guard action such as `loaded_verified_phase_14_package_no_resave`.

If the final package path exists but the manifest/artifact is missing, any hash differs, provenance changes, or semantic configuration is invalid, stop with an explicit integrity error. Do not silently repair, delete, overwrite, redownload, select another checkpoint, or mark PASS.

Required integrity checks:

- all expected package files exist, are regular files and are non-empty;
- no unexpected training-state files are packaged;
- every recorded size and SHA-256 matches;
- source checkpoint still has the authoritative Phase 11 SHA-256;
- package model type is DistilBERT;
- `num_labels == 2`;
- mappings are exactly `0 -> NEGATIVE`, `1 -> POSITIVE`;
- tokenizer contract and special tokens match;
- `MAX_TOKEN_LENGTH == 80`, truncation and dynamic-padding methodology remain unchanged;
- Phase 13 custom input signature/order/count match;
- equivalence checks PASS;
- no training/backward/optimizer/scheduler operation exists;
- no Test provider, split, predictions or evaluator is accessed;
- Phase 11 manifest still reports `test_evaluation_count == 1`;
- authoritative checkpoint is neither modified nor reranked.

## H. Completion Criteria

Phase 14 implementation is complete only when:

- `processing_own_phase/phase_14_save_reload.py` owns all core logic;
- checkpoint-1068 epoch 2/step 1068 and its source fingerprint are verified before save;
- model and tokenizer are saved once with Hugging Face `save_pretrained()` into the specified package directory;
- the final package contains all functional model/config/tokenizer assets and excludes training-only state;
- every package file is non-empty, inventoried and hashed;
- model and tokenizer reload exclusively from the saved local package;
- source and reloaded tokenizer batches are identical;
- state dictionaries are exactly equivalent;
- labels match exactly and logits/probabilities/confidence meet the declared tolerances for all four Phase 13 inputs;
- output count/order and locked preprocessing contract are preserved;
- package manifest and Phase 14 verification JSON contain real executed evidence and pass read-back checks;
- notebook Phase 14 calls only the module and displays the required evidence;
- notebook Run All through Phase 14 completes with no Phase 0–13 regression;
- Phase 9 does not retrain, Phase 11 does not reevaluate Test, Phase 13 remains guarded, and Test count remains 1;
- a dated Phase 14 process log is saved under the workflow log directory;
- subsequent guarded execution verifies/loads without overwriting the package;
- Phase 15 has not started.

Expected files in the later implementation turn:

| File | Planned action |
|---|---|
| `processing_own_phase/phase_14_save_reload.py` | Create core save/reload/integrity/equivalence module |
| `docs/result/phase_14_saved_model/` | Create verified reusable local model/tokenizer package |
| `docs/result/phase_14_saved_model/package_manifest.json` | Create package inventory and provenance manifest |
| `docs/result/phase_14_save_reload_verification.json` | Create executed equivalence evidence |
| `notebook_practice_3/practice_3.ipynb` | Add thin Phase 14 call and presentation |
| `docs/save_process_proceduce_own_phase_refactor&fix/<dated-phase-14-log>.md` | Create implementation/verification process log |

## I. Out of Scope

Phase 14 must not:

- train, fine-tune, backpropagate, create optimizer/scheduler steps, or run epochs;
- select, rerank, replace, modify, or improve checkpoint-1068;
- access, reload, predict, evaluate, or analyze the Test split again;
- use Test results to change package content;
- load the reloaded model/tokenizer from the Hub, a base model, or another checkpoint;
- copy optimizer, scheduler, RNG or Trainer state into the reusable inference package;
- change DistilBERT architecture, `num_labels`, label mapping, tokenizer, maximum length 80, truncation or dynamic padding methodology;
- repeat Phase 12 error analysis or change Phase 13 custom inputs without invalidating the guard;
- implement Phase 15 Final Summary or Final Review;
- fabricate a package file, fingerprint, prediction, equivalence result or PASS artifact.

No implementation module, package, verification artifact, notebook cell, process log, model file, tokenizer file, or Phase 15 work is created during this planning-only step.
