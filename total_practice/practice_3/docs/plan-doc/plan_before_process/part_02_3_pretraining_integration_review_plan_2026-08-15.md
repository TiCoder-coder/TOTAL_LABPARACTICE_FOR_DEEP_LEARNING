# Part 2.3 — Pre-training Integration Review Plan

## Objective

Perform the final zero-training integration gate for frozen protocol `practice_3_v2.0` using real v2 Train/Validation, tokenizer, model, collator, TrainingArguments, callbacks, optimizer construction path, and three isolated Trainer instances. Create a readiness/freeze manifest only if all checks pass.

## Frozen protocol

Part 2.1 plan, protocol manifest, split manifest, run configs, registry, and Holdout state are read-only inputs. LR candidates, split/counts, seed, max epochs, patience, threshold, ranking tolerance, model/tokenizer, batches, max length, optimizer, and scheduler cannot change.

## Components under review

- Protocol/config/hash validation.
- Real v2 Train/Validation materialization.
- DistilBERT tokenizer, preprocessing, dynamic collator and dry batches.
- Fresh model construction and forward-only sanity.
- TrainingArguments, EarlyStoppingCallback and Trainer construction for each run.
- Fused AdamW optimizer construction without step.
- Output/checkpoint/TensorBoard/run-state isolation.
- Registry lifecycle/cache/ranking schemas using temporary synthetic copies.
- Pre-lock Holdout denial before any provider callback.
- Execution protocol freeze hash.

## Allowed operations

- Read official Train and Validation from existing cache.
- Select only v2 Train/Validation source IDs.
- Tokenize only selected Train/Validation records.
- Construct fresh models, callbacks, Trainer instances and optimizers.
- Run inference-mode forward checks on Train/Validation batches.
- Use temporary synthetic registry/winner/guard state.
- Write the readiness manifest, Part 2.3 tests/log, and a narrowly scoped bug-fix plan if required.

## Prohibited operations

- `Trainer.train()`, backward, optimizer/scheduler step, or model weight mutation.
- Official Test loading.
- v2 Holdout indexing, selection, tokenization, batching, inference, evaluation, examples, or EDA.
- Real registry transition/ranking/winner creation.
- Any v1 artifact, frozen protocol, notebook, or current-flow modification.

## Dataset-access boundary

Load the official Train and Validation Dataset objects only. Parse the frozen manifest IDs, select/index only IDs assigned to v2 Train or v2 Validation, and return only those two datasets. Holdout IDs may be counted/hashed from metadata but must never be used to index the source dataset in Part 2.3. Official Test must never be requested.

## Model/tokenizer construction

Use cached `distilbert-base-uncased`, `AutoTokenizer`, and a fresh `AutoModelForSequenceClassification(num_labels=2)` for each run. Tokenization remains max length 80 with truncation and dynamic padding. Forward checks use inference mode and finite logits/loss assertions only.

## Trainer construction strategy

For each run, construct isolated TrainingArguments, EarlyStoppingCallback, fresh model, fused AdamW optimizer and Trainer using v2 Train/Validation only. Do not call any training/evaluation method. Verify attached datasets/callback/config and dispose each instance before the next run.

## Run isolation checks

Programmatically verify distinct run ID, config hash, checkpoint/output path, TensorBoard path, model identity, optimizer identity and Trainer identity. Experimental fixed fields must match; only LR differs.

## TensorBoard checks

Verify three unique declared log directories and required scalar contract. Trainer construction may create empty directories; no event or scalar is written.

## Freeze/readiness strategy

Canonical SHA-256 over the Part 2.1 plan, protocol manifest, split manifest, three run configs, and frozen ranking representation becomes `execution_protocol_hash`. The readiness manifest records only integration evidence and booleans, never fake training metrics.

## Validation

- Existing 11 synthetic infrastructure tests.
- New integration tests/checks for real Train/Validation counts, labels, tokenizer fields/dtypes, dry batches, fresh models, forward-only output, three Trainer/optimizer/callback constructions, isolation, registry/cache schema, pre-lock Holdout denial and official-Test exclusion.
- Static search for forbidden executed operations.
- v1 Test count and checkpoint/package hash comparison.
- Protocol/readiness manifest schema and hash read-back.

## Definition of done

All real zero-training checks pass; official Test is not loaded; Holdout remains sealed, unmaterialized and count zero; real registry stays PLANNED; no real ranking/winner/training occurs; v1 remains unchanged; execution protocol hash is stored; and `pretraining_readiness_manifest.json` records `ready_for_training=true`.

