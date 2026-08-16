# PART 3.2.F — Tokenizer Correction v2.1 Process Log

Date: 2026-08-15

## Root cause

v2.0 reused the model alias for tokenizer loading. The namespaced cache contained valid model weights but no canonical tokenizer assets, producing a five-special-token tokenizer and all-`[UNK]` lexical input. v2.1 separates model `distilbert/distilbert-base-uncased` from tokenizer `distilbert-base-uncased`.

## Files created/changed

- `docs/plan-doc/plan_before_process/part_03_2_f_tokenizer_correction_v2_1_plan_2026-08-15.md`
- `processing_own_phase/tokenizer_correction_v2_1.py`
- `processing_own_phase/experiment_runner_v2_1.py`
- `tests/test_tokenizer_correction_v2_1.py`
- `tests/test_training_runner_v2.py` (updated stale expectation: completed v2.0 run must be rejected)
- `docs/result/practice_3_v2_1/` metadata/artifacts
- `runs/practice_3_v2_1/` three isolated empty log directories
- this process log

## Tokenizer correction and evidence

- Tokenizer identifier: `distilbert-base-uncased`
- Tokenizer class: `BertTokenizer`
- Vocab size: 30,522
- Known words `this`, `movie`, `good`, `bad`, `positive`, `negative`: all recognized
- Deterministic Train sample count: 128
- Unique texts/encodings: 128/128
- Lexical `[UNK]` ratio: 0.0
- Tokenizer integrity: PASS

## Split and protocol

- Reused the unchanged validated v2.0 split manifest.
- Train/Validation/Holdout: 7,676 / 960 / 960.
- Split manifest hash: `4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13`.
- Protocol version: `practice_3_v2.1`.
- Execution protocol hash: `be7e3900b6c25fecd8212eedcdfbfba9c3119e231539da4169b7cf5806f00427`.
- Run states: `PLANNED / PLANNED / PLANNED`.
- Holdout: SEALED, evaluation count 0.

## Smoke and zero-training integration

The corrected tokenizer passed a manual PyTorch Train-only known-easy smoke test at 100% accuracy. The smoke used no Trainer training, Validation, Holdout or Test.

Zero-training integration materialized/tokenized only Train and Validation, constructed three fresh models, Trainers, fused AdamW optimizers, linear schedulers and Early Stopping callbacks, inspected dynamic batches and ran inference-only forward sanity. No `Trainer.train()`, evaluation, optimizer step, TensorBoard event, checkpoint or real metric was produced.

## Authorization

- Artifact: `docs/result/practice_3_v2_1/part_03_2_f_training_authorization.json`
- Tokenizer integrity: PASS
- Smoke: PASS
- Zero-training integration: PASS
- Authorization self-hash: PASS
- `training_authorized=true`

## Tests

Command:

```bash
cd total_practice/practice_3
HF_HUB_OFFLINE=1 ../../.venv/bin/python -m unittest -v \
  tests.test_tokenizer_correction_v2_1 \
  tests.test_training_runner_v2 \
  tests.test_protocol_v2
```

Result: 24 tests PASS. Coverage includes resolver separation, valid vocabulary, known-token checks, high-UNK rejection, unique-encoding rejection, config hashes, registry states, Holdout seal, authorization self-hash, CLI parsing and read-only preflight.

## Explicit safety confirmation

- v2.0 registry/status/evidence modified: false
- `Trainer.train()` executed: false
- official experiment executed: false
- real checkpoint/metric/event created: false
- Holdout accessed/materialized: false
- official Test loaded: false
- winner selected: false

## First manual command

From repository root:

```bash
source .venv/bin/activate
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m total_practice.practice_3.processing_own_phase.experiment_runner_v2_1 --run-id p3v21_lr_2e-5
```
