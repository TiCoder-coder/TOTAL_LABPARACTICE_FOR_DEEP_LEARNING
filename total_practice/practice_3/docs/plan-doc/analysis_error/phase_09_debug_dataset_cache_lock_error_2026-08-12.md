# Phase 09 Debug Dataset Cache Lock Error — 2026-08-12

## Error

The mandatory Phase 9 debug command stopped before Trainer construction with:

```text
PermissionError: [Errno 1] Operation not permitted:
~/.cache/huggingface/datasets/_Users_vientu_.cache_huggingface_datasets_
cornell-movie-review-data___rotten_tomatoes_default_0.0.0_
aa13bc287fa6fcab6daf52f0dfb9994269ffea28.lock
```

## Phase

Phase 9 — mandatory debug run, before full fine-tuning.

## Context

- Repository `.venv` Python 3.11 was used.
- Hugging Face Hub, Transformers and Datasets were placed in offline mode.
- The Rotten Tomatoes dataset already exists in the local cache.
- The failure occurred while `load_dataset()` attempted to acquire its normal
  cache lock, before tokenization, model construction or `run_debug()`.

## Root Cause

The workspace sandbox permits reading the existing Hugging Face cache but does
not permit the datasets library to open/acquire its lock file under
`~/.cache/huggingface/datasets`. This is an execution-permission issue, not a
dataset, preprocessing, Trainer or Phase 9 logic failure.

## Evidence and Safety State

- Debug Trainer created: false.
- Debug training started: false.
- Optimizer/scheduler step: false.
- Full model constructed after debug: false.
- Full training started: false.
- Test accessed: false.
- No Phase 9 checkpoint or fabricated training artifact was created.

## Required Resolution

After explicit approval, rerun only the mandatory debug command with permission
to use the existing Hugging Face cache lock. Keep all Hugging Face components
offline and do not download or replace the dataset/model. If and only if the
executed debug verification reports PASS, proceed to the approved fresh full
baseline.

## Resolution

After explicit user approval, the same offline command was rerun with access
to the existing cache lock. No dataset or model was downloaded or replaced.
The 1,000/200 one-epoch debug run completed with 63 optimizer steps and PASS,
after which the separately authorized fresh full baseline completed.

## Status

RESOLVED.
