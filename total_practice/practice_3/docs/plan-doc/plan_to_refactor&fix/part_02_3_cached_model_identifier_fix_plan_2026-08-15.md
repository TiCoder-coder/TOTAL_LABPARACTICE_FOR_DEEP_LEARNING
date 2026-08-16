# Part 2.3 — Cached Model Identifier Resolution Fix Plan

## Change

Add a strict resolver in the integration module:

`distilbert-base-uncased → distilbert/distilbert-base-uncased`

Only this frozen alias is accepted. Run configs and protocol manifests remain unchanged.

## Verification

- Load canonical cached checkpoint with `local_files_only=True`.
- Confirm model type `distilbert`, `num_labels=2`, label mapping and tokenizer family.
- Record requested/resolved IDs.
- Rerun integration from the beginning without training.

## Failure behavior

Any other missing identifier or model-family/config mismatch stops Part 2.3. No download, alternative model or protocol edit is allowed.

