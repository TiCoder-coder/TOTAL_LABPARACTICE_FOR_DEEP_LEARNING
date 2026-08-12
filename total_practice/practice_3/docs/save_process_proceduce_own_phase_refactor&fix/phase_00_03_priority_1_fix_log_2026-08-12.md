# Phase 0–3 Priority 1 Fix Log — 2026-08-12

## Scope

Only Priority 1 — Make the pipeline runnable.

## Changes

- Normalized internal config imports to package-relative imports.
- Scoped `PROJECT_ROOT` and `RESULT_DIR` to `total_practice/practice_3`.
- Fixed the Phase 3 notebook syntax error.
- Updated the Phase 3 notebook call to display the token table, token IDs,
  attention mask, decode sanity check and tokenizer comparison.
- Normalized the notebook config import to
  `processing_own_phase.config`.
- Installed the missing dependencies in the repository `.venv` using the
  versions declared by Practice 3.
- Aligned the expected Torch version with `requirements.txt` (`2.13.0`);
  the installed package was not replaced.

## Verification Evidence

- Package imports for Phase 1, Phase 2 and Phase 3: PASS.
- Practice root resolved to `total_practice/practice_3`: PASS.
- Result directory resolved to `total_practice/practice_3/docs/result`: PASS.
- Phase 1 environment check: PASS; Python 3.11.14, device `mps`, Hugging Face
  connectivity OK.
- Phase 2 pretrained inference: PASS for three example sentences; tokenizer
  vocabulary size 30,522.
- Phase 3 token table: PASS.
- Phase 3 token IDs and attention mask: PASS.
- Phase 3 decode sanity check: PASS, word overlap ratio 1.0.
- Fine-tuned/base tokenizer comparison: PASS (`are_identical=True`).

## Artifacts

- `docs/result/2026-08-10_phase01-environment-log.json`
- Executed outputs in `notebook_practice_3/practice_3.ipynb`, cells 13–16.

## Remaining Scope

Phase 4–6 issues from the audit remain intentionally untouched. The Phase 6
handoff remains unapproved until Priority 2 and later verification are complete.
