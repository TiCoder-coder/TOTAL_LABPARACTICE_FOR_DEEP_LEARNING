# Phase 05 EDA Enhancement Log — 2026-08-12

## Scope

Enhanced Phase 5 NLP EDA using the presentation principle from Practice 2:

```text
EDA → TABLE → FIGURE → INTERPRETATION → PREPROCESSING DECISION
```

Phase 0–4 methodology, official splits, tokenizer, `MAX_TOKEN_LENGTH`, Phase 6
implementation and Phase 7 onward were not changed. No training ran.

## Implementation

- Preserved schema, text-quality, label, duplicate, overlap and existing length
  checks.
- Added all-split character/word/token statistics.
- Added deterministic representative samples: first three indexed examples per
  label from Train.
- Added shortest/longest samples by token length with stable index tie-breaking.
- Added calculated truncation impact for all official splits.
- Added label, character-length and word-length figures; regenerated the
  existing token-length figure.
- Updated the notebook to call Phase 5 functions and display real tables,
  figures, decision evidence and conclusions.

## Executed Phase 5 Evidence

- Schema: PASS for train/validation/test.
- Null/empty/whitespace-only: zero for every split.
- Labels: exactly balanced.
  - Train: 4,265 Negative / 4,265 Positive.
  - Validation: 533 / 533.
  - Test: 533 / 533.
- Exact within-split duplicates: zero.
- Exact cross-split overlaps: zero.
- Train token P95/P99/max: 47 / 56 / 78.
- Validation max: 72.
- Test max: 67.
- Selected `MAX_TOKEN_LENGTH`: 80.
- Observed samples over length 80: zero in all splits.

## Phase 6 Re-verification

- `input_ids`: PASS.
- `attention_mask`: PASS.
- Labels `{0,1}`: PASS.
- Sample counts preserved: PASS (8,530 / 1,066 / 1,066).
- Maximum sequence lengths: 78 / 72 / 67, all within 80.
- Decode sanity checks: PASS, three overlap ratios 1.0.
- Dynamic padding: PASS.
  - Original lengths: 3, 47, 78.
  - Batch shape: `[3, 78]`.
  - Padding counts: 75, 31, 0.
  - Attention-mask padding matched expected counts.

## Notebook Verification

- Offline cache only; no dataset/model download.
- Seven code cells executed continuously with execution counts 1–7.
- Error outputs: zero.
- Phase 5 marker: PASS.
- Phase 6 marker: PASS.

The first attempt selected an external Anaconda kernel and failed before Phase
5. The resolved kernel mismatch is documented separately in
`docs/plan-doc/analysis_error/phase_05_notebook_kernel_mismatch_2026-08-12.md`.

## Final Status

- Phase 5 enhancement: PASS.
- Phase 6 re-verification: PASS.
- No regression detected in the verified Phase 0–6 contract.
- Ready for Phase 7, but Phase 7 was not started.

