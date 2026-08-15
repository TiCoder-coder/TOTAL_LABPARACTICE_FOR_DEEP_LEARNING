# Phase 05 EDA Enhancement Plan — 2026-08-12

## Objective

Enhance Phase 5 with task-focused NLP EDA presented as tables, figures,
interpretation and an evidence-based preprocessing decision. Preserve the
already verified Phase 0–6 methodology, official dataset splits, tokenizer and
`MAX_TOKEN_LENGTH=80` unless executed evidence disproves the existing result.

## Input

- `cornell-movie-review-data/rotten_tomatoes`
- Official train/validation/test splits: 8,530 / 1,066 / 1,066
- `distilbert-base-uncased` tokenizer
- Existing Phase 5 summaries and token-length histogram
- Existing Phase 6 preprocessing contract

## Existing EDA to Preserve

- Schema verification for `text` and `label`
- Null, empty-string and whitespace-only checks
- Label validation and per-split counts/percentages
- Exact within-split duplicates
- Exact cross-split overlap
- Character, word and token statistics
- Token P95, P99 and maximum
- Evidence-based `MAX_TOKEN_LENGTH` justification
- Existing Phase 5 JSON artifacts and token histogram

## EDA Enhancements

1. Produce presentation-ready schema, quality, label, duplicate and overlap
   records without moving processing logic into the notebook.
2. Compute character, word and token statistics for all three splits.
3. Select deterministic representative Positive and Negative samples using a
   fixed, documented index rule.
4. Select deterministic shortest and longest examples by token length, with
   stable index tie-breaking and associated label/length fields.
5. Quantify expected truncation impact at the configured maximum length for
   every split.
6. Generate a concise EDA conclusion that connects executed evidence to Phase
   6 preprocessing.

## Visualizations

- `phase_05_label_distribution.png`
- `phase_05_character_length_distribution.png`
- `phase_05_word_length_distribution.png`
- Reuse `phase_05_token_length_distribution.png`

No CV-specific RGB, brightness, image-gallery, raw-pixel PCA or t-SNE analysis
will be added.

## Result Artifacts

- Update `phase_05_eda_summary.json` with all-split statistics, deterministic
  sample evidence, truncation impact and conclusion fields.
- Update `phase_05_token_length_statistics.json` with per-split token evidence
  and the final max-length decision.
- Create the three new Phase 5 PNG figures listed above.
- Preserve/rebuild the existing token-length histogram at its current path.
- Re-run and update `phase_06_preprocessing_verification.json` only through the
  existing Phase 6 verification functions.

## Notebook Presentation

The Phase 5 notebook cell will only call reusable functions and display:

- dataset schema and quality tables;
- label counts/percentages plus figure;
- duplicate/overlap table;
- all-split length-statistics table;
- character, word and token figures;
- deterministic representative and extreme-sample tables;
- P95/P99/max → selected max length → expected truncation impact;
- concise EDA conclusion and explicit transition to Phase 6.

The Phase 6 cell will retain its existing calls and verification output.

## Files Affected

- `processing_own_phase/phase_05_dataset_eda.py`
- `notebook_practice_3/practice_3.ipynb`
- `docs/result/phase_05_eda_summary.json`
- `docs/result/phase_05_token_length_statistics.json`
- Phase 5 PNG artifacts in `docs/result/`
- `docs/result/phase_06_preprocessing_verification.json` after re-verification
- Phase 5 enhancement process log

Phase 0–4 and Phase 7 onward are out of scope. Phase 6 implementation will not
be refactored unless verification exposes a blocking defect.

## Validation / Sanity Checks

- Official splits and sample counts remain unchanged.
- Required schema is present in all splits.
- Every label belongs to `{0, 1}`.
- Existing null/empty/whitespace, duplicate and overlap checks remain active.
- Length statistics and sample tables are derived from executed data.
- Deterministic sample selection returns stable indices.
- Figure files exist and are non-empty.
- `MAX_TOKEN_LENGTH=80` is at least every observed sequence maximum, or any
  contrary evidence is explicitly reported before changing it.
- Expected truncation counts are computed, not asserted unconditionally.
- Phase 6 sample counts, fields, labels, decode checks and dynamic-padding
  assertions all pass after the Phase 5 enhancement.
- Notebook executes continuously through Phase 6 without unresolved errors.

## Completion Criteria

- Plan exists before implementation.
- Phase 5 processing module owns all added computation and plotting logic.
- Notebook calls functions and presents real tables/figures/conclusions.
- Phase 5 is executed and all result artifacts are saved.
- Final P95/P99/max and truncation impact are reported from real data.
- Phase 6 is re-run and all preprocessing/dynamic-padding checks pass.
- No dataset, split, tokenizer, Phase 0–4 methodology or Phase 7 code changes.

