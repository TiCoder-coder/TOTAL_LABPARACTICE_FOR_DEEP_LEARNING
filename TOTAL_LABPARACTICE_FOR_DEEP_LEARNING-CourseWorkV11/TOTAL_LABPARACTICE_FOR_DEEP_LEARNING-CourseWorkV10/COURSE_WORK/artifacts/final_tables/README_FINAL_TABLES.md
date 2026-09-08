# Phase 58 — Final Tables (README)

This README explains the canonical Phase 58 final-report table package.

## Main vs Appendix

- **Main tables** (FT01–FT10): presented in the report body.
- **Appendix tables** (FA01–FA12): full detail for audit.

## Authoritative Upstream Sources

| Table | Source phase | Evidence class |
|-------|--------------|----------------|
| FT01 | Phase 45 final lock | FROZEN_CONFIG |
| FT02 | Phase 47 Test metrics | HELD_OUT_TEST |
| FT03 | Phase 44 rolling-origin | DEVELOPMENT |
| FT04 | Phase 48 + Phase 49 | POST_TEST_DIAGNOSTIC |
| FT05 | Phase 50 + Phase 51 | POST_TEST_DIAGNOSTIC |
| FT06 | Phase 54 | POST_TEST_DIAGNOSTIC |
| FT07 | Phase 55 | POST_TEST_DIAGNOSTIC |
| FT08 | Phase 56 | POST_TEST_DIAGNOSTIC |
| FT09 | Phase 57 | POST_TEST_DIAGNOSTIC |
| FT10 | upstream-supported claim trace | EVIDENCE_AND_LIMITATION |

## Why full-precision CSV precedes report rounding

`tables/csv/FT*.csv` preserves upstream full precision. Markdown and LaTeX emit display-rounded values only. Display rounding is applied at output time; aggregates were computed on full precision.

## Why three-seed summary is NOT an ensemble

Mean ± SD (ddof=1) across three seed-level metrics is descriptive run-variability, not a forecast ensemble. Persistence and Tuned LSTM baselines are also single-value.

## Why development and Test evidence are separated

FT03 (rolling-origin) carries `DEVELOPMENT_EVIDENCE` and is presented in a separate panel. FT02 (Phase 47) carries `HELD_OUT_TEST_EVIDENCE` and is the primary final generalization evidence.

## Why no best-value highlighting is used

No bold/color-coded winner. Tables report Held-Out Test evidence without creating post-Test selection.

## Why attention tables are diagnostic

Phase 54–57 attention analyses are post-Test diagnostics, not model-selection inputs.

## Why Phase 56 error cohorts are not deployment regimes

HIGH_ERROR / LOW_ERROR are Test-relative diagnostic cohorts. NOT deployment regimes, NOT retuning evidence.

## Why same-index attention heads are not assumed aligned across seeds

Phase 57 canonical JSD matching resolves head permutation. Same-index across seeds does NOT have guaranteed semantic alignment.

## How table checksums work

`final_table_checksums.json` records SHA256 over CSV / Markdown / LaTeX for each table.

## How Phase 59 should use the claim-traceability file

`table_claim_traceability.csv` lists which phases / tables support which claim templates (C1..C10) and what is prohibited. Phase 59 may ONLY use allowed-claim scope; never derive a new claim beyond that.
