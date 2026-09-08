# FINAL TABLE CATALOG (Phase 58)
_Version: FINAL_TABLES-v1  •  Date: 2026-09-04  •  Seeds: 42 / 123 / 2026_

## Main Tables (FT)

### FT01 — Final Experimental and Model Configuration
- Evidence class: `METHOD / LOCKED_CONFIG`
- Rows: 29
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Frozen upstream source. See final_table_source_ledger.csv for cell lineage.; Mean ± SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.

### FT02 — Final Held-Out Test Performance
- Evidence class: `HELD_OUT_TEST_EVIDENCE`
- Rows: 6
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Frozen upstream source. See final_table_source_ledger.csv for cell lineage.; Mean ± SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.

### FT03 — Rolling-Origin Temporal Robustness (DEVELOPMENT_EVIDENCE)
- Evidence class: `DEVELOPMENT_EVIDENCE`
- Rows: 5
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: DEVELOPMENT_EVIDENCE only. Pooled RMSE is the primary criterion; mean fold RMSE is secondary.

### FT04 — Final Prediction and Residual Diagnostics
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 30
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Post-Test diagnostics only. No ensemble residual.

### FT05 — Error-by-Regime and Worst-Error Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 165
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Phase 50 regime thresholds FROZEN.; Worst cases are NOT removed from final Test metrics.

### FT06 — Last-Query Temporal Attention Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 432
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Attention values describe temporal allocation, not raw feature importance.

### FT07 — Within-Seed Head Comparison Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 6
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Similarity does NOT prove functional redundancy.

### FT08 — Error-Conditioned Attention Summary
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 108
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: HIGH/LOW are Test diagnostic cohorts; not deployment regimes.; Error-attention associations are descriptive; not causal.

### FT09 — Seed-Stability Attention Summary (S57-A + S57-B + S57-C)
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 80
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Same-index heads NOT assumed semantically aligned (Phase 57 canonical JSD matching).; Layer 0 cycle consistency 1/4; Layer 1 4/4 (descriptive).

### FT10 — Evidence and Limitation Summary
- Evidence class: `EVIDENCE_AND_LIMITATION`
- Rows: 10
- Insertion recommendation: main report body / appendix (per type)
- Important footnotes: Single-house dataset. No multi-house generalization possible.

## Appendix Tables (FA)

### FA01 — Per-Seed Final Test Metrics (full precision)
- Evidence class: `HELD_OUT_TEST_EVIDENCE`
- Rows: 3
- Footer: Frozen upstream source. See final_table_source_ledger.csv for cell lineage.; Mean ± SD (ddof=1) of seed-level metrics; descriptive only; NOT an ensemble forecast.

### FA02 — Rolling-Origin Fold-Level Metrics (DEVELOPMENT_EVIDENCE)
- Evidence class: `DEVELOPMENT_EVIDENCE`
- Rows: 12
- Footer: DEVELOPMENT evidence only.

### FA03 — Residual Distribution Details
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 3
- Footer: Residual = y_true - y_pred; UNDER = residual > 0; OVER = residual < 0.

### FA04 — Full Error-by-Regime Results
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 54
- Footer: Phase 50 regime thresholds FROZEN.

### FA05 — Shared Worst-Error Cases (Phase 51 frozen)
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 160
- Footer: Worst-error cases were NOT removed from final Test metrics.

### FA06 — Full Last-Query Head Metrics
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 71064
- Footer: Attention = temporal allocation only. NOT feature importance.

### FA07 — Full Head Pairwise Comparison
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 36
- Footer: No composite diversity score; no head pruning implication.

### FA08 — Full Error-Conditioned Attention Coefficients
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 432
- Footer: Diagnostic only; not deployment regimes; not causal.

### FA09 — Head Matching and Ambiguity Details
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 24
- Footer: Anchor = seed42 (first predeclared final seed, NOT performance-based).; Same-index heads NOT assumed semantically aligned.

### FA10 — Attention Seed-Stability Details
- Evidence class: `POST_TEST_DIAGNOSTIC_EVIDENCE`
- Rows: 168
- Footer: No target-specific rematch; consensus profiles computed AFTER matching.

### FA11 — Provenance and Population Audit
- Evidence class: `EVIDENCE_AND_LIMITATION`
- Rows: 19
- Footer: Critical for reproducibility.

### FA12 — Upstream Warnings and Reporting Caveats
- Evidence class: `EVIDENCE_AND_LIMITATION`
- Rows: 13
- Footer: Caveats propagated from upstream phases.

## Forbidden Post-Test Selection Reminders

- No best-value bolding; no color-coded winner.
- Three-seed summary = mean ± sample SD (ddof=1) of seed-level metrics. NOT an ensemble.
- Phase 50 regimes FROZEN; thresholds unchanged.
- Phase 51 worst-case ranks FROZEN; cases were NOT removed.
- Same-index heads across seeds NOT assumed semantically aligned (Phase 57 canonical JSD).
- Attention labels = descriptive allocation only; NOT feature importance; NOT causal.
- No MAPE; no new significance test; no new confidence interval.
