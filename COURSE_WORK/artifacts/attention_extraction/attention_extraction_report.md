# Phase 52 -- Attention Extraction Report

## 1. Objective

Phase 52 is the extraction + integrity phase for the temporal self-attention
of the three frozen final Transformer seeds (42/123/2026). Phase 52 does NOT
interpret attention causally or as feature importance.

## 2. Scope

- Per-seed dense attention for the frozen Phase 51 worst-error case set (44 unique cases)
- Per-seed all-Test last-query attention (N_TEST=2961)
- Audit / ordering / summary / handoff artifacts
- Phase 53-57 handoff files (handoff only, no implementation)

## 3. Hard contracts (frozen before extraction)

- lookback L = 72, feature count F = 33
- pooling = LAST_STEP, RevIN = disabled
- attention_axes = B_H_Q_S, self_attention_shape = B_H_L_L
- need_weights = True, average_attn_weights = False
- last_query_definition = A[:, :, L-1, :]
- prediction_equivalence tolerance: rtol = atol = 1e-5
- raw dtype = float32, no averaging, no thresholding, no smoothing
- same dense case set + same order for all three seeds

## 4. Upstream evidence (verified, unmodified)

- Phase 51 signoff: PASS, ready_for_phase52 = true
- Test population: FINAL_TEST_POP-v1 (N_TEST = 2961, SHA256 = d7dbc0b3cde772dc...)
- Selection contract: ec798326cb03586e... (frozen by Phase 51)
- Final scalers: FINAL_SCALING-v1 (transform-only, fit forbidden)
- Final lock SHA: 81fb87c44b6af31b...
- Checkpoints verified strict-load for seeds 42/123/2026

## 5. Per-seed extraction results

| Seed | Prediction Eq | Max Abs Diff (Wh) | Model Mutation | Reproducibility |
|------|---------------|-------------------|----------------|------------------|
| 42 | PASS | 2.99e-04 | PASS | PASS |
| 123 | PASS | 5.23e-04 | PASS | PASS |
| 2026 | PASS | 4.60e-04 | PASS | PASS |

## 6. Raw storage (two-tier, float32)

| File | Shape | Dtype | Size |
|------|-------|-------|------|
| dense_case_attention_seed42.npz | [44, 2, 4, 72, 72] | float32 | ~7 MB |
| dense_case_attention_seed123.npz | [44, 2, 4, 72, 72] | float32 | ~7 MB |
| dense_case_attention_seed2026.npz | [44, 2, 4, 72, 72] | float32 | ~7 MB |
| last_query_attention_seed42.npz | [2961, 2, 4, 72] | float32 | ~6.6 MB |
| last_query_attention_seed123.npz | [2961, 2, 4, 72] | float32 | ~6.6 MB |
| last_query_attention_seed2026.npz | [2961, 2, 4, 72] | float32 | ~6.6 MB |

## 7. Integrity audits

- attention finite: PASS for all (sample, layer, head) batches
- attention nonnegative (>= -1e-7): PASS for all
- max(attention) <= 1 + 1e-6: PASS for all
- attention row sums = 1.0: PASS for all (atol=rtol=1e-5)
- no renormalization applied to raw source
- dense vs last-query consistency: PASS for all (K x L x H) pairs

## 8. Position/lag mapping

- Position 0 = oldest historical input: lag 72, 720 min
- Position L-1 = newest historical input: lag 1, 10 min
- Forecast target is NOT an attention token
- Last-query attention = A[:, :, L-1, :] (NOT A[:, :, :, L-1])

## 9. Findings

See attention_extraction_findings.csv (21 findings: PASS or CAVEAT).
CAVEAT findings: ATTENTION_IS_TEMPORAL, ATTENTION_IS_NOT_CAUSAL,
HEAD_ALIGNMENT_NOT_ASSUMED. PASS findings include API verification,
prediction equivalence, shape verification, probability audits,
no model mutation, reproducibility, coverage completeness.

## 10. Discrepancies

See attention_extraction_discrepancies.json. 7 issues, all RESOLVED.

## 11. Phase 53-57 handoffs

All 5 handoff JSONs written. Each: ready_for_*=true, *_authorized=false.
Implementation requires separate Human governance gate.

## 12. O52 inventory

- O52.01: PASS
- O52.02: PASS
- O52.03: PASS
- O52.04: PASS
- O52.05: PASS
- O52.06: PASS
- O52.07: PASS
- O52.08: PASS
- O52.09: PASS
- O52.10: PASS
- O52.11: PASS
- O52.12: PASS
- O52.13: PASS
- O52.14: PASS
- O52.15: PASS
- O52.16: PASS
- O52.17: PASS
- O52.18: PASS
- O52.19: PASS
- O52.20: PASS
- O52.21: PASS
- O52.22: PASS
- O52.23: PASS
- O52.24: PASS
- O52.25: PASS
- O52.26: PASS
- O52.27: PASS
- O52.28: PASS
- O52.29: PASS
- O52.30: PASS
- O52.31: PASS
- O52.32: PASS
- O52.33: PASS
- O52.34: PASS
- O52.35: PASS
- O52.36: PASS
- O52.37: PASS

## 13. Conclusion

Phase 52 PASSES its strict extraction + integrity gate. Raw attention is
frozen, audit-verified, and SHA256-anchored. Phase 53-57 may read these
artifacts under separate Human authorization.
