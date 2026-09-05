# Phase 51 — Worst-Error Analysis Report

**Phase**: 51 (finalization 51-G)  
**Status**: PASS (after both correctives — frozen-contract on 51-C, exact-input on 51-F)  
**Model**: Transformer (FS2_TF1), seeds 42 / 123 / 2026  
**Test population**: N = 2961  
**Frozen selection contract SHA256**: `ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4`  

## 1. Objective

Phase 51 performs descriptive, post-hoc worst-case error analysis on the final Transformer (FS2_TF1) Test predictions across three seeds, without reranking, retraining, manual case selection, or attention analysis.

## 2. Frozen Contracts

- Selection contract SHA256: `ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4`
- Test population fingerprint SHA256: `d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87`
- Phase 50 regime assignment SHA256: `e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac`
- Phase 50 train-only threshold SHA256: `2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109`
- Lookback = `72`
- Feature count = `33`
- Feature set = `FS2_TF1`
- Boundary protocol = `WB0_CONTEXT_CARRY_OVER`

## 3. Source Integrity

All Phase 51-B/C/D/E/F artifacts are verified against their documented SHAs and remained unchanged during Phase 51-G. Phase 47-50 upstream scientific artifacts (Test predictions, residuals, regime labels, thresholds, feature contract, scaling contract, boundary contract, data region contract, LSTM eligibility) remained unchanged.

## 4. Worst-Case Selection (W1)

- Per-seed Top20 (W1): 60 rows across 3 seeds.
- Shared across all 3 seeds Top20 (W2): 20 rows.
- Three-seed W1 intersection: 17 targets (descriptive, descriptive-only).

## 5. Cross-Seed Consistency

W1 pairwise Top20 Jaccard similarity was high across all three seed pairs, indicating that the worst-error signal is shared across Final Transformer re-runs:

- w1_pairwise_123-2026: intersection=18, union=22, Jaccard=0.8182
- w1_pairwise_123-42: intersection=18, union=22, Jaccard=0.8182
- w1_pairwise_2026-42: intersection=19, union=21, Jaccard=0.9048

## 6. Error Concentration

Top 20 cases (≈ 0.7% of Test) account for a disproportionate share of SAE/SSE:

- seed 42: SAE share = 10.29%, SSE share = 33.13%
- seed 123: SAE share = 10.77%, SSE share = 33.55%
- seed 2026: SAE share = 10.38%, SSE share = 32.48%

## 7. Regime Context

- Phase 50 regime assignment SHA256: `e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac` (frozen).
- Regime overrepresentation rows: 216.
Descriptive observation: certain regime labels (e.g., R2_EXTREME_HIGH, R1_TARGET_LEVEL=TL_HIGH) tend to be over-represented among the selected worst-case lists, consistent with the global Test behavior.

## 8. Persistence / LSTM Context

- Persistence rows (case-level): 160.
- LSTM status: **NOT_ELIGIBLE_CONFIG_MISMATCH**.
- Canonical reason (verbatim): 
No LSTM inference is performed. No LSTM casebook exists. The casebook uses Transformer-only canonical evidence.

## 9. Local Temporal Context

- Local context audit rows: 160.
Centers are present for all cases; boundary/gap rows are marked UNAVAILABLE_BOUNDARY / UNAVAILABLE_GAP. No padding, no interpolation.

## 10. Exact Model-Input Context

- Exact 72×33 input-window reconstructions: 44
- Verified (44) reconstructions pass shape/dimensionality/finiteness checks.
Coordinates are split into RAW_FEATURE_CONTEXT (from FEATURES-v1) and MODEL_VISIBLE_FEATURE_CONTEXT (transformed via FINAL_SCALING-v1 transform-only). Reconstruction is fully read-only: no model inference, no checkpoint loading, no scaler fit.

## 11. Casebook

- Case memberships: 160
- Unique targets: 44
Case IDs are deterministic: CASE_{family_short}_{seed}_rank{NNN}_{target_id}.

## 12. Scientific Findings (descriptive, post-hoc, non-causal)

- 19 structured findings, all descriptive.
Top20 cases concentrate the predictive error: a small fraction of Test accounts for a large share of SSE; case identity is highly shared across seeds; underprediction shared membership is more consistent than overprediction shared membership; extreme-high target levels are over-represented. None of these observations constitute root-cause claims about the model.

## 13. Limitations

Phase 51 analysis is descriptive, post-hoc, and selection-conditioned. Conclusions drawn from selected worst cases do not generalize to the broader Test population. LSTM analysis is NOT_APPLICABLE in Phase 51. No attention analysis is performed; that is reserved for Phase 52.

## 14. Safety / Governance

No new Test inference, no checkpoint loading, no training, no scaler fit. Phase 47-50 upstream artifacts unchanged. Best seed was not selected; no ensemble was produced; no prediction correction was applied.

## 15. Phase 52 Handoff

The handoff is stored at `artifacts/worst_error_analysis/phase52_attention_extraction_handoff.json`. It contains:

- final candidate lineage (model, seeds, N_TEST, lookback, feature_count, feature_set, WB0)
- selection contract SHA256
- ranking / casebook / exact-input reconstruction SHAs and counts
- residual convention
- LSTM canonical status & verbatim reason
- phase50 regime context reference
- explicit `phase52_authorized = false`
- safety statement.

Phase 52 may consume this handoff ONLY after a separate human approval gate.

## 16. Conclusion

Phase 51-G finalizes the descriptive worst-error analysis produced by Phase 51-A through F. All required artifacts, figures, findings, discrepancies, tests artifact, summary, report, README, handoff, and signoff are present. The Phase 51 signoff gate passes. Execution of Phase 52 attention analysis requires its own human authorization.
