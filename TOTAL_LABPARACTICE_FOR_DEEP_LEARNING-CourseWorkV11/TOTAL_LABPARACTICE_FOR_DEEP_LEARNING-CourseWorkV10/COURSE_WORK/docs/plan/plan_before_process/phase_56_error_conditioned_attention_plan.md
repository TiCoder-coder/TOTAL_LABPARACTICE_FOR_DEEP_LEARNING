# PHASE 56 PRE-PROCESS PLAN — Error-Conditioned Attention

## 1. Phase role

Phase 56 trả lời câu hỏi:

> Temporal attention của Final Transformer có thay đổi một cách có hệ thống khi forecast error lớn hơn, nhỏ hơn, underprediction hay overprediction hay không?

Phase 56 chỉ kết nối hai nhánh phân tích đã frozen:

- Error branch: Phase 47 → Phase 49 → Phase 50 → Phase 51
- Attention branch: Phase 52 → Phase 53 → Phase 54 → Phase 55

Phase 56 KHÔNG thay đổi mô hình, không training, không fine-tune, không optimizer step, không scaler fit, không new Test inference, không new attention extraction.

## 2. Forbidden actions (locked from amendment v1.18)

- training / fine-tune / optimizer.step / .backward / scaler.fit
- new Test inference; new attention extraction
- model checkpoint loading; model.forward; return_attention
- materialize_phase52, extract_attention, materialize_phase54, materialize_phase55
- destructive overwrite of any Phase 47-55 canonical artifact
- best-seed selection, ensemble
- best-head selection, head ranking, head pruning, head ablation, head clustering
- cross-seed head matching, cross-seed same-index averaging
- cartesian subgroup mining
- squared error as independent primary conditioning
- Test error cohort used as deployment regime
- regime threshold modification
- attention labeled as feature importance, causal attribution, predictive head quality
- causal claim
- weighted evidence score
- post-hoc redundancy threshold
- heatmap pixels used as numeric source
- implementing Phase 57 / 58 (handoff files only)
- notebook modification in this run

## 3. Primary scope

- Seeds: [42, 123, 2026]
- Layers: 2 (Layer 0, Layer 1)
- Heads: 4 (H1, H2, H3, H4)
- Test population: 2961 targets
- Lookback: 72 steps = 720 min = 12h
- Residual convention: `residual = y_true - y_pred`
- Primary error variable: `AE = |residual|` (Wh)
- Signed secondary variable: `residual`

## 4. Frozen CORE_ATTENTION_METRICS-v1 (locked before analysis)

1. `normalized_entropy`
2. `expected_lag_minutes`
3. `recent_1h_mass`
4. `recent_6h_mass`
5. `top5_mass`
6. `lag80_minutes`

## 5. Required inputs (frozen upstream)

- Phase 49: `artifacts/residual_analysis/residual_long_table.csv`
- Phase 50: `artifacts/error_by_regime/test_regime_assignment.csv`
- Phase 51: shared hardness derived from `residual_long_table.csv` (`mean_abs_error_wh = (|e_42|+|e_123|+|e_2026|)/3`)
- Phase 52: `artifacts/attention_extraction/raw/last_query_attention_seed*.npz` (verification fallback) + `attention_full_matrix_summary.csv`
- Phase 54: `artifacts/last_query_attention/last_query_metrics_long.csv` + `last_query_profile_by_lag.csv` + `last_query_recent_mass_summary.csv`
- Phase 55: head behavior summary + head-pair comparison + layer diversity (handoff reference only)

## 6. Required outputs (O56.1–O56.38)

Under `artifacts/error_conditioned_attention/`:

- O56.1  Analysis manifest
- O56.2  Analysis contract
- O56.3  Preflight audit
- O56.4  Source verification
- O56.5  Frozen cohort assignment
- O56.6  Cohort assignment audit
- O56.7  Shared cohort audit
- O56.8  Assignment fingerprint
- O56.9  Join audit
- O56.10 Continuous association long table
- O56.11 Association matrices (per seed/layer, AE + signed)
- O56.12 Error-decile metric summary
- O56.13 Error-decile profile by lag
- O56.14 High-vs-low metric comparison
- O56.15 High-vs-low Cliff's-delta matrices
- O56.16 High-vs-low profile comparison
- O56.17 High-minus-low profile difference by lag
- O56.18 Under-vs-over metric comparison
- O56.19 Under-vs-over profile comparison
- O56.20 Under-minus-over profile difference by lag
- O56.21 Layer head-mean per-target metrics long
- O56.22 Layer head-mean association
- O56.23 Layer head-mean high-low
- O56.24 Shared-cohort layer summary
- O56.25 Cross-seed layer summary
- O56.26 Full-matrix association
- O56.27 Error-cohort regime composition
- O56.28 Worst-case context
- O56.29 Core figures (14)
- O56.30 Findings
- O56.31 Phase57 handoff
- O56.32 Phase58 context handoff
- O56.33 Tests
- O56.34 Discrepancy log
- O56.35 Summary JSON
- O56.36 Human-readable report
- O56.37 README
- O56.38 Sign-off

## 7. Required analyses (executed in this order)

1. Verify all upstream sources.
2. Derive shared_hardness per target_id from residual_long_table.csv.
3. Freeze cohort assignment per seed (rank-based 20/60/20 + deciles + sign group).
4. Freeze shared cohort assignment.
5. Compute and store SHA256 of cohort assignment.
6. Join Phase 54 attention metrics by (seed, target_id).
7. Audit join row counts.
8. Compute C1 continuous Spearman associations (AE + signed + shared × 6 metrics × 24 seed/layer/head combos).
9. Compute error-decile metric summaries + per-decile mean temporal profiles.
10. Compute C2 HIGH vs LOW metric deltas + Cliff's delta + profile JSD/L1/Cosine/Wasserstein minutes.
11. Build D_HL profile-by-lag tables (sum≈0 audit).
12. Compute UNDER vs OVER metric deltas + Cliff's delta + profile distances.
13. Build D_UO profile-by-lag tables (sum≈0 audit).
14. Reconstruct layer head-mean vectors from raw Phase 52 attention.
15. Compute layer head-mean CORE_ATTENTION_METRICS-v1.
16. Compute layer head-mean continuous + high-low + shared-cohort analyses.
17. Compute cross-seed layer summary.
18. Compute secondary full-matrix Spearman associations.
19. Annotate error cohorts with frozen Phase 50 regime composition.
20. Attach deterministic Phase 51 W2 shared ranks 1–5 worst-case examples.
21. Generate all figures in architectural order.
22. Write findings with explicit non-causal language.
23. Write Phase 57 + Phase 58 handoffs.
24. Run integrity/scope/discrepancy tests.
25. Write summary JSON, report, README, sign-off.

## 8. Tests

All required audit checks from the canonical Phase 56 plan:

- Source verification
- Cohort coverage (LOW + MID + HIGH = N_test)
- Decile coverage (10 deciles × N_test/10 ≈ equal)
- Shared cohort consistency across seeds
- Assignment SHA256 frozen before attention join
- Join row count: 3 × 2961 × 2 × 4 = 71,064
- Profile probability sum ≈ 1 per cohort/head/seed/layer
- D_HL sum ≈ 0
- D_UO sum ≈ 0
- JSD ∈ [0, ln(2)]
- Wasserstein units = minutes
- Cliff's delta ∈ [-1, 1]
- Spearman target alignment (no NaN; constant cases marked NOT_DEFINED)
- Architectural head ordering (H1..H4)
- No best-head, no best-seed, no pruning, no ablation
- No retraining, no new inference, no new attention extraction
- No upstream Phase 47-55 mutation

## 9. Execution sequence

1. Verify upstream (Phase 49/50/51/52/54/55).
2. Build cohort assignments + freeze SHA.
3. Join attention.
4. C1 + C2 + signed + decile + layer-mean + shared + full-matrix + regime + worst-case.
5. Figures + findings + handoffs + tests + summary + report + README + sign-off.

## 10. Sub-phase gates

- 56-A: Source verification + audit ✓
- 56-B: Contract + cohort assignment freeze ✓
- 56-C: Continuous associations + decile analyses ✓
- 56-D: HIGH/LOW + UNDER/OVER + layer head-mean ✓
- 56-E: Shared-cohort + full-matrix + regime + worst-case ✓
- 56-F: Figures + findings + handoffs ✓
- 56-G: Tests + summary + report + README + sign-off ✓
- 56-H: Notebook visualization (DEFERRED — not in this run)

## 11. Hard governance checklist

- [ ] No model training or new inference
- [ ] No new attention extraction
- [ ] No modification of Phase 47-55 artifacts
- [ ] Cohort assignment frozen before attention join (SHA verified)
- [ ] C1 uses only Spearman (no p-value headline)
- [ ] C2 uses Cliff's delta (no canned magnitude labels)
- [ ] Deciles cover all targets (1..10, exact rank)
- [ ] Layer head-mean vectors built per target/layer FIRST, then metrics recomputed
- [ ] Shared cohorts identical across seeds
- [ ] No regime threshold modification
- [ ] No Test error cohort used as deployment regime
- [ ] No cartesian subgroup mining
- [ ] No cross-seed same-index averaging
- [ ] No causal claim from error-attention association
- [ ] No best-head / best-seed selection
- [ ] No head pruning / ablation
- [ ] Notebook untouched

## 12. Downstream boundary

- `phase57_seed_stability_attention_handoff.json` is created.
- `phase58_attention_results_context_handoff.json` is created.
- `PHASE57_AUTHORIZED = NO`
- `PHASE58_AUTHORIZED = NO`
- Phase 57 implementation NOT started.
- Phase 58 implementation NOT started.
