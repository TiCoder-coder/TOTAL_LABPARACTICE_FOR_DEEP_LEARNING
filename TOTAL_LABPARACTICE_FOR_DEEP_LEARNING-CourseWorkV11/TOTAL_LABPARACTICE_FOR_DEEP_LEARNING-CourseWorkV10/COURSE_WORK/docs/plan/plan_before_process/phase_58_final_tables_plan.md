# Phase 58 — Final Tables Pre-Process Plan

**Phase ID:** PHASE_58_FINAL_TABLES
**Output version:** FINAL_TABLES-v1
**Date:** 2026-09-05
**Author:** Phase 58 orchestrator
**Approval status:** APPROVED (per architecture amendment v1.20)

---

## 1. Phase Role

Phase 58 is the final **reporting synthesis and governance** phase. It does NOT train, infer, extract attention, or recompute upstream analyses. It only:

- copies / joins / reshapes existing machine-readable evidence,
- audits frozen upstream artifacts,
- recomputes already-authorized display aggregates (e.g. three-seed mean ± SD with ddof=1),
- formats / rounds / packages tables (CSV + Markdown + LaTeX + per-table metadata),
- freezes the source-of-truth ledger and critical-cell lineage,
- emits cross-table consistency / population / model-lock / seed / unit / rounding audits,
- emits a course-requirement coverage audit,
- emits a Phase 59 final-conclusions handoff.

## 2. Forbidden Actions (HARD)

- NO new training, fine-tune, optimizer.step, .backward, model.train, scaler.fit.
- NO new Test inference, NO new attention extraction, NO return_attention.
- NO model checkpoint loading, NO model.forward, NO torch.load.
- NO destructive overwrite of any Phase 43–57 canonical artifact.
- NO best-seed selection, NO best-head selection, NO ensemble reconstruction, NO weighted overall stability score.
- NO Test reranking by metric, NO best-value highlighting (bold/color) by Test performance.
- NO MAPE / new accuracy %.
- NO new hypothesis test (paired t-test / Wilcoxon / Diebold-Mariano / bootstrap).
- NO new confidence interval.
- NO Phase 50 regime redefinition, NO Phase 51 worst-case reselection.
- NO attention labeled as feature importance or causal contribution; NO attention stability labeled as functional-equivalence proof.
- NO cross-seed head semantic alignment assumption by same numeric index.
- NO reading of numeric attention values from PNG heatmaps.
- NO manual typing of scientific values without source lineage.
- NO implementation of Phase 59 substantive conclusions (handoff JSON only).
- NO notebook modification in this run (Phase 58-H deferred to separate Human-approved run).

## 3. Primary Scope

Build the canonical final-report table package FT01–FT10 + FA01–FA12 from frozen upstream artifacts.

| Table ID | Source Phase | Evidence Class |
|----------|--------------|----------------|
| FT01 | Phase 45 final lock | METHOD / LOCKED_CONFIG |
| FT02 | Phase 47 Test metrics | HELD_OUT_TEST_EVIDENCE |
| FT03 | Phase 44 rolling-origin | DEVELOPMENT_EVIDENCE |
| FT04 | Phase 48 + Phase 49 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT05 | Phase 50 + Phase 51 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT06 | Phase 54 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT07 | Phase 55 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT08 | Phase 56 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT09 | Phase 57 | POST_TEST_DIAGNOSTIC_EVIDENCE |
| FT10 | upstream-supported claim trace | EVIDENCE_AND_LIMITATION |
| FA01–FA12 | mirrors of upstream detail | matching parent |

## 4. Frozen Core Variables (locked)

- three official seeds = [42, 123, 2026]
- three-seed summary = mean ± sample SD (ddof=1) of seed-level metrics, NOT an ensemble
- FT02 model order = [Persistence, Tuned LSTM, Final Transformer Seed 42, 123, 2026, Three-Seed Summary]
- lookback = 72 (12h); horizon = 1; sampling interval = 10 min
- target unit = Wh; residual = y_true - y_pred (signed Wh)
- display precision: Wh 2dp; R² 3dp; dimensionless 3dp; minutes 1dp; percent 1dp
- aggregate-before-round; full-precision CSV BEFORE Markdown/LaTeX display rounding
- no best-value bolding; no color-coded winner

## 5. Required Inputs

- Phase 45 final lock: `final_model_scientific_config.json`, `final_seed_contract.json`, `final_loss_contract.json`, `final_optimizer_contract.json`, `final_boundary_contract.json`, `final_feature_contract.json`, `final_data_region_contract.json`, `final_epoch_policy.json`, `final_training_recipe.json`
- Phase 47 final Test metrics: `final_test_summary.json`, `transformer_seed_aggregate_metrics.csv`, `final_test_summary_table.csv`, `final_test_baseline_metrics.csv`, `final_test_model_comparison.csv`
- Phase 44 rolling-origin: `rolling_origin_pooled_metrics.csv`, `rolling_origin_fold_table.csv`, `rolling_origin_fold_metrics.csv`
- Phase 49 residual: `residual_long_table.csv`, `phase49_residual_distribution_summary.csv`
- Phase 50 regimes: `regime_metrics_long.csv`, `regime_cross_seed_summary.csv`, `regime_reference_train_audit.csv`
- Phase 51 worst-case: `phase51_target_level_working_table.csv`, `phase51_finding_id_summary` (within signoff), `phase51_attention_handoff_cases.csv`
- Phase 54 last-query: `last_query_metrics_long.csv`, `last_query_profile_by_lag.csv`, `last_query_layer_head_mean_profile.csv`, `last_query_top1_lag_frequency.csv`, `last_query_lag_bin_mass.csv`
- Phase 55 head comparison: `head_behavior_summary.csv`, `head_pair_comparison_long.csv`, `layer_head_diversity_summary.csv`
- Phase 56 error-conditioned: `error_attention_*.csv`, `error_conditioning_assignment.csv`, `layer_*.csv`
- Phase 57 seed stability: `layer_head_mean_seed_stability_summary.csv`, `head_matching_assignments.csv`, `head_matching_cycle_consistency.csv`, `head_matching_wasserstein_sensitivity.csv`, `head_matching_independence_audit.csv`, `matched_head_*.csv`, `dense_case_attention_seed_stability.csv`, `layer_error_conditioned_seed_stability.csv`, `matched_head_error_conditioned_stability.csv`, `prediction_attention_disagreement_association.csv`, `attention_seed_stability_evidence_summary.csv`
- Phase 48 prediction spread: `prediction_seed_spread.csv`, `prediction_long_table.csv`, `prediction_range_compression.csv`

## 6. Required Outputs (O58.1–O58.43)

Per the canonical plan (Phase_58_Final_tables.md §171). All persisted under `artifacts/final_tables/`.

## 7. Tests / Audits

Run the full Phase 58 integrity checks. Verify:

- all 14 upstream signoffs PASS / PASS_WITH_WARNING
- final_lock_sha256 + final_test_population_sha256 stable across all tables
- seeds exactly {42, 123, 2026}
- baseline population comparability (FT02 fairness rule)
- three-seed ddof=1
- aggregate before rounding
- no ensemble reconstruction
- no best-seed / head selection
- no new metric / analysis
- no Test reranking
- no development / Test evidence mixing
- no untraceable numeric cells
- no hidden upstream warnings
- deterministic CSV / Markdown / LaTeX
- critical cell lineage complete

## 8. Execution Sequence

1. preflight → upstream signoff verification + handoff readiness
2. freeze inventory → main + appendix + figure + render config
3. build source ledger from frozen artifacts (one row per scientific cell)
4. build FT01 from Phase 45 lock
5. build FT02 from Phase 47 metrics + recompute three-seed mean/SD (full precision)
6. build FT03 from Phase 44 rolling-origin
7. build FT04 from Phase 48/49
8. build FT05 from Phase 50/51
9. build FT06 from Phase 54
10. build FT07 from Phase 55
11. build FT08 from Phase 56
12. build FT09 from Phase 57
13. build FT10 from upstream-supported claim trace
14. build FA01–FA12 from frozen detail artifacts
15. write full-precision CSV → Markdown → LaTeX for each table
16. cross-table consistency + population + model-lock + seed + unit + rounding audits
17. coursework coverage + claim traceability + Phase 59 handoff
18. findings + tests + discrepancies
19. summary + report + README + sign-off

## 9. Sub-phase Gates

Each sub-phase (FT01..FT10 + audits) requires all checks to PASS before writing the display artefact. A single FAIL or BLOCKED_CONSISTENCY_FAILURE halts Phase 58.

## 10. Hard Governance Checklist

- [x] Phase 44 PASS
- [x] Phase 45 PASS
- [x] Phase 46 PASS
- [x] Phase 47 PASS
- [x] Phase 48 PASS
- [x] Phase 49 PASS
- [x] Phase 50 PASS
- [x] Phase 51 PASS
- [x] Phase 52 PASS
- [x] Phase 53 PASS
- [x] Phase 54 PASS
- [x] Phase 55 PASS
- [x] Phase 56 PASS
- [x] Phase 57 PASS
- [x] phase58_ready=true (from Phase 57 handoff)
- [x] Final lock SHA known
- [x] Final Test population SHA known
- [x] Final seed set exactly {42, 123, 2026}
- [x] Table inventory frozen
- [x] Figure inventory frozen
- [x] Render config frozen
- [x] No new scientific analysis requested

## 11. Downstream Boundary

Phase 59 remains UNAUTHORIZED for substantive implementation. Phase 58 may emit only `phase59_final_conclusions_handoff.json`. `phase59_ready` may become true after Phase 58 PASS.

## 12. Sign-off Conditions

- All FT01–FT10 are READY or READY_WITH_WARNING
- All FA01–FA12 are built
- All critical-cell lineage complete
- Cross-table consistency passes
- Population / model-lock / seed / unit / rounding audits pass
- No new selection / analysis / metric
- Phase 59 handoff emitted with allowed-claims / prohibited-claims lists
