# PHASE 57 PRE-PROCESS PLAN — Seed-Stability Attention Check

## 1. Phase role

Phase 57 closes the attention branch by answering:

> Do the temporal-attention patterns (Phase 52–54) and error-conditioned attention findings (Phase 56) remain stable across the three Final Transformer seeds, once the head permutation problem is handled correctly?

Phase 57 connects three frozen analyses:

- Attention branch: Phase 52 (extraction) → Phase 54 (last-query metrics) → Phase 55 (head comparison)
- Error branch: Phase 49 → Phase 50 → Phase 51 → Phase 56
- Prediction seed spread (secondary): Phase 48

Phase 57 does NOT change the model, does NOT retrain, does NOT run new Test inference, does NOT extract new attention.

## 2. Forbidden actions

- training, fine-tune, optimizer.step, scaler.fit, model checkpoint loading
- new Test inference
- new attention extraction
- modifying any Phase 47 / 48 / 49 / 50 / 51 / 52 / 53 / 54 / 55 / 56 frozen artifact
- best-seed selection, ensemble, weighted overall stability score
- best-head selection, head ranking, head pruning, head ablation, head clustering
- target/error/regime/case-specific rematching
- anchor change after seeing results
- Wasserstein replacing canonical JSD mapping post hoc
- Test error cohort used as deployment regime
- attention = feature importance / causal explanation / functional equivalence proof
- reading PNG pixels for numeric attention values
- implementing Phase 58 / 59
- notebook modification in this run

## 3. Primary scope

Three seeds: 42, 123, 2026.

Two parallel views:

```text
S57-A — permutation-invariant layer head-mean stability
S57-B — permutation-aware matched-head stability
```

## 4. Frozen core variables

```text
seeds = (42, 123, 2026)
N_TEST = 2961
LAYERS = 2
HEADS = 4
LOOKBACK = 72
LAG_STEP_MINUTES = 10
H2_LAYER_HEAD_COUNTS = [2, 4]  # locked H2/H4 protocol
MATCH_TIE_TOL = 1e-12
ANCHOR_SEED = 42
ANCHOR_REASON = FIRST_PREDECLARED_FINAL_SEED
```

## 5. Required inputs (READ-ONLY)

- `artifacts/residual_analysis/residual_long_table.csv` (Phase 49)
- `artifacts/error_conditioned_attention/*.csv` (Phase 56)
- `artifacts/attention_extraction/raw/last_query_attention_seed{42,123,2026}.npz` (Phase 52)
- `artifacts/attention_extraction/raw/dense_case_attention_seed{42,123,2026}.npz` (Phase 52)
- `artifacts/attention_extraction/raw_attention_checksums.json` (Phase 52)
- `artifacts/attention_extraction/attention_test_target_order.csv` (Phase 52)
- `artifacts/attention_extraction/attention_dense_case_order.csv` (Phase 52)
- `artifacts/last_query_attention/last_query_metrics_long.csv` (Phase 54)
- `artifacts/last_query_attention/last_query_profile_by_lag.csv` (Phase 54)
- `artifacts/last_query_attention/last_query_layer_head_mean_profile.csv` (Phase 54)
- `artifacts/last_query_attention/last_query_top1_lag_frequency.csv` (Phase 54)
- `artifacts/last_query_attention/last_query_lag_bin_mass.csv` (Phase 54)
- `artifacts/last_query_attention/last_query_seed_overall_profile.csv` (Phase 54)
- `artifacts/head_comparison/head_behavior_summary.csv` (Phase 55 handoff ref)
- `artifacts/head_comparison/layer_head_diversity_summary.csv` (Phase 55 handoff ref)
- `artifacts/head_comparison/phase57_seed_stability_head_context_handoff.json` (Phase 55)
- `artifacts/error_conditioned_attention/error_attention_*.csv` (Phase 56 effects)
- `artifacts/error_conditioned_attention/error_conditioning_assignment.csv` (Phase 56)
- `artifacts/error_conditioned_attention/phase57_seed_stability_attention_handoff.json` (Phase 56)
- `artifacts/prediction_analysis/prediction_seed_spread.csv` (Phase 48 secondary diagnostic)

## 6. Required outputs (O57.1–O57.42)

All under `artifacts/seed_stability_attention/`:

```text
seed_stability_attention_manifest.json
seed_stability_attention_contract.json
phase57_preflight_audit.csv
seed_stability_source_verification.csv
seed_pair_manifest.csv
layer_head_mean_seed_stability.csv
layer_head_mean_seed_stability_summary.csv
layer_head_mean_per_target_stability.csv
layer_attention_disagreement_by_target.csv
layer_attention_metric_seed_stability.csv
head_matching_cost_matrices.csv
head_matching_assignments.csv
head_matching_ambiguity_audit.csv
head_matching_edge_margin_audit.csv
head_matching_cycle_consistency.csv
head_matching_wasserstein_sensitivity.csv
canonical_matched_head_groups.csv
head_matching_fingerprint.json
head_matching_independence_audit.csv
matched_head_profile_seed_stability.csv
matched_head_profile_stability_summary.csv
matched_head_consensus_profile.csv
matched_head_per_target_stability.csv
matched_head_disagreement_by_target.csv
matched_head_metric_seed_stability.csv
matched_head_top1_lag_stability.csv
dense_case_attention_seed_stability.csv
dense_case_attention_stability_summary.csv
layer_error_conditioned_seed_stability.csv
matched_head_error_conditioned_stability.csv
prediction_attention_disagreement_association.csv
matched_head_prediction_disagreement_association.csv
attention_seed_stability_evidence_summary.csv
seed_stability_attention_findings.csv
seed_stability_attention_tests.csv
seed_stability_attention_discrepancies.json
phase58_final_tables_handoff.json
phase59_conclusions_context_handoff.json
seed_stability_attention_summary.json
seed_stability_attention_report.md
README_SEED_STABILITY_ATTENTION.md
figures/SEEDATTN_57_01..14.png
phase_57_signoff.json
```

## 7. Analyses (canonical order)

1. **Preflight + source verification**: confirm seed set, architecture, FINAL_TEST_POP-v1, target order, dense-case order, Phase 52 raw checksums, Phase 54/55/56 source availability.
2. **Seed-pair manifest**: P1=42-123, P2=42-2026, P3=123-2026.
3. **S57-A — Layer head-mean stability**:
   - Compute layer head-mean vector per (seed, target, layer) = mean over heads.
   - Compute mean layer head-mean profile per (seed, layer) for layer-level stability.
   - For each (layer, seed_pair) compute JSD, L1, L2, cosine, Pearson, Spearman, Wasserstein-minutes.
   - Per-target layer head-mean vectors: JSD, cosine, L1, Wasserstein-minutes per (target, layer, seed_pair).
   - Per-target three-seed disagreement (mean/max pairwise JSD and Wasserstein).
   - Layer attention metric stability (Spearman + mean/median abs diff) for 6 CORE_ATTENTION_METRICS-v1.
4. **S57-B — Canonical head matching**:
   - Build mean head profile per (seed, layer, head) = mean of full-Test last-query vectors.
   - For each (layer, seed_pair): build H×H JSD cost matrix (and Wasserstein matrix).
   - Enumerate all H! permutations lexicographically.
   - For each perm compute (total_JSD, total_Wasserstein).
   - Select canonical: min total_JSD → tie-break: min total_Wasserstein → tie-break: lex smallest perm.
   - Use MATCH_TIE_TOL = 1e-12.
   - Audit ambiguity: best vs second-best, edge margins.
   - Compute cycle consistency (direct 123-2026 vs anchor-induced 123-42-2026).
   - Compute Wasserstein-only sensitivity matching + agreement fraction.
   - Freeze `head_matching_fingerprint.json` BEFORE applying Phase 56 effects.
5. **Canonical three-seed matched groups** anchored at seed42.
6. **Matched-head mean-profile stability**: JSD, Wasserstein-min, cosine, Pearson, Spearman, L1, L2 per (layer, group, seed_pair).
7. **Matched-head consensus profile** per (layer, group, lag) — sum ≈ 1.
8. **Per-target matched-head stability**: same metrics per (target, layer, group, seed_pair).
9. **Matched-head metric stability**: Spearman + mean/median abs diff for 6 CORE_ATTENTION_METRICS-v1 per matched group.
10. **Matched-head top1-lag stability**: TVD + JSD + modal lag agreement.
11. **Dense-case stability**: apply frozen global mapping to frozen Phase 51 cases; compute rowwise JSD/cosine + normalized Frobenius.
12. **Layer error-conditioned stability** (PRIMARY error-effect robustness): Spearman + HIGH vs LOW + HIGH-LOW Cliff's + profile JSD + Wasserstein for (analysis_type, conditioning_variable, attention_metric) per (layer), with sign agreement counts.
13. **Matched-head error-conditioned stability** (SECONDARY): reindexed Phase 56 effects using frozen matching; ambiguity warnings propagated.
14. **Shared-cohort layer stability**: identical target IDs across seeds → compare shared-cohort effects descriptively.
15. **Prediction-attention disagreement**: Spearman between Phase 48 prediction range/SD and layer mean-pairwise JSD/Wasserstein.
16. **Stability evidence summary**: transparent rows per analysis without composite score.

## 8. Tests

- `tests/unit/test_phase57_*.py`
- Source immutability (Phase 47–56)
- Three seeds retained
- Same Test population, target order, dense-case order
- Probability-vector integrity (sum ≈ 1)
- JSD bounds, Wasserstein units = minutes
- H! permutation count: 4! = 24 for H=4 layer, 2! = 2 for H=2 layer
- Bijective assignments
- Deterministic matching (same input → same perm)
- Matching fingerprint before Phase56 effect mapping
- Canonical group uniqueness
- Cycle consistency computed (no rematch)
- Wasserstein sensitivity does not replace canonical JSD
- No same-index semantic assumption
- No target/error/regime/case-specific rematching
- Dense attention rows sum ≈ 1
- Consensus profile sums ≈ 1
- Phase56 effects use frozen cohorts
- Phase48 target alignment has no drops
- No best seed/head; no pruning/ablation
- No training; no new inference; no new attention extraction
- No weighted stability score
- No causal attribution
- Upstream artifacts unchanged

## 9. Execution sequence

1. Preflight + source verification + freeze contract
2. Seed-pair manifest
3. S57-A layer head-mean stability
4. Per-target layer stability + three-seed disagreement
5. Layer attention metric stability
6. S57-B head matching: cost matrices + permutation enumeration + canonical assignment + ambiguity + edge margins + cycle consistency + Wasserstein sensitivity
7. Freeze matching fingerprint
8. Build canonical matched groups
9. Matched-head mean-profile stability + consensus profile
10. Per-target matched-head stability + three-seed disagreement
11. Matched-head metric + top1 stability
12. Dense-case stability
13. Layer + matched-head error-conditioned stability + shared-cohort
14. Prediction-attention disagreement
15. Findings + tests + discrepancies + summary + report + README + handoffs + signoff

## 10. Sub-phase gates (per architecture v1.19)

- Phase 57-A (Preflight + S57-A layer stability): GO
- Phase 57-B (S57-B matching + audits): GO
- Phase 57-C (Matched-head + dense-case): GO
- Phase 57-D (Error-conditioned stability): GO
- Phase 57-E (Prediction disagreement): GO
- Phase 57-F (Findings + tests + discrepancies + summary + report + README + handoffs): GO
- Phase 57-G (Sign-off): GO

All sub-phases within Phase 57 are authorized in this run by the user's "execute PHASE 57" command.

## 11. Hard governance checklist

- [x] Phase 47–56 signoffs PASS (verified above)
- [x] phase57_ready = true (Phase 56 signoff)
- [x] Three official seeds present
- [x] Architecture amendment v1.19 applied
- [x] Pre-process plan approved (this document)
- [ ] All canonical sources verified SHA-frozen before scientific execution
- [ ] Matching contract frozen before results
- [ ] No best seed/head, no pruning, no causal claim
- [ ] Upstream artifacts unchanged after Phase 57 execution
- [ ] Phase 58 handoff ready
- [ ] Phase 59 context handoff ready

## 12. Downstream boundary

- Phase 58 (Final Tables): handoff-ready but UNAUTHORIZED for implementation.
- Phase 59 (Conclusions): context-ready but UNAUTHORIZED for implementation.

## 13. Phase 57 signoff conditions

PASS requires:

- All three seed sources verified
- Layer head-mean stability complete (no threshold invention)
- Canonical head matching complete (within each layer, JSD-exhaustive, deterministic)
- Ambiguity + cycle + Wasserstein-sensitivity audits complete
- Matched-head mean-profile + per-target + metric + top1 stability complete
- Dense-case full-map stability complete (raw numerical only)
- Error-conditioned layer + matched-head stability complete
- Prediction-attention disagreement computed (secondary)
- No best seed/head selected
- No upstream artifact mutated
- Phase 58 handoff ready; Phase 59 context ready
- Phase 57 signoff JSON with `overall_status: PASS` (or `PASS_WITH_WARNING` when scientifically allowed)

FAIL if any forbidden action occurs.
