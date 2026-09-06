# Phase 55 — Head Comparison Analysis Pre-Process Plan

**Phase ID:** `PHASE_55_HEAD_COMPARISON`
**Output version:** `HEAD_COMPARISON-v1`
**Phase before:** `PHASE_54_LAST_QUERY_ATTENTION`
**Phase after:** `PHASE_56_ERROR_CONDITIONED_ATTENTION`

This pre-process plan is the canonical governance-level planning document for
Phase 55. It must faithfully map the full canonical Phase 55 requirement
defined in `docs/plan-doc/plan_detail_for_each_phase/Phase_55_Head_comparison.md`
before any scientific Phase 55 implementation is allowed.

---

## 1. Status

```text
STATUS: APPROVED
```

This document is approved for Phase 55 sub-phase execution under architecture
amendment `phase-55-architecture-amendment-v1.17` and the existing Phase 55
detail file (`docs/plan-doc/plan_detail_for_each_phase/Phase_55_Head_comparison.md`).

Human approval timestamp: 2026-09-04 (see governance log
`docs/save_log_in_processing/phase_55_architecture_amendment_log.json`).

---

## 2. Phase 55 role

Phase 55 = **head-behavior comparison**.

Primary goal: quantify whether attention heads within the same encoder layer
learn non-identical temporal allocation patterns.

Specifically:

```text
Q55.1  Do heads within the same layer have different temporal profiles?
Q55.2  Which head pairs are highly similar vs highly distinct?
Q55.3  Is diversity driven by recent-vs-old allocation or concentration?
Q55.4  Are some heads broad/diffuse while others are narrow/concentrated?
Q55.5  Are some heads centered on systematically different expected lags?
Q55.6  Do some heads repeatedly attend to similar top1 lag regions?
Q55.7  Is one encoder layer more internally diverse than another?
Q55.8  Are there near-redundant heads by profile similarity?
Q55.9  Do qualitative heatmap differences agree with quantitative profile metrics?
Q55.10 Which head-comparison artifacts should Phase56/57 consume?
```

Central principle:

```text
Compare heads
+ within fixed seed/layer
+ behavioral diversity
+ redundancy diagnostics
+ NO best head
+ NO pruning
+ NO error conditioning
```

---

## 3. What Phase 55 is NOT

```text
- head pruning
- head ablation
- best-head selection
- model retraining
- Test metric recomputation
- error-conditioned comparison
- regime-conditioned comparison
- worst-case statistical comparison
- cross-seed head matching
- cross-seed same-index semantic alignment
- cross-layer same-index semantic identity
- weighted composite head score
- post-hoc redundancy threshold
- attention labeled as feature importance
- attention labeled as causal explanation
- Phase 56 or Phase 57 implementation
```

---

## 4. Primary scope

```text
Within seed: 42, 123, 2026
Within layer: 0, 1
Across heads: 0, 1, 2, 3
Pair count per (seed, layer) = H(H-1)/2 = 6 for H=4
Total pairs across all (seed, layer) = 3 seeds * 2 layers * 6 pairs = 36
Head order: ARCHITECTURAL (no reordering by similarity)
```

Primary comparison unit:

```text
(seed, layer, head_a, head_b) with head_a < head_b
```

---

## 5. Required inputs (frozen)

Required inputs read by Phase 55 (all from frozen Phase 54 canonical
artifacts):

```text
artifacts/last_query_attention/last_query_metrics_long.csv             # per-vector metrics
artifacts/last_query_attention/last_query_metric_summary_by_head.csv   # head-level summary
artifacts/last_query_attention/last_query_profile_by_lag.csv           # mean profile by lag
artifacts/last_query_attention/last_query_layer_head_mean_profile.csv  # layer head-mean
artifacts/last_query_attention/last_query_lag_bin_mass.csv             # non-overlap lag bins
artifacts/last_query_attention/last_query_recent_mass_summary.csv      # 1h/6h/12h/24h
artifacts/last_query_attention/last_query_coverage_radius_summary.csv  # Lag50/80/90
artifacts/last_query_attention/last_query_top1_lag_frequency.csv       # top1 freq
artifacts/last_query_attention/last_query_top1_tie_summary.csv         # tie summary
artifacts/last_query_attention/last_query_report_case_manifest.csv    # case selection
artifacts/last_query_attention/last_query_report_case_metrics.csv      # case metrics
artifacts/last_query_attention/phase_54_signoff.json                   # Phase 54 sign-off
artifacts/last_query_attention/phase55_head_comparison_handoff.json    # Phase 55 handoff
```

Raw fallback source (verification only):

```text
artifacts/attention_extraction/raw/last_query_attention_seed{42,123,2026}.npz
artifacts/attention_extraction/raw_attention_checksums.json
```

Phase 53 PNG:

```text
visual context only. NEVER used as numeric input.
```

---

## 6. Required outputs (O55)

Per canonical Phase 55 detail file (`Phase_55_Head_comparison.md` §104–§105,
§172–§175):

```text
O55.1   head_comparison_manifest.json
O55.2   head_comparison_contract.json
O55.3   phase55_preflight_audit.csv
O55.4   head_comparison_source_verification.csv
O55.5   head_profile_integrity_audit.csv
O55.6   head_target_alignment_audit.csv
O55.7   head_pair_comparison_long.csv
O55.8   head_pair_profile_similarity.csv
O55.9   head_pair_metric_difference.csv
O55.10  head_pair_paired_difference_summary.csv
O55.11  head_pair_top1_distribution_distance.csv
O55.12  head_pair_wasserstein_distance.csv
O55.13  head_behavior_summary.csv
O55.14  head_to_layer_mean_distance.csv
O55.15  layer_head_diversity_summary.csv
O55.16  head_similarity_matrix_jsd.csv
O55.17  head_similarity_matrix_cosine.csv
O55.18  head_similarity_matrix_pearson.csv
        head_similarity_matrix_spearman.csv
O55.19  head_distance_matrix_l1.csv
        head_distance_matrix_wasserstein.csv
O55.20  head_expected_lag_difference_matrix.csv
O55.21  head_recent1h_difference_matrix.csv
O55.22  head_top1_tvd_matrix.csv
O55.23  core figures (HEAD_55_01..HEAD_55_14)
O55.24  head_comparison_findings.csv
O55.25  phase56_error_conditioned_attention_handoff.json
O55.26  phase57_seed_stability_head_context_handoff.json
O55.27  head_comparison_tests.csv
O55.28  head_comparison_discrepancies.json
O55.29  head_comparison_summary.json
O55.30  head_comparison_report.md
O55.31  README_HEAD_COMPARISON.md
O55.32  phase_55_signoff.json
```

Total: **32 required outputs**.

---

## 7. Required analyses

### 7.1 Pairwise profile metrics (per head pair)

Required (canonical Phase 55 §14, §75):

```text
Pearson correlation
Spearman correlation
Cosine similarity
Jensen-Shannon divergence (natural log)
L1 distance
L2 distance
Wasserstein-1 distance (minutes)
```

### 7.2 Pairwise behavioral differences (per head pair, median-based)

Required (canonical Phase 55 §112, §114):

```text
Delta(A-B) + abs_delta for:
- median normalized_entropy
- median expected_lag_minutes
- median recent_1h_mass
- median recent_6h_mass
- median recent_12h_mass (if supported)
- median recent_24h_mass (if supported)
- median lag80_minutes
- median top5_mass
```

### 7.3 Paired same-target differences (per head pair × metric)

Required (canonical Phase 55 §43, §115, §163):

```text
Metrics: normalized_entropy, expected_lag_minutes, recent_1h_mass, lag80_minutes
For each (seed, layer, head_a, head_b, metric):
- N, mean_difference, median_difference, sample_sd_difference
- p05, p25, p75, p95
- fraction_positive, fraction_zero, fraction_negative
- identical target_ids across pair (no row drops)
```

### 7.4 Top1 lag distribution comparison

Required (canonical Phase 55 §116):

```text
TVD = 0.5 * sum(|P_k - Q_k|)
JSD (natural log) on top1 lag frequency distribution
Inherit Phase 52 NEWEST_SOURCE tie rule
Top1 freq distribution must sum to 1 per head
```

### 7.5 Head behavior summary (per head)

Required (canonical Phase 55 §118):

```text
median_normalized_entropy
mean_normalized_entropy
median_effective_source_count
median_expected_lag_minutes
median_lag_sd_minutes
median_top1_weight
median_top5_mass
median_recent1h_mass
median_recent6h_mass
median_recent12h_mass (if supported)
median_recent24h_mass (if supported)
median_lag50_minutes
median_lag80_minutes
median_lag90_minutes
modal_top1_lag_minutes
modal_top1_lag_fraction
jsd_to_layer_mean
l1_to_layer_mean
cosine_to_layer_mean
```

No rank. No score.

### 7.6 Head-to-layer-mean distance

Required (canonical Phase 55 §119):

```text
jsd_to_layer_head_mean_profile
l1_to_layer_head_mean_profile
l2_to_layer_head_mean_profile
cosine_to_layer_head_mean_profile
wasserstein_to_layer_head_mean_minutes
```

### 7.7 Layer-level diversity summary

Required (canonical Phase 55 §120, §166):

```text
seed, layer_idx0, head_count, pair_count
mean_pairwise_jsd
median_pairwise_jsd
max_pairwise_jsd
mean_pairwise_l1
mean_pairwise_l2
mean_pairwise_wasserstein_minutes
mean_pairwise_abs_expected_lag_diff_minutes
mean_pairwise_top1_tvd
min_pairwise_cosine
mean_pairwise_cosine
```

No weighted composite score.

### 7.8 Square matrices (architectural order, audit-compliant)

Required matrices (canonical Phase 55 §66–§67, §121):

```text
JSD matrix (diagonal=0, symmetric, [0, ln2])
Cosine matrix (diagonal=1, symmetric, [0, 1])
Pearson matrix (diagonal=1, symmetric, [-1, 1])
Spearman matrix (diagonal=1, symmetric, [-1, 1])
L1 matrix (diagonal=0, symmetric, nonnegative)
Wasserstein matrix (diagonal=0, symmetric, units=minutes, nonnegative)
Expected-lag absolute difference matrix (diagonal=0, symmetric, nonnegative)
Recent1h absolute difference matrix (diagonal=0, symmetric, nonnegative)
Top1 TVD matrix (diagonal=0, symmetric, [0, 1])
```

### 7.9 Core figures (HEAD_55_01..HEAD_55_14)

Required (canonical Phase 55 §129–§142):

```text
HEAD_55_01 — JSD matrices (one per seed/layer, scale 0..ln(2))
HEAD_55_02 — Cosine matrices (scale 0..1)
HEAD_55_03 — Wasserstein matrices (common max across seed/layer)
HEAD_55_04 — Expected-lag difference matrices (common max, minutes)
HEAD_55_05 — Top1 TVD matrices (scale 0..1)
HEAD_55_06 — Entropy by head (architectural order, no rank labels)
HEAD_55_07 — Expected lag by head (minutes/hours)
HEAD_55_08 — Recent 1h mass by head (no recency-quality judgment)
HEAD_55_09 — Recent 6h mass by head
HEAD_55_10 — Lag80 by head
HEAD_55_11 — Head-to-layer-mean JSD
HEAD_55_12 — Layer diversity summary (separate panels per metric)
HEAD_55_13 — Mean temporal profiles by head (per seed/layer)
HEAD_55_14 — Paired-difference distributions (selected core metrics)
```

### 7.10 Findings

Allowed findings codes (canonical Phase 55 §144, §145):

```text
HEAD_PROFILES_HIGHLY_SIMILAR_DESCRIPTIVE
HEAD_PROFILES_DIVERSE_DESCRIPTIVE
HEAD_PAIR_LOW_JSD
HEAD_PAIR_HIGH_JSD
HEAD_PAIR_SMALL_WASSERSTEIN
HEAD_PAIR_LARGE_WASSERSTEIN
HEAD_RECENCY_ALLOCATION_DIFFERS
HEAD_CONCENTRATION_DIFFERS
HEAD_EXPECTED_LAG_DIFFERS
HEAD_TOP1_LAG_DISTRIBUTIONS_DIFFERS
HEADS_CLOSE_TO_LAYER_MEAN
HEADS_DEVIATE_FROM_LAYER_MEAN
LAYER_HEAD_DIVERSITY_HIGHER_DESCRIPTIVE
LAYER_HEAD_DIVERSITY_LOWER_DESCRIPTIVE
PAIRWISE_DIFFERENCES_TARGET_DEPENDENT
PAIRWISE_DIFFERENCES_DIRECTIONALLY_CONSISTENT
NO_CLEAR_HEAD_DIVERSITY
NO_BEST_HEAD_SELECTED
NO_HEAD_PRUNING
NO_ERROR_CONDITIONING
NO_CROSS_SEED_HEAD_MATCHING
ATTENTION_TEMPORAL_NOT_FEATURE_IMPORTANCE
READY_FOR_ERROR_CONDITIONED_ATTENTION
READY_FOR_SEED_STABILITY_CONTEXT
```

All findings **descriptive**. No causal claim. No feature importance claim.

### 7.11 Phase 56 handoff

Required (canonical Phase 55 §151):

```text
phase56_error_conditioned_attention_handoff.json:
  source_phase55_version
  source_phase54_version
  source_phase52_version
  final_lock_sha256
  test_population_sha256
  seed_list
  head_behavior_summary_ref
  head_pair_comparison_long_ref
  layer_diversity_summary_ref
  last_query_metrics_long_ref
  raw_last_query_refs
  Phase49_residual_refs
  Phase50_regime_refs
  Phase51_worst_case_refs
  head_order=ARCHITECTURAL
  best_head_selected=false
  head_pruning=false
  error_conditioning_performed_in_phase55=false
  ready_for_phase56=true
```

### 7.12 Phase 57 context handoff

Required (canonical Phase 55 §153):

```text
phase57_seed_stability_head_context_handoff.json:
  source_phase55_version
  within_seed_pairwise_jsd
  within_seed_pairwise_wasserstein
  head_to_layer_mean_distance
  layer_head_diversity_summary
  per_seed_head_profiles_ref
  head_count
  layer_count
  same_index_semantic_alignment_assumed=false
  head_matching_performed=false
  ready_for_phase57_context=true
```

---

## 8. Required tests (focused)

Per canonical Phase 55 §158–§169:

```text
PREFLIGHT
1.  Phase52 signoff PASS
2.  Phase53 signoff PASS
3.  Phase54 signoff PASS
4.  Phase55 handoff ready=true
5.  Same seed list (42, 123, 2026)
6.  Same layers (2)
7.  Same heads (4)
8.  Same target population (N=2961)
9.  Profile integrity (sum approx 1)
10. Top1 distribution integrity (sum=1)

PAIR CONSTRUCTION
11. Heads in architectural order
12. Within-seed within-layer only
13. head_a < head_b (no duplicates, no symmetric)
14. Pair count = H(H-1)/2 = 6 per (seed, layer)
15. Total pairs = 36 (3 seeds * 2 layers * 6)

PROFILE SIMILARITY
16. Pearson computed
17. Spearman computed
18. Cosine computed
19. JSD computed with natural log
20. JSD within [0, ln2]
21. L1 computed
22. L2 computed
23. Wasserstein in minutes
24. Same lag support across all metrics

MATRIX AUDITS
25. JSD diagonal=0
26. JSD symmetric
27. Cosine diagonal=1
28. Cosine symmetric
29. Pearson diagonal=1
30. Spearman diagonal=1
31. L1 diagonal=0
32. Wasserstein diagonal=0
33. TVD diagonal=0
34. All distances nonnegative
35. Architectural order preserved

METRIC DIFFERENCES
36. Median expected lag differences
37. Median entropy differences
38. Median recent1h differences
39. Median recent6h differences
40. Median Lag80 differences
41. Delta convention A-B documented
42. Absolute delta fields available

PAIRED TARGET
43. Exact same target IDs across pair
44. N unchanged
45. Mean difference
46. Median difference
47. SD
48. p05/p25/p75/p95
49. Fraction positive/zero/negative
50. No iid significance test

TOP1 DISTRIBUTION
51. Lag support identical
52. Frequency sum=1 per head
53. TVD in [0, 1]
54. JSD valid
55. NEWEST_SOURCE tie rule inherited

HEAD BEHAVIOR
56. Every head has behavior row
57. Architectural order
58. No score, no rank
59. JSD to layer mean
60. L1 to layer mean
61. Cosine to layer mean
62. No outlier threshold

LAYER DIVERSITY
63. Pair count correct
64. Mean pairwise JSD
65. Median/max JSD
66. Mean L1
67. Mean Wasserstein
68. Mean abs expected-lag diff
69. Mean top1 TVD
70. No weighted composite
71. No layer "winner"

SCOPE (SAFETY)
72. No best head selected
73. No head pruning
74. No head ablation
75. No model retraining
76. No Test metric recomputation
77. No error conditioning
78. No regime conditioning
79. No worst-case statistical comparison
80. No cross-seed head matching
81. No same-index semantic assumption across seeds
82. No same-index semantic assumption across layers
83. No feature importance claim
84. No causal claim

PROVENANCE
85. Source Phase 54 version stored
86. Raw Phase 52 refs preserved (no modification)
87. Final lock SHA stored
88. Test population SHA stored
89. Head/layer counts stored
90. Phase56 handoff references exact outputs
91. Phase57 context handoff references exact outputs
```

---

## 9. Execution sequence

```text
1.  Verify Phase 52/53/54 signoffs + Phase 55 handoff (preflight).
2.  Verify Phase 54 source tables exist with deterministic SHA256.
3.  Verify same seed/layer/head/target ordering.
4.  Freeze Phase 55 head-comparison contract BEFORE any numerical result.
5.  Build canonical architectural head-pair list per (seed, layer).
6.  Verify pair count H(H-1)/2 = 6 per (seed, layer).
7.  Compute pairwise profile metrics (Pearson, Spearman, Cosine).
8.  Compute pairwise JSD / L1 / L2.
9.  Compute Wasserstein-1 distance in minutes.
10. Compute pairwise behavioral differences (signed + absolute).
11. Compute paired per-target difference summaries.
12. Compute top1 lag TVD + JSD.
13. Compute head-to-layer-mean distance.
14. Compute head behavior cards.
15. Compute layer-level diversity summary.
16. Build square similarity/distance matrices.
17. Run symmetry / diagonal / range audits.
18. Generate comparison figures in architectural order.
19. Write findings with no head winner.
20. Write Phase 56 handoff.
21. Write Phase 57 context handoff.
22. Run tests / discrepancy audit.
23. Write summary / report / README.
24. Sign off Phase 55.
```

---

## 10. Hard governance checklist

```text
[x] Phase 52 signoff PASS verified
[x] Phase 53 signoff PASS verified
[x] Phase 54 signoff PASS verified
[x] Phase 55 handoff ready=true verified
[x] Raw NPZ SHAs unchanged before processing
[x] Architecture amendment v1.17 applied
[x] Phase 55 pre-process plan APPROVED
[x] No new attention extraction
[x] No new Test inference
[x] No checkpoint loading
[x] No model.forward
[x] No training
[x] No scaler fitting
[x] No best-seed selection
[x] No best-head selection
[x] No head pruning / ablation
[x] No head clustering core
[x] No error-conditioned analysis
[x] No regime-conditioned analysis
[x] No cross-seed head matching
[x] No same-index semantic alignment assumption
[x] No feature importance claim
[x] No causal claim
[x] No Phase 56 implementation
[x] No Phase 57 implementation
[x] No notebook modification (Phase 55-H deferred)
```

---

## 11. Sub-phase gates

Per canonical Phase 55 detail, Phase 55 is decomposed into:

```text
Phase 55-A: governance + preflight (this document + amendment log)
Phase 55-B: source verification + integrity + target alignment
Phase 55-C: profile similarity + Wasserstein
Phase 55-D: behavioral differences + paired target-level differences
Phase 55-E: head behavior cards + head-to-layer-mean + layer diversity
Phase 55-F: matrices + audits + figures
Phase 55-G: findings + discrepancies + tests + summary + report + README + sign-off
Phase 55-H: notebook visualization (DEFERRED, requires separate Human approval)
```

This prompt authorizes Phases 55-A through 55-G as one continuous scientific
run. Phase 55-H is **DEFERRED** to a separate Human-approved run after Phase
55-G sign-off.

---

## 12. Source-of-truth precedence

```text
1. Canonical Phase 55 detail (docs/plan-doc/plan_detail_for_each_phase/Phase_55_Head_comparison.md)
2. This pre-process plan
3. Architecture rule v1.17 §7.37.16
4. Phase 54 signoff + handoff (frozen inputs)
5. Any conflict: canonical detail wins.
```
