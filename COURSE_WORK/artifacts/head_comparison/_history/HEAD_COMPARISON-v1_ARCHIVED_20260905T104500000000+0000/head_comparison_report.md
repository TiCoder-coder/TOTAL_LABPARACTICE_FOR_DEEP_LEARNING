# Phase 55 — Head Comparison Analysis Report

- Version: `HEAD_COMPARISON-v1`
- Generated (UTC): 2026-09-04T16:17:26+00:00
- Phase 54 sign-off SHA: `9247e0353495c60acc192f87314542d95e4df38fffbb4e0a7b2b6dba75c00340`
- Phase 55 handoff SHA: `dab1d327f589417a6969be4bc33ab55311c83758c05ee5eac52784d79f7f4fcc`
- Phase 55 contract SHA: `c5663f2a21de4985548d8471ad4e87c9e844bd7e81672f2a363c71573f1d9692`
- Seeds: [42, 123, 2026]
- Lookback: 72
- Layers: 2; Heads: 4
- N_TEST: 2961
- Pair count per (seed, layer): 6
- Total head pairs across all (seed, layer): 36
- Behavior cards: 24; Layer diversity rows: 6
- Findings: 123; Paired-difference summaries: 144; Top1 distance rows: 36
- Tests: 46/46 PASS
- Discrepancies: 0
- Overall status: **PASS**

## 1. Objective

Phase 55 quantifies whether attention heads within the same encoder
layer of the Final Transformer learn non-identical temporal-allocation
patterns, using only the frozen Phase 54 last-query-attention artifacts.

## 2. Why head comparison is behavioral, not predictive ranking

Two heads with similar attention profiles can still produce different
outputs because their `value` projections differ. Phase 55 therefore
describes head behavior in profile, recency, concentration, coverage
and top1-lag dimensions, without claiming predictive superiority.

## 3. Upstream sources and integrity

- Phase 52 raw last-query NPZ: SHA256 verified at preflight (raw SHA
  preserved, never modified for analysis).
- Phase 53 context handoff: visual context only, never used as
  numeric input.
- Phase 54 derived CSVs (mean profile, head-level summary, per-
  vector metrics, top1 frequency, layer head-mean profile, lag-bin
  mass, recent-mass, coverage) are the primary numeric source.
- Profile integrity (sum approx 1) and top1 frequency integrity
  (sum = 1) verified per (seed, layer, head).

## 4. Within-seed/within-layer comparison scope

- Primary unit: `(seed, layer, head_a, head_b)` with `head_a < head_b`.
- Total pairs: 3 seeds * 2 layers * 6 pairs = 36 (H=4).
- Head order: architectural (no reordering by similarity).

## 5. Temporal profile similarity metrics

- **Pearson**: linear shape association.
- **Spearman**: rank-order similarity.
- **Cosine**: directional similarity (0..1 for nonnegative profiles).
- **Jensen-Shannon divergence (nat-log)**: bounded [0, ln 2].
- **L1**: absolute probability-mass redistribution.
- **L2**: Euclidean difference.
- **Wasserstein-1 (minutes)**: average temporal distance to move one
  profile's mass into the other.

## 6. Why JSD and Wasserstein are useful

- JSD is symmetric and bounded; suitable for probability distributions.
- Wasserstein adds lag-axis geometry: two profiles shifted in time
  are close in JSD/L1 but separated in Wasserstein.

## 7. Head-level behavior summaries

- 24 behavior cards produced in architectural order,
  with no rank, score, or outlier threshold.

## 8. Pairwise profile similarity results

- 36 head pairs computed with full similarity metric set.

## 9. Recency-allocation differences

- Median `recent_1h_mass` differences per pair (signed Δ(A−B) and |Δ|).
- 6h, 12h, 24h windows where supported.

## 10. Concentration differences

- Median normalized entropy differences per pair.

## 11. Temporal coverage differences

- Median Lag50/80/90 differences.

## 12. Top1-lag distribution differences

- 36 head pairs compared via TVD and JSD on the top1 lag
  frequency distribution. Phase 52 NEWEST_SOURCE tie rule preserved.

## 13. Paired same-target behavioral differences

- 144 paired-difference summaries (N=2961, identical
  target IDs per pair) for normalized_entropy, expected_lag_minutes,
  recent_1h_mass, lag80_minutes.
- No iid significance test performed (Test targets are temporally
  dependent).

## 14. Head-to-layer-mean deviation

- JSD, L1, L2, cosine, Wasserstein-minutes per head.
- No outlier threshold.

## 15. Layer-level head diversity

- 6 layer diversity rows with mean/median/max pairwise
  JSD, mean pairwise L1/L2/Wasserstein, mean pairwise |Δ expected
  lag|, mean pairwise top1 TVD.
- No weighted composite score.

## 16. Attention-allocation redundancy caveat

- Two heads with near-identical attention profiles may still have
  different value projections. Phase 55 only describes attention-
  allocation redundancy, NOT functional redundancy.

## 17. Why no best head is selected

- Best-head selection requires predictive superiority, which
  attention statistics alone cannot establish.

## 18. Why no head pruning is performed

- Pruning is out of scope. Phase 56 (error-conditioned attention)
  and Phase 57 (seed stability) consume Phase 55 behavior metrics
  as inputs.

## 19. Handoff to error-conditioned attention (Phase 56)

- `phase56_error_conditioned_attention_handoff.json` references all
  behavior metrics and pair comparisons, with `best_head_selected`
  = false and `error_conditioning_performed_in_phase55` = false.

## 20. Handoff to seed-stability analysis (Phase 57)

- `phase57_seed_stability_head_context_handoff.json` provides within-
  seed pairwise JSD + Wasserstein + head-to-layer-mean context with
  `same_index_semantic_alignment_assumed` = false and
  `head_matching_performed` = false.

## 21. Limitations

- All comparisons are descriptive and within (seed, layer).
- Cross-seed head matching is explicitly out of scope.
- Heads are observed in TEST only; attention-training dynamics not
  available.
- 12h/24h windows are reported only when supported by lookback
  (L=72 ⇒ 12h supported; 24h truncated).

## 22. Definition of Done

- Verified head profiles with sum approx 1 per (seed, layer, head).
- All within-layer head pairs (H(H-1)/2 = 6 per (seed, layer))
  computed exactly once.
- Full pairwise profile metric set, Wasserstein, top1 TVD/JSD.
- Paired same-target differences for 4 metrics.
- Behavior cards in architectural order.
- Layer diversity summaries without composite scores.
- Phase 56 + Phase 57 handoffs emitted.
- Findings: descriptive only.
- All Phase 47-54 artifacts unchanged.

## Sample findings (first 10)

- Within seed 42 layer 0, heads H1-H2 have large Wasserstein distance (226.9 min) indicating temporally-shifted allocations.
- Within seed 42 layer 0, heads H1-H3 have large Wasserstein distance (122.3 min) indicating temporally-shifted allocations.
- Within seed 42 layer 0, heads H1-H4 have large Wasserstein distance (141.5 min) indicating temporally-shifted allocations.
- Within seed 42 layer 1, heads H1-H2 have small Wasserstein distance (25.9 min) indicating similar temporal centroids.
- Within seed 123 layer 0, heads H1-H2 have large Wasserstein distance (163.3 min) indicating temporally-shifted allocations.
- Within seed 123 layer 0, heads H2-H3 have high JSD (0.2105) over mean temporal profile; distinct temporal allocations.
- Within seed 123 layer 0, heads H2-H3 have large Wasserstein distance (234.8 min) indicating temporally-shifted allocations.
- Within seed 123 layer 0, heads H3-H4 have large Wasserstein distance (139.1 min) indicating temporally-shifted allocations.
- Within seed 123 layer 1, heads H2-H3 have large Wasserstein distance (127.7 min) indicating temporally-shifted allocations.
- Within seed 123 layer 1, heads H2-H4 have high JSD (0.2314) over mean temporal profile; distinct temporal allocations.

## Sample head pairs (first 6)

- seed=42 layer=0 H1-H2: JSD=0.1875, cosine=0.4115, Wasserstein=226.9 min
- seed=42 layer=0 H1-H3: JSD=0.1794, cosine=0.5396, Wasserstein=122.3 min
- seed=42 layer=0 H1-H4: JSD=0.1340, cosine=0.5645, Wasserstein=141.5 min
- seed=42 layer=0 H2-H3: JSD=0.1274, cosine=0.5379, Wasserstein=106.4 min
- seed=42 layer=0 H2-H4: JSD=0.1167, cosine=0.6307, Wasserstein=91.9 min
- seed=42 layer=0 H3-H4: JSD=0.1039, cosine=0.6746, Wasserstein=41.7 min

---

End of Phase 55 report.