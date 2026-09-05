# Phase 57 — Seed-Stability Attention Check Report

## 1. Objective

Phase 57 quantifies whether the temporal-attention patterns of the Final Transformer remain stable across the three official final seeds (42 / 123 / 2026), once the head permutation problem is handled correctly. Two parallel views are reported:

- **S57-A**: permutation-invariant layer head-mean stability (mean over heads first, then compare across seeds).
- **S57-B**: permutation-aware matched-head stability, using a deterministic canonical JSD matching within each encoder layer.

## 2. Why head permutation matters

In Multi-Head Attention, heads in the same encoder layer do not have fixed semantic labels. Two independently trained models can learn similar roles but place them in different head indices. Same-index comparisons across seeds are NOT assumed to be semantically equivalent.

## 3. Frozen sources and three-seed contract

- Seeds: [42, 123, 2026] (immutable).
- Anchor seed = 42 (anchor reason: FIRST_PREDECLARED_FINAL_SEED). Anchor is NOT performance-based.
- Layers: 2; Heads: 4; Lookback: 72 steps (720 min = 12h).
- Final lock SHA: `81fb87c44b6af31b...`
- Test population SHA: `d7dbc0b3cde772dc...`

## 4. Permutation-invariant layer head-mean stability

Mean profiles and per-target vectors are compared across all three seed pairs (42-123, 42-2026, 123-2026) using JSD, L1, L2, cosine, Pearson, Spearman, Wasserstein (minutes). No threshold invented; exact values reported.

## 5. Per-target layer attention stability

Per-target layer head-mean vectors are compared across all seed pairs. Distribution summaries (N / mean / SD / median / percentiles) are reported, not iid confidence intervals.

## 6. Canonical head-matching methodology

- Representation: full-Test mean last-query temporal profile per (seed, layer, head).
- Cost: JSD (natural log), range [0, ln(2)].
- Search: exhaustive enumeration of all H! permutations under the locked H2/H4 protocol (24 perms per layer per pair for H=4).
- Tie-break: minimum total JSD -> minimum total Wasserstein (minutes) -> lexicographically smallest permutation.
- MATCH_TIE_TOL = 1e-12 (frozen before any computation).

## 7. Matching cost matrices

Cost matrices are stored in `head_matching_cost_matrices.csv` for every (layer, seed_pair, head_a, head_b). Canonical assignments are highlighted.

## 8. Matching ambiguity and edge margins

Best total JSD, second-best total JSD, assignment gap, edge margins are recorded for every (layer, seed_pair). Matches within MATCH_TIE_TOL trigger `ambiguous_match_warning`.

## 9. Cycle-consistency audit

Direct 123-2026 mapping vs anchor-induced 123-42-2026 mapping. Per-head boolean + per-layer fraction are recorded.

## 10. Wasserstein matching sensitivity

Wasserstein-only matching is computed independently. Agreement fraction with canonical JSD matching is reported. Wasserstein never replaces canonical JSD.

## 11. Canonical three-seed matched-head groups

Anchored at seed42. Each (layer, group) records (seed42_head, seed123_head, seed2026_head, ambiguity flags, cycle-consistent flag).

## 12. Matched-head mean-profile stability

Three pairwise stability views per matched group; consensus profiles (mean of three seeds) with sum ~1 audit.

## 13. Matched-head per-target stability

Frozen global mapping reused for every Test target. No target-specific rematch.

## 14. Matched-head metric/top1 stability

Six CORE_ATTENTION_METRICS-v1 compared across matched heads; top1 lag distribution TVD + JSD + modal-lag agreement.

## 15. Dense worst-case full-map stability

Frozen Phase 51 cases, raw [L,L] matrices, rowwise JSD + cosine + normalized Frobenius. NO PNG/image-pixel similarity. Selection-conditioned caveat documented.

## 16. Error-conditioned layer-level stability

Primary error-effect robustness view. Phase 56 layer-head-mean effects compared across seeds for Spearman / HIGH-LOW Cliff's / HIGH-LOW median delta / shared cohort / profile JSD / profile Wasserstein.

## 17. Error-conditioned matched-head stability

Secondary. Phase 56 per-head effects reindexed to canonical matched groups. Ambiguity warnings propagated.

## 18. Shared-cohort seed consistency

Identical target IDs across seeds → directly comparable effect magnitudes.

## 19. Prediction-spread vs attention-disagreement analysis

Secondary diagnostic. Spearman between Phase 48 prediction range/SD and layer mean-pairwise attention disagreement. No causal claim.

## 20. Main attention robustness findings

Total findings: 17.

## 21. Why no best seed/head is selected

All three seeds are official stochastic realizations. Cross-seed head semantic alignment is NOT assumed.

## 22. Why attention stability is not functional equivalence

Stable attention weights do not prove stable value (V) projections or functional contribution. The matching operates on full-Test mean last-query profiles only.

## 23. Why stability is not causal explanation

Stable error-attention association does not establish causal mechanism.

## 24. Three-seed limitation

Only three seeds; mean/SD across seeds describe the chosen stochastic realizations, not the full seed population.

## 25. Handoff to Final Tables

Phase 58 (Final Tables) is handoff-ready with frozen attention robustness evidence.

## 26. Definition of Done

- All three seed sources verified.
- Layer head-mean + per-target stability complete.
- Canonical JSD matching complete with ambiguity + cycle + sensitivity audits.
- Matched-head stability + dense-case stability complete.
- Error-effect layer + matched-head stability complete.
- Prediction-spread secondary association complete.
- No best seed/head; no pruning; no ablation; no retraining.
- Phase 58 handoff ready; Phase 59 context ready.