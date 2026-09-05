# Phase 57 — README (Seed-Stability Attention Check)

## Why same-index heads cannot be assumed equivalent

In Multi-Head Attention, heads in the same encoder layer have no fixed semantic label. Two independently trained models can learn similar roles but assign them to different head indices. Same-index diagnostic comparisons are explicitly flagged as `semantic_alignment_not_guaranteed`.

## Why layer head-mean is permutation-invariant

Averaging over heads within a layer FIRST (before cross-seed comparison) removes the head-index issue. This is the primary robust cross-seed stability view in Phase 57 (S57-A).

## Why JSD is the canonical matching cost

JSD is symmetric, bounded [0, ln(2)], designed for probability distributions, and already used in Phase 55. It does NOT depend on forecast error.

## Why H2/H4 allows exhaustive permutation matching

Under the locked H2/H4 protocol, the canonical Final Transformer architecture uses H=4 heads. With H=4, exhaustive enumeration of all 4! = 24 permutations is fully tractable.

## How the deterministic tie-break works

(1) minimum total JSD; (2) if tied within MATCH_TIE_TOL = 1e-12, minimum total Wasserstein (minutes); (3) if still tied, lexicographically smallest permutation.

## Why seed42 is only an alignment anchor
Seed42 is the first predeclared final seed (`ANCHOR_REASON = FIRST_PREDECLARED_FINAL_SEED`). It is NOT selected because it has lower Test RMSE, cleaner attention, or better matching.

## What matching ambiguity means

When multiple permutations produce total JSD within MATCH_TIE_TOL of the minimum, an ambiguous-match warning is recorded. Ambiguity is a legitimate stability finding, not a methodological failure.

## What cycle consistency means

Cycle consistency compares the direct 123-2026 mapping against the anchor-induced 123-42-2026 mapping. Inconsistency indicates that head identities are weakly identifiable across seeds.

## Why Wasserstein matching is sensitivity-only

Wasserstein-only matching is computed independently. Agreement with canonical JSD is reported. Wasserstein never replaces canonical JSD, regardless of how the result looks.

## Why matching is frozen before error-conditioned analysis

Freezing the matching SHA BEFORE applying Phase 56 effects prevents outcome-conditioned alignment (which would inflate apparent stability).

## How per-target matched-head stability is computed

The same global frozen mapping is applied to every Test target. NO target-specific rematching.

## Why dense-case stability is supplementary

Dense-case attention is only available for the frozen Phase 51 worst-case set. The dense case set is enriched for shared hard cases, so dense-case stability is selection-conditioned, not unbiased full-Test stability.

## Why error-conditioned stability prioritizes layer head-mean

Layer head-mean averages over heads FIRST, removing permutation. This avoids the ambiguity-propagation chain that matched-head results carry when head matching is weak.

## Why prediction-attention disagreement is non-causal

Both quantities are co-observed outcomes of the same forward pass + attention extraction. Correlation does not establish that attention instability CAUSED prediction instability.

## Why no stability score/best seed/best head is created

Weighted overall stability scores, ad-hoc stable/unstable thresholds, and best-seed/best-head selection are all explicitly forbidden. Stability is reported as transparent, separate evidence.

## Forbidden actions summary

- training, fine-tune, optimizer.step, scaler.fit
- new Test inference; new attention extraction
- destructive overwrite of any Phase 47-56 canonical artifact
- best-seed/head selection; ensemble; weighted overall stability score
- head ranking, pruning, ablation, clustering
- target-specific, error-specific, regime-specific, case-specific rematching
- anchor change after seeing results
- Wasserstein replacing canonical JSD mapping post hoc
- Test error cohort used as deployment regime
- attention = feature importance / causal explanation / functional equivalence proof
- cartesian subgroup mining
- reading PNG pixels for numeric attention values
- implementing Phase 58 / 59 (handoff files only)
- notebook modification in this run