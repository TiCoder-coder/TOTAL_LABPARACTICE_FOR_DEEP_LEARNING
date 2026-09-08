# Phase 55 - Head Comparison Analysis (HEAD_COMPARISON-v2)

## Why comparisons are within seed/layer

Cross-seed same-index head comparison is not assumed to imply same semantic
role. Layer depth and seed identity alter the parameter context, so the
primary comparison unit is `(seed, layer, head_a, head_b)` with
`head_a < head_b`.

## Why same-index heads across seeds are not assumed identical

`H1` in seed 42 and `H1` in seed 2026 are different parameter sets. The
project does not assume any implicit alignment.

## What Pearson / Spearman / Cosine / JSD / L1 / L2 mean

- Pearson: linear shape correlation.
- Spearman: rank-order correlation.
- Cosine: directional similarity in [0, 1].
- JSD: Jensen-Shannon divergence (natural log, [0, ln 2]).
- L1: sum of absolute probability-mass differences.
- L2: Euclidean distance.

## Why Wasserstein uses lag minutes

Wasserstein-1 over the cumulative distribution on the lag-minutes axis
captures temporal displacement that JSD / L1 cannot see: two profiles
shifted by an hour may have small JSD but meaningful Wasserstein.

## What TVD on top1 lag distribution means

Total Variation Distance between the two top1-lag frequency distributions
inherited from Phase 52 (NEWEST_SOURCE tie rule).

## Why paired same-target differences are useful

Both heads are observed on the SAME Test target IDs, so the difference
distribution is paired per target. The paired distribution reveals
whether head A is consistently more recent than head B on the same
target set, or whether the difference is target-dependent.

## Why no iid p-values are central

Test targets are temporally dependent. iid significance tests would
underestimate uncertainty.

## Why architectural head order is preserved

Canonical matrices use architectural order (H1..HH). Any clustering or
sorting by metric would create an implicit ranking.

## Why redundancy is only attention-allocation redundancy

Phase 55 measures similarity of attention profiles. Two heads with
similar attention profiles may still output different values because
their `value` projections differ. The same applies to `Q`/`K`/`O`
projections.

## Why no best head / pruning is allowed

Best-head selection requires predictive superiority, which attention
statistics alone cannot establish. Phase 55 does not prune, ablate,
or rank heads.

## How Phase 56 / 57 consume outputs

- `phase56_error_conditioned_attention_handoff.json`: head behavior +
  pair comparisons + layer diversity + raw last-query NPZ refs.
- `phase57_seed_stability_head_context_handoff.json`: within-seed
  pairwise JSD + Wasserstein + head-to-layer-mean + diversity context.

## Safety constraints (HARD)

- No training, fine-tune, optimizer.step, scaler.fit.
- No new attention extraction, no new Test inference.
- No model load, no model.forward.
- No best-seed selection, ensemble.
- No best-head, head pruning, head ablation, head clustering core,
  unsupervised clustering.
- No error conditioning (deferred to Phase 56).
- No regime conditioning.
- No cross-seed head matching.
- No same-index semantic alignment assumption.
- No feature importance or causal claim.
- No Phase 56 or Phase 57 implementation.

## Source-of-truth precedence

1. `docs/plan-doc/plan_detail_for_each_phase/Phase_55_Head_comparison.md` (canonical detail)
2. `docs/plan-doc/plan_before_process/phase_55_head_comparison_plan.md` (pre-process plan)
3. `docs/RULE_BASE/architecture_rule.md` §7.37.16 (amendment v1.17)
4. Frozen Phase 54 signoff + handoff
5. Conflicts: canonical detail wins.
