"""Phase 55 - report and README generators."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sources import (
    SEEDS,
    NUM_LAYERS,
    NUM_HEADS,
    N_TEST,
    LOOKBACK,
)


def write_report(
    fp: Path,
    sources: Any,
    contract: dict[str, Any],
    n_pair_metrics: int,
    n_layer_div: int,
    n_behavior_cards: int,
    n_findings: int,
    n_paired_diff: int,
    n_top1_dist: int,
    tests_pass: int,
    tests_total: int,
    discrepancies_count: int,
    overall_status: str,
    sample_findings: list[str],
    sample_pairs: list[dict[str, Any]],
) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    md_lines = [
        "# Phase 55 — Head Comparison Analysis Report",
        "",
        f"- Version: `{contract.get('version', 'HEAD_COMPARISON-v2')}`",
        f"- Generated (UTC): {now}",
        f"- Phase 54 sign-off SHA: `{sources.p54_signoff_sha}`",
        f"- Phase 55 handoff SHA: `{sources.handoff_p55_sha}`",
        f"- Phase 55 contract SHA: `{contract.get('contract_sha256', '')}`",
        f"- Seeds: {list(SEEDS)}",
        f"- Lookback: {LOOKBACK}",
        f"- Layers: {NUM_LAYERS}; Heads: {NUM_HEADS}",
        f"- N_TEST: {N_TEST}",
        f"- Pair count per (seed, layer): {NUM_HEADS * (NUM_HEADS - 1) // 2}",
        f"- Total head pairs across all (seed, layer): {n_pair_metrics}",
        f"- Behavior cards: {n_behavior_cards}; Layer diversity rows: {n_layer_div}",
        f"- Findings: {n_findings}; Paired-difference summaries: {n_paired_diff}; Top1 distance rows: {n_top1_dist}",
        f"- Tests: {tests_pass}/{tests_total} PASS",
        f"- Discrepancies: {discrepancies_count}",
        f"- Overall status: **{overall_status}**",
        "",
        "## 1. Objective",
        "",
        "Phase 55 quantifies whether attention heads within the same encoder",
        "layer of the Final Transformer learn non-identical temporal-allocation",
        "patterns, using only the frozen Phase 54 last-query-attention artifacts.",
        "",
        "## 2. Why head comparison is behavioral, not predictive ranking",
        "",
        "Two heads with similar attention profiles can still produce different",
        "outputs because their `value` projections differ. Phase 55 therefore",
        "describes head behavior in profile, recency, concentration, coverage",
        "and top1-lag dimensions, without claiming predictive superiority.",
        "",
        "## 3. Upstream sources and integrity",
        "",
        "- Phase 52 raw last-query NPZ: SHA256 verified at preflight (raw SHA",
        "  preserved, never modified for analysis).",
        "- Phase 53 context handoff: visual context only, never used as",
        "  numeric input.",
        "- Phase 54 derived CSVs (mean profile, head-level summary, per-",
        "  vector metrics, top1 frequency, layer head-mean profile, lag-bin",
        "  mass, recent-mass, coverage) are the primary numeric source.",
        "- Profile integrity (sum approx 1) and top1 frequency integrity",
        "  (sum = 1) verified per (seed, layer, head).",
        "",
        "## 4. Within-seed/within-layer comparison scope",
        "",
        "- Primary unit: `(seed, layer, head_a, head_b)` with `head_a < head_b`.",
        "- Total pairs: 3 seeds * 2 layers * 6 pairs = 36 (H=4).",
        "- Head order: architectural (no reordering by similarity).",
        "",
        "## 5. Temporal profile similarity metrics",
        "",
        "- **Pearson**: linear shape association.",
        "- **Spearman**: rank-order similarity.",
        "- **Cosine**: directional similarity (0..1 for nonnegative profiles).",
        "- **Jensen-Shannon divergence (nat-log)**: bounded [0, ln 2].",
        "- **L1**: absolute probability-mass redistribution.",
        "- **L2**: Euclidean difference.",
        "- **Wasserstein-1 (minutes)**: average temporal distance to move one",
        "  profile's mass into the other.",
        "",
        "## 6. Why JSD and Wasserstein are useful",
        "",
        "- JSD is symmetric and bounded; suitable for probability distributions.",
        "- Wasserstein adds lag-axis geometry: two profiles shifted in time",
        "  are close in JSD/L1 but separated in Wasserstein.",
        "",
        "## 7. Head-level behavior summaries",
        "",
        f"- {n_behavior_cards} behavior cards produced in architectural order,",
        "  with no rank, score, or outlier threshold.",
        "",
        "## 8. Pairwise profile similarity results",
        "",
        f"- {n_pair_metrics} head pairs computed with full similarity metric set.",
        "",
        "## 9. Recency-allocation differences",
        "",
        "- Median `recent_1h_mass` differences per pair (signed Δ(A−B) and |Δ|).",
        "- 6h, 12h, 24h windows where supported.",
        "",
        "## 10. Concentration differences",
        "",
        "- Median normalized entropy differences per pair.",
        "",
        "## 11. Temporal coverage differences",
        "",
        "- Median Lag50/80/90 differences.",
        "",
        "## 12. Top1-lag distribution differences",
        "",
        f"- {n_top1_dist} head pairs compared via TVD and JSD on the top1 lag",
        "  frequency distribution. Phase 52 NEWEST_SOURCE tie rule preserved.",
        "",
        "## 13. Paired same-target behavioral differences",
        "",
        f"- {n_paired_diff} paired-difference summaries (N={N_TEST}, identical",
        "  target IDs per pair) for normalized_entropy, expected_lag_minutes,",
        "  recent_1h_mass, lag80_minutes.",
        "- No iid significance test performed (Test targets are temporally",
        "  dependent).",
        "",
        "## 14. Head-to-layer-mean deviation",
        "",
        "- JSD, L1, L2, cosine, Wasserstein-minutes per head.",
        "- No outlier threshold.",
        "",
        "## 15. Layer-level head diversity",
        "",
        f"- {n_layer_div} layer diversity rows with mean/median/max pairwise",
        "  JSD, mean pairwise L1/L2/Wasserstein, mean pairwise |Δ expected",
        "  lag|, mean pairwise top1 TVD.",
        "- No weighted composite score.",
        "",
        "## 16. Attention-allocation redundancy caveat",
        "",
        "- Two heads with near-identical attention profiles may still have",
        "  different value projections. Phase 55 only describes attention-",
        "  allocation redundancy, NOT functional redundancy.",
        "",
        "## 17. Why no best head is selected",
        "",
        "- Best-head selection requires predictive superiority, which",
        "  attention statistics alone cannot establish.",
        "",
        "## 18. Why no head pruning is performed",
        "",
        "- Pruning is out of scope. Phase 56 (error-conditioned attention)",
        "  and Phase 57 (seed stability) consume Phase 55 behavior metrics",
        "  as inputs.",
        "",
        "## 19. Handoff to error-conditioned attention (Phase 56)",
        "",
        "- `phase56_error_conditioned_attention_handoff.json` references all",
        "  behavior metrics and pair comparisons, with `best_head_selected`",
        "  = false and `error_conditioning_performed_in_phase55` = false.",
        "",
        "## 20. Handoff to seed-stability analysis (Phase 57)",
        "",
        "- `phase57_seed_stability_head_context_handoff.json` provides within-",
        "  seed pairwise JSD + Wasserstein + head-to-layer-mean context with",
        "  `same_index_semantic_alignment_assumed` = false and",
        "  `head_matching_performed` = false.",
        "",
        "## 21. Limitations",
        "",
        "- All comparisons are descriptive and within (seed, layer).",
        "- Cross-seed head matching is explicitly out of scope.",
        "- Heads are observed in TEST only; attention-training dynamics not",
        "  available.",
        "- 12h/24h windows are reported only when supported by lookback",
        "  (L=72 ⇒ 12h supported; 24h truncated).",
        "",
        "## 22. Definition of Done",
        "",
        "- Verified head profiles with sum approx 1 per (seed, layer, head).",
        "- All within-layer head pairs (H(H-1)/2 = 6 per (seed, layer))",
        "  computed exactly once.",
        "- Full pairwise profile metric set, Wasserstein, top1 TVD/JSD.",
        "- Paired same-target differences for 4 metrics.",
        "- Behavior cards in architectural order.",
        "- Layer diversity summaries without composite scores.",
        "- Phase 56 + Phase 57 handoffs emitted.",
        "- Findings: descriptive only.",
        "- All Phase 47-54 artifacts unchanged.",
        "",
        "## Sample findings (first 10)",
        "",
    ]
    for s in sample_findings[:10]:
        md_lines.append(f"- {s}")
    md_lines.append("")
    md_lines.append("## Sample head pairs (first 6)")
    md_lines.append("")
    for r in sample_pairs[:6]:
        md_lines.append(
            f"- seed={r.get('seed')} layer={r.get('layer_idx0')} "
            f"H{int(r.get('head_a_idx0', 0)) + 1}-H{int(r.get('head_b_idx0', 0)) + 1}: "
            f"JSD={r.get('jsd_profile', float('nan')):.4f}, "
            f"cosine={r.get('cosine_profile', float('nan')):.4f}, "
            f"Wasserstein={r.get('wasserstein_minutes', float('nan')):.1f} min"
        )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("End of Phase 55 report.")

    fp.write_text("\n".join(md_lines), encoding="utf-8")


def write_readme(fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    text = """# Phase 55 - Head Comparison Analysis (HEAD_COMPARISON-v2)

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
"""
    fp.write_text(text, encoding="utf-8")
