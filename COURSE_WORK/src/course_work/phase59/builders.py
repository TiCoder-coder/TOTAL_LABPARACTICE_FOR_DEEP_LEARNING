# -*- coding: utf-8 -*-
"""Phase 59 — RQ matrix + claim-strength ledger + outcome matrix.

These builders read frozen Phase 58 FT01..FT10 and FA12 evidence, then
construct the canonical conclusion ledgers using only already-frozen
numbers and qualitative evidence classes. No new analysis is performed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from . import constants as C
from .sources import FrozenSources59
from .writers import fmt_display, fmt_full


# ---------------------------------------------------------------------------
# RQ1–RQ10 conclusion matrix
# ---------------------------------------------------------------------------

def build_rq_matrix(sources: FrozenSources59) -> list[dict]:
    """Build the research-question conclusion matrix from frozen Phase 58 evidence."""
    ft02 = sources.ft("FT02")
    ft03 = sources.ft("FT03")
    ft05 = sources.ft("FT05")
    ft06 = sources.ft("FT06")
    ft07 = sources.ft("FT07")
    ft08 = sources.ft("FT08")
    ft09 = sources.ft("FT09")
    ft10 = sources.ft("FT10")

    # Pre-extract a few canonical numeric facts
    ft02_by_model = {r.get("model", "").strip(): r for r in ft02}

    persistence = ft02_by_model.get("Persistence Baseline", {})
    lstm = ft02_by_model.get("Tuned LSTM Baseline", {})
    s42 = ft02_by_model.get("Final Transformer — Seed 42", {})
    s123 = ft02_by_model.get("Final Transformer — Seed 123", {})
    s2026 = ft02_by_model.get("Final Transformer — Seed 2026", {})
    summary_row = ft02_by_model.get("Final Transformer — Three-Seed Summary", {})

    def _mae(r) -> str:
        return fmt_display(r.get("mae_wh", ""), "Wh")

    def _rmse(r) -> str:
        return fmt_display(r.get("rmse_wh", ""), "Wh")

    def _r2(r) -> str:
        return fmt_display(r.get("r2", ""), "R2")

    # RQ1
    rq1_allowed = (
        f"On the frozen chronological Held-Out Test segment "
        f"(FINAL_TEST_POP-v1), the final selected Transformer achieved "
        f"three-seed mean MAE {_mae(summary_row)} Wh, "
        f"RMSE {_rmse(summary_row)} Wh, "
        f"and R² {_r2(summary_row)} across seeds {C.SEED_LABEL}."
    )

    # RQ2
    rq2_allowed = (
        f"Persistence baseline MAE {_mae(persistence)} Wh, RMSE {_rmse(persistence)} Wh, R² {_r2(persistence)}. "
        f"Tuned LSTM baseline was not evaluated on FINAL_TEST_POP-v1 due to lookback "
        f"mismatch (L36 vs L72). The final Transformer RMSE is "
        f"{_rmse(summary_row)} Wh vs Persistence {_rmse(persistence)} Wh."
    )

    # RQ3
    rq3_allowed = (
        "Rolling-origin pooled RMSE and fold variability are reported in FT03 "
        "as DEVELOPMENT_EVIDENCE only."
    )

    # RQ4
    rq4_allowed = (
        f"FT05 reports error by regime and worst-error concentration "
        f"with frozen Phase 50/51 definitions; "
        f"worst-error cases remain valid frozen Test observations."
    )

    # RQ5
    rq5_allowed = (
        "FT06 reports last-query temporal attention summary (recent-1h mass, "
        "recent-6h mass, expected lag, normalized entropy) per seed, per layer. "
        "Lookback = 72 steps = 12 h, so the most recent 24-h mass is truncated."
    )

    # RQ6
    rq6_allowed = (
        "FT07 reports within-seed head-comparison descriptive metrics "
        "(pairwise JSD, Wasserstein minutes, cosine, top1 TVD) per seed, per layer. "
        "Higher values indicate higher pairwise temporal-profile diversity, "
        "NOT functional redundancy."
    )

    # RQ7
    rq7_allowed = (
        "FT08 reports Phase-56 error-conditioned Spearman ρ and HIGH_ERROR vs LOW_ERROR "
        "differences. HIGH/LOW are Test-relative diagnostic cohorts, NOT deployment regimes."
    )

    # RQ8
    rq8_allowed = (
        "FT09 reports Phase-57 seed-stability evidence. Layer head-mean "
        "(permutation-invariant) is the primary view; permutation-aware matched-head "
        "and matching sensitivity are secondary. Same numeric head indices across seeds "
        "are NOT assumed semantically equivalent."
    )

    # RQ9
    rq9_allowed = (
        "Attention describes temporal token allocation only — NOT raw-feature importance, "
        "NOT causal contribution. Similarity in attention profiles does NOT prove functional "
        "redundancy. Matching ambiguity and cycle consistency (Layer 0 = 1/4; Layer 1 = 4/4) "
        "are reported when partial."
    )

    # RQ10
    rq10_allowed = (
        "FA12 propagates upstream caveats; limitations are grouped into L1 (dataset), "
        "L2 (forecast scope), L3 (model selection), L4 (statistical), "
        "L5 (attention interpretability), L6 (deployment/generalization)."
    )

    rows = []
    rq_records = [
        ("RQ1", C.RESEARCH_QUESTIONS[0][1], "FT02", "FT01", 58,
         C.EVIDENCE_HELD_OUT_TEST, C.LEVEL_3, C.SUPPORTED,
         rq1_allowed,
         "mean ± sample SD (ddof=1), NOT an ensemble; scoped to held-out chronological segment",
         "Test performance does NOT generalize to other households / climates / buildings",
         ),
        ("RQ2", C.RESEARCH_QUESTIONS[1][1], "FT02", "FT03", 58,
         C.EVIDENCE_HELD_OUT_TEST, C.LEVEL_3, C.MIXED,
         rq2_allowed,
         "mixed evidence preserved if applicable; Persistence and LSTM both reported",
         "Transformer is NOT universally superior; LSTM / Persistence results not omitted",
         ),
        ("RQ3", C.RESEARCH_QUESTIONS[2][1], "FT03", "FT01", 58,
         C.EVIDENCE_DEVELOPMENT, C.LEVEL_2, C.PARTIALLY_SUPPORTED,
         rq3_allowed,
         "rolling-origin is DEVELOPMENT_EVIDENCE only; pooled RMSE primary, mean fold RMSE secondary",
         "rolling-origin is NOT a second Test set",
         ),
        ("RQ4", C.RESEARCH_QUESTIONS[3][1], "FT05", "FT04", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_1, C.SUPPORTED,
         rq4_allowed,
         "residual sign convention: y_true - y_pred; positive = UNDERPREDICTION",
         "regime does NOT cause error; worst cases not deleted from final metrics",
         ),
        ("RQ5", C.RESEARCH_QUESTIONS[4][1], "FT06", "FA12", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_2, C.PARTIALLY_SUPPORTED,
         rq5_allowed,
         "24-h mass truncated because lookback = 72 = 12 h; last-query is one token",
         "attention ≠ feature importance; attention ≠ causal",
         ),
        ("RQ6", C.RESEARCH_QUESTIONS[5][1], "FT07", "FA12", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_2, C.SUPPORTED,
         rq6_allowed,
         "similarity ≠ functional redundancy; value/output projections can differ",
         "no head-pruning recommendation as result",
         ),
        ("RQ7", C.RESEARCH_QUESTIONS[6][1], "FT08", "FA12", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_1, C.PARTIALLY_SUPPORTED,
         rq7_allowed,
         "associations are descriptive; weak/mixed outcomes preserved",
         "HIGH/LOW ≠ deployment regime; association ≠ causality",
         ),
        ("RQ8", C.RESEARCH_QUESTIONS[7][1], "FT09", "FA12", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_2, C.PARTIALLY_SUPPORTED,
         rq8_allowed,
         "layer head-mean prioritized; matching ambiguity / cycle inconsistency surfaced",
         "same-index heads NOT semantically equivalent; no best-seed",
         ),
        ("RQ9", C.RESEARCH_QUESTIONS[8][1], "FT10", "FA12", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_0, C.SUPPORTED,
         rq9_allowed,
         "interpretive boundary is permanent; partial / ambiguous matching acknowledged",
         "no causal attention claim; no functional-equivalence proof",
         ),
        ("RQ10", C.RESEARCH_QUESTIONS[9][1], "FA12", "FT10", 58,
         C.EVIDENCE_POST_TEST_DIAGNOSTIC, C.LEVEL_0, C.SUPPORTED,
         rq10_allowed,
         "single-house dataset; three seeds; sequential tuning; attention diagnostic only",
         "no external generalization; no deployment-ready claim",
         ),
    ]

    for (rq_id, q, pt, st, sp, ev, lvl, status, allowed, caveat, prohibited) in rq_records:
        rows.append({
            "rq_id": rq_id,
            "question": q,
            "primary_table": pt,
            "secondary_table": st,
            "source_phase": str(sp),
            "evidence_class": ev,
            "answer_status": status,
            "claim_level": lvl,
            "allowed_conclusion": allowed,
            "required_caveat": caveat,
            "prohibited_overclaim": prohibited,
            "status": "CLOSED",
        })
    return rows


# ---------------------------------------------------------------------------
# Claim-strength ledger (candidates from FT findings)
# ---------------------------------------------------------------------------

def build_claim_ledger(sources: FrozenSources59, ft_rows: dict[str, list[dict]]) -> list[dict]:
    """Build candidate claim ledger; auto-reject LEVEL_4."""
    # Numeric facts from FT02
    ft02 = {r.get("model", "").strip(): r for r in ft_rows.get("FT02", [])}
    summary = ft02.get("Final Transformer — Three-Seed Summary", {})
    persistence = ft02.get("Persistence Baseline", {})
    s42 = ft02.get("Final Transformer — Seed 42", {})
    s123 = ft02.get("Final Transformer — Seed 123", {})
    s2026 = ft02.get("Final Transformer — Seed 2026", {})

    def _disp(v, unit):
        return fmt_display(v, unit)

    claims: list[dict] = []

    # C01 — final performance (FT02)
    claims.append({
        "claim_id": "C01",
        "topic": "final_test_performance",
        "claim_text": (
            f"On the frozen chronological Held-Out Test segment, the final selected Transformer "
            f"obtained three-seed mean MAE {_disp(summary.get('mae_wh'), 'Wh')} Wh, "
            f"RMSE {_disp(summary.get('rmse_wh'), 'Wh')} Wh, "
            f"and R² {_disp(summary.get('r2'), 'R2')} across seeds {C.SEED_LABEL}."
        ),
        "claim_level": C.LEVEL_3,
        "evidence_class": C.EVIDENCE_HELD_OUT_TEST,
        "supporting_table": "FT02",
        "supporting_source": "Phase 47 final Test metrics; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1 (2961 targets per seed)",
        "seed_scope": "42,123,2026",
        "numeric_support_if_any": (
            f"MAE={summary.get('mae_wh', '')}; "
            f"RMSE={summary.get('rmse_wh', '')}; "
            f"R2={summary.get('r2', '')}"
        ),
        "required_caveat": "Three-Seed Summary = mean ± sample SD (ddof=1); NOT an ensemble forecast.",
        "forbidden_extension": "do not extend to other households/climates/buildings",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C02 — baseline comparison (FT02)
    claims.append({
        "claim_id": "C02",
        "topic": "baseline_comparison",
        "claim_text": (
            f"The Persistence baseline obtained MAE {_disp(persistence.get('mae_wh'), 'Wh')} Wh, "
            f"RMSE {_disp(persistence.get('rmse_wh'), 'Wh')} Wh, "
            f"and R² {_disp(persistence.get('r2'), 'R2')} on FINAL_TEST_POP-v1. "
            "Tuned LSTM was not evaluated on FINAL_TEST_POP-v1 due to lookback "
            "mismatch (L36 vs L72)."
        ),
        "claim_level": C.LEVEL_3,
        "evidence_class": C.EVIDENCE_HELD_OUT_TEST,
        "supporting_table": "FT02",
        "supporting_source": "Phase 47 final Test metrics; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1",
        "seed_scope": "single final Test realization (no fake SD)",
        "numeric_support_if_any": f"persistence_rmse={persistence.get('rmse_wh', '')}",
        "required_caveat": "Both baselines reported even when unfavorable to the Transformer.",
        "forbidden_extension": "do not omit LSTM / Persistence results",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C03 — temporal robustness (FT03)
    claims.append({
        "claim_id": "C03",
        "topic": "temporal_robustness",
        "claim_text": (
            "Phase 44 rolling-origin pooled RMSE is the primary Phase 44 robustness "
            "quantity; mean fold RMSE is secondary. Fold-by-fold variability is reported "
            "in FT03."
        ),
        "claim_level": C.LEVEL_2,
        "evidence_class": C.EVIDENCE_DEVELOPMENT,
        "supporting_table": "FT03",
        "supporting_source": "Phase 44 rolling-origin; Phase 58 reporting synthesis",
        "population": "Phase 44 rolling-origin folds",
        "seed_scope": "selected final Transformer seed=42 development run",
        "numeric_support_if_any": "",
        "required_caveat": "Rolling-origin is DEVELOPMENT_EVIDENCE only; NOT a second Test set.",
        "forbidden_extension": "do not conflate with Held-Out Test",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C04 — error concentration / regime (FT04/FT05)
    claims.append({
        "claim_id": "C04",
        "topic": "error_behavior",
        "claim_text": (
            "Forecast accuracy deteriorated in high-consumption and rapid-change regimes "
            "relative to the reference baseline; the largest errors were retained as "
            "valid Test observations and were used for diagnostic analysis rather than "
            "excluded from final metrics."
        ),
        "claim_level": C.LEVEL_1,
        "evidence_class": C.EVIDENCE_POST_TEST_DIAGNOSTIC,
        "supporting_table": "FT05",
        "supporting_source": "Phase 50/51 frozen regime + worst-case definitions; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1",
        "seed_scope": "selected final Transformer seed=42",
        "numeric_support_if_any": "",
        "required_caveat": "Residual sign convention: y_true − y_pred; positive = UNDERPREDICTION.",
        "forbidden_extension": "regime does NOT cause error; no deletion recommendation",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C05 — last-query attention (FT06)
    claims.append({
        "claim_id": "C05",
        "topic": "temporal_attention",
        "claim_text": (
            "Across the final Test, last-query attention allocated substantial mass to "
            "recent historical positions (recent-1h and recent-6h mass from FT06). "
            "Lookback = 72 steps = 12 h, so the most recent 24-h mass is truncated."
        ),
        "claim_level": C.LEVEL_2,
        "evidence_class": C.EVIDENCE_POST_TEST_DIAGNOSTIC,
        "supporting_table": "FT06",
        "supporting_source": "Phase 54 last-query metrics; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1 (2961 last-query rows per seed)",
        "seed_scope": "42,123,2026 (per-layer/per-seed means)",
        "numeric_support_if_any": "",
        "required_caveat": "Attention is temporal token allocation; NOT raw-feature importance; NOT causal.",
        "forbidden_extension": "no feature-importance claim; no causal claim",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C06 — head diversity (FT07)
    claims.append({
        "claim_id": "C06",
        "topic": "head_diversity",
        "claim_text": (
            "Different heads within each layer learned non-identical temporal allocation "
            "profiles, as reported by FT07 (pairwise JSD, Wasserstein minutes, cosine, "
            "top1 TVD)."
        ),
        "claim_level": C.LEVEL_1,
        "evidence_class": C.EVIDENCE_POST_TEST_DIAGNOSTIC,
        "supporting_table": "FT07",
        "supporting_source": "Phase 55 within-seed head comparison; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1 (within-seed, within-layer, across-head)",
        "seed_scope": "42,123,2026 (within-seed)",
        "numeric_support_if_any": "",
        "required_caveat": "Similarity in attention allocation does NOT prove functional redundancy.",
        "forbidden_extension": "no head-pruning recommendation",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C07 — error-conditioned attention (FT08)
    claims.append({
        "claim_id": "C07",
        "topic": "error_conditioned_attention",
        "claim_text": (
            "Attention metrics were associated with forecast-error magnitude as reported "
            "by FT08 (Spearman ρ, HIGH_ERROR vs LOW_ERROR differences). HIGH/LOW are "
            "Test-relative diagnostic cohorts."
        ),
        "claim_level": C.LEVEL_1,
        "evidence_class": C.EVIDENCE_POST_TEST_DIAGNOSTIC,
        "supporting_table": "FT08",
        "supporting_source": "Phase 56 error-conditioned analysis; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1 with frozen Phase 56 cohorts",
        "seed_scope": "42,123,2026 (per-seed effects)",
        "numeric_support_if_any": "",
        "required_caveat": "Associations are descriptive and non-causal; HIGH/LOW are NOT deployment regimes.",
        "forbidden_extension": "no causal claim; no deployment-regime claim",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C08 — seed stability (FT09)
    claims.append({
        "claim_id": "C08",
        "topic": "seed_stability_attention",
        "claim_text": (
            "Layer-level head-mean attention was more reproducible across the three "
            "predefined final seeds than individual matched-head patterns, as reported "
            "by FT09 (permutation-invariant layer head-mean). Matching ambiguity and "
            "cycle consistency are surfaced when partial."
        ),
        "claim_level": C.LEVEL_2,
        "evidence_class": C.EVIDENCE_POST_TEST_DIAGNOSTIC,
        "supporting_table": "FT09",
        "supporting_source": "Phase 57 seed-stability analysis; Phase 58 reporting synthesis",
        "population": "FINAL_TEST_POP-v1 (three-seed)",
        "seed_scope": "42,123,2026 (permutation-aware matching within each layer)",
        "numeric_support_if_any": "",
        "required_caveat": (
            "Same numeric head indices across seeds are NOT assumed semantically equivalent. "
            "Three seeds provide limited stochastic robustness."
        ),
        "forbidden_extension": "no semantic-head proof; no best-seed selection",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C09 — overall coursework goal
    claims.append({
        "claim_id": "C09",
        "topic": "coursework_goal",
        "claim_text": (
            "The coursework objective was completed by implementing a Transformer Encoder "
            "for one-step-ahead multivariate energy regression, benchmarking it against a "
            "tuned LSTM and persistence baseline, and analyzing temporal attention behavior "
            "at map, head, error-conditioned, and cross-seed levels."
        ),
        "claim_level": C.LEVEL_2,
        "evidence_class": C.EVIDENCE_FROZEN_CONFIG,
        "supporting_table": "FT01",
        "supporting_source": "Phase 45 final lock + Phase 47-57 evidence + Phase 58 reporting",
        "population": "FINAL_TEST_POP-v1 + rolling-origin + attention diagnostic",
        "seed_scope": "42,123,2026",
        "numeric_support_if_any": "",
        "required_caveat": "Completion status means protocol completion, NOT that Transformer beat every baseline.",
        "forbidden_extension": "do not claim universal superiority; do not claim deployment readiness",
        "approved": "YES",
        "status": "ACTIVE",
    })

    # C10 — forbidden LEVEL_4 claim example (REJECTED — kept for audit visibility)
    claims.append({
        "claim_id": "C10",
        "topic": "external_generalization",
        "claim_text": (
            "[REJECTED] The Transformer generalizes to all households and climates and is "
            "ready for deployment."
        ),
        "claim_level": C.LEVEL_4,
        "evidence_class": C.EVIDENCE_HELD_OUT_TEST,
        "supporting_table": "N/A",
        "supporting_source": "N/A",
        "population": "N/A",
        "seed_scope": "N/A",
        "numeric_support_if_any": "",
        "required_caveat": "N/A",
        "forbidden_extension": "explicit forbidden by Phase 59 policy",
        "approved": "NO",
        "status": "REJECTED_LEVEL_4",
    })

    return claims
