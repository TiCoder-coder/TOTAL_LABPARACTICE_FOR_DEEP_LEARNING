"""Phase 51-G — descriptive scientific findings.

Reads frozen Phase 51-B/C/D/E/F artifacts and produces a structured
FINDINGS document. All findings are descriptive, post-hoc, non-causal.

Potential findings (with actual numbers from frozen artifacts):

- Top20 SAE/SSE concentration
- Cross-seed overlap / triple intersection
- Shared hardness Spearman correlation with seed spread
- Regime composition / overrepresentation
- Persistence case-level comparison
- LSTM NOT_ELIGIBLE_CONFIG_MISMATCH
- Local context availability
- Exact input reconstruction coverage

No causal claims, no retraining prescriptions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


def build_findings(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    findings: list[dict[str, Any]] = []
    artifact_refs: dict[str, str] = {}

    # 1. Error concentration (Top20 SAE/SSE share per seed)
    ec_fp = root / PHASE51_DIR_REL / "error_concentration_table.csv"
    if ec_fp.exists():
        rows = _load_csv(ec_fp)
        artifact_refs["error_concentration_table"] = _sha(ec_fp)
        # Top20 rows (k=20) per seed
        for r in rows:
            if r.get("k") == "20":
                findings.append({
                    "code": "TOP20_SAE_SSE_CONCENTRATION",
                    "severity": "DESCRIPTIVE",
                    "scope": f"seed={r.get('seed')}",
                    "summary": (
                        f"Top 20 worst-error cases account for "
                        f"{_round(float(r['sae_share'])) * 100:.2f}% of SAE and "
                        f"{_round(float(r['sse_share'])) * 100:.2f}% of SSE "
                        f"in Test population for seed {r['seed']}."
                    ),
                    "evidence": {
                        "n_population": int(r["n_population"]),
                        "sae_top_k": _round(float(r["sae_top_k"]), 3),
                        "sse_top_k": _round(float(r["sse_top_k"]), 3),
                        "sae_share": _round(float(r["sae_share"])),
                        "sse_share": _round(float(r["sse_share"])),
                    },
                    "source": "artifacts/worst_error_analysis/error_concentration_table.csv",
                })

    # 2. Cross-seed overlap / Jaccard
    ov_fp = root / PHASE51_DIR_REL / "seed_overlap_table.csv"
    if ov_fp.exists():
        rows = _load_csv(ov_fp)
        artifact_refs["seed_overlap_table"] = _sha(ov_fp)
        # W1 pairwise
        for r in rows:
            if r.get("section") == "W1_PAIRWISE":
                findings.append({
                    "code": "W1_CROSS_SEED_OVERLAP",
                    "severity": "DESCRIPTIVE",
                    "scope": f"set={r.get('group_label', '')}",
                    "summary": (
                        f"W1 pairwise Top20 overlap "
                        f"{r.get('set_a', '')} vs {r.get('set_b', '')}: "
                        f"intersection={r.get('intersection_count', '')}, "
                        f"union={r.get('union_count', '')}, "
                        f"Jaccard={_round(float(r['jaccard']))}."
                    ),
                    "evidence": {
                        "intersection_count": int(r["intersection_count"]),
                        "union_count": int(r["union_count"]),
                        "jaccard": _round(float(r["jaccard"])),
                    },
                    "source": "artifacts/worst_error_analysis/seed_overlap_table.csv",
                })
        # Triple intersection (W1)
        for r in rows:
            if r.get("section") == "W1_3WAY":
                findings.append({
                    "code": "W1_TRIPLE_INTERSECTION",
                    "severity": "DESCRIPTIVE",
                    "scope": "W1 × 3 seeds",
                    "summary": (
                        f"All three Final Transformer seeds share "
                        f"{r.get('intersection_count', '')} Test target(s) "
                        f"in their W1 Top20 worst-error lists."
                    ),
                    "evidence": {
                        "intersection_count": int(r["intersection_count"]),
                        "intersection_target_ids": r.get("extra", "").replace("intersection=", ""),
                    },
                    "source": "artifacts/worst_error_analysis/seed_overlap_table.csv",
                })

    # 3. Hardness vs seed spread (Spearman)
    h_fp = root / PHASE51_DIR_REL / "hardness_vs_seed_disagreement.csv"
    if h_fp.exists():
        rows = _load_csv(h_fp)
        artifact_refs["hardness_vs_seed_disagreement"] = _sha(h_fp)
        if rows:
            try:
                xs = [float(r["mean_abs_error_wh"]) for r in rows]
                ys = [float(r["seed_range_prediction"]) for r in rows]
                # Spearman
                def rankify(vals):
                    indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
                    ranks = [0] * len(vals)
                    for r, (i, _) in enumerate(indexed):
                        ranks[i] = r + 1
                    return ranks
                rx = rankify(xs)
                ry = rankify(ys)
                n = len(rx)
                mx = sum(rx) / n
                my = sum(ry) / n
                cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
                varx = sum((rx[i] - mx) ** 2 for i in range(n))
                vary = sum((ry[i] - my) ** 2 for i in range(n))
                rho = cov / ((varx * vary) ** 0.5) if varx and vary else 0.0
                findings.append({
                    "code": "HARDNESS_VS_SEED_SPREAD_SPEARMAN",
                    "severity": "DESCRIPTIVE",
                    "scope": "All Test population (N=2961)",
                    "summary": (
                        f"Spearman rank correlation between per-target mean "
                        f"absolute error across seeds and Phase48 "
                        f"seed_range_prediction is "
                        f"{_round(rho, 4)}."
                    ),
                    "evidence": {
                        "spearman_rho": _round(rho, 4),
                        "n": n,
                    },
                    "source": "artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv",
                })
            except (KeyError, ValueError, ZeroDivisionError):
                pass

    # 4. Regime composition / overrepresentation (focus on shared worst)
    rr_fp = root / PHASE51_DIR_REL / "regime_overrepresentation.csv"
    if rr_fp.exists():
        rows = _load_csv(rr_fp)
        artifact_refs["regime_overrepresentation"] = _sha(rr_fp)
        # Restrict to seed=42 Top20 for descriptive emphasis.
        rows_w1 = [
            r for r in rows
            if r["seed"] == "42" and r["selection_k"] == "20"
        ]
        by_family: dict[str, list[dict[str, str]]] = {}
        for r in rows_w1:
            by_family.setdefault(r["regime_family"], []).append(r)
        for fam, fam_rows in by_family.items():
            sorted_rows = sorted(
                [r for r in fam_rows if r["enrichment_ratio"] not in ("inf", "nan", "")],
                key=lambda r: float(r["enrichment_ratio"]),
                reverse=True,
            )
            if not sorted_rows:
                continue
            top = sorted_rows[0]
            try:
                ratio = float(top["enrichment_ratio"])
                share_w = float(top["selected_prevalence"])
                share_test = float(top["global_prevalence"])
            except ValueError:
                continue
            findings.append({
                "code": "REGIME_OVERREPRESENTATION",
                "severity": "DESCRIPTIVE",
                "scope": f"family={fam} (W1 seed=42 Top20 vs Test population)",
                "summary": (
                    f"Worst seed=42 Top20 has share {share_w:.4f} in "
                    f"{fam}={top['regime_label']} vs Test population share "
                    f"{share_test:.4f}; enrichment ratio "
                    f"{_round(ratio, 4)}. Phase 50 regime labels reused "
                    f"without modification."
                ),
                "evidence": {
                    "regime_family": fam,
                    "regime_label": top["regime_label"],
                    "selected_prevalence": _round(share_w, 4),
                    "global_prevalence": _round(share_test, 4),
                    "enrichment_ratio": _round(ratio, 4),
                },
                "source": "artifacts/worst_error_analysis/regime_overrepresentation.csv",
            })

    # 5. Persistence case-level comparison
    bc_fp = root / PHASE51_DIR_REL / "baseline_context.csv"
    if bc_fp.exists():
        rows = _load_csv(bc_fp)
        artifact_refs["baseline_context"] = _sha(bc_fp)
        counts: dict[str, int] = {}
        for r in rows:
            cls = r.get("transformer_vs_persistence_outcome", "UNKNOWN")
            counts[cls] = counts.get(cls, 0) + 1
        if counts:
            total = sum(counts.values())
            summary = "; ".join(
                f"{k}={v} ({v/total*100:.2f}%)" for k, v in sorted(counts.items())
            )
            findings.append({
                "code": "PERSISTENCE_CASE_LEVEL_COMPARISON",
                "severity": "DESCRIPTIVE",
                "scope": "Phase 51 selected cases (case-level)",
                "summary": (
                    f"Selection-conditioned Persistence-vs-Transformer "
                    f"comparison across {total} selected cases: {summary}. "
                    f"This is a case-level diagnostic; not a global "
                    f"model comparison."
                ),
                "evidence": counts,
                "source": "artifacts/worst_error_analysis/baseline_context.csv",
            })

    # 6. LSTM context
    lstm_fp = root / PHASE51_DIR_REL / "lstm_eligibility_context.json"
    if lstm_fp.exists():
        ctx = json.loads(lstm_fp.read_text())
        artifact_refs["lstm_eligibility_context"] = _sha(lstm_fp)
        findings.append({
            "code": "LSTM_NOT_APPLICABLE",
            "severity": "DESCRIPTIVE",
            "scope": "global",
            "summary": (
                f"LSTM is NOT_ELIGIBLE_CONFIG_MISMATCH in this Phase 51 "
                f"analysis. Canonical reason: {ctx.get('phase51_f_canonical_reason', ctx.get('reason', ''))}. "
                f"No LSTM inference, ranking, or comparison is performed."
            ),
            "evidence": {
                "status": ctx.get("phase51_f_status", ctx.get("eligibility_status")),
                "reason": ctx.get("phase51_f_canonical_reason", ctx.get("reason")),
            },
            "source": "artifacts/worst_error_analysis/lstm_eligibility_context.json",
        })

    # 7. Local context completeness
    cia_fp = root / PHASE51_DIR_REL / "context_integrity_audit.csv"
    if cia_fp.exists():
        rows = _load_csv(cia_fp)
        artifact_refs["context_integrity_audit"] = _sha(cia_fp)
        centers_present = sum(1 for r in rows if int(r["center_present"]) == 1)
        with_gaps = sum(1 for r in rows if int(r["gap_count"]) > 0)
        total = len(rows)
        findings.append({
            "code": "TEMPORAL_CONTEXT_COVERAGE",
            "severity": "DESCRIPTIVE",
            "scope": "Phase 51 cases",
            "summary": (
                f"Local ±6 temporal context: centers present in "
                f"{centers_present}/{total} cases; "
                f"{with_gaps}/{total} cases have at least one unavailable "
                f"row due to boundary/gap. No padding/interpolation applied."
            ),
            "evidence": {
                "centers_present": centers_present,
                "n_cases": total,
                "with_gaps": with_gaps,
            },
            "source": "artifacts/worst_error_analysis/context_integrity_audit.csv",
        })

    # 8. Exact input reconstruction coverage
    eir_fp = root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv"
    if eir_fp.exists():
        rows = _load_csv(eir_fp)
        artifact_refs["exact_input_window_reconstruction"] = _sha(eir_fp)
        verified = sum(
            1 for r in rows
            if int(r["input_window_values_verified"]) == 1
        )
        target_free = sum(
            1 for r in rows if int(r["target_row_in_window"]) == 0
        )
        findings.append({
            "code": "EXACT_INPUT_RECONSTRUCTION_COVERAGE",
            "severity": "DESCRIPTIVE",
            "scope": "Phase 51 unique selected targets",
            "summary": (
                f"Exact 72x33 input-window values reconstructed for "
                f"{verified}/{len(rows)} unique targets; "
                f"target row excluded in {target_free}/{len(rows)} cases. "
                f"Reconstruction is read-only: FEATURES-v1 + WINDOWPOP-v1 "
                f"+ FINAL_SCALING-v1 transform_only. No model inference."
            ),
            "evidence": {
                "n_unique_targets": len(rows),
                "verified": verified,
                "target_row_excluded": target_free,
            },
            "source": "artifacts/worst_error_analysis/exact_input_window_reconstruction.csv",
        })

    # 9. Casebook statistics
    cb_fp = root / PHASE51_DIR_REL / "casebook_index.csv"
    um_fp = root / PHASE51_DIR_REL / "casebook_unique_case_master.csv"
    if cb_fp.exists() and um_fp.exists():
        cases = _load_csv(cb_fp)
        unique = _load_csv(um_fp)
        family_counts: dict[str, int] = {}
        for c in cases:
            family_counts[c["selection_family"]] = family_counts.get(c["selection_family"], 0) + 1
        findings.append({
            "code": "CASEBOOK_STRUCTURE",
            "severity": "DESCRIPTIVE",
            "scope": "Phase 51 casebook",
            "summary": (
                f"Casebook contains {len(cases)} case memberships "
                f"covering {len(unique)} unique targets. Selection-family "
                f"counts: {family_counts}. Deterministic case IDs from "
                f"(family, seed, rank, target_id)."
            ),
            "evidence": {
                "n_cases": len(cases),
                "n_unique_targets": len(unique),
                "family_counts": family_counts,
            },
            "source": [
                "artifacts/worst_error_analysis/casebook_index.csv",
                "artifacts/worst_error_analysis/casebook_unique_case_master.csv",
            ],
        })

    return {
        "n_findings": len(findings),
        "findings": findings,
        "artifact_refs": artifact_refs,
    }


def write_phase51_findings_json(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    import os
    fp = root / PHASE51_DIR_REL / "phase51_findings.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = build_findings(root)
    envelope = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_FINDINGS-v1",
        "kind": "DESCRIPTIVE_POST_HOC_NON_CAUSAL",
        "ready_for_phase51_signoff": True,
        **payload,
    }
    content = canonical_json_bytes(envelope)
    atomic_write_bytes(fp, content)
    os.chmod(fp, 0o444)
    return {"findings_sha256": _sha(fp), "n_findings": envelope["n_findings"]}
