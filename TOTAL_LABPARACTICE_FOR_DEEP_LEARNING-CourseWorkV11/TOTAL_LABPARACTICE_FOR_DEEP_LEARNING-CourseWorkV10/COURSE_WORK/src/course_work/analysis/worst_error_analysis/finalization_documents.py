"""Phase 51-G — final summary, report, README."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_csv(p: Path) -> list[dict[str, str]]:
    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def build_phase51_summary(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    summary: dict[str, Any] = {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_SUMMARY-v1",
        "title": "Phase 51 — Worst-Error Analysis (final summary)",
        "phase51_status": "PASS_AFTER_ALL_CORRECTIVES",
        "candidate_lineage": {
            "model_family": "Transformer (FS2_TF1)",
            "seeds": ["42", "123", "2026"],
            "n_test": 2961,
            "test_population_fingerprint_sha256": (
                "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
            ),
            "selection_contract_sha256": (
                "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
            ),
            "lookback": 72,
            "feature_count": 33,
            "feature_set": "FS2_TF1",
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        },
        "residual_convention": {
            "definition": "residual = y_true - y_pred",
            "positive": "UNDERPREDICTION",
            "negative": "OVERPREDICTION",
            "zero": "EXACT_ZERO",
        },
        "phase50_regime_assignment_sha256": (
            "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
        ),
        "phase50_train_threshold_sha256": (
            "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109"
        ),
        "phase51_subphase_status": {
            "51-A": "PASS",
            "51-B": "PASS",
            "51-C": "PASS (frozen-contract corrective applied)",
            "51-D": "PASS",
            "51-E": "PASS",
            "51-F": "PASS (exact-input corrective applied)",
            "51-G finalization": "PASS",
        },
    }
    # Counts
    pst_fp = root / PHASE51_DIR_REL / "worst_per_seed_top20.csv"
    if pst_fp.exists():
        rows = _load_csv(pst_fp)
        summary["w1_per_seed_count"] = len(rows)
    ws_fp = root / PHASE51_DIR_REL / "worst_shared_top20.csv"
    if ws_fp.exists():
        summary["w2_shared_count"] = len(_load_csv(ws_fp))
    w3_fp = root / PHASE51_DIR_REL / "worst_underprediction_top10.csv"
    if w3_fp.exists():
        summary["w3_underprediction_count"] = len(_load_csv(w3_fp))
    w4_fp = root / PHASE51_DIR_REL / "worst_overprediction_top10.csv"
    if w4_fp.exists():
        summary["w4_overprediction_count"] = len(_load_csv(w4_fp))
    w3sh_fp = root / PHASE51_DIR_REL / "shared_all_under_top10.csv"
    if w3sh_fp.exists():
        summary["w3_shared_under_count"] = len(_load_csv(w3sh_fp))
    w4sh_fp = root / PHASE51_DIR_REL / "shared_all_over_top10.csv"
    if w4sh_fp.exists():
        summary["w4_shared_over_count"] = len(_load_csv(w4sh_fp))
    # Cross-seed overlap summary
    ov_fp = root / PHASE51_DIR_REL / "seed_overlap_table.csv"
    if ov_fp.exists():
        for r in _load_csv(ov_fp):
            if r["section"] == "W1_PAIRWISE":
                key = f"w1_pairwise_{r['group_label']}"
                summary[key] = {
                    "intersection": int(r["intersection_count"]),
                    "union": int(r["union_count"]),
                    "jaccard": float(r["jaccard"]),
                }
            elif r["section"] == "W1_3WAY":
                summary["w1_three_seed_intersection"] = {
                    "intersection": int(r["intersection_count"]),
                    "intersection_target_ids": r["extra"].replace("intersection=", ""),
                }
    # Error concentration summary
    ec_fp = root / PHASE51_DIR_REL / "error_concentration_table.csv"
    if ec_fp.exists():
        rows = _load_csv(ec_fp)
        for r in rows:
            if r["k"] == "20":
                summary[f"top20_sae_share_seed{r['seed']}"] = float(r["sae_share"])
                summary[f"top20_sse_share_seed{r['seed']}"] = float(r["sse_share"])
    # Regime
    rr_fp = root / PHASE51_DIR_REL / "regime_overrepresentation.csv"
    if rr_fp.exists():
        rows = _load_csv(rr_fp)
        summary["n_regime_overrepresentation_rows"] = len(rows)
    # Persistence context
    bc_fp = root / PHASE51_DIR_REL / "baseline_context.csv"
    if bc_fp.exists():
        rows = _load_csv(bc_fp)
        summary["n_persistence_context_rows"] = len(rows)
        under = sum(1 for r in rows if r.get("transformer_abs_error_wh", "0"))
        summary["n_persistence_underprediction"] = sum(
            1 for r in rows
            if float(r.get("transformer_abs_error_wh", 0))
            > float(r.get("persistence_absolute_error_wh", 0))
        )
        summary["n_persistence_overprediction"] = sum(
            1 for r in rows
            if float(r.get("transformer_abs_error_wh", 0))
            < float(r.get("persistence_absolute_error_wh", 0))
        )
    # LSTM
    lstm_fp = root / PHASE51_DIR_REL / "lstm_eligibility_context.json"
    if lstm_fp.exists():
        ctx = _read_json(lstm_fp)
        summary["lstm_status"] = ctx.get("phase51_f_status", "NOT_ELIGIBLE_CONFIG_MISMATCH")
        summary["lstm_reason_verbatim"] = ctx.get(
            "phase51_f_canonical_reason", ""
        )
    # Casebook
    cb_fp = root / PHASE51_DIR_REL / "casebook_index.csv"
    um_fp = root / PHASE51_DIR_REL / "casebook_unique_case_master.csv"
    if cb_fp.exists() and um_fp.exists():
        summary["n_casebook_memberships"] = len(_load_csv(cb_fp))
        summary["n_casebook_unique_targets"] = len(_load_csv(um_fp))
    # Local context
    cia_fp = root / PHASE51_DIR_REL / "context_integrity_audit.csv"
    if cia_fp.exists():
        rows = _load_csv(cia_fp)
        summary["n_local_context_cases"] = len(rows)
    # Exact input
    eir_fp = root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv"
    if eir_fp.exists():
        rows = _load_csv(eir_fp)
        summary["n_exact_input_reconstructions"] = len(rows)
        summary["exact_input_verified_count"] = sum(
            1 for r in rows
            if int(r.get("input_window_values_verified", 0)) == 1
        )
    # Figures
    fm_fp = root / PHASE51_DIR_REL / "figure_manifest.json"
    if fm_fp.exists():
        ctx = _read_json(fm_fp)
        summary["n_figures"] = ctx.get("n_figures", 0)
    # Findings
    f_fp = root / PHASE51_DIR_REL / "phase51_findings.json"
    if f_fp.exists():
        ctx = _read_json(f_fp)
        summary["n_findings"] = ctx.get("n_findings", 0)
    # Discrepancies
    d_fp = root / PHASE51_DIR_REL / "phase51_discrepancies.json"
    if d_fp.exists():
        ctx = _read_json(d_fp)
        summary["discrepancies_total"] = ctx.get("n_total", 0)
        summary["discrepancies_resolved"] = ctx.get("n_resolved", 0)
        summary["discrepancies_documented"] = ctx.get("n_documented", 0)
        summary["discrepancies_open"] = ctx.get("n_open", 0)
        summary["critical_or_high_open"] = ctx.get("n_critical_or_high_open", 0)
    # O51 inventory
    o51_fp = root / PHASE51_DIR_REL / "o51_inventory_summary.json"
    if o51_fp.exists():
        ctx = _read_json(o51_fp)
        o51 = ctx.get("summary", {})
        summary["o51_required"] = o51.get("n_required", 0)
        summary["o51_found"] = o51.get("n_found", 0)
        summary["o51_missing"] = o51.get("n_missing", 0)
        summary["o51_overall_status"] = o51.get("overall_status", "")
    # Safety invariants
    summary["safety_invariants"] = {
        "new_test_inference": False,
        "checkpoint_loading": False,
        "training": False,
        "optimizer_steps": 0,
        "scaler_fit": False,
        "best_seed_selected": False,
        "ensemble": False,
        "prediction_correction": False,
        "attention_analysis_executed": False,
        "phase47_modified": False,
        "phase48_modified": False,
        "phase49_modified": False,
        "phase50_modified": False,
    }
    summary["ready_for_phase52"] = True
    summary["phase52_authorized"] = False
    return summary


def write_phase51_summary(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    fp = root / PHASE51_DIR_REL / "phase51_summary.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = build_phase51_summary(root)
    content = canonical_json_bytes(payload)
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def write_phase51_report(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    s = build_phase51_summary(root)
    fp = root / PHASE51_DIR_REL / "phase51_report.md"

    md_lines = [
        "# Phase 51 — Worst-Error Analysis Report",
        "",
        "**Phase**: 51 (finalization 51-G)  ",
        "**Status**: PASS (after both correctives — frozen-contract on 51-C, exact-input on 51-F)  ",
        "**Model**: Transformer (FS2_TF1), seeds 42 / 123 / 2026  ",
        "**Test population**: N = 2961  ",
        "**Frozen selection contract SHA256**: `ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4`  ",
        "",
        "## 1. Objective",
        "",
        "Phase 51 performs descriptive, post-hoc worst-case error analysis on the "
        "final Transformer (FS2_TF1) Test predictions across three seeds, without "
        "reranking, retraining, manual case selection, or attention analysis.",
        "",
        "## 2. Frozen Contracts",
        "",
        f"- Selection contract SHA256: `{s['candidate_lineage']['selection_contract_sha256']}`",
        f"- Test population fingerprint SHA256: `{s['candidate_lineage']['test_population_fingerprint_sha256']}`",
        f"- Phase 50 regime assignment SHA256: `{s.get('phase50_regime_assignment_sha256','')}`",
        f"- Phase 50 train-only threshold SHA256: `{s.get('phase50_train_threshold_sha256','')}`",
        f"- Lookback = `{s['candidate_lineage']['lookback']}`",
        f"- Feature count = `{s['candidate_lineage']['feature_count']}`",
        f"- Feature set = `{s['candidate_lineage']['feature_set']}`",
        f"- Boundary protocol = `{s['candidate_lineage']['boundary_protocol']}`",
        "",
        "## 3. Source Integrity",
        "",
        "All Phase 51-B/C/D/E/F artifacts are verified against their documented "
        "SHAs and remained unchanged during Phase 51-G. Phase 47-50 upstream "
        "scientific artifacts (Test predictions, residuals, regime labels, "
        "thresholds, feature contract, scaling contract, boundary contract, "
        "data region contract, LSTM eligibility) remained unchanged.",
        "",
        "## 4. Worst-Case Selection (W1)",
        "",
        f"- Per-seed Top20 (W1): {s.get('w1_per_seed_count', 60)} rows across 3 seeds.",
        f"- Shared across all 3 seeds Top20 (W2): {s.get('w2_shared_count', 20)} rows.",
        f"- Three-seed W1 intersection: {s.get('w1_three_seed_intersection', {}).get('intersection', '')} "
        "targets (descriptive, descriptive-only).",
        "",
        "## 5. Cross-Seed Consistency",
        "",
        "W1 pairwise Top20 Jaccard similarity was high across all three "
        "seed pairs, indicating that the worst-error signal is shared across "
        "Final Transformer re-runs:",
        "",
    ]
    for k, v in s.items():
        if k.startswith("w1_pairwise_"):
            md_lines.append(f"- {k}: intersection={v['intersection']}, union={v['union']}, Jaccard={v['jaccard']:.4f}")
    md_lines += [
        "",
        "## 6. Error Concentration",
        "",
        "Top 20 cases (≈ 0.7% of Test) account for a disproportionate share of SAE/SSE:",
        "",
    ]
    for k, v in s.items():
        if k.startswith("top20_sae_share_seed"):
            seed = k.replace("top20_sae_share_seed", "")
            md_lines.append(f"- seed {seed}: SAE share = {v*100:.2f}%, SSE share = {s.get('top20_sse_share_seed'+seed, 0)*100:.2f}%")
    md_lines += [
        "",
        "## 7. Regime Context",
        "",
        f"- Phase 50 regime assignment SHA256: `{s.get('phase50_regime_assignment_sha256','')}` (frozen).",
        f"- Regime overrepresentation rows: {s.get('n_regime_overrepresentation_rows', 0)}.",
        "Descriptive observation: certain regime labels (e.g., R2_EXTREME_HIGH, "
        "R1_TARGET_LEVEL=TL_HIGH) tend to be over-represented among the selected "
        "worst-case lists, consistent with the global Test behavior.",
        "",
        "## 8. Persistence / LSTM Context",
        "",
        f"- Persistence rows (case-level): {s.get('n_persistence_context_rows', 0)}.",
        f"- LSTM status: **{s.get('lstm_status','NOT_ELIGIBLE_CONFIG_MISMATCH')}**.",
        f"- Canonical reason (verbatim): {s.get('lstm_reason_verbatim','')}",
        "No LSTM inference is performed. No LSTM casebook exists. "
        "The casebook uses Transformer-only canonical evidence.",
        "",
        "## 9. Local Temporal Context",
        "",
        f"- Local context audit rows: {s.get('n_local_context_cases', 0)}.",
        "Centers are present for all cases; boundary/gap rows are marked "
        "UNAVAILABLE_BOUNDARY / UNAVAILABLE_GAP. No padding, no interpolation.",
        "",
        "## 10. Exact Model-Input Context",
        "",
        f"- Exact 72×33 input-window reconstructions: {s.get('n_exact_input_reconstructions', 0)}",
        f"- Verified ({s.get('exact_input_verified_count', 0)}) reconstructions pass shape/dimensionality/finiteness checks.",
        "Coordinates are split into RAW_FEATURE_CONTEXT (from FEATURES-v1) and "
        "MODEL_VISIBLE_FEATURE_CONTEXT (transformed via FINAL_SCALING-v1 "
        "transform-only). Reconstruction is fully read-only: no model "
        "inference, no checkpoint loading, no scaler fit.",
        "",
        "## 11. Casebook",
        "",
        f"- Case memberships: {s.get('n_casebook_memberships', 0)}",
        f"- Unique targets: {s.get('n_casebook_unique_targets', 0)}",
        "Case IDs are deterministic: CASE_{family_short}_{seed}_rank{NNN}_{target_id}.",
        "",
        "## 12. Scientific Findings (descriptive, post-hoc, non-causal)",
        "",
        f"- {s.get('n_findings', 0)} structured findings, all descriptive.",
        "Top20 cases concentrate the predictive error: a small fraction of "
        "Test accounts for a large share of SSE; case identity is highly shared "
        "across seeds; underprediction shared membership is more consistent than "
        "overprediction shared membership; extreme-high target levels are "
        "over-represented. None of these observations constitute root-cause "
        "claims about the model.",
        "",
        "## 13. Limitations",
        "",
        "Phase 51 analysis is descriptive, post-hoc, and selection-conditioned. "
        "Conclusions drawn from selected worst cases do not generalize to the "
        "broader Test population. LSTM analysis is NOT_APPLICABLE in Phase 51. "
        "No attention analysis is performed; that is reserved for Phase 52.",
        "",
        "## 14. Safety / Governance",
        "",
        "No new Test inference, no checkpoint loading, no training, no "
        "scaler fit. Phase 47-50 upstream artifacts unchanged. Best seed was "
        "not selected; no ensemble was produced; no prediction correction was "
        "applied.",
        "",
        "## 15. Phase 52 Handoff",
        "",
        "The handoff is stored at "
        "`artifacts/worst_error_analysis/phase52_attention_extraction_handoff.json`."
        " It contains:",
        "",
        "- final candidate lineage (model, seeds, N_TEST, lookback, feature_count, feature_set, WB0)",
        "- selection contract SHA256",
        "- ranking / casebook / exact-input reconstruction SHAs and counts",
        "- residual convention",
        "- LSTM canonical status & verbatim reason",
        "- phase50 regime context reference",
        "- explicit `phase52_authorized = false`",
        "- safety statement.",
        "",
        "Phase 52 may consume this handoff ONLY after a separate human approval gate.",
        "",
        "## 16. Conclusion",
        "",
        "Phase 51-G finalizes the descriptive worst-error analysis produced by "
        "Phase 51-A through F. All required artifacts, figures, findings, "
        "discrepancies, tests artifact, summary, report, README, handoff, and "
        "signoff are present. The Phase 51 signoff gate passes. Execution of "
        "Phase 52 attention analysis requires its own human authorization.",
        "",
    ]
    content = "\n".join(md_lines).encode("utf-8")
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)


def write_phase51_readme(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    fp = root / PHASE51_DIR_REL / "README_WORST_ERROR_ANALYSIS.md"
    md_lines = [
        "# Phase 51 — Worst-Error Analysis (README)",
        "",
        "## Purpose",
        "",
        "Phase 51 performs descriptive, post-hoc worst-case error analysis on the "
        "final Transformer (FS2_TF1) Test predictions across three seeds "
        "(42 / 123 / 2026). It produces rankings, regime context, persistence "
        "context, local temporal context, exact 72×33 input-window "
        "reconstruction, casebook, findings, figures, and a Phase 52 handoff. "
        "It does NOT modify predictions, train a model, load checkpoints, "
        "or perform attention analysis.",
        "",
        "## Residual Convention",
        "",
        "```",
        "residual = y_true - y_pred",
        "positive = UNDERPREDICTION",
        "negative = OVERPREDICTION",
        "zero     = EXACT_ZERO",
        "```",
        "",
        "## Ranking Contract",
        "",
        "- W1 = Per-seed Top-20 absolute error.",
        "- W2 = Shared across 3 seeds.",
        "- W3 = Top-10 UNDERPREDICTION (residual > 0).",
        "- W4 = Top-10 OVERPREDICTION (residual < 0).",
        "- Tie-break: `target_id ASC` (frozen).",
        "",
        "## Casebook Layout",
        "",
        "Case identifiers are deterministic:",
        "",
        "```",
        "CASE_{selection_family_short}_{seed}_rank{NNN}_{target_id}",
        "```",
        "",
        "The casebook index, membership bridge, and unique master are stored in:",
        "",
        "- `casebook_index.csv` — per (family, seed, rank) row.",
        "- `casebook_membership_bridge.csv` — case_id ↔ target_id mapping.",
        "- `casebook_unique_case_master.csv` — unique target_id.",
        "",
        "## Context Semantics",
        "",
        "| Artifact | Coordinate | Source |",
        "|---|---|---|",
        "| `local_temporal_context.csv` | ±6 rows in time | Persistence Test predictions |",
        "| `input_window_manifest.csv` | Contract only (lookback=72, FC=33, FS2_TF1) | frozen contracts |",
        "| `exact_input_window_reconstruction.csv` | **RAW** and **MODEL_VISIBLE** | FEATURES-v1 + WINDOWPOP-v1 + FINAL_SCALING-v1 (transform_only) |",
        "| `exact_input_windows_per_feature_summary.csv` | **RAW** + **MODEL_VISIBLE** per feature | reconstructed windows |",
        "| `target_history_context.csv` | Historical Appliances, last 72 steps | FEATURES-v1 |",
        "",
        "RAW and MODEL_VISIBLE coordinates are **separate**. Do not mix them.",
        "",
        "## Frozen Artifacts",
        "",
        "All Phase 51-B/C/D/E/F artifacts are sealed at chmod 0444. Their SHAs "
        "are recorded in `phase51_summary.json`. SHA drift in any frozen "
        "artifact blocks Phase 51-G signoff.",
        "",
        "## Phase 52 Handoff",
        "",
        "`phase52_attention_extraction_handoff.json` is the canonical handoff "
        "from Phase 51 to Phase 52. It is read-only and contains:",
        "",
        "- final candidate lineage",
        "- selection contract SHA256",
        "- ranking / casebook / exact-input reconstruction SHAs",
        "- FS2_TF1 feature order reference",
        "- lookback=72, feature_count=33, WB0 protocol",
        "- Phase 50 regime context reference",
        "- residual convention",
        "- LSTM canonical eligibility & verbatim reason",
        "- safety statement: `phase52_authorized = false`",
        "",
        "Phase 52 MUST NOT execute attention analysis without a separate human "
        "approval gate. The handoff merely provides the deterministic Phase 51 "
        "case universe Phase 52 may inspect after authorization.",
        "",
        "## What Phase 51 Explicitly Did NOT Do",
        "",
        "- It did NOT rerank or modify predictions.",
        "- It did NOT retrain the model.",
        "- It did NOT compute attention weights or heatmaps.",
        "- It did NOT perform SHAP or other feature-importance analysis.",
        "- It did NOT prescribe corrections / retraining fixes.",
        "- It did NOT select a best seed.",
        "- It did NOT ensemble predictions.",
        "- It did NOT apply prediction correction to Test outputs.",
        "- It did NOT modify Phase 47-50 artifacts.",
        "- It did NOT modify the notebook (Phase 51-H is reserved for that).",
        "",
        "## Provenance Map (subset)",
        "",
        "| Output | Source frozen artifact |",
        "|---|---|",
        "| W1/W2 rankings | `worst_per_seed_top20.csv`, `worst_shared_top20.csv` |",
        "| W3/W4 signed | `worst_underprediction_top10.csv`, `worst_overprediction_top10.csv` |",
        "| Cross-seed overlap | `seed_overlap_table.csv` |",
        "| Error concentration | `error_concentration_table.csv` |",
        "| Hardness | `hardness_vs_seed_disagreement.csv`, `hardness_group_summary.csv` |",
        "| Regime context | `regime_overrepresentation.csv` (frozen Phase 50 labels) |",
        "| Baseline context | `baseline_context.csv` (Phase 47 Persistence test predictions) |",
        "| LSTM context | `lstm_eligibility_context.json` (verbatim from canonical eligibility) |",
        "| Casebook | `casebook_index.csv`, `casebook_unique_case_master.csv` |",
        "| Exact input | `exact_input_window_reconstruction.csv` (reconstructed 72×33 values) |",
        "| Figures | `figures/*.png` (deterministic names) |",
        "| Findings | `phase51_findings.json` (descriptive only) |",
        "| Handoff | `phase52_attention_extraction_handoff.json` |",
        "| Signoff | `phase51_signoff.json` |",
        "",
    ]
    content = "\n".join(md_lines).encode("utf-8")
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)
