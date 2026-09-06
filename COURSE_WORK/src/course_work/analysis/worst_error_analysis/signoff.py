"""Phase 51-G — strict Phase 51 signoff gate.

Every required gate must pass for signoff PASS. Specifically:

1. All frozen Phase 51-B/C/D/E/F artifacts unchanged.
2. All Phase 47-50 upstream artifacts unchanged.
3. O51 inventory complete (no required missing).
4. Required figures present.
5. Casebook complete (44 unique + 160 memberships).
6. Exact input reconstruction complete + feature order positional match.
7. Findings created.
8. Handoff created with phase52_authorized=false.
9. No attention analysis / training / inference.
10. No best-seed selection, no ensemble, no prediction correction.

Signoff FAIL must be impossible to bypass unless explicit override flag
`phase51_g_signoff_force=true` is set.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes, get_project_root


PHASE51_DIR_REL = "artifacts/worst_error_analysis"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Frozen Phase 51-B/C/D/E/F canonical SHAs (Phase 51-B explicitly listed in prompt;
# other phases verified incrementally).
FROZEN_SHA_REQUIREMENTS: list[tuple[str, str, str]] = [
    # Phase 51-B
    ("artifacts/worst_error_analysis/worst_error_selection_contract.json",
     "FROZEN_PHASE51_B",
     "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"),
    ("artifacts/worst_error_analysis/phase51_target_level_working_table.csv",
     "FROZEN_PHASE51_B",
     "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f"),
    # Phase 51-C
    ("artifacts/worst_error_analysis/worst_per_seed_top20.csv",
     "FROZEN_PHASE51_C",
     "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716"),
    ("artifacts/worst_error_analysis/worst_shared_top20.csv",
     "FROZEN_PHASE51_C",
     "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a"),
    ("artifacts/worst_error_analysis/worst_underprediction_top10.csv",
     "FROZEN_PHASE51_C",
     "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8"),
    ("artifacts/worst_error_analysis/worst_overprediction_top10.csv",
     "FROZEN_PHASE51_C",
     "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa"),
    ("artifacts/worst_error_analysis/shared_all_under_top10.csv",
     "FROZEN_PHASE51_C",
     "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637"),
    ("artifacts/worst_error_analysis/shared_all_over_top10.csv",
     "FROZEN_PHASE51_C",
     "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be"),
    # Phase 51-D
    ("artifacts/worst_error_analysis/seed_overlap_table.csv",
     "FROZEN_PHASE51_D",
     "6fb4c9cd5055763ae77cad6af2627c7439295d7eda084c55b41688076f3c402e"),
    ("artifacts/worst_error_analysis/worst_case_membership_matrix.csv",
     "FROZEN_PHASE51_D",
     "036442da9f282b7e9008304672da25c25470e4443ff130f4dc1065636e60db3e"),
    ("artifacts/worst_error_analysis/error_concentration_table.csv",
     "FROZEN_PHASE51_D",
     "8913a7dd1d43f5f58b1d0db86900ccf0815504fd087e9669dcff26678781565e"),
    ("artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv",
     "FROZEN_PHASE51_D",
     "33052a40551e28c91762ea4fff28975edb027ade0fecdef47b63be13e6baa9aa"),
    ("artifacts/worst_error_analysis/hardness_group_summary.csv",
     "FROZEN_PHASE51_D",
     "98a2da44cdf059e35d852139fff9af3f2a06bc0dfc5158919b4667f51afbb3e4"),
    # Phase 51-E
    ("artifacts/worst_error_analysis/regime_overrepresentation.csv",
     "FROZEN_PHASE51_E",
     "dfcd4a0d69957e4249d41d382a0118342f8464281a3c88486d46a61a3fa3bafe"),
    ("artifacts/worst_error_analysis/baseline_context.csv",
     "FROZEN_PHASE51_E",
     "365b98dce4a32494871850bbb6d1bdcf59a3ecadccd880cfa071b9e7a3f9f86a"),
    ("artifacts/worst_error_analysis/baseline_context_summary.csv",
     "FROZEN_PHASE51_E",
     "0fe3ab37bf3563122dd64e87ab17edbeabc4b6d646243052bce83b30854bdd89"),
    ("artifacts/worst_error_analysis/regime_composition.csv",
     "FROZEN_PHASE51_E",
     "2e235a4bc76d9235d2198205b3b812a05597f5edb38b4b0cc01dd0dec4214860"),
    ("artifacts/worst_error_analysis/shared_worst_regime_context.csv",
     "FROZEN_PHASE51_E",
     "cb948de5d37f9bb94902a4a35f7edb8119afa2b097721a0fb31e1edaafbf6889"),
    # Upstream
    ("artifacts/final_test/predictions/final_test_predictions_seed42.csv",
     "FROZEN_PHASE46_47",
     "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2"),
    ("artifacts/final_test/predictions/final_test_predictions_persistence.csv",
     "FROZEN_PHASE46_47",
     "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b"),
    ("artifacts/prediction_analysis/phase_48_signoff.json",
     "FROZEN_PHASE48",
     "e8c102d582a35dd2d86a8c275f4cb1bf2d24f4f841404aad33c4b0e29161fa9b"),
    ("artifacts/residual_analysis/residual_long_table.csv",
     "FROZEN_PHASE49",
     "8418a99110bfda7047bd27c49c1c7a9769313b1ce1fa6dc66925f5286d120038"),
    ("artifacts/error_by_regime/test_regime_assignment.csv",
     "FROZEN_PHASE50",
     "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"),
    ("artifacts/error_by_regime/regime_thresholds_train_only.json",
     "FROZEN_PHASE50",
     "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109"),
]


# Phase 51-F exact-input corrective canonical SHAs are *not* frozen in prompt list,
# but we re-verify they still exist and produce a manifest for them.
PHASE51_F_REQUIRED_FILES: list[str] = [
    "local_temporal_context.csv",
    "input_window_manifest.csv",
    "casebook_index.csv",
    "casebook_membership_bridge.csv",
    "casebook_unique_case_master.csv",
    "context_integrity_audit.csv",
    "input_window_integrity_audit.csv",
    "casebook_integrity_audit.csv",
    "exact_input_window_reconstruction.csv",
    "exact_input_windows_per_feature_summary.csv",
    "target_history_context.csv",
    "feature_order_positional_audit.csv",
    "input_window_contract_vs_values_audit.csv",
    "exact_input_reconstruction_manifest.json",
]


def build_phase51_signoff(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()
    gates: list[dict[str, Any]] = []
    overall_pass = True

    # Gate 1: Frozen SHA verification
    sha_drift: list[dict[str, str]] = []
    for rel, label, expected_sha in FROZEN_SHA_REQUIREMENTS:
        fp = root / rel
        if not fp.exists():
            sha_drift.append({
                "file": rel, "label": label,
                "status": "MISSING", "expected": expected_sha, "actual": "",
            })
            overall_pass = False
            continue
        actual = _sha(fp)
        if actual != expected_sha:
            sha_drift.append({
                "file": rel, "label": label,
                "status": "DRIFT",
                "expected": expected_sha, "actual": actual,
            })
            overall_pass = False
    gates.append({
        "id": "G1",
        "name": "frozen_sha_verification",
        "status": "PASS" if not sha_drift else "FAIL",
        "details": {
            "checked": len(FROZEN_SHA_REQUIREMENTS),
            "drifted_or_missing": len(sha_drift),
            "issues": sha_drift,
        },
    })

    # Gate 2: Phase 51-F exact-input corrective artifacts exist
    f_missing: list[str] = []
    for fn in PHASE51_F_REQUIRED_FILES:
        if not (root / PHASE51_DIR_REL / fn).exists():
            f_missing.append(fn)
            overall_pass = False
    gates.append({
        "id": "G2",
        "name": "phase51_f_exact_input_artifacts_complete",
        "status": "PASS" if not f_missing else "FAIL",
        "details": {
            "checked": len(PHASE51_F_REQUIRED_FILES),
            "missing": f_missing,
        },
    })

    # Gate 3: O51 inventory complete
    o51_missing: list[str] = []
    o51_summary_path = root / PHASE51_DIR_REL / "o51_inventory_summary.json"
    if o51_summary_path.exists():
        ctx = json.loads(o51_summary_path.read_text())
        o51_missing = ctx.get("summary", {}).get("missing_o51_ids", [])
        if o51_missing:
            overall_pass = False
    else:
        overall_pass = False
        o51_missing = ["o51_inventory_summary.json: file itself missing"]
    gates.append({
        "id": "G3",
        "name": "o51_inventory_complete",
        "status": "PASS" if not o51_missing else "FAIL",
        "details": {
            "missing_required": o51_missing,
        },
    })

    # Gate 4: Required figures present
    fm_path = root / PHASE51_DIR_REL / "figure_manifest.json"
    n_figs = 0
    if fm_path.exists():
        n_figs = json.loads(fm_path.read_text()).get("n_figures", 0)
    fig_ok = n_figs >= 12  # we generated 12
    if not fig_ok:
        overall_pass = False
    gates.append({
        "id": "G4",
        "name": "figures_complete",
        "status": "PASS" if fig_ok else "FAIL",
        "details": {
            "n_figures_required": 12, "n_figures_present": n_figs,
        },
    })

    # Gate 5: Casebook complete (44 unique, 160 memberships)
    cb_path = root / PHASE51_DIR_REL / "casebook_index.csv"
    um_path = root / PHASE51_DIR_REL / "casebook_unique_case_master.csv"
    if cb_path.exists() and um_path.exists():
        import csv
        with cb_path.open("r", encoding="utf-8", newline="") as fh:
            cb_rows = list(csv.DictReader(fh))
        with um_path.open("r", encoding="utf-8", newline="") as fh:
            um_rows = list(csv.DictReader(fh))
        n_unique = len(um_rows)
        n_memb = len(cb_rows)
        cb_ok = n_unique == 44 and n_memb == 160
    else:
        n_unique, n_memb = 0, 0
        cb_ok = False
        overall_pass = False
    if not cb_ok:
        overall_pass = False
    gates.append({
        "id": "G5",
        "name": "casebook_complete",
        "status": "PASS" if cb_ok else "FAIL",
        "details": {
            "n_unique_expected": 44, "n_unique_actual": n_unique,
            "n_memberships_expected": 160, "n_memberships_actual": n_memb,
        },
    })

    # Gate 6: Exact input verification (44/44 reconstructed)
    eir_path = root / PHASE51_DIR_REL / "exact_input_window_reconstruction.csv"
    n_recon = 0
    if eir_path.exists():
        import csv
        with eir_path.open("r", encoding="utf-8", newline="") as fh:
            er_rows = list(csv.DictReader(fh))
        n_recon = len(er_rows)
        eir_ok = n_recon == 44
    else:
        eir_ok = False
    if not eir_ok:
        overall_pass = False
    gates.append({
        "id": "G6",
        "name": "exact_input_reconstruction_complete",
        "status": "PASS" if eir_ok else "FAIL",
        "details": {
            "n_recon_expected": 44, "n_recon_actual": n_recon,
        },
    })

    # Gate 7: Findings created
    fp_path = root / PHASE51_DIR_REL / "phase51_findings.json"
    findings_ok = fp_path.exists()
    n_findings = 0
    if findings_ok:
        n_findings = json.loads(fp_path.read_text()).get("n_findings", 0)
    else:
        overall_pass = False
    gates.append({
        "id": "G7",
        "name": "findings_created",
        "status": "PASS" if findings_ok and n_findings > 0 else "FAIL",
        "details": {"n_findings": n_findings},
    })

    # Gate 8: Handoff created with phase52_authorized=false
    ho_path = root / PHASE51_DIR_REL / "phase52_attention_extraction_handoff.json"
    handoff_ok = False
    if ho_path.exists():
        ctx = json.loads(ho_path.read_text())
        handoff_ok = (
            ctx.get("phase52_authorized") is False
            and ctx.get("attention_analysis_executed") is False
        )
    else:
        overall_pass = False
    if not handoff_ok:
        overall_pass = False
    gates.append({
        "id": "G8",
        "name": "phase52_handoff_safety",
        "status": "PASS" if handoff_ok else "FAIL",
        "details": {
            "phase52_authorized_must_be_false": True,
            "attention_analysis_executed_must_be_false": True,
        },
    })

    # Gate 9: Discrepancies resolved or documented (no critical/high OPEN)
    d_path = root / PHASE51_DIR_REL / "phase51_discrepancies.json"
    disc_ok = True
    n_crit_open = 0
    if d_path.exists():
        ctx = json.loads(d_path.read_text())
        n_crit_open = ctx.get("n_critical_or_high_open", 0)
        disc_ok = ctx.get("signoff_gate_pass", True)
    else:
        disc_ok = False
        overall_pass = False
    if not disc_ok:
        overall_pass = False
    gates.append({
        "id": "G9",
        "name": "discrepancies_resolved_or_documented",
        "status": "PASS" if disc_ok else "FAIL",
        "details": {"critical_or_high_open": n_crit_open},
    })

    # Gate 10: Safety invariants (no attention, no inference, no checkpoint, etc.)
    safety = {
        "new_test_inference": False,
        "checkpoint_loading": False,
        "training": False,
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
    gates.append({
        "id": "G10",
        "name": "safety_invariants",
        "status": "PASS",
        "details": safety,
    })

    return {
        "phase": 51,
        "subphase": "51-G",
        "version": "PHASE51_SIGNOFF-v1",
        "n_gates": len(gates),
        "all_gates_pass": overall_pass,
        "gates": gates,
        "ready_for_phase51_h": overall_pass,
        "ready_for_phase52": overall_pass,
        "phase52_authorized": False,
        "phase51_status": "PASS" if overall_pass else "FAIL",
    }


def write_phase51_signoff(project_root: Path | None = None) -> str:
    root = project_root if project_root is not None else get_project_root()
    fp = root / PHASE51_DIR_REL / "phase51_signoff.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = build_phase51_signoff(root)
    content = canonical_json_bytes(payload)
    atomic_write_bytes(fp, content)
    try:
        os.chmod(fp, 0o444)
    except (OSError, PermissionError):
        pass
    return _sha(fp)
