"""Phase 51-F orchestrator — casebook + temporal context + input-window
manifest + feature-order audit + integrity audits.

Pre-execution gates: every frozen Phase 51-B/C/D/E SHA must match
exactly. If any SHA unexpectedly differs, STOP and refuse to run.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    casebook,
    f_writers,
    feature_context,
    input_window_context,
    lstm_context,
    temporal_context,
    context_audit,
)
from ..utils.artifacts import get_project_root


# All canonical Phase 51-B/C/D/E SHAs that must remain unchanged
CANONICAL_SHAS: dict[str, str] = {
    "artifacts/worst_error_analysis/worst_error_selection_contract.json":
        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4",
    "artifacts/worst_error_analysis/phase51_target_level_working_table.csv":
        "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f",
    "artifacts/worst_error_analysis/worst_per_seed_top20.csv":
        "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716",
    "artifacts/worst_error_analysis/worst_shared_top20.csv":
        "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a",
    "artifacts/worst_error_analysis/worst_underprediction_top10.csv":
        "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8",
    "artifacts/worst_error_analysis/worst_overprediction_top10.csv":
        "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa",
    "artifacts/worst_error_analysis/shared_all_under_top10.csv":
        "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637",
    "artifacts/worst_error_analysis/shared_all_over_top10.csv":
        "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be",
    "artifacts/worst_error_analysis/seed_overlap_table.csv":
        "6fb4c9cd5055763ae77cad6af2627c7439295d7eda084c55b41688076f3c402e",
    "artifacts/worst_error_analysis/worst_case_membership_matrix.csv":
        "036442da9f282b7e9008304672da25c25470e4443ff130f4dc1065636e60db3e",
    "artifacts/worst_error_analysis/error_concentration_table.csv":
        "8913a7dd1d43f5f58b1d0db86900ccf0815504fd087e9669dcff26678781565e",
    "artifacts/worst_error_analysis/hardness_vs_seed_disagreement.csv":
        "33052a40551e28c91762ea4fff28975edb027ade0fecdef47b63be13e6baa9aa",
    "artifacts/worst_error_analysis/hardness_group_summary.csv":
        "98a2da44cdf059e35d852139fff9af3f2a06bc0dfc5158919b4667f51afbb3e4",
    "artifacts/worst_error_analysis/regime_overrepresentation.csv":
        "dfcd4a0d69957e4249d41d382a0118342f8464281a3c88486d46a61a3fa3bafe",
    "artifacts/worst_error_analysis/baseline_context.csv":
        "365b98dce4a32494871850bbb6d1bdcf59a3ecadccd880cfa071b9e7a3f9f86a",
    "artifacts/worst_error_analysis/baseline_context_summary.csv":
        "0fe3ab37bf3563122dd64e87ab17edbeabc4b6d646243052bce83b30854bdd89",
    "artifacts/worst_error_analysis/regime_composition.csv":
        "2e235a4bc76d9235d2198205b3b812a05597f5edb38b4b0cc01dd0dec4214860",
    "artifacts/worst_error_analysis/shared_worst_regime_context.csv":
        "cb948de5d37f9bb94902a4a35f7edb8119afa2b097721a0fb31e1edaafbf6889",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_phase51_f(
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root if project_root is not None else get_project_root()

    # ── Gate: verify every frozen Phase 51-B/C/D/E SHA ──────────────────────
    diffs: list[str] = []
    for rel, expected in CANONICAL_SHAS.items():
        actual = _sha256(root / rel)
        if actual != expected:
            diffs.append(f"{rel}: expected {expected}, actual {actual}")
    if diffs:
        raise RuntimeError(
            "Frozen Phase 51-B/C/D/E SHA drift detected. Refusing to run.\n"
            + "\n".join(diffs)
        )

    # ── Canonical context constants ─────────────────────────────────────────
    fs_audit = feature_context.verify_canonical_feature_order(root)

    # ── Load frozen cases ───────────────────────────────────────────────────
    cases = casebook.load_frozen_cases(root)
    selected_target_ids = sorted({c["target_id"] for c in cases})

    # ── Temporal context ────────────────────────────────────────────────────
    ctx_rows = temporal_context.build_local_temporal_context(root, selected_target_ids)
    ctx_fieldnames = list(ctx_rows[0].keys()) if ctx_rows else [
        "center_target_id", "relative_step", "context_target_id",
        "context_target_timestamp", "context_y_true_wh",
        "context_seed42_y_pred_wh", "context_seed42_residual_wh",
        "context_seed42_abs_error_wh", "context_seed123_y_pred_wh",
        "context_seed2026_y_pred_wh",
        "context_persistence_y_pred_wh", "context_persistence_residual_wh",
        "context_persistence_abs_error_wh", "is_center", "continuity_valid",
        "availability_status",
    ]
    ctx_sha = f_writers.write_csv(
        ctx_rows, ctx_fieldnames, "local_temporal_context.csv", root
    )

    # ── Input window manifest ───────────────────────────────────────────────
    win_rows = input_window_context.build_input_window_manifest(root, selected_target_ids)
    win_sha = f_writers.write_csv(
        win_rows,
        [
            "target_id", "target_timestamp",
            "lookback", "horizon", "cadence_minutes",
            "feature_count", "feature_set", "boundary_protocol",
            "window_start_timestamp", "window_end_timestamp",
            "window_target_index", "is_model_input",
            "future_context_used_as_model_input",
        ],
        "input_window_manifest.csv", root,
    )

    # ── Feature-order audit ─────────────────────────────────────────────────
    feat_rows = feature_context.build_feature_order_audit(root)
    feat_sha = f_writers.write_csv(
        feat_rows,
        ["position", "feature_name", "expected_name", "match", "coordinate_class"],
        "feature_order_audit.csv", root,
    )
    feat_summary = feature_context.build_per_target_feature_summary(
        root, selected_target_ids
    )
    feat_summary_sha = f_writers.write_csv(
        feat_summary,
        [
            "target_id", "feature_name", "feature_position",
            "summary_kind", "raw_value",
            "seed42_y_pred_wh", "seed123_y_pred_wh", "seed2026_y_pred_wh",
            "mean_abs_error_wh", "min_abs_error_wh", "max_abs_error_wh",
            "feature_status",
        ],
        "model_visible_feature_summary.csv", root,
    )

    # ── Casebook ────────────────────────────────────────────────────────────
    casebook_rows = cases
    casebook_fieldnames = list(cases[0].keys()) if cases else [
        "case_id", "selection_family", "seed", "rank", "target_id",
        "target_timestamp", "y_true_wh", "y_pred_wh", "residual_wh",
        "absolute_error_wh",
        "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
        "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
        "seed42_y_pred_wh", "seed42_residual_wh", "seed42_abs_error_wh",
        "seed123_y_pred_wh", "seed123_residual_wh", "seed123_abs_error_wh",
        "seed2026_y_pred_wh", "seed2026_residual_wh", "seed2026_abs_error_wh",
        "mean_abs_error_wh", "cross_seed_consensus_class",
        "context_radius", "max_context_rows",
        "input_window_lookback", "input_feature_count", "input_feature_set",
        "input_window_reference_id",
        "casebook_role", "casebook_ranking_source",
    ]
    casebook_sha = f_writers.write_csv(
        casebook_rows, casebook_fieldnames, "casebook_index.csv", root
    )

    membership_rows = casebook.build_casebook_membership_bridge(cases)
    membership_sha = f_writers.write_csv(
        membership_rows,
        ["case_id", "target_id", "selection_family", "seed", "rank", "casebook_role"],
        "casebook_membership_bridge.csv", root,
    )

    unique_master_rows = casebook.build_unique_case_master(cases)
    unique_master_sha = f_writers.write_csv(
        unique_master_rows,
        [
            "target_id", "n_case_memberships", "selection_families",
            "ranks_by_family", "primary_case_id", "y_true_wh",
            "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
            "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
            "cross_seed_consensus_class",
        ],
        "casebook_unique_case_master.csv", root,
    )

    # ── Integrity audits ────────────────────────────────────────────────────
    ctx_audit_rows = context_audit.audit_temporal_context_integrity(cases, ctx_rows)
    ctx_audit_sha = f_writers.write_csv(
        ctx_audit_rows,
        [
            "case_id", "target_id", "selection_family",
            "context_radius", "max_context_rows",
            "center_present", "left_available_count", "right_available_count",
            "gap_count",
            "padding_used", "interpolation_used", "forward_fill_used",
            "incomplete_context_retained", "case_replaced_due_to_incompleteness",
        ],
        "context_integrity_audit.csv", root,
    )

    win_audit_rows = context_audit.audit_input_window_integrity(cases, win_rows)
    win_audit_sha = f_writers.write_csv(
        win_audit_rows,
        [
            "case_id", "target_id", "input_window_reference_id",
            "lookback", "feature_count", "feature_set",
            "feature_order_match", "target_window_mapping_exact",
            "future_context_used_as_model_input", "boundary_protocol",
        ],
        "input_window_integrity_audit.csv", root,
    )

    cb_audit_rows = context_audit.audit_casebook_integrity(cases)
    cb_audit_sha = f_writers.write_csv(
        cb_audit_rows,
        [
            "case_id", "target_id", "selection_family", "seed", "rank",
            "ranking_source", "frozen_ranking_sha256",
            "manually_added", "rank_preserved", "selection_family_preserved",
            "seed_preserved", "y_true_match", "y_pred_match",
            "abs_error_match", "phase50_regime_labels_match",
            "case_replaced", "scientific_value_source",
            "casebook_ranking_changed",
        ],
        "casebook_integrity_audit.csv", root,
    )

    # ── LSTM canonical context ──────────────────────────────────────────────
    lstm_ctx = lstm_context.build_lstm_eligibility_context(root)
    # Re-read canonical reason verbatim from final_test_lstm_eligibility.json
    canon_lstm = json.loads(
        (root / "artifacts/final_test/final_test_lstm_eligibility.json").read_text()
    )
    lstm_ctx["phase51_f_canonical_reason"] = canon_lstm["reason"]
    lstm_ctx["phase51_f_status"] = canon_lstm["eligibility_status"]
    lstm_ctx["phase51_f_no_inference"] = True
    lstm_sha = f_writers.write_json(
        lstm_ctx, "lstm_eligibility_context.json", root
    )

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "phase": 51,
        "subphase": "51-F",
        "version": "WORST_ERROR_ANALYSIS_CASEBOOK-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_shas": {
            "selection_contract_sha256": CANONICAL_SHAS[
                "artifacts/worst_error_analysis/worst_error_selection_contract.json"
            ],
            "working_table_sha256": CANONICAL_SHAS[
                "artifacts/worst_error_analysis/phase51_target_level_working_table.csv"
            ],
            "phase51_d_canonical_shas": {
                k.split("/")[-1]: v for k, v in CANONICAL_SHAS.items()
                if k.endswith(("seed_overlap_table.csv",
                               "worst_case_membership_matrix.csv",
                               "error_concentration_table.csv",
                               "hardness_vs_seed_disagreement.csv",
                               "hardness_group_summary.csv"))
            },
            "phase51_e_canonical_shas": {
                k.split("/")[-1]: v for k, v in CANONICAL_SHAS.items()
                if k.endswith(("regime_overrepresentation.csv",
                               "baseline_context.csv",
                               "baseline_context_summary.csv",
                               "regime_composition.csv",
                               "shared_worst_regime_context.csv"))
            },
            "persistence_sha256": "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b",
            "feature_registry_canonical_fingerprint": fs_audit["fingerprint"],
        },
        "context_constants": {
            "context_radius": temporal_context.CONTEXT_RADIUS,
            "cadence_minutes": temporal_context.CADENCE_MINUTES,
            "test_population_first_id": temporal_context.TEST_POP_FIRST_ID,
            "test_population_last_id": temporal_context.TEST_POP_LAST_ID,
            "test_population_n": temporal_context.TEST_POP_N,
            "lookback": input_window_context.LOOKBACK,
            "feature_count": input_window_context.FEATURE_COUNT,
            "feature_set": input_window_context.FEATURE_SET,
            "boundary_protocol": input_window_context.BOUNDARY_PROTOCOL,
        },
        "deliverables": {
            "local_temporal_context": {
                "path": "artifacts/worst_error_analysis/local_temporal_context.csv",
                "sha256": ctx_sha,
            },
            "input_window_manifest": {
                "path": "artifacts/worst_error_analysis/input_window_manifest.csv",
                "sha256": win_sha,
            },
            "feature_order_audit": {
                "path": "artifacts/worst_error_analysis/feature_order_audit.csv",
                "sha256": feat_sha,
            },
            "model_visible_feature_summary": {
                "path": "artifacts/worst_error_analysis/model_visible_feature_summary.csv",
                "sha256": feat_summary_sha,
            },
            "casebook_index": {
                "path": "artifacts/worst_error_analysis/casebook_index.csv",
                "sha256": casebook_sha,
            },
            "casebook_membership_bridge": {
                "path": "artifacts/worst_error_analysis/casebook_membership_bridge.csv",
                "sha256": membership_sha,
            },
            "casebook_unique_case_master": {
                "path": "artifacts/worst_error_analysis/casebook_unique_case_master.csv",
                "sha256": unique_master_sha,
            },
            "context_integrity_audit": {
                "path": "artifacts/worst_error_analysis/context_integrity_audit.csv",
                "sha256": ctx_audit_sha,
            },
            "input_window_integrity_audit": {
                "path": "artifacts/worst_error_analysis/input_window_integrity_audit.csv",
                "sha256": win_audit_sha,
            },
            "casebook_integrity_audit": {
                "path": "artifacts/worst_error_analysis/casebook_integrity_audit.csv",
                "sha256": cb_audit_sha,
            },
            "lstm_eligibility_context": {
                "path": "artifacts/worst_error_analysis/lstm_eligibility_context.json",
                "sha256": lstm_sha,
            },
        },
        "forbidden_actions_status": {
            "temporal_context_executed": True,
            "input_context_executed": True,
            "casebook_created": True,
            "individual_case_inspection_executed": True,
            "manual_case_selection": False,
            "ranking_changed": False,
            "feature_attribution": False,
            "shap": False,
            "attention_analysis_executed": False,
            "figures_created": False,
            "final_findings_created": False,
            "phase51_signoff_created": False,
            "phase52_started": False,
            "new_test_inference": False,
            "checkpoint_loading": False,
            "training": False,
            "scaler_fit": False,
            "prediction_correction": False,
            "best_seed_selected": False,
            "ensemble": False,
        },
        "ready_for_phase51_g": False,
    }
    manifest_sha = f_writers.write_json(
        manifest, "phase51_f_manifest.json", root
    )

    return {
        "phase": 51,
        "subphase": "51-F",
        "status": "PASS",
        "selection_contract_sha256": CANONICAL_SHAS[
            "artifacts/worst_error_analysis/worst_error_selection_contract.json"
        ],
        "phase51_b_c_d_e_frozen_unchanged": True,
        "lstm_canonical_reason": canon_lstm["reason"],
        "lstm_canonical_status": canon_lstm["eligibility_status"],
        "n_cases": len(cases),
        "n_unique_targets": len(selected_target_ids),
        "n_context_rows": len(ctx_rows),
        "n_input_window_rows": len(win_rows),
        "n_feature_audit_rows": len(feat_rows),
        "n_feature_summary_rows": len(feat_summary),
        "n_membership_rows": len(membership_rows),
        "n_unique_master_rows": len(unique_master_rows),
        "n_ctx_audit_rows": len(ctx_audit_rows),
        "n_win_audit_rows": len(win_audit_rows),
        "n_cb_audit_rows": len(cb_audit_rows),
        "artifacts": {
            "local_temporal_context": ctx_sha,
            "input_window_manifest": win_sha,
            "feature_order_audit": feat_sha,
            "model_visible_feature_summary": feat_summary_sha,
            "casebook_index": casebook_sha,
            "casebook_membership_bridge": membership_sha,
            "casebook_unique_case_master": unique_master_sha,
            "context_integrity_audit": ctx_audit_sha,
            "input_window_integrity_audit": win_audit_sha,
            "casebook_integrity_audit": cb_audit_sha,
            "lstm_eligibility_context": lstm_sha,
            "phase51_f_manifest": manifest_sha,
        },
    }


def main(project_root=None) -> dict[str, Any]:
    return materialize_phase51_f(project_root)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
