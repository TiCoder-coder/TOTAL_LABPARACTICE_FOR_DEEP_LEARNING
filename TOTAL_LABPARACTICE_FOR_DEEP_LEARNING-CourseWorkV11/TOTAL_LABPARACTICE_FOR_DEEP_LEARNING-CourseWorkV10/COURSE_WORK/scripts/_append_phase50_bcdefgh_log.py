"""Append Phase 50-B through 50-H records to the processing log."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    log_path = Path("docs/save_log_in_processing/phase_50_error_by_regime_analysis_log.json")
    log = json.loads(log_path.read_text())

    log["log_version"] = "3"
    log["task"] = "PHASE_50_B_THROUGH_50_H_FULL_EXECUTION"
    log["subphase"] = "50-B + 50-C + 50-D + 50-E + 50-F + 50-G + 50-H"
    log["subphase_name"] = (
        "Train reference + threshold freeze → Test assignment freeze → residual join + metrics "
        "+ contributions → cross-seed aggregation + contrasts + prevalence → Persistence + LSTM "
        "+ seed-spread + sign-consensus → figures + findings + discrepancies + report + Phase51 "
        "handoff + signoff → notebook presentation dashboard"
    )
    log["timestamp_utc"] = now_iso()

    log.setdefault("timestamp_history", []).append(
        {"subphase": "50-B → 50-H full execution", "timestamp_utc": now_iso()}
    )

    log["implementation_started"] = True
    log["thresholds_computed"] = True
    log["test_regime_assignment_created"] = True
    log["train_regime_assignment_created"] = True
    log["residual_join_executed"] = True
    log["regime_metrics_computed"] = True
    log["rmse_lift_computed"] = True
    log["sae_sse_computed"] = True
    log["figures_generated"] = True
    log["notebook_modified"] = True

    err_dir = Path("artifacts/error_by_regime")
    canonical_files = [
        "regime_reference_train_manifest.json",
        "regime_reference_train_audit.csv",
        "regime_thresholds_train_only.json",
        "regime_threshold_audit.csv",
        "regime_threshold_fingerprint.json",
        "train_regime_assignment.csv",
        "test_regime_assignment.csv",
        "test_regime_assignment_audit.csv",
        "test_regime_assignment_fingerprint.json",
        "regime_error_join_audit.csv",
        "regime_metrics_long.csv",
        "regime_cross_seed_summary.csv",
        "regime_rmse_lift.csv",
        "regime_pairwise_contrasts.csv",
        "regime_train_vs_test_prevalence.csv",
        "regime_rank_stability.csv",
        "regime_metrics_persistence.csv",
        "regime_metrics_lstm.csv",
        "regime_seed_spread.csv",
        "regime_seed_spread_status.json",
        "regime_sign_consensus.csv",
        "regime_sign_consensus_mapping.json",
        "regime_lstm_status.json",
        "phase50_findings.json",
        "phase50_discrepancies.json",
        "phase50_report.md",
        "README.md",
        "phase51_handoff.json",
        "phase_50_signoff.json",
    ]
    canonical_shas = {}
    for fn in canonical_files:
        p = err_dir / fn
        if p.exists():
            canonical_shas[fn] = hashlib.sha256(p.read_bytes()).hexdigest()
    log["phase50_canonical_artifact_sha256"] = canonical_shas

    fig_dir = err_dir / "figures"
    fig_shas = {}
    for p in sorted(fig_dir.glob("*.png")):
        fig_shas[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    log["phase50_figures_sha256"] = fig_shas

    backups = sorted(Path("notebook_course_work").glob("CourseWork_1.ipynb.bak_before_phase50_*"))
    log["notebook_backup_file"] = backups[-1].name if backups else None
    nb = Path("notebook_course_work/CourseWork_1.ipynb")
    log["notebook_sha256"] = hashlib.sha256(nb.read_bytes()).hexdigest()

    so = json.loads((err_dir / "phase_50_signoff.json").read_text())
    log["phase_50_signoff_summary"] = {
        "status": so["status"],
        "n_gates": so["n_gates"],
        "n_gates_passed": so["n_gates_passed"],
        "ready_for_phase51": so["ready_for_phase51"],
        "ready_for_phase52_plus": so["ready_for_phase52_plus"],
        "n_figures": so["n_figures"],
        "n_canonical_artifacts": so["n_canonical_artifacts"],
    }

    history = log.setdefault("subphase_history", [])

    history.append({
        "subphase": "50-B",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/regime_reference_train_manifest.json",
            "artifacts/error_by_regime/regime_reference_train_audit.csv",
            "artifacts/error_by_regime/regime_thresholds_train_only.json",
            "artifacts/error_by_regime/regime_threshold_audit.csv",
            "artifacts/error_by_regime/regime_threshold_fingerprint.json",
        ],
        "q25": 50.0,
        "q75": 100.0,
        "q90": 210.0,
        "q90_abs_delta": 80.0,
        "n_train_reference": 13670,
        "n_classified_delta_pairs": 13669,
        "quantile_method": "linear",
        "quantile_library": "numpy",
        "tests": "20 passed (tests/unit/test_phase50_b_train_reference_and_thresholds.py)",
        "defects": "0",
        "safety_invariants": ["training=false", "test_derived_threshold_used=false"],
        "upstream_integrity": "Phase 47/48/49 canonical artifacts unchanged (sha256 verified).",
    })

    history.append({
        "subphase": "50-C",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/test_regime_assignment.csv",
            "artifacts/error_by_regime/test_regime_assignment_audit.csv",
            "artifacts/error_by_regime/test_regime_assignment_fingerprint.json",
            "artifacts/error_by_regime/train_regime_assignment.csv",
        ],
        "n_test_target_ids": 2961,
        "first_target_id": "TGT_00016774",
        "wb0_predecessor": "TGT_00016773 (last Validation)",
        "audit_checks_passed": 11,
        "tests": "25 passed (tests/unit/test_phase50_c_regime_assignment.py)",
        "defects": "0",
        "safety_invariants": ["no_prediction_columns", "no_seed_or_model_id_columns"],
        "upstream_integrity": "Phase 47/48/49 unchanged; thresholds frozen from 50-B.",
    })

    history.append({
        "subphase": "50-D",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/regime_error_join_audit.csv",
            "artifacts/error_by_regime/regime_metrics_long.csv",
        ],
        "n_join_rows": 8883,
        "n_unmatched": 0,
        "n_duplicate": 0,
        "global_mae_per_seed": {"42": 29.529, "123": 27.115, "2026": 28.942},
        "global_rmse_per_seed": {"42": 64.943, "123": 61.986, "2026": 64.560},
        "tests": "15 passed (tests/unit/test_phase50_d_residual_join_metrics.py)",
        "defects": "0",
        "safety_invariants": ["sample_share_sum_to_1", "sae_share_sum_to_1", "sse_share_sum_to_1"],
        "upstream_integrity": "Phase 47/48/49 unchanged.",
    })

    history.append({
        "subphase": "50-E",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/regime_cross_seed_summary.csv",
            "artifacts/error_by_regime/regime_rmse_lift.csv",
            "artifacts/error_by_regime/regime_pairwise_contrasts.csv",
            "artifacts/error_by_regime/regime_train_vs_test_prevalence.csv",
            "artifacts/error_by_regime/regime_rank_stability.csv",
        ],
        "cross_seed_ddof": 1,
        "predeclared_contrasts": 6,
        "tests": "16 passed (tests/unit/test_phase50_e_cross_seed.py)",
        "defects": "0",
        "safety_invariants": ["ddof=1", "no_3n_pooling"],
        "upstream_integrity": "Phase 47/48/49 unchanged.",
    })

    history.append({
        "subphase": "50-F",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/regime_metrics_persistence.csv",
            "artifacts/error_by_regime/regime_metrics_lstm.csv",
            "artifacts/error_by_regime/regime_seed_spread.csv",
            "artifacts/error_by_regime/regime_seed_spread_status.json",
            "artifacts/error_by_regime/regime_sign_consensus.csv",
            "artifacts/error_by_regime/regime_sign_consensus_mapping.json",
            "artifacts/error_by_regime/regime_lstm_status.json",
        ],
        "persistence_global_mae_wh": 26.73758865248227,
        "persistence_global_rmse_wh": 66.83691534084765,
        "lstm_status": "NOT_APPLICABLE",
        "tests": "17 passed (tests/unit/test_phase50_f_baselines_context.py)",
        "defects": "0",
        "safety_invariants": ["no_lstm_inference", "no_lstm_retrain", "no_lstm_fabrication"],
        "upstream_integrity": "Phase 47/48/49 unchanged.",
    })

    history.append({
        "subphase": "50-G",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "artifacts/error_by_regime/figures/*.png (14 figures)",
            "artifacts/error_by_regime/phase50_findings.json",
            "artifacts/error_by_regime/phase50_discrepancies.json",
            "artifacts/error_by_regime/phase50_report.md",
            "artifacts/error_by_regime/README.md",
            "artifacts/error_by_regime/phase51_handoff.json",
            "artifacts/error_by_regime/phase_50_signoff.json",
        ],
        "n_figures": 14,
        "n_discrepancies": 9,
        "n_unresolved_critical_discrepancies": 0,
        "phase51_worst_error_ranking_executed": False,
        "tests": "19 passed (tests/unit/test_phase50_g_finalization.py)",
        "defects": "0",
        "safety_invariants": [
            "training=false",
            "new_test_inference=false",
            "checkpoint_loading=false",
            "optimizer_steps=0",
            "scaler_fit=false",
            "best_seed_selected=false",
            "ensemble=false",
            "three_n_iid_interpretation=false",
            "test_derived_threshold_used=false",
            "cartesian_regime_mining=false",
            "prediction_correction=false",
            "worst_error_ranking_executed=false",
            "attention_analysis_executed=false",
            "phase47_modified=false",
            "phase48_modified=false",
            "phase49_modified=false",
        ],
        "upstream_integrity": "Phase 47/48/49 unchanged.",
    })

    history.append({
        "subphase": "50-H",
        "status": "PASS",
        "timestamp_utc": now_iso(),
        "outputs": [
            "src/course_work/reporting/phase_50_dashboard.py (read-only renderer)",
            "notebook_course_work/CourseWork_1.ipynb (130 → 132 cells, +1 markdown +1 code)",
        ],
        "tests": "14 passed (tests/unit/test_phase50_h_notebook_dashboard.py)",
        "defects": "0",
        "safety_invariants": [
            "no_training_text_in_dashboard",
            "no_scaler_text_in_dashboard",
            "no_checkpoint_text_in_dashboard",
            "phase0_49_cells_preserved_source",
            "phase0_49_cells_preserved_outputs",
        ],
        "upstream_integrity": "Phase 47/48/49 unchanged; Phase 0-49 notebook cells preserved.",
    })

    log["final_safety_invariants"] = {
        "training": False,
        "new_test_inference": False,
        "checkpoint_loading": False,
        "optimizer_steps": 0,
        "scaler_fit": False,
        "best_seed_selected": False,
        "ensemble": False,
        "three_n_iid_interpretation": False,
        "test_derived_threshold_used": False,
        "cartesian_regime_mining": False,
        "prediction_correction": False,
        "worst_error_ranking_executed": False,
        "attention_analysis_executed": False,
        "phase47_modified": False,
        "phase48_modified": False,
        "phase49_modified": False,
    }

    log["ready_for_phase50_b"] = True
    log["blockers"] = []

    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"log updated: {log_path}")
    print(f"log_version: {log['log_version']}")
    print(f"subphase_history entries: {len(history)}")


if __name__ == "__main__":
    main()
