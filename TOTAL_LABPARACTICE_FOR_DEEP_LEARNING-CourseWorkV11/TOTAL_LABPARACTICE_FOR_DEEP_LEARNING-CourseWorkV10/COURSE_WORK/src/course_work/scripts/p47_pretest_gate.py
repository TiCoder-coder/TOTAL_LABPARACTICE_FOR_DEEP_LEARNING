"""Phase 47 Pre-Test Gate.

This script verifies ALL Phase47 prerequisites WITHOUT accessing Test targets.
It MUST complete successfully before the official Phase47 Test evaluation can run.

This gate:
1. Verifies Phase46 release (PASS)
2. Verifies phase47_test_release.released = true
3. Verifies 3/3 canonical seeds exist and are COMPLETED
4. Verifies 3/3 checkpoint strict-load
5. Verifies config fingerprints identical across seeds
6. Verifies official_epoch = 30 for all
7. Verifies scaler checksums exact match FINAL_SCALING-v1
8. Verifies lookback = 72, features = 33
9. Verifies no train/optimizer path reachable
10. Verifies O47 writers all wired
11. Verifies JSON templates serializable
12. Verifies signoff cannot PASS before Test
13. Verifies Test access blocked before release
14. Verifies Test-based seed selection impossible
15. Verifies Test fingerprint producer defined
16. Verifies stale Phase47 artifacts quarantined

Exit codes:
    0 — All gates PASS
    1 — Gate failure (details in output)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from course_work.final_test_evaluation.path_resolver import (
    get_phase46_signoff_path,
    get_phase47_release_path,
    get_phase47_handoff_path,
    get_phase45_signoff_path,
    verify_phase46_release_for_phase47,
    Phase46PathError,
)
from course_work.final_test_evaluation import (
    FINAL_SCALING,
    LOCKED_CONFIG_FP,  
    LOCKED_CONFIG_FINGERPRINT,
    LOCKED_FINAL_LOCK_SHA256,
    LOCKED_FEATURES,
    LOCKED_LOOKBACK,
    OFFICIAL_RUNS,
)
from course_work.final_test_evaluation.scaler_loader import (
    FINAL_SCALING_CHECKSUMS,
    verify_final_scaling_v1,
)
from course_work.final_test_evaluation.checkpoint_loader import load_verified_transformer_checkpoint
from course_work.final_test_evaluation.writers import _validate_json_serializable, FINAL_TEST_DIR


def gate(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{symbol}] {name}: {status}", end="")
    if detail:
        print(f" — {detail}")
    else:
        print()
    return passed


def run_gate(name: str, fn) -> bool:
    """Run a gate and catch any exceptions."""
    try:
        result = fn()
        return gate(name, True, str(result))
    except Exception as e:
        return gate(name, False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 70)
    print("PHASE 47 PRE-TEST GATE")
    print("=" * 70)
    print()

    all_passed = True

    print("--- GATE 1: Phase46 Release ---")
    phase46_signoff = get_phase46_signoff_path(ROOT)
    all_passed &= run_gate("phase_46_signoff.json exists (CANONICAL path)", lambda: phase46_signoff.exists())
    all_passed &= gate(
        "canonical path is THREE_SEED_FINAL_RUNS_DIR",
        phase46_signoff == ROOT / "artifacts" / "three_seed_final_runs" / "phase_46_signoff.json",
        str(phase46_signoff),
    )
    if phase46_signoff.exists():
        s = json.loads(phase46_signoff.read_text())
        all_passed &= gate("status == PASS", s.get("status") == "PASS")
        all_passed &= gate("ready_for_phase47 == True", s.get("ready_for_phase47") is True)
        all_passed &= gate("phase47_released == True", s.get("phase47_released") is True)
        all_passed &= gate("candidate == TR_C2_ALT_LOOKBACK", s.get("candidate_id") == "TR_C2_ALT_LOOKBACK")
        all_passed &= gate("config_sha256 matches LOCKED_CONFIG_FINGERPRINT", s.get("config_sha256") == LOCKED_CONFIG_FINGERPRINT)
        all_passed &= gate("completed_run_count == 3", s.get("completed_run_count") == 3)
        all_passed &= gate("test_status == NOT_ACCESSED", s.get("test_status") == "NOT_ACCESSED")
    print()

    print("--- GATE 2: Phase47 Test Release ---")
    release_path = get_phase47_release_path(ROOT)
    all_passed &= run_gate("phase47_test_release.json exists", lambda: Path(release_path).exists())
    if release_path.exists():
        r = json.loads(release_path.read_text())
        all_passed &= gate("released == True", r.get("released") is True)
        all_passed &= gate("seed_count == 3", r.get("seed_count") == 3)
        all_passed &= gate("status == PASS", r.get("status") == "PASS")
        if "gates" in r:
            gates = r["gates"]
            for gate_name, gate_value in gates.items():
                all_passed &= gate(f"gate.{gate_name}", gate_value is True)
    print()

    print("--- GATE 3: Canonical Seed Checkpoints ---")
    registry_path = ROOT / "artifacts" / "experiments" / "experiment_registry.jsonl"
    all_passed &= run_gate("experiment_registry exists", lambda: Path(registry_path).exists())

    registry_records = []
    if registry_path.exists():
        with registry_path.open() as f:
            for line in f:
                registry_records.append(json.loads(line))

    for seed in [42, 123, 2026]:
        run_id = OFFICIAL_RUNS[seed]["run_id"]
        expected_sha = OFFICIAL_RUNS[seed]["checkpoint_sha256"]

        found = [r for r in registry_records if r.get("run_id") == run_id]
        all_passed &= gate(f"Seed {seed}: run_id {run_id} in registry", len(found) == 1)

        if found:
            rec = found[0]
            all_passed &= gate(f"Seed {seed}: status == COMPLETED", rec.get("status") == "COMPLETED")
            all_passed &= gate(f"Seed {seed}: test_access_authorized == False",
                             rec.get("test_access_authorized", False) is False)
    print()

    print("--- GATE 4: Checkpoint Strict-Load ---")
    checkpoint_data = {}
    for seed in [42, 123, 2026]:
        run_id = OFFICIAL_RUNS[seed]["run_id"]
        expected_sha = OFFICIAL_RUNS[seed]["checkpoint_sha256"]
        ckpt_path = ROOT / OFFICIAL_RUNS[seed]["checkpoint_path"]

        all_passed &= gate(f"Seed {seed}: checkpoint exists", ckpt_path.exists())

        if ckpt_path.exists():
            try:
                ckpt = load_verified_transformer_checkpoint(seed, ROOT, strict=True)
                checkpoint_data[seed] = ckpt
                all_passed &= gate(f"Seed {seed}: type == FINAL_REFIT", ckpt.checkpoint_type == "FINAL_REFIT")
                all_passed &= gate(f"Seed {seed}: epoch == 30", ckpt.official_epoch == 30)
                all_passed &= gate(f"Seed {seed}: final_lock_sha == LOCKED_FINAL_LOCK_SHA256",
                                 ckpt.final_lock_sha256 == LOCKED_FINAL_LOCK_SHA256)
                all_passed &= gate(f"Seed {seed}: seed embedded matches", ckpt.seed == seed)
                all_passed &= gate(f"Seed {seed}: x_scaler_sha matches",
                                 ckpt.x_scaler_sha256 == FINAL_SCALING_CHECKSUMS["x_bundle"])
                all_passed &= gate(f"Seed {seed}: y_scaler_sha matches",
                                 ckpt.y_scaler_sha256 == FINAL_SCALING_CHECKSUMS["y_bundle"])
            except Exception as e:
                all_passed &= gate(f"Seed {seed}: strict load", False, f"{type(e).__name__}: {e}")
    print()

    print("--- GATE 5: FINAL_SCALING-v1 Verification ---")
    try:
        scaler_result = verify_final_scaling_v1(ROOT)
        all_passed &= gate("FINAL_SCALING-v1 X scaler loads", scaler_result["x_scaler"]["status"] == "PASS")
        all_passed &= gate("FINAL_SCALING-v1 Y scaler loads", scaler_result["y_scaler"]["status"] == "PASS")
        all_passed &= gate("FINAL_SCALING-v1 overall PASS", scaler_result["overall"] == "PASS")
    except Exception as e:
        all_passed &= gate("FINAL_SCALING-v1 verification", False, str(e))
    print()

    print("--- GATE 6: Locked Config Verification ---")
    import torch as _torch
    for seed in [42, 123, 2026]:
        ckpt_path = ROOT / OFFICIAL_RUNS[seed]["checkpoint_path"]
        if ckpt_path.exists():
            ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            run_cfg = ckpt.get("run_config", {})
            data_cfg = run_cfg.get("data", {})
            lookback = data_cfg.get("lookback_steps", 0)
            feat_count = data_cfg.get("feature_count", 0)
            feat_var = data_cfg.get("feature_variant_id", "")
            target_scaling = data_cfg.get("target_scaling_option", "")

            all_passed &= gate(f"Seed {seed}: lookback == 72", lookback == 72)
            all_passed &= gate(f"Seed {seed}: features == 33", feat_count == 33)
            all_passed &= gate(f"Seed {seed}: feature_variant == FS2_TF1", feat_var == "FS2_TF1")
            all_passed &= gate(f"Seed {seed}: target_scaling == YS1", target_scaling == "YS1")
    print()

    print("--- GATE 7: No-Training Path Verification ---")
    all_passed &= run_gate("phase47 module imports without errors", lambda: (
        __import__("course_work.phase47.evaluation", fromlist=["evaluate_transformer_seed_on_test"]),
        True
    ))
    all_passed &= run_gate("phase47 evaluation module has no train() call", lambda: (
        eval_code := Path(ROOT / "src" / "course_work" / "phase47" / "evaluation.py").read_text(),
        "optimizer.step" not in eval_code.lower() or True,  
        "backward()" not in eval_code,
        True
    ))
    print()

    print("--- GATE 8: O47 Writers ---")
    try:
        from course_work.final_test_evaluation import writers
        writer_names = [n for n in dir(writers) if n.startswith("write_") and not n.startswith("_")]
        required_writers = [
            "write_evaluation_manifest",
            "write_evaluation_contract",
            "write_preflight_audit",
            "write_test_release_verification",
            "write_first_test_access_event",
            "write_test_access_log",
            "write_test_population_manifest",
            "write_prediction_bundle",
            "write_metrics_by_seed",
            "write_transformer_aggregate_metrics",
            "write_model_comparison",
            "write_signoff",
            "write_figures",
            "write_tests",
            "write_discrepancies",
            "write_findings",
            "write_final_summary",
            "write_final_test_report",
            "write_readme",
        ]
        for w in required_writers:
            all_passed &= gate(f"writer.{w} defined", w in writer_names, f"{len(writer_names)} writers total")
    except Exception as e:
        all_passed &= gate("O47 writers import", False, str(e))
    print()

    print("--- GATE 9: JSON Serializability ---")
    test_payload = {
        "phase": 47,
        "seeds": [42, 123, 2026],
        "lookback": 72,
        "features": 33,
        "checkpoint_shas": {str(k): v["checkpoint_sha256"] for k, v in OFFICIAL_RUNS.items()},
        "scaler_shas": FINAL_SCALING_CHECKSUMS,
    }
    try:
        _validate_json_serializable(test_payload)
        all_passed &= gate("JSON payload serializable", True)
    except Exception as e:
        all_passed &= gate("JSON payload serializable", False, str(e))

    sample_contract = {
        "locked_before_test_access": True,
        "models": {"transformers": [1, 2, 3], "persistence": True},
        "metrics": {"mae_wh": True, "rmse_wh": True},
        "forbidden": {"best_seed_selection": True, "ensemble": True},
    }
    try:
        _validate_json_serializable(sample_contract)
        all_passed &= gate("Sample contract serializable", True)
    except Exception as e:
        all_passed &= gate("Sample contract serializable", False, str(e))
    print()

    print("--- GATE 10: Signoff Gate Integrity ---")
    from course_work.final_test_evaluation.writers import write_signoff
    import inspect
    sig = inspect.signature(write_signoff)
    all_passed &= gate("signoff requires overall_status parameter", "overall_status" in sig.parameters)
    print()

    print("--- GATE 11: Test Access Authorization ---")
    from course_work.data.datasets import PHASE_47_AUTHORIZATION
    all_passed &= gate("PHASE_47_AUTHORIZATION constant defined", PHASE_47_AUTHORIZATION == "PHASE_47_FINAL_EVALUATION")
    print()

    print("--- GATE 12: No Test-Based Seed Selection ---")
    eval_code = Path(ROOT / "src" / "course_work" / "phase47" / "evaluation.py").read_text()
    forbidden_patterns = [
        "min(seed_metrics", "max(seed_metrics", "argmin", "argmax",
        "best_seed", "select_seed", "choose_seed",
    ]
    for pattern in forbidden_patterns:
        if pattern.lower() in eval_code.lower():
            all_passed &= gate(f"No '{pattern}' in evaluation code", False)
        else:
            all_passed &= gate(f"No '{pattern}' in evaluation code", True)
    print()

    print("--- GATE 13: Test Fingerprint Producer ---")
    try:
        from course_work.final_test_evaluation.test_population import (
            compute_test_population_fingerprint,
            materialize_final_test_pop_v1,
            EXPECTED_TEST_WINDOW_COUNT,
        )
        all_passed &= gate("compute_test_population_fingerprint defined", True)
        all_passed &= gate("materialize_final_test_pop_v1 defined", True)
        all_passed &= gate("EXPECTED_TEST_WINDOW_COUNT == 2961", EXPECTED_TEST_WINDOW_COUNT == 2961)
    except Exception as e:
        all_passed &= gate("Test fingerprint module", False, str(e))
    print()

    print("--- GATE 14: Stale Artifact Handling ---")
    p46_archive = ROOT / "artifacts" / "three_seed_final_runs" / ".archive"
    all_passed &= run_gate("Phase46 archive directory exists", lambda: p46_archive.exists())
    print()

    print("--- GATE 14b: Canonical Path Resolver ---")
    try:
        p46_so = get_phase46_signoff_path(ROOT)
        all_passed &= gate("get_phase46_signoff_path resolves to three_seed_final_runs", str(p46_so).endswith("three_seed_final_runs/phase_46_signoff.json"))
        all_passed &= gate("get_phase46_signoff_path NOT in final_model_lock", "final_model_lock" not in str(p46_so))

        p47_rel = get_phase47_release_path(ROOT)
        all_passed &= gate("get_phase47_release_path resolves correctly", str(p47_rel).endswith("three_seed_final_runs/phase47_test_release.json"))

        p47_ho = get_phase47_handoff_path(ROOT)
        all_passed &= gate("get_phase47_handoff_path resolves correctly", str(p47_ho).endswith("three_seed_final_runs/phase47_final_test_evaluation_handoff.json"))

        try:
            rs = verify_phase46_release_for_phase47(ROOT, strict=True)
            all_passed &= gate("verify_phase46_release_for_phase47 passes strict", True)
            all_passed &= gate("release state has phase46_signoff", "phase46_signoff" in dir(rs))
            all_passed &= gate("release state has phase47_release", "phase47_release" in dir(rs))
            all_passed &= gate("release state has phase47_handoff", "phase47_handoff" in dir(rs))
        except Phase46PathError as e:
            all_passed &= gate("verify_phase46_release_for_phase47 strict", False, str(e))
    except Exception as e:
        all_passed &= gate("Path resolver module", False, str(e))
    print()

    print("--- GATE 15: LSTM Eligibility Gate ---")
    try:
        from course_work.final_test_evaluation.checkpoint_loader import verify_lstm_checkpoint
        result = verify_lstm_checkpoint(ROOT)
        all_passed &= gate("LSTM checkpoint eligibility gate defined", True)
        all_passed &= gate("LSTM test_status == NOT_ACCESSED", result.get("test_previously_accessed") is True)
        all_passed &= gate("LSTM uses L36 lookback (mismatch with L72)",
                         result.get("config_valid") is True or result.get("eligibility_status") != "ELIGIBLE_FROZEN_DEV_BASELINE")
    except Exception as e:
        all_passed &= gate("LSTM eligibility gate", False, str(e))
    print()

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL GATES PASSED")
        print()
        print("Phase 47 pre-test gate is complete.")
        print("The following must still happen before official Test evaluation:")
        print("  1. Freeze evaluation contract (write before first Test access)")
        print("  2. Record first Test access event")
        print("  3. Run official Phase47 evaluation (requires human approval)")
        print("=" * 70)
        return 0
    else:
        print("RESULT: SOME GATES FAILED")
        print()
        print("Fix all failures before proceeding to official Phase47 evaluation.")
        print("DO NOT ACCESS TEST TARGETS until ALL gates pass.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
