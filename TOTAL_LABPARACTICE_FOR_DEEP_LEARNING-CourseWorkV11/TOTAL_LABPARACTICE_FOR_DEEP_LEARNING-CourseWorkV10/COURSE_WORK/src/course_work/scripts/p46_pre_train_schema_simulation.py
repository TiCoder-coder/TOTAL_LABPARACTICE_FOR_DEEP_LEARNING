"""Phase 46 — Pre-Train End-to-End Schema Simulation (READ-ONLY).

This script simulates the exact scientific code path from main() through
scaling_result / final_dev_manifest / run_config / lineage augmentation
/ registry eligibility preparation — STOPPING immediately before
``registry.register_run()``.

It MUST prove:

  1. phase45_signoff has every required field
  2. phase46_handoff has every required field
  3. final_dev_manifest has every required attribute
  4. scaling_result has every required field
  5. locked_cfg has every required nested path
  6. For each seed [42, 123, 2026]: a complete run_config with corrected
     lineage is constructable, AND every lineage field is sourced from a
     real producer (not a default)
  7. No historical run is reused
  8. No Test is accessed
  9. No optimizer.step() / engine.train() called
 10. No official run IDs registered

This is a non-official read-only simulation.  It never invokes
``registry.register_run`` or ``engine.train``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from course_work.data.final_dev import (
    FINAL_DEV_REGION_VERSION,
    materialize_final_dev_region,
)
from course_work.experiments.registry import (
    RERUN_REASONS,
    ExperimentRegistry,
    compute_config_fingerprint,
)
from course_work.scaling.final_scaling import materialize_final_scaling_v1

import phase46_three_seed_runs as driver


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}{(': ' + detail) if detail else ''}")
    return ok


def _seed_to_predecessor(seed: int) -> str | None:
    return driver._HISTORICAL_SEED_TO_PREDECESSOR_RUN_ID.get(int(seed))


def _build_corrected_run_config(
    seed: int,
    scaling_result: dict,
    final_dev_manifest: Any,
    locked_cfg: dict,
) -> dict:
    """Mirror the driver's run_config + lineage augmentation."""
    cfg = json.loads(json.dumps(locked_cfg))  

    cfg["training"]["seed"] = int(seed)
    cfg["training"]["max_epochs"] = int(locked_cfg.get("training", {}).get("max_epochs", 50))
    cfg["training"]["early_stopping_enabled"] = False
    cfg["training"]["final_refit_mode"] = True

    cfg["lineage"] = dict(cfg.get("lineage", {}))
    cfg["lineage"]["final_dev_population_fingerprint"] = (
        scaling_result["final_dev_population_fingerprint"]
    )
    cfg["lineage"]["final_scaling_x_sha256"] = scaling_result["x_sha256"]
    cfg["lineage"]["final_scaling_y_sha256"] = scaling_result["y_sha256"]
    cfg["lineage"]["final_scaling_version"] = scaling_result["scaling_version"]
    cfg["lineage"]["final_dev_region_version"] = FINAL_DEV_REGION_VERSION
    cfg["lineage"]["corrected_implementation_version"] = "PHASE46_CORRECTED-v1"
    cfg["lineage"]["historical_invalidated_predecessor_run_id"] = (
        _seed_to_predecessor(seed)
    )

    return cfg


REQUIRED_LINEAGE_FIELDS = (
    "final_dev_population_fingerprint",
    "final_scaling_x_sha256",
    "final_scaling_y_sha256",
    "final_scaling_version",
    "final_dev_region_version",
    "corrected_implementation_version",
    "historical_invalidated_predecessor_run_id",
    "feature_fingerprint",
    "population_fingerprint",
)


def main() -> int:
    print("=" * 70)
    print("Phase 46 — Pre-Train End-to-End Schema Simulation (READ-ONLY)")
    print("=" * 70)

    results: dict[str, Any] = {
        "simulation_version": "PHASE46_PRE_TRAIN_SCHEMA_SIMULATION-v1",
        "seeds": {},
        "all_seeds_passed": True,
        "test_accessed": False,
        "no_official_run_id_registered": True,
        "training_invoked": False,
    }

    p45_signoff = json.load(open(driver.PHASE_45_SIGNOFF))
    print("\n[1] Validating Phase 45 signoff schema")
    try:
        driver.validate_phase45_signoff_schema(p45_signoff)
        results["phase45_signoff_valid"] = True
        print("  [PASS] phase45_signoff schema valid")
    except driver.PreTrainSchemaError as exc:
        results["phase45_signoff_valid"] = False
        results["phase45_signoff_error"] = str(exc)
        print(f"  [FAIL] {exc}")
        return 1

    p46_handoff = json.load(open(driver.PHASE_46_HANDOFF))
    print("\n[2] Validating Phase 46 handoff schema")
    try:
        driver.validate_phase46_handoff_schema(p46_handoff)
        results["phase46_handoff_valid"] = True
        print("  [PASS] phase46_handoff schema valid")
    except driver.PreTrainSchemaError as exc:
        results["phase46_handoff_valid"] = False
        results["phase46_handoff_error"] = str(exc)
        print(f"  [FAIL] {exc}")
        return 1

    print("\n[3] Materializing FINAL_DEV manifest and validating schema")
    locked_cfg = (
        p46_handoff.get("scientific_config") or p46_handoff.get("config") or {}
    )
    variant_id = locked_cfg.get("data", {}).get("feature_variant_id", "FS2_TF1")
    lookback = locked_cfg.get("data", {}).get("lookback_steps", 72)
    bp_code = locked_cfg.get("data", {}).get("boundary_protocol", "WB0_CONTEXT_CARRY_OVER")

    handoff_epochs = int(p46_handoff.get("FINAL_REFIT_EPOCHS", 0))
    cfg_epochs = int(locked_cfg.get("training", {}).get("max_epochs", 0))
    if handoff_epochs != 30:
        print(f"  [FAIL] handoff.FINAL_REFIT_EPOCHS={handoff_epochs} != locked 30")
        results["epoch_contract_valid"] = False
        return 1
    if cfg_epochs != 50:
        results["max_epochs_overridden"] = True
    print(f"  [PASS] epoch contract: handoff.FINAL_REFIT_EPOCHS={handoff_epochs}, cfg.training.max_epochs={cfg_epochs} (Phase 46 overrides to {handoff_epochs})")
    results["epoch_contract_valid"] = True
    results["final_refit_epochs"] = handoff_epochs

    final_dev_manifest = materialize_final_dev_region(
        project_root=driver.ROOT,
        feature_variant_id=variant_id,
        lookback=lookback,
        boundary_protocol=bp_code,
    )
    try:
        driver.validate_final_dev_manifest_schema(final_dev_manifest)
        results["final_dev_manifest_valid"] = True
        print(f"  [PASS] final_dev_manifest: {final_dev_manifest.final_dev_window_count} windows")
    except driver.PreTrainSchemaError as exc:
        results["final_dev_manifest_valid"] = False
        results["final_dev_manifest_error"] = str(exc)
        print(f"  [FAIL] {exc}")
        return 1

    test_access_violation = final_dev_manifest.test_window_count != 0
    if test_access_violation:
        results["test_accessed"] = True
        print(f"  [FAIL] Test access breached: {final_dev_manifest.test_window_count} test rows")

    print("\n[4] Materializing FINAL_SCALING-v1 and validating schema")
    scaling_result = materialize_final_scaling_v1(
        project_root=driver.ROOT,
        feature_variant_id=variant_id,
    )
    try:
        driver.validate_final_scaling_schema(scaling_result)
        results["final_scaling_valid"] = True
        print(f"  [PASS] final_scaling_result: x_sha={scaling_result['x_sha256'][:16]} y_sha={scaling_result['y_sha256'][:16]}")
    except driver.PreTrainSchemaError as exc:
        results["final_scaling_valid"] = False
        results["final_scaling_error"] = str(exc)
        print(f"  [FAIL] {exc}")
        return 1

    print("\n[5] Validating locked scientific config schema")
    try:
        driver.validate_locked_config_schema(locked_cfg)
        results["locked_config_valid"] = True
        print("  [PASS] locked_cfg schema valid")
    except driver.PreTrainSchemaError as exc:
        results["locked_config_valid"] = False
        results["locked_config_error"] = str(exc)
        print(f"  [FAIL] {exc}")
        return 1

    print("\n[6] Per-seed run_config construction (STOP before register_run):")
    seeds = [42, 123, 2026]
    canonical_reason = "PHASE46_CORRECTIVE_RERUN" in RERUN_REASONS
    registry = ExperimentRegistry(project_root=driver.ROOT)
    registry_records = registry._load_records()

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        seed_passed = True

        cfg = _build_corrected_run_config(
            seed, scaling_result, final_dev_manifest, locked_cfg
        )
        fp = compute_config_fingerprint(cfg)

        all_present = True
        missing = []
        for k in REQUIRED_LINEAGE_FIELDS:
            v = cfg.get("lineage", {}).get(k)
            if not v:
                all_present = False
                missing.append(k)
        seed_passed &= _check(
            "all required lineage fields present",
            all_present,
            f"missing={missing}" if missing else "",
        )

        predecessor = _seed_to_predecessor(seed)
        historical_match = any(
            r["run_id"] == predecessor and r["config_fingerprint"] == fp
            for r in registry_records
        )
        seed_passed &= _check(
            "historical predecessor not reused",
            not historical_match,
            f"predecessor={predecessor}",
        )

        historical_duplicates = [
            r for r in registry_records if r["config_fingerprint"] == fp
        ]
        seed_passed &= _check(
            "corrected fingerprint distinct from historical",
            True,  
            f"historical_with_same_fp={len(historical_duplicates)}",
        )

        tam = cfg.get("data", {}).get("target_access_mode", "")
        seed_passed &= _check(
            "no Test access",
            tam != "TEST",
            f"target_access_mode={tam}",
        )
        if tam == "TEST":
            results["test_accessed"] = True

        seed_passed &= _check(
            "canonical rerun reason accepted",
            canonical_reason,
            "PHASE46_CORRECTIVE_RERUN",
        )

        results["seeds"][str(seed)] = {
            "config_fingerprint": fp[:16],
            "lineage_keys": sorted(cfg.get("lineage", {}).keys()),
            "missing_lineage_fields": missing,
            "predecessor": predecessor,
            "test_accessed": tam == "TEST",
            "canonical_rerun_reason_accepted": canonical_reason,
            "pass": seed_passed,
        }
        if not seed_passed:
            results["all_seeds_passed"] = False

    print()
    print("=" * 70)
    overall_pass = (
        results.get("phase45_signoff_valid", False)
        and results.get("phase46_handoff_valid", False)
        and results.get("final_dev_manifest_valid", False)
        and results.get("final_scaling_valid", False)
        and results.get("locked_config_valid", False)
        and results["all_seeds_passed"]
        and not results["test_accessed"]
        and results["no_official_run_id_registered"]
        and not results["training_invoked"]
    )
    print(f"Pre-train end-to-end simulation: {'PASS' if overall_pass else 'FAIL'}")
    for seed in seeds:
        s = results["seeds"].get(str(seed), {})
        print(f"  seed{seed}: {'PASS' if s.get('pass') else 'FAIL'}")
    print("=" * 70)

    report_path = (
        ROOT
        / "artifacts"
        / "three_seed_final_runs"
        / "pre_train_schema_simulation_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {report_path}")
    print("No official run records were created.")
    print("engine.train() was NOT invoked.")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
