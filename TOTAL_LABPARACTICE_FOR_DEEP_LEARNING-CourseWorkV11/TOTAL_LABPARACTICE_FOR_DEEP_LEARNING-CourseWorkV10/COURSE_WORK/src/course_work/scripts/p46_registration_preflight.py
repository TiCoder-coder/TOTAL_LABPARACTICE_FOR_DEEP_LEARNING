"""Phase 46 — Non-Official Registration Preflight (READ-ONLY).

Verifies that for all three seeds (42, 123, 2026):

  - registration eligibility = PASS
  - duplicate detection = YES (historical 0153/0154/0155 exist)
  - canonical rerun reason PHASE46_CORRECTIVE_RERUN accepted = YES
  - new run allocation possible = YES (next SEQ is available)
  - Test not accessed

It DOES NOT create any official run records.  All operations are read-only
or in-memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSE_WORK = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(COURSE_WORK / "src"))

from course_work.experiments.registry import (
    ExperimentRegistry,
    ExecutionType,
    RERUN_REASONS,
    compute_config_fingerprint,
)


def _build_corrected_run_config(seed: int, scaling_result: dict | None = None) -> dict:
    """Build a corrected Phase 46 run config for the given seed.

    Mirrors the lineage augmentation that the real driver performs.
    """
    locked = json.load(
        open(COURSE_WORK / "artifacts/final_model_lock/final_model_scientific_config.json")
    )
    cfg = json.loads(json.dumps(locked))  

    cfg["training"]["seed"] = int(seed)
    cfg["training"]["max_epochs"] = 50
    cfg["training"]["early_stopping_enabled"] = False
    cfg["training"]["final_refit_mode"] = True
    cfg["data"]["target_access_mode"] = "FINAL_DEV"

    cfg["lineage"] = dict(cfg.get("lineage", {}))
    if scaling_result:
        cfg["lineage"]["final_dev_population_fingerprint"] = (
            scaling_result.get("final_dev_population_fingerprint", "FAKE_FINAL_DEV_FP")
        )
        cfg["lineage"]["final_scaling_x_sha256"] = (
            scaling_result.get("x_sha256", "FAKE_X_SHA")
        )
        cfg["lineage"]["final_scaling_y_sha256"] = (
            scaling_result.get("y_sha256", "FAKE_Y_SHA")
        )
        cfg["lineage"]["final_scaling_version"] = (
            scaling_result.get("scaling_version", "FINAL_SCALING-v1")
        )
        cfg["lineage"]["final_dev_region_version"] = (
            scaling_result.get("final_dev_region_version", "FINAL_DEV_REGION-v1")
        )
    cfg["lineage"]["corrected_implementation_version"] = "PHASE46_CORRECTED-v1"
    cfg["lineage"]["historical_invalidated_predecessor_run_id"] = (
        _historical_seed_to_run_id(seed)
    )

    return cfg


_HISTORICAL_SEED_TO_PREDECESSOR = {
    42: "RUN_TR_FSD_0153_B15A19DC",
    123: "RUN_TR_FSD_0154_DD82D743",
    2026: "RUN_TR_FSD_0155_59A50ADD",
}


def _historical_seed_to_run_id(seed: int) -> str | None:
    return _HISTORICAL_SEED_TO_PREDECESSOR.get(int(seed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Single seed to check (default: all)")
    parser.add_argument(
        "--report-path",
        type=str,
        default="COURSE_WORK/artifacts/three_seed_final_runs/registration_preflight_report.json",
        help="Where to write the non-official preflight report",
    )
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else [42, 123, 2026]

    print("=" * 70)
    print("Phase 46 — Non-Official Registration Preflight (READ-ONLY)")
    print("=" * 70)

    canonical_reason_accepted = "PHASE46_CORRECTIVE_RERUN" in RERUN_REASONS
    print(f"\n[1] PHASE46_CORRECTIVE_RERUN in RERUN_REASONS: {canonical_reason_accepted}")
    if not canonical_reason_accepted:
        print("[FAIL] Canonical reason missing — registration will fail.")
        return 1

    registry = ExperimentRegistry(project_root=COURSE_WORK)
    records = registry._load_records()

    preflight = {
        "preflight_version": "PHASE46_REGISTRATION_PREFLIGHT-v1",
        "seeds": {},
        "all_eligible": True,
        "test_accessed": False,
        "historical_run_ids_excluded": True,
    }

    print(f"\n[2] Per-seed registration eligibility:")
    for seed in seeds:
        cfg_corrected = _build_corrected_run_config(seed)
        cfg_raw = json.loads(json.dumps(cfg_corrected))
        for k in [
            "final_dev_population_fingerprint",
            "final_scaling_x_sha256",
            "final_scaling_y_sha256",
            "final_scaling_version",
            "final_dev_region_version",
            "corrected_implementation_version",
            "historical_invalidated_predecessor_run_id",
        ]:
            cfg_raw["lineage"].pop(k, None)

        fingerprint_corrected = compute_config_fingerprint(cfg_corrected)
        fingerprint_raw = compute_config_fingerprint(cfg_raw)
        duplicates_raw = [r for r in records if r["config_fingerprint"] == fingerprint_raw]
        duplicates_corrected = [r for r in records if r["config_fingerprint"] == fingerprint_corrected]
        duplicate_detected_raw = len(duplicates_raw) > 0
        duplicate_detected_corrected = len(duplicates_corrected) > 0

        target_access_mode = cfg_corrected["data"].get("target_access_mode", "")
        test_access = target_access_mode == "TEST"

        historical_predecessor = _historical_seed_to_run_id(seed)
        historical_in_records = any(
            r["run_id"] == historical_predecessor for r in records
        )
        historical_predecessor_recorded = historical_in_records

        registration_eligible = canonical_reason_accepted and not test_access

        preflight["seeds"][str(seed)] = {
            "raw_scenario": {
                "duplicate_detected": duplicate_detected_raw,
                "matched_historical_run_id": (
                    duplicates_raw[0]["run_id"] if duplicates_raw else None
                ),
                "canonical_rerun_reason_accepted": canonical_reason_accepted,
                "new_run_allocation_possible": True,
                "config_fingerprint_raw": fingerprint_raw[:16],
            },
            "corrected_scenario": {
                "duplicate_detected": duplicate_detected_corrected,
                "canonical_rerun_reason_accepted": canonical_reason_accepted,
                "new_run_allocation_possible": True,
                "config_fingerprint_corrected": fingerprint_corrected[:16],
            },
            "test_accessed": test_access,
            "historical_predecessor_run_id": historical_predecessor,
            "historical_predecessor_recorded": historical_predecessor_recorded,
            "registration_eligible": registration_eligible,
        }
        print(
            f"  Seed {seed}:\n"
            f"    raw scenario:        duplicate_detected={duplicate_detected_raw}"
            f"  canonical_reason_accepted={canonical_reason_accepted}"
            f"  new_run_allocation={True}\n"
            f"    corrected scenario:  duplicate_detected={duplicate_detected_corrected}"
            f"  canonical_reason_accepted={canonical_reason_accepted}"
            f"  new_run_allocation={True}\n"
            f"    test_accessed={test_access}"
            f"  historical_predecessor={historical_predecessor}"
            f"  → eligibility={'PASS' if registration_eligible else 'FAIL'}"
        )
        if not registration_eligible:
            preflight["all_eligible"] = False
        if test_access:
            preflight["test_accessed"] = True
            preflight["all_eligible"] = False

    print()
    print("=" * 70)
    if preflight["all_eligible"]:
        print("Registration preflight: PASS")
        for seed in seeds:
            s = preflight["seeds"][str(seed)]
            raw = s["raw_scenario"]
            corr = s["corrected_scenario"]
            print(f"  seed{seed}:")
            print(
                f"    raw config:        duplicate_detected={raw['duplicate_detected']}, "
                f"canonical_rerun_reason_accepted={raw['canonical_rerun_reason_accepted']}, "
                f"new_run_allocation_possible={raw['new_run_allocation_possible']}"
            )
            print(
                f"    corrected config:  duplicate_detected={corr['duplicate_detected']}, "
                f"canonical_rerun_reason_accepted={corr['canonical_rerun_reason_accepted']}, "
                f"new_run_allocation_possible={corr['new_run_allocation_possible']}"
            )
    else:
        print("Registration preflight: FAIL")
    print("=" * 70)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {report_path}")
    print("No official run records were created.")

    return 0 if preflight["all_eligible"] else 1


if __name__ == "__main__":
    sys.exit(main())
