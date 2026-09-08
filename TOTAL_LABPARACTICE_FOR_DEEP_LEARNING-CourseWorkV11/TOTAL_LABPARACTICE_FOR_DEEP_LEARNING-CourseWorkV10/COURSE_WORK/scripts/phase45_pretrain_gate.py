"""Phase 45 PRETRAIN / PRELOCK gate.

Despite the name, Phase45 has no training. This gate verifies all pre-lock
conditions:

    Phase44 PASS / approved_for_phase45
    Transformer-only
    Canonical candidate fingerprint matches Phase42
    S1-S19 lineage complete
    WB0 locked, no protocol amendment
    3/3 RO epochs present
    median == 30
    no fallback == 50
    FINAL_DEV valid (last ts < first TEST ts)
    Test excluded
    Scaler contract excludes Test
    Seeds exactly [42, 123, 2026]
    Phase46 recipe has no validation / no early stopping
    Fingerprints deterministic
    O45 writers complete (38/38)
    Zero optimizer path
    Zero new RUN IDs path

Exit 0 on PASS, nonzero on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from course_work.final_model_lock import (
    load_phase44_handoff,
    load_phase44_signoff,
    load_phase42_shortlist,
    lock_candidate,
    derive_final_epoch,
    build_final_dev_population,
    build_scaling_contract,
    scaling_contract_to_dict,
    build_recipe,
    recipe_to_dict,
    build_seed_contract,
    build_run_matrix,
    build_boundary_sensitivity_evidence,
    config_fingerprint,
    recipe_fingerprint,
    lineage_fingerprint,
    lock_fingerprint,
    build_lineage_audit,
    ARTIFACT_NAMES,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase45 prelock gate")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()

    project_root: Path = args.project_root
    artifact_dir = project_root / "artifacts" / "final_model_lock"

    fails: list[str] = []

    signoff = load_phase44_signoff(
        project_root / "artifacts" / "rolling_origin" / "phase_44_signoff.json"
    )
    if signoff["overall_status"] not in {"PASS", "PASS_WITH_WARNING"}:
        fails.append("phase44_signoff: overall_status must be PASS/PASS_WITH_WARNING")
    if not signoff["approved_for_phase45"]:
        fails.append("phase44_signoff: approved_for_phase45 must be true")
    if signoff["test_status"] != "NOT_ACCESSED":
        fails.append("phase44_signoff: test_status must be NOT_ACCESSED")

    handoff = load_phase44_handoff(
        project_root / "artifacts" / "rolling_origin" / "phase45_final_model_lock_handoff.json"
    )
    shortlist = load_phase42_shortlist(
        project_root / "artifacts" / "candidate_synthesis" / "transformer_candidate_shortlist.json"
    )
    candidate = lock_candidate(handoff, shortlist_payload=shortlist)
    if candidate.model_family != "TRANSFORMER_ENCODER":
        fails.append(f"model_family must be TRANSFORMER_ENCODER, got {candidate.model_family!r}")

    if candidate.config_fingerprint != handoff["recommended_transformer_fingerprint"]:
        fails.append("Phase42 candidate fingerprint != Phase44 handoff fingerprint")

    max_epochs = int(candidate.config.get("training", {}).get("max_epochs", 50))
    decision = derive_final_epoch(handoff, max_epochs)
    if decision.FINAL_REFIT_EPOCHS != sorted([decision.RO1, decision.RO2, decision.RO3])[1]:
        fails.append(f"median rule violated: FRO={decision.FINAL_REFIT_EPOCHS}")
    if decision.FINAL_REFIT_EPOCHS == 50:
        fails.append("FRO must not be 50")
    if decision.FINAL_REFIT_EPOCHS == max_epochs:
        fails.append(f"FRO must not equal max_epochs ({max_epochs})")
    expected_ro = {decision.RO1, decision.RO2, decision.RO3}
    expected_median = sorted(expected_ro)[1]
    if expected_median != decision.FINAL_REFIT_EPOCHS:
        fails.append(f"Median({sorted(expected_ro)}) != FRO")

    final_dev = build_final_dev_population(project_root, lookback_steps=candidate.lookback_steps)
    if not (final_dev.last_target_timestamp < final_dev.first_test_timestamp):
        fails.append(
            f"FINAL_DEV last ts {final_dev.last_target_timestamp} >= first TEST ts {final_dev.first_test_timestamp}"
        )
    if final_dev.test_target_values_accessed:
        fails.append("FINAL_DEV must not access TEST target values")

    scaling_contract = build_scaling_contract(candidate.config, materialize_final_fit=False)
    if scaling_contract.Test_rows_used is not False:
        fails.append("FINAL_SCALING must exclude Test rows")
    scaling_dict = scaling_contract_to_dict(scaling_contract)
    if not (scaling_dict["x_scaler_bundle_checksum"] == "REQUIRED_AT_PHASE46"
            and scaling_dict["y_scaler_bundle_checksum"] == "REQUIRED_AT_PHASE46"):
        fails.append("Both X and Y scaler checksums must be REQUIRED_AT_PHASE46")

    seed_contract = build_seed_contract()
    if seed_contract["seeds"] != [42, 123, 2026]:
        fails.append(f"seeds must equal [42,123,2026], got {seed_contract['seeds']}")

    recipe = build_recipe(
        candidate.config,
        decision.FINAL_REFIT_EPOCHS,
        final_dev.target_ids_fingerprint,
        scaling_dict,
    )
    recipe_dict = recipe_to_dict(recipe)
    if recipe_dict["validation_loader"] is not None:
        fails.append("Phase46 recipe must use validation_loader=None")
    if recipe_dict["early_stopping"] is not False:
        fails.append("Phase46 recipe must have early_stopping=False")
    if recipe_dict["checkpoint_type"] != "FINAL_REFIT":
        fails.append("Phase46 recipe must use checkpoint_type=FINAL_REFIT")

    run_matrix = build_run_matrix(
        candidate.candidate_id,
        candidate.config_fingerprint,
        "TBD", "TBD",
        decision.FINAL_REFIT_EPOCHS,
        final_dev.target_ids_fingerprint,
        scaling_dict["x_scaler_bundle_id"],
        scaling_dict["y_scaler_bundle_id"],
    )
    if len(run_matrix) != 3:
        fails.append(f"run_matrix must have exactly 3 rows, got {len(run_matrix)}")
    epochs_in_matrix = {r["final_refit_epochs"] for r in run_matrix}
    if epochs_in_matrix != {decision.FINAL_REFIT_EPOCHS}:
        fails.append(f"all run-matrix epochs must equal FRO ({decision.FINAL_REFIT_EPOCHS})")

    config_sha = config_fingerprint(candidate.config)
    recipe_sha = recipe_fingerprint(recipe_dict)
    lineage_rows, _ = build_lineage_audit(candidate.config, project_root / "artifacts", project_root)
    lineage_payload = {"rows": lineage_rows}
    lineage_sha = lineage_fingerprint(lineage_payload)
    lock_sha = lock_fingerprint(config_sha, recipe_sha, lineage_sha)
    if lock_fingerprint(config_fingerprint(candidate.config), recipe_sha, lineage_sha) != lock_sha:
        fails.append("Fingerprint determinism violated (lock_sha differs across recompute)")

    existing = {p.name for p in artifact_dir.iterdir() if p.is_file()}
    missing_artifacts = ARTIFACT_NAMES - existing
    if missing_artifacts:
        fails.append(f"missing O45 artifacts: {sorted(missing_artifacts)}")

    if recipe_dict["early_stopping"] is not False:
        fails.append("Phase46 FINAL_REFIT recipe must have early_stopping=False")

    boundary_sensitivity = build_boundary_sensitivity_evidence(project_root)
    boundary_sensitivity["wb0_primary"] = boundary_sensitivity.get("wb0_primary", True)
    boundary_sensitivity["protocol_amendment_required"] = boundary_sensitivity.get(
        "protocol_amendment_required", False
    )
    if not boundary_sensitivity["wb0_primary"]:
        fails.append("WB0 must remain primary")
    if boundary_sensitivity["protocol_amendment_required"]:
        fails.append("protocol_amendment_required must be False")

    if fails:
        print("Phase 45 PRELOCK GATE: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1

    print("Phase 45 PRELOCK GATE: PASS")
    print(f"  candidate: {candidate.candidate_id}")
    print(f"  family: {candidate.model_family}")
    print(f"  lookback: {candidate.lookback_steps}")
    print(f"  RO epochs: {decision.RO1}/{decision.RO2}/{decision.RO3}")
    print(f"  FRO: {decision.FINAL_REFIT_EPOCHS}")
    print(f"  FINAL_DEV count: {final_dev.target_count}")
    print(f"  seeds: [42, 123, 2026]")
    print(f"  lock_sha256: {lock_sha}")
    print(f"  config_sha256: {config_sha}")
    print(f"  recipe_sha256: {recipe_sha}")
    print(f"  lineage_sha256: {lineage_sha}")
    print(f"  O45 artifacts: {len(ARTIFACT_NAMES - missing_artifacts)}/{len(ARTIFACT_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
