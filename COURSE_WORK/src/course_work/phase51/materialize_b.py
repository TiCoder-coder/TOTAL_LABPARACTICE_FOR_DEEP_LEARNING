"""Phase 51-B orchestrator — selection-contract freeze + alignment audit.

Executes Phase 51-B (governance + selection-contract + alignment + working table).

DOES NOT:
  - rank worst errors
  - inspect worst cases
  - compute Top-K
  - build casebook
  - generate worst-error figures
  - start Phase 51-C
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import alignment
from . import contract
from . import sources
from . import writers


# Default output paths
PHASE51_DIR = Path("artifacts") / "worst_error_analysis"


def build_selection_contract_payload() -> dict[str, Any]:
    """Build the frozen Phase 51 selection-contract payload (O51.5).

    Timestamps are pinned to the human-approved approval moment so that
    re-running materialize_phase51_b does NOT change the contract SHA256.
    The canonical contract SHA is held externally by the user.
    """
    # Frozen approval timestamps — DO NOT regenerate.
    # These exact strings reproduce the human-approved canonical SHA:
    #   ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4
    # Modifying them invalidates the contract SHA.
    _CREATED_AT_UTC = "2026-09-04T03:23:24.068851+00:00"
    _APPROVAL_AT_UTC = "2026-09-04T03:23:24.068855+00:00"
    return {
        "schema": "WORST_ERROR_SELECTION_CONTRACT-v1",
        "version": "WORST_ERROR_ANALYSIS-v1",
        "phase": 51,
        "contract_id": "phase51-selection-contract-2026-09-04-v1.13",
        "subphase": "51-B",
        "created_at_utc": _CREATED_AT_UTC,
        "approved_by": "human_owner",
        "approval_at_utc": _APPROVAL_AT_UTC,
        "contract_frozen": True,
        "selection_executed": False,
        "worst_error_ranking_executed": False,
        "individual_case_inspection": False,
        "phase52_authorized": False,
        # Upstream frozen state
        "seed_list": list(contract.SEEDS),
        "n_test": contract.N_TEST,
        "test_population_fingerprint": contract.TEST_POPULATION_FINGERPRINT,
        "residual_convention": contract.RESIDUAL_CONVENTION,
        "positive_residual_semantics": contract.POSITIVE_RESIDUAL_SEMANTICS,
        "negative_residual_semantics": contract.NEGATIVE_RESIDUAL_SEMANTICS,
        "zero_policy": contract.ZERO_POLICY,
        "cadence_minutes": contract.CADENCE_MINUTES,
        "local_context_radius_steps": contract.LOCAL_CONTEXT_RADIUS,
        "phase50_test_assignment_sha256": contract.PHASE50_TEST_ASSIGNMENT_SHA256,
        "phase50_train_threshold_sha256": contract.PHASE50_TRAIN_THRESHOLD_SHA256,
        "phase50_thresholds": contract.PHASE50_THRESHOLDS,
        # K values (HUMAN-APPROVED)
        "K_PER_SEED": contract.K_ABS_PER_SEED,
        "K_SHARED": contract.K_SHARED,
        "K_UNDER_PER_SEED": contract.K_UNDER_PER_SEED,
        "K_OVER_PER_SEED": contract.K_OVER_PER_SEED,
        "K_SHARED_SIGNED": contract.K_SHARED_SIGNED,
        # Primary ranking
        "primary_ranking_metric": contract.PRIMARY_RANKING_METRIC,
        "primary_ranking_direction": contract.PRIMARY_RANKING_DIRECTION,
        "tie_break": contract.TIE_BREAK,
        "random_tiebreak_forbidden": contract.RANDOM_TIEBREAK_FORBIDDEN,
        "manual_preselection_forbidden": contract.MANUAL_PRESELECTION_FORBIDDEN,
        "cherry_picking_forbidden": contract.CHERRY_PICKING_FORBIDDEN,
        # Ranking families
        "ranking_families": contract.RANKING_FAMILIES,
        # Regime contract (inherited from Phase 50, NOT recomputed)
        "regime_families": list(contract.REGIME_FAMILIES),
        "regime_labels": {
            k: list(v) for k, v in contract.REGIME_LABELS.items()
        },
        # Source checksums (loaded from disk at runtime)
        "source_checksums_to_verify": {
            "phase47_seed42_predictions": "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2",
            "phase47_seed123_predictions": "1bb55c445ffe132d3cfdc22e09439d77c76a2defd2f918bfb8f47029f666a08b",
            "phase47_seed2026_predictions": "bfb575357dd6a8e3a23fd08230c69f8e3d3297582ea73c16cf006601fcce79d8",
            "phase47_persistence_predictions": "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b",
            "phase49_residual_long": "8418a99110bfda7047bd27c49c1c7a9769313b1ce1fa6dc66925f5286d120038",
            "phase49_residual_wide": "931ff9109aef236d9517d8964d29168316c7d5903633441443c1ae5e71e8f8bd",
            "phase50_test_regime_assignment": "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac",
            "phase50_thresholds": "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109",
        },
        # Persistence / LSTM
        "persistence_context_only": contract.PERSISTENCE_CONTEXT_ONLY,
        "lstm_eligibility": contract.LSTM_ELIGIBILITY,
        "lstm_not_eligible_reason": contract.LSTM_NOT_ELIGIBLE_REASON,
        # Forbidden actions in Phase 51 (carry forward)
        "forbidden_actions": [
            "training",
            "new_test_inference",
            "checkpoint_loading",
            "scaler_fitting",
            "best_seed_selection",
            "ensemble_promotion",
            "prediction_correction",
            "recalibration",
            "test_derived_threshold_tuning",
            "phase50_regime_recomputation",
            "3N_iid_pooling",
            "manual_cherry_picking",
            "random_tiebreak_in_ranking",
            "phase52_attention_analysis",
        ],
    }


def materialize_phase51_b(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Execute Phase 51-B: governance + selection-contract freeze + alignment + working table.

    Returns a summary dict suitable for the processing log.
    """
    from ..utils.artifacts import get_project_root
    root = project_root if project_root is not None else get_project_root()

    # 1. Verify upstream SHA256
    source_checks = sources.verify_all_sources(root)

    # 2. Run alignment audit (pure-read, no ranking)
    audit = alignment.audit_alignment(root)
    audit_dict = audit.to_dict()

    # 3. Write alignment audit (JSON + CSV)
    audit_json_sha = writers.write_alignment_audit(audit_dict, root)
    audit_csv_sha = writers.write_alignment_audit_csv(audit_dict, root)

    # 4. Build and write frozen selection contract (O51.5)
    contract_payload = build_selection_contract_payload()
    contract_path = str((root / writers.PHASE51_DIR_REL / writers.SELECTION_CONTRACT_REL).relative_to(root))
    contract_sha = writers.write_selection_contract(contract_payload, root)

    # 5. Write selection contract fingerprint (O51.6)
    fp_sha = writers.write_selection_contract_fingerprint(contract_sha, contract_path, root)

    # 6. Build target-level working table (prerequisite for Phase 51-C)
    rows = alignment.build_target_level_working_table(root)
    working_sha = writers.write_working_table(
        rows, alignment.WORKING_TABLE_FIELDS, root,
    )

    # 7. Build Phase 51 manifest (O51.1)
    manifest_payload = {
        "phase": 51,
        "version": "WORST_ERROR_ANALYSIS-v1",
        "subphase": "51-B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": "FINAL_TEST_EVAL-v1",
        "test_population_sha256": contract.TEST_POPULATION_FINGERPRINT,
        "regime_assignment_sha256": contract.PHASE50_TEST_ASSIGNMENT_SHA256,
        "seed_list": list(contract.SEEDS),
        "K_ABS": contract.K_ABS_PER_SEED,
        "K_SHARED": contract.K_SHARED,
        "K_UNDER": contract.K_UNDER_PER_SEED,
        "K_OVER": contract.K_OVER_PER_SEED,
        "K_SHARED_SIGNED": contract.K_SHARED_SIGNED,
        "LOCAL_CONTEXT_RADIUS_STEPS": contract.LOCAL_CONTEXT_RADIUS,
        "new_inference": False,
        "new_training": False,
        "best_seed_selection": False,
        "regime_threshold_changes": False,
        "prediction_modification": False,
        "attention_extraction": False,
        "phase52_authorized": False,
        "worst_error_ranking_executed": False,
        "individual_case_inspection": False,
        "selection_contract_path": contract_path,
        "selection_contract_sha256": contract_sha,
        "selection_contract_fingerprint_sha256": fp_sha,
        "alignment_audit_json_sha256": audit_json_sha,
        "alignment_audit_csv_sha256": audit_csv_sha,
        "working_table_sha256": working_sha,
        "alignment_all_pass": audit.all_pass,
        "n_defects": len(audit.defects),
        "status": "PASS" if audit.all_pass else "FAIL",
    }
    manifest_sha = writers.write_phase51_manifest(manifest_payload, root)

    return {
        "phase": 51,
        "subphase": "51-B",
        "status": "PASS" if audit.all_pass else "FAIL",
        "source_checks": source_checks,
        "alignment_audit": audit_dict,
        "selection_contract_path": contract_path,
        "selection_contract_sha256": contract_sha,
        "selection_contract_fingerprint_sha256": fp_sha,
        "manifest_sha256": manifest_sha,
        "working_table_sha256": working_sha,
        "n_defects": len(audit.defects),
    }


def main(project_root=None) -> dict[str, Any]:
    """CLI entry point."""
    return materialize_phase51_b(project_root)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
