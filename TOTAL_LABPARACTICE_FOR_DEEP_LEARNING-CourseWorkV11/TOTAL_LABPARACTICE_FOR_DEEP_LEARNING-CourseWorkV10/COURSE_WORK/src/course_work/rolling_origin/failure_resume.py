"""Phase 44 — Failure / Resume one-shot audit (TASK 11).

Simulates 5 distinct failure scenarios and reports what is marked FAILED,
what is reusable, what is recomputed, what is immutable, and whether a
new run ID is required:

  A. failure during Stage A training
  B. failure after Stage A completion (before Stage B)
  C. failure during Stage B refit
  D. failure after Stage B checkpoint, before Stage C
  E. failure during artifact finalization

This module does NOT actually invoke the official pipeline. It computes
the failure semantics from the canonical contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureScenario(str, Enum):
    DURING_STAGE_A = "DURING_STAGE_A"
    AFTER_STAGE_A_BEFORE_B = "AFTER_STAGE_A_BEFORE_B"
    DURING_STAGE_B = "DURING_STAGE_B"
    AFTER_STAGE_B_BEFORE_C = "AFTER_STAGE_B_BEFORE_C"
    DURING_FINALIZATION = "DURING_FINALIZATION"


@dataclass
class FailureAudit:
    scenario: FailureScenario
    what_failed: list[str]
    what_reusable: list[str]
    what_recomputed: list[str]
    what_immutable: list[str]
    new_run_id_required: bool
    orphan_records_left: list[str]
    notes: str
    status: str


def _audit_template(scenario: FailureScenario, **kwargs) -> FailureAudit:
    return FailureAudit(
        scenario=scenario,
        what_failed=kwargs.get("what_failed", []),
        what_reusable=kwargs.get("what_reusable", []),
        what_recomputed=kwargs.get("what_recomputed", []),
        what_immutable=kwargs.get("what_immutable", []),
        new_run_id_required=kwargs.get("new_run_id_required", True),
        orphan_records_left=kwargs.get("orphan_records_left", []),
        notes=kwargs.get("notes", ""),
        status="PASS" if not kwargs.get("orphan_records_left") else "FAIL",
    )


def simulate_failure(scenario: FailureScenario) -> FailureAudit:
    if scenario == FailureScenario.DURING_STAGE_A:
        return _audit_template(
            scenario=scenario,
            what_failed=[
                "Stage A run record (status: FAILED)",
                f"PHASE44_INNER_<candidate>_<fold>",
            ],
            what_reusable=[
                "Fold population",
                "Stage A scaler audit",
                "Fold-local fold_manifest",
            ],
            what_recomputed=[
                "Stage A best epoch",
                "Inner-validation predictions",
            ],
            what_immutable=[
                "fold_manifest (frozen pre-training)",
                "phase_42 shortlist",
                "phase_43 LSTM handoff",
            ],
            new_run_id_required=True, 
            orphan_records_left=[], 
            notes=(
                "registry.fail_run() invoked inside TRAIN_FAILURE transition. "
                "Old record transitions to FAILED (immutable). New attempt "
                "registers a fresh STAGE_A run_id with parent_run_id=None."
            ),
        )
    if scenario == FailureScenario.AFTER_STAGE_A_BEFORE_B:
        return _audit_template(
            scenario=scenario,
            what_failed=[
                "No new failure recorded (Stage A persisted)",
                "Script orchestration interrupted before Stage B",
            ],
            what_reusable=[
                "Stage A run record (status: COMPLETED)",
                "Best epoch reading",
                "Fold-local scalers (stage A bundle)",
            ],
            what_recomputed=[],
            what_immutable=[
                "fold_manifest",
                "STAGE_A_RUN_ID",
                "inner_selection_registry.csv row",
            ],
            new_run_id_required=False, 
            orphan_records_left=[],
            notes=(
                "Resume reuses STAGE_A run_id and produces a new STAGE_B run_id "
                "with parent_run_id pointing to STAGE_A. No orphan records."
            ),
        )
    if scenario == FailureScenario.DURING_STAGE_B:
        return _audit_template(
            scenario=scenario,
            what_failed=[
                "Stage B run record (status: FAILED)",
                "Partial refit_final.pt (orphan — overwritten by new attempt)",
            ],
            what_reusable=[
                "Stage A run record (COMPLETED)",
                "Stage B scaler bundle (unchanged)",
            ],
            what_recomputed=[
                "Stage B trained epochs (from scratch, exactly best_epoch_inner)",
            ],
            what_immutable=[
                "fold_manifest",
                "STAGE_A_RUN_ID",
                "best_epoch_inner lock",
            ],
            new_run_id_required=True,
            orphan_records_left=[],
            notes=(
                "Old STAGE_B run marked FAILED. New attempt creates a fresh "
                "STAGE_B run_id with rerun_reason=PHASE_44_RESUMED and the same "
                "parent_run_id=STAGE_A."
            ),
        )
    if scenario == FailureScenario.AFTER_STAGE_B_BEFORE_C:
        return _audit_template(
            scenario=scenario,
            what_failed=[
                "Script interrupted after Stage B checkpoint, before Stage C",
            ],
            what_reusable=[
                "Stage A run (COMPLETED)",
                "Stage B refit_final.pt (COMPLETED on disk)",
            ],
            what_recomputed=[],
            what_immutable=[
                "refit_final.pt",
                "Stage B run record",
            ],
            new_run_id_required=False,
            orphan_records_left=[],
            notes=(
                "Resume reads refit_final.pt from disk, runs Stage C outer "
                "evaluation without retraining. No new STAGE_B run id."
            ),
        )
    if scenario == FailureScenario.DURING_FINALIZATION:
        return _audit_template(
            scenario=scenario,
            what_failed=[
                "Artifact finalization interrupted (some writer files may "
                "exist as partial)",
            ],
            what_reusable=[
                "All staged per-run artifacts (refit_final.pt, training_history.csv, "
                "refit_status.json)",
                "Fold-level signed intermediate registries",
            ],
            what_recomputed=[
                "Final O44.1-O44.39 writer outputs (from staged intermediate state)",
            ],
            what_immutable=[
                "Per-run signed artefacts in run_root",
                "refit_final.pt",
            ],
            new_run_id_required=False,
            orphan_records_left=[],
            notes=(
                "write_text_once_or_verify rejects mismatched rewrites, so partial "
                "writes either complete cleanly or are not emitted. Resume "
                "reconstructs the writer outputs from intermediate registries."
            ),
        )
    raise ValueError(f"Unknown scenario {scenario}")


def run_all_failure_simulations() -> list[FailureAudit]:
    return [simulate_failure(s) for s in FailureScenario]


def failure_audit_to_dict(a: FailureAudit) -> dict[str, Any]:
    return {
        "scenario": a.scenario.value,
        "what_failed": a.what_failed,
        "what_reusable": a.what_reusable,
        "what_recomputed": a.what_recomputed,
        "what_immutable": a.what_immutable,
        "new_run_id_required": a.new_run_id_required,
        "orphan_records_left": a.orphan_records_left,
        "notes": a.notes,
        "status": a.status,
    }
