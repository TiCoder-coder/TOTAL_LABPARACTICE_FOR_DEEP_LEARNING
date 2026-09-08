"""Phase 44 — Stage A/B registration payload builder (TASK 4 / TASK 8).

`build_stage_a_payload` and `build_stage_b_payload` construct the exact
registry-shaped payload that `ExperimentRegistry.register_run` would accept
for each (candidate, fold) pair, without actually calling register_run.

These payloads are returned alongside the run-id reservation so the dry-run
preflight and rehearsal can prove the registration contract is satisfiable
for ALL 12 + 12 = 24 expected jobs.

The payload includes:
  candidate_id           (string)
  fold_id                (e.g. "RO1")
  stage                  ("A" inner validation | "B" exact-epoch refit)
  seed                   (always 42 for Phase 44)
  config_fingerprint     (the candidate's frozen cfg fingerprint)
  population_fingerprint (fold-level fingerprint)
  scaler_fingerprint     (per-stage audit's bundle checksum)
  parent_run_id          (Stage A: None; Stage B: stage_a_run_id)
  rerun_reason            ("PHASE_44_INITIAL" | "PHASE_44_RESUMED")
  notes                  (human-readable)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from course_work.rolling_origin.candidate_loader import CandidateSpec
from course_work.rolling_origin.folds import FoldDefinition


RERUN_REASON_INITIAL = "PHASE_44_INITIAL"
RERUN_REASON_RESUMED = "PHASE_44_RESUMED"


@dataclass
class StageRegistrationPayload:
    candidate_id: str
    fold_id: str
    stage: str
    seed: int
    config_fingerprint: str
    population_fingerprint: str
    scaler_fingerprint: str
    parent_run_id: str | None
    rerun_reason: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _stage_run_id(candidate_id: str, fold_id: str, stage: str) -> str:
    short = candidate_id.split("_")[0] + "_" + candidate_id.split("_")[1]
    return f"RUN_RO_{stage}_{short}_{fold_id}"


def build_stage_a_payload(
    *,
    candidate: CandidateSpec,
    fold: FoldDefinition,
    population_fingerprint: str,
    scaler_a_fingerprint: str,
) -> StageRegistrationPayload:
    return StageRegistrationPayload(
        candidate_id=candidate.candidate_id,
        fold_id=str(fold.fold_id),
        stage="A",
        seed=42,
        config_fingerprint=candidate.config_fingerprint,
        population_fingerprint=population_fingerprint,
        scaler_fingerprint=scaler_a_fingerprint,
        parent_run_id=None,
        rerun_reason=RERUN_REASON_INITIAL,
        notes=(
            f"Stage A: nested inner epoch selection for "
            f"{candidate.candidate_id} on fold {fold.fold_id}."
        ),
    )


def build_stage_b_payload(
    *,
    candidate: CandidateSpec,
    fold: FoldDefinition,
    population_fingerprint: str,
    scaler_b_fingerprint: str,
    stage_a_run_id: str,
) -> StageRegistrationPayload:
    return StageRegistrationPayload(
        candidate_id=candidate.candidate_id,
        fold_id=str(fold.fold_id),
        stage="B",
        seed=42,
        config_fingerprint=candidate.config_fingerprint,
        population_fingerprint=population_fingerprint,
        scaler_fingerprint=scaler_b_fingerprint,
        parent_run_id=stage_a_run_id,
        rerun_reason=RERUN_REASON_INITIAL,
        notes=(
            f"Stage B: fresh full-history exact-epoch refit for "
            f"{candidate.candidate_id} on fold {fold.fold_id}; "
            f"parent_run_id={stage_a_run_id}."
        ),
    )


def build_all_payloads(
    *,
    candidates: list[CandidateSpec],
    folds: list[FoldDefinition],
    scaler_a_audits: list,
    scaler_b_audits: list,
) -> tuple[list[StageRegistrationPayload], list[StageRegistrationPayload]]:
    """Build ALL Stage A and Stage B registration payloads (12 each).

    `scaler_a_audits` and `scaler_b_audits` are lists of ScalerFitAudit
    (or ScalerProbeResult) rows; each must contain (candidate_id, fold_id,
    stage, bundle_checksum).

    Returns (stage_a_payloads, stage_b_payloads).
    """
    if len(scaler_a_audits) != len(candidates) * len(folds):
        raise ValueError(
            f"Expected {len(candidates) * len(folds)} Stage A audits, "
            f"got {len(scaler_a_audits)}"
        )
    if len(scaler_b_audits) != len(candidates) * len(folds):
        raise ValueError(
            f"Expected {len(candidates) * len(folds)} Stage B audits, "
            f"got {len(scaler_b_audits)}"
        )

    a_by_key: dict[tuple[str, str], str] = {
        (row.candidate_id, row.fold_id): row.bundle_checksum
        for row in scaler_a_audits
    }
    b_by_key: dict[tuple[str, str], str] = {
        (row.candidate_id, row.fold_id): row.bundle_checksum
        for row in scaler_b_audits
    }

    a_payloads: list[StageRegistrationPayload] = []
    b_payloads: list[StageRegistrationPayload] = []
    for f in folds:
        for c in candidates:
            key = (c.candidate_id, str(f.fold_id))
            a_fp = a_by_key[key]
            b_fp = b_by_key[key]
            stage_a_run_id = _stage_run_id(c.candidate_id, str(f.fold_id), "A")
            a_payloads.append(
                build_stage_a_payload(
                    candidate=c,
                    fold=f,
                    population_fingerprint=f.fold_population_fingerprint,
                    scaler_a_fingerprint=a_fp,
                )
            )
            b_payloads.append(
                build_stage_b_payload(
                    candidate=c,
                    fold=f,
                    population_fingerprint=f.fold_population_fingerprint,
                    scaler_b_fingerprint=b_fp,
                    stage_a_run_id=stage_a_run_id,
                )
            )
    return a_payloads, b_payloads
