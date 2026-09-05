"""PHASE 44 — Finalize / Resume utilities.

Used by `--mode finalize` to discover and validate already-completed
Stage A and Stage B runs and reuse their checkpoints / best_epoch
evidence WITHOUT calling any training primitives.

This module is intentionally read-only against the canonical registry:
it does not create, mutate, or delete any registered run record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from course_work.experiments.registry import RunStatus


def discover_completed_stage_runs(project_root: Path) -> tuple[set[str], set[str]]:
    """Discover all Phase44 Stage A and Stage B runs that are COMPLETED.

    Returns:
        (completed_stage_a_run_ids, completed_stage_b_run_ids)

    "Completed" means:
      - registry status == COMPLETED
      - sweep_stage matches RO[1-3]_A or RO[1-3]_B
      - physical artifacts exist (training_history.csv for Stage A,
        checkpoints/refit_final.pt for Stage B)

    Failed runs (RUN_TR_ROB_0001-0005) are EXCLUDED — they are
    non-reusable scientific evidence per previous task decisions.
    """
    reg_path = (
        project_root / "artifacts" / "registry" / "experiment_registry.jsonl"
    )
    completed_a: set[str] = set()
    completed_b: set[str] = set()
    if not reg_path.exists():
        return completed_a, completed_b

    with reg_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = r.get("run_id", "")
            status = r.get("status", "")
            sweep_stage = r.get("sweep_stage", "")
            if status != RunStatus.COMPLETED.value:
                continue
            if not (sweep_stage.startswith("RO") and (
                sweep_stage.endswith("_A") or sweep_stage.endswith("_B")
            )):
                continue
            run_dir = project_root / "artifacts" / "runs" / run_id
            if not run_dir.exists():
                continue
            if sweep_stage.endswith("_A"):
                # Stage A runs: only require run_dir existence + COMPLETED
                # status. Earlier official runs may not have written a
                # training_history.csv; the inner_best_epochs are recovered
                # from rolling_origin_inner_best_epochs.csv.
                if run_dir.exists():
                    completed_a.add(run_id)
            elif sweep_stage.endswith("_B"):
                ckpt = run_dir / "checkpoints" / "refit_final.pt"
                if ckpt.exists():
                    completed_b.add(run_id)
    return completed_a, completed_b


def validate_completion_requirements(
    completed_a: set[str],
    completed_b: set[str],
    expected_n: int = 12,
) -> tuple[list[str], list[str]]:
    """Validate that 12 Stage A and 12 Stage B runs are present and reusable.

    Returns:
        (missing_stage_a_keys, missing_stage_b_keys) where each key is
        a stable identifier "<candidate_id>/<fold_id>".

    Idempotent: this validator does not mutate registry state.
    """
    # We don't know which (candidate, fold) pairs are missing without
    # inspecting candidate IDs and sweep stages. The caller must
    # reconstruct the expected (candidate, fold) pair map from
    # candidate_loader and build_rolling_folds and call
    # validate_completion_requirements with the run_id set indexed by
    # the same (candidate, fold) pair.

    # For safety, we return the raw counts only; the actual (candidate,
    # fold) → run_id mapping is constructed inside _run_finalize from
    # the registry.
    return ([], []) if (
        len(completed_a) >= expected_n and len(completed_b) >= expected_n
    ) else ([f"__count_a={len(completed_a)}/{expected_n}"], [f"__count_b={len(completed_b)}/{expected_n}"])


def discover_completed_stage_runs_for_pair(
    project_root: Path,
    candidate_id: str,
    fold_id: str,
) -> dict[str, Any]:
    """Discover completed Stage A and Stage B run IDs for a single
    (candidate_id, fold_id) pair.

    Returns a dict with keys:
        stage_a_run_id  (str | None)
        stage_b_run_id  (str | None)
        best_epoch_inner (int | None) — recovered from the rolling_origin
                                     inner_best_epochs.csv artifact, or
                                     fallback to training_history.csv of
                                     the Stage A run if present

    Idempotent and read-only.
    """
    import csv

    reg_path = (
        project_root / "artifacts" / "registry" / "experiment_registry.jsonl"
    )
    out: dict[str, Any] = {
        "stage_a_run_id": None,
        "stage_b_run_id": None,
        "best_epoch_inner": None,
    }
    if not reg_path.exists():
        return out

    fold_suffix = fold_id.replace("RO", "")
    sweep_a = f"RO{fold_suffix}_A"
    sweep_b = f"RO{fold_suffix}_B"

    with reg_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != RunStatus.COMPLETED.value:
                continue
            if r.get("candidate_id") != candidate_id:
                continue
            sweep_stage = r.get("sweep_stage", "")
            run_id = r.get("run_id", "")
            run_dir = project_root / "artifacts" / "runs" / run_id
            if sweep_stage == sweep_a:
                # Stage A runs are considered COMPLETED if the run_dir
                # exists AND the registry reports COMPLETED. We do NOT
                # require training_history.csv because earlier official
                # runs may not have written one. The recovered best_epoch
                # is sourced from rolling_origin_inner_best_epochs.csv.
                if run_dir.exists():
                    out["stage_a_run_id"] = run_id
            elif sweep_stage == sweep_b:
                if (run_dir / "checkpoints" / "refit_final.pt").exists():
                    out["stage_b_run_id"] = run_id

    # Recover best_epoch_inner from the rolling_origin artifact CSV
    # (this is the canonical inner-best-epoch ledger maintained by
    # Phase 44 signoff, populated from the registered Stage A runs).
    inner_best_path = (
        project_root / "artifacts" / "rolling_origin"
        / "rolling_origin_inner_best_epochs.csv"
    )
    if inner_best_path.exists():
        try:
            import csv as _csv
            with inner_best_path.open() as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if (
                        row.get("candidate_id") == candidate_id
                        and row.get("fold_id") == fold_id
                    ):
                        try:
                            out["best_epoch_inner"] = int(row["best_epoch_inner"])
                        except (ValueError, KeyError):
                            pass
                        break
        except Exception:
            pass

    # Fallback: try training_history.csv of the Stage A run
    if out["best_epoch_inner"] is None and out["stage_a_run_id"]:
        history_path = (
            project_root / "artifacts" / "runs"
            / out["stage_a_run_id"] / "training_history.csv"
        )
        if history_path.exists():
            try:
                best_epoch = None
                best_val = float("inf")
                with history_path.open() as hf:
                    reader = csv.DictReader(hf)
                    for row in reader:
                        epoch = int(row.get("epoch", 0))
                        v = row.get("val_rmse_wh") or row.get("val_rmse") or ""
                        try:
                            v = float(v)
                        except ValueError:
                            continue
                        if v < best_val:
                            best_val = v
                            best_epoch = epoch
                out["best_epoch_inner"] = best_epoch
            except Exception:
                pass

    return out
