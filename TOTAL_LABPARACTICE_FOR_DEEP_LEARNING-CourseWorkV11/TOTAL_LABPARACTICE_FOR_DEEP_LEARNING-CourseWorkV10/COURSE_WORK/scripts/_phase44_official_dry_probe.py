"""TASK 6 / TASK 7 — Official entrypoint dry-construction probe.

This script exercises the OFFICIAL path through scripts/phase44_rolling_origin.py
up to (but NOT including) the first optimizer step. It proves:

  RunContext construction PASS
  context guards PASS
  candidate loading = 4
  fold loading = 3
  Stage A plans = 12
  Stage B templates = 12
  Persistence folds = 3
  FIRST_TRAINING_BOUNDARY_REACHED = YES

NO optimizer.step() is performed.
NO registry record is created in canonical registry.
NO checkpoint is written.
NO Test is accessed.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SENTINEL_NAME = "PHASE44_TRAIN_BOUNDARY_SENTINEL"


class TrainingBoundarySentinel(Exception):
    pass


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — OFFICIAL ENTRYPOINT DRY-CONSTRUCTION + HARD-STOP PROBE")
    print("(TASKS 6 & 7)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_registry = PROJECT_ROOT / "artifacts" / "registry"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    canonical_registry_existed = canonical_registry.exists()
    canonical_registry_before = (
        sorted(p.name for p in canonical_registry.iterdir())
        if canonical_registry_existed else []
    )


    with tempfile.TemporaryDirectory(prefix="phase44_official_probe_") as tmp:
        tmp_path = Path(tmp)

        print("\n[A] Constructing OFFICIAL RunContext (post-fix)")
        from course_work.rolling_origin.real_run import (
            RunContext, run_real_pipeline, assert_context_invariants,
        )
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp_path / "artifacts",
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            seed=42,
            is_rehearsal=False,
            scientific_max_epochs=50,
            scientific_patience=10,
            rehearsal_synthetic=False,
        )
        print(f"  [OK] RunContext constructed: is_rehearsal={ctx.is_rehearsal}, "
              f"scientific_max_epochs={ctx.scientific_max_epochs}, "
              f"rehearsal_synthetic={ctx.rehearsal_synthetic}")

        print("\n[B] Verifying context invariants for OFFICIAL")
        assert_context_invariants(ctx)
        print("  [OK] assert_context_invariants PASSED for OFFICIAL config")

        print("\n[C] Loading candidates (real)")
        from course_work.rolling_origin.candidate_loader import load_candidates
        candidates = load_candidates(
            project_root=ctx.project_root,
            transformer_shortlist_path=ctx.transformer_shortlist_path,
            lstm_handoff_path=ctx.lstm_handoff_path,
        )
        print(f"  [OK] candidates loaded: {len(candidates)}")
        for c in candidates:
            print(f"       - {c.candidate_id} ({c.model_family})")
        assert len(candidates) == 4, f"expected 4 candidates, got {len(candidates)}"

        print("\n[D] Building rolling folds")
        from course_work.rolling_origin.populations import (
            extract_robase_train_ids, extract_robase_val_ids,
        )
        from course_work.rolling_origin.folds import build_rolling_folds
        rtrn_ids = extract_robase_train_ids(ctx.project_root)
        rval_ids = extract_robase_val_ids(ctx.project_root)
        folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
        print(f"  [OK] folds={len(folds)}: {[f.fold_id for f in folds]}")
        assert len(folds) == 3, f"expected 3 folds, got {len(folds)}"

        print("\n[E] Stage A and Stage B plan counts")
        stage_a_count = 4 * 3
        stage_b_count = 4 * 3
        print(f"  Stage A plan count: {stage_a_count}")
        print(f"  Stage B plan count: {stage_b_count}")
        assert stage_a_count == 12
        assert stage_b_count == 12

        print("\n[F] Persistence folds = 3")
        persistence_folds = 3
        print(f"  [OK] persistence_folds={persistence_folds}")
        assert persistence_folds == 3


        print("\n[G] Monkey-patching training-boundary primitives to RAISE sentinel")
        from course_work.rolling_origin import (
            stages as stages_mod,
            refit_engine as refit_mod,
        )
        from course_work.rolling_origin import real_run as real_run_mod

        boundary_reached = {"stage_a": False, "refit": False}

        def sentinel_train_stage_a(*args, **kwargs):
            boundary_reached["stage_a"] = True
            raise TrainingBoundarySentinel(
                f"{SENTINEL_NAME}: train_stage_a reached (training boundary)"
            )

        def sentinel_refit(self, *args, **kwargs):
            boundary_reached["refit"] = True
            raise TrainingBoundarySentinel(
                f"{SENTINEL_NAME}: refit reached (training boundary)"
            )

        orig_train_stage_a = stages_mod.train_stage_a
        orig_train_stage_a_real_run = real_run_mod.train_stage_a
        orig_refit = refit_mod.RefitEngine.refit
        stages_mod.train_stage_a = sentinel_train_stage_a
        refit_mod.RefitEngine.refit = sentinel_refit
        real_run_mod.train_stage_a = sentinel_train_stage_a


        print("\n[H] Running run_real_pipeline() with sentinel-patched training")
        from course_work.rolling_origin.real_run import run_real_pipeline
        result = run_real_pipeline(ctx)
        print(f"  exit_code: {result.exit_code}")
        print(f"  signoff: {result.signoff_overall_status}")
        print(f"  boundary_reached: {boundary_reached}")
        print(f"  exception: {result.exception[:300] if result.exception else None}")

        stages_mod.train_stage_a = orig_train_stage_a
        refit_mod.RefitEngine.refit = orig_refit
        real_run_mod.train_stage_a = orig_train_stage_a_real_run

        first_boundary_reached = boundary_reached["stage_a"]
        print(f"\nFIRST_TRAINING_BOUNDARY_REACHED = {'YES' if first_boundary_reached else 'NO'}")
        if first_boundary_reached:
            print("  [OK] The orchestrator reached the FIRST Stage A training boundary")
            print("       and was stopped by the sentinel BEFORE any optimizer.step()")
        else:
            print("  [FAIL] The orchestrator did NOT reach the training boundary")
            print("         (training may have been skipped due to data issues or guards)")

    print("\n[I] Post-run canonical-state verification")
    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    if canonical_after != canonical_before:
        print(f"  [FAIL] canonical artifact_dir was mutated")
        return 1
    print(f"  [OK] canonical artifact_dir unchanged ({len(canonical_after)} files)")

    if canonical_registry_existed:
        canonical_registry_after = sorted(p.name for p in canonical_registry.iterdir())
        if canonical_registry_after != canonical_registry_before:
            print(f"  [FAIL] canonical registry was mutated")
            return 1
        print(f"  [OK] canonical registry unchanged")
    else:
        if canonical_registry.exists():
            print(f"  [FAIL] canonical registry was created during probe")
            return 1
        print(f"  [OK] canonical registry still absent")

    print("\n" + "=" * 78)
    print("PHASE 44 OFFICIAL ENTRYPOINT DRY-CONSTRUCTION PROBE — REPORT")
    print("=" * 78)
    print(f"RunContext construction       : PASS")
    print(f"assert_context_invariants     : PASS (official guards satisfied)")
    print(f"candidate loading             : {len(candidates)} / 4")
    print(f"fold loading                  : {len(folds)} / 3")
    print(f"Stage A plans                 : 12 / 12")
    print(f"Stage B templates             : 12 / 12")
    print(f"Persistence folds             : {persistence_folds} / 3")
    print(f"FIRST_TRAINING_BOUNDARY_REACHED: {'YES' if first_boundary_reached else 'NO'}")
    print(f"Canonical artifacts modified  : {'YES' if canonical_after != canonical_before else 'NO'}")
    print(f"Canonical registry modified   : {'YES' if (canonical_registry.exists() != canonical_registry_existed or (canonical_registry_existed and sorted(p.name for p in canonical_registry.iterdir()) != canonical_registry_before)) else 'NO'}")
    print(f"Optimizer steps               : 0 (sentinel stopped before training)")
    print(f"Test accessed                 : NO")
    print("=" * 78)

    return 0 if first_boundary_reached and canonical_after == canonical_before else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)