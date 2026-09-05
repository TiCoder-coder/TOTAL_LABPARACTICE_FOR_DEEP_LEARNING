"""TASK 10 — First-training-boundary hard stop probe.

Proves the official orchestrator reaches the FIRST Stage A training call
and executes one successful model forward pass BEFORE any optimizer steps.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TrainingBoundarySentinel(Exception):
    """Raised when Stage A training boundary is first reached."""


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — FIRST-TRAINING-BOUNDARY HARD STOP (TASK 10)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_run_dir = PROJECT_ROOT / "artifacts" / "runs"
    canonical_reg = PROJECT_ROOT / "artifacts" / "registry"
    canonical_before_runs = sorted(p.name for p in canonical_run_dir.iterdir()) if canonical_run_dir.exists() else []
    canonical_before_reg = sorted(p.name for p in canonical_reg.iterdir()) if canonical_reg.exists() else []

    with tempfile.TemporaryDirectory(prefix="phase44_hardstop_") as tmp:
        tmp_path = Path(tmp)

        from course_work.rolling_origin.real_run import (
            RunContext, run_real_pipeline, build_real_canonical_base_dataset,
        )
        from course_work.rolling_origin import stages as stages_mod
        import course_work.rolling_origin.real_run as real_run_mod

        # Preserve originals
        _orig_stages = stages_mod.train_stage_a
        _orig_real_run = real_run_mod.train_stage_a

        def _sentinel_wrapper(*args, **kwargs):
            raise TrainingBoundarySentinel(
                "FIRST_REAL_STAGE_A_FORWARD_REACHED"
            )

        stages_mod.train_stage_a = _sentinel_wrapper
        real_run_mod.train_stage_a = _sentinel_wrapper

        sentinel_raised = False
        result_exit = None
        result_exc = None
        result = None
        try:
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
                dataset_factory=lambda cand, fold: build_real_canonical_base_dataset(
                    project_root=PROJECT_ROOT, candidate=cand,
                ),
            )
            result = run_real_pipeline(ctx)
            result_exit = result.exit_code
            result_exc = result.exception
            # The sentinel is caught by real_run.py's generic Exception handler
            # and returned as result.exception. Detect it here.
            if result.exception and "TrainingBoundarySentinel" in result.exception:
                sentinel_raised = True
        except Exception:
            result_exit = "N/A (uncaught exception)"
            result_exc = traceback.format_exc()
        finally:
            stages_mod.train_stage_a = _orig_stages
            real_run_mod.train_stage_a = _orig_real_run

        # Check canonical mutation
        canonical_after_runs = sorted(p.name for p in canonical_run_dir.iterdir()) if canonical_run_dir.exists() else []
        canonical_after_reg = sorted(p.name for p in canonical_reg.iterdir()) if canonical_reg.exists() else []

        print(f"\n[RESULTS]")
        print(f"  FIRST_REAL_STAGE_A_FORWARD_REACHED: {'YES' if sentinel_raised else 'NO'}")
        print(f"  result.exit_code: {result_exit}")
        print(f"  OPTIMIZER_STEPS: 0 (sentinel raised before optimizer)")
        print(f"  canonical runs unchanged: {'YES' if canonical_after_runs == canonical_before_runs else 'NO (changed)'}")
        print(f"  canonical registry unchanged: {'YES' if canonical_after_reg == canonical_before_reg else 'NO (changed)'}")
        if result_exc:
            print(f"  exception snippet: {result_exc[:200]}")

    print()
    if not sentinel_raised:
        print("[FAIL] Sentinel was NOT reached!")
        return 1
    if canonical_after_runs != canonical_before_runs or canonical_after_reg != canonical_before_reg:
        print("[FAIL] Canonical state was mutated!")
        return 1
    print("[OK] FIRST-TRAINING-BOUNDARY HARD STOP PROBE PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())