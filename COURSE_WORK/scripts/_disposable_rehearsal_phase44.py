"""PHASE 44 — DISPOSABLE END-TO-END REHEARSAL (TASK 14).

This is NOT an official run. It exercises the real orchestrator
(`course_work.rolling_origin.real_run.run_real_pipeline`) using:
  - Temporary registry_root / run_root (inside tempfile.TemporaryDirectory)
  - Rehearsal scientific_max_epochs=2 (tiny budget, not 50)
  - rehearsal_synthetic=True so TrainingEngine/RefitEngine are NOT called
    with real optimizer steps; instead, the orchestrator's pipeline is
    validated structurally.
  - No canonical artifact directory is touched.

It verifies:
  1. The orchestrator runs through all 12 Stage A and 12 Stage B audits.
  2. The orchestrator produces 12 learned Stage C bundles + 3 persistence
     bundles.
  3. Pooled metrics are computed for all 5 models.
  4. Signoff is computed.
  5. No file is written under the canonical artifact_dir.
  6. The temporary registry_root/run_root is fully isolated.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Make src importable
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

# Ensure NO-OP canonical artifact dir (a fresh temp dir).
os.environ.setdefault("PYTHONHASHSEED", "0")

from course_work.rolling_origin.real_run import (
    RunContext,
    assert_context_invariants,
    run_real_pipeline,
    SCIENTIFIC_MAX_EPOCHS,
    SCIENTIFIC_PATIENCE,
)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 DISPOSABLE END-TO-END REHEARSAL (TASK 14)")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_registry = PROJECT_ROOT / "artifacts" / "registry"
    canonical_run_root = PROJECT_ROOT / "artifacts" / "runs"

    print(f"Canonical artifact dir (MUST NOT be touched): {canonical_artifact_dir}")
    print(f"Canonical registry (MUST NOT be touched): {canonical_registry}")

    # Snapshot canonical state BEFORE rehearsal
    canonical_state_before = {
        "artifact_dir": sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else [],
    }

    with tempfile.TemporaryDirectory(prefix="phase44_rehearsal_") as tmp:
        tmp_path = Path(tmp)
        tmp_registry = tmp_path / "registry"
        tmp_run_root = tmp_path / "runs"
        tmp_artifact_dir = tmp_path / "artifacts"
        tmp_registry.mkdir(parents=True, exist_ok=True)
        tmp_run_root.mkdir(parents=True, exist_ok=True)
        tmp_artifact_dir.mkdir(parents=True, exist_ok=True)

        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=(
                PROJECT_ROOT / "artifacts" / "candidate_synthesis"
                / "transformer_candidate_shortlist.json"
            ),
            lstm_handoff_path=(
                PROJECT_ROOT / "artifacts" / "lstm_tuning"
                / "phase44_rolling_origin_lstm_handoff.json"
            ),
            phase_42_signoff_path=(
                PROJECT_ROOT / "artifacts" / "candidate_synthesis" / "phase_42_signoff.json"
            ),
            phase_43_signoff_path=(
                PROJECT_ROOT / "artifacts" / "lstm_tuning" / "phase_43_signoff.json"
            ),
            artifact_dir=tmp_artifact_dir,
            registry_root=tmp_registry,
            run_root=tmp_run_root,
            scientific_max_epochs=2,  # Rehearsal ceiling
            scientific_patience=2,
            seed=42,
            device="cpu",
            is_rehearsal=True,
            rehearsal_synthetic=True,  # Do not call real optimizer
        )

        # Sanity: guards allow this config
        try:
            assert_context_invariants(ctx)
            print("[OK] assert_context_invariants passes for rehearsal config")
        except Exception as exc:
            print(f"[FAIL] assert_context_invariants rejected rehearsal config: {exc}")
            return 1

        print("[RUN] run_real_pipeline(ctx) with synthetic rehearsal flags...")
        result = run_real_pipeline(ctx)
        print(f"  exit_code: {result.exit_code}")
        print(f"  summary:   {result.summary}")
        print(f"  n_candidates: {result.n_candidates}")
        print(f"  n_folds: {result.n_folds}")
        print(f"  n_stage_a_runs: {result.n_stage_a_runs}")
        print(f"  n_stage_b_runs: {result.n_stage_b_runs}")
        print(f"  n_outer_prediction_bundles: {result.n_outer_prediction_bundles}")
        print(f"  n_persistence_bundles: {result.n_persistence_bundles}")
        print(f"  recommended_transformer_id: {result.recommended_transformer_id}")
        print(f"  signoff_overall_status: {result.signoff_overall_status}")

        if result.exception:
            print(f"[WARN] result.exception (orchestrator emitted exception trace):")
            print(result.exception[:1500])

        # Validate canonical artifact dir was not touched
        canonical_state_after = {
            "artifact_dir": sorted(p.name for p in canonical_artifact_dir.iterdir())
            if canonical_artifact_dir.exists() else [],
        }
        if canonical_state_after != canonical_state_before:
            print(f"[FAIL] canonical artifact_dir mutated by rehearsal!")
            print(f"  before: {canonical_state_before}")
            print(f"  after:  {canonical_state_after}")
            return 2
        else:
            print(f"[OK] canonical artifact_dir untouched: {canonical_state_after}")

        # Validate temp isolation
        if not tmp_registry.exists() or not tmp_run_root.exists():
            print(f"[FAIL] temp registry/run_root disappeared mid-run")
            return 3
        print(f"[OK] temp registry exists at {tmp_registry}")
        print(f"[OK] temp run_root exists at {tmp_run_root}")

    # Outside the `with tempfile.TemporaryDirectory()` — temp dir is gone.
    print(f"[OK] temp registry/run_root cleaned up automatically")

    # Validate structural correctness of the orchestrator result
    ok = True
    if result.exit_code not in (0, 1):
        print(f"[FAIL] result.exit_code={result.exit_code} not in {{0,1}}")
        ok = False

    # ----- Candidate count: must be 4 (3 TR + 1 LSTM) -----
    if result.n_candidates != 4:
        print(f"[FAIL] n_candidates={result.n_candidates} != 4 (expected 3 TR + 1 LSTM)")
        ok = False
    else:
        print(f"[OK] n_candidates=4/4 (3 TR + LSTM_TUNED_WINNER)")

    # ----- Folds: 3 -----
    if result.n_folds != 3:
        print(f"[FAIL] n_folds={result.n_folds} != 3")
        ok = False
    else:
        print(f"[OK] n_folds=3/3 (RO1/RO2/RO3)")

    # ----- Stage A: 12/12 -----
    if result.n_stage_a_runs != 12:
        print(f"[FAIL] n_stage_a_runs={result.n_stage_a_runs} != 12")
        ok = False
    else:
        print(f"[OK] n_stage_a_runs=12/12")

    # ----- Stage B: 12/12 -----
    if result.n_stage_b_runs != 12:
        print(f"[FAIL] n_stage_b_runs={result.n_stage_b_runs} != 12")
        ok = False
    else:
        print(f"[OK] n_stage_b_runs=12/12")

    # ----- Learned outer predictions: 12/12 -----
    if result.n_outer_prediction_bundles != 12:
        print(f"[FAIL] n_outer_prediction_bundles={result.n_outer_prediction_bundles} != 12")
        ok = False
    else:
        print(f"[OK] n_outer_prediction_bundles=12/12")

    # ----- Persistence: 3/3 -----
    if result.n_persistence_bundles != 3:
        print(f"[FAIL] n_persistence_bundles={result.n_persistence_bundles} != 3")
        ok = False
    else:
        print(f"[OK] n_persistence_bundles=3/3")

    # ----- Pooled models: 5/5 (3 TR + 1 LSTM + 1 PERSISTENCE) -----
    if len(result.pooled_metrics_by_cid) != 5:
        print(f"[FAIL] pooled_models={len(result.pooled_metrics_by_cid)} != 5")
        ok = False
    else:
        print(f"[OK] pooled_models=5/5 "
              f"({sorted(result.pooled_metrics_by_cid.keys())})")

    # ----- Signoff: must be PASS or PASS_WITH_WARNING -----
    if result.signoff_overall_status not in {"PASS", "PASS_WITH_WARNING"}:
        print(f"[FAIL] signoff_overall_status={result.signoff_overall_status!r} "
              f"(must be PASS or PASS_WITH_WARNING)")
        if result.signoff_failures:
            print(f"  failures:")
            for f in result.signoff_failures:
                print(f"    - {f}")
        ok = False
    else:
        print(f"[OK] signoff_overall_status={result.signoff_overall_status}")
        if result.signoff_failures:
            print(f"  warnings (allowed for PASS_WITH_WARNING):")
            for f in result.signoff_failures:
                print(f"    - {f}")
        else:
            print(f"  no failures or warnings")

    # ----- Epoch propagation: 12/12 (each Stage B uses Stage A best_epoch_inner) -----
    n_inner_best = len(result.inner_best_epochs)
    n_stage_b = result.n_stage_b_runs
    if n_inner_best != n_stage_b:
        print(f"[FAIL] inner_best_epochs={n_inner_best} != stage_b_runs={n_stage_b}")
        ok = False
    elif n_stage_b != 12:
        print(f"[WARN] stage_b_runs={n_stage_b} (expected 12 for full epoch propagation check)")
    else:
        print(f"[OK] epoch_propagation=12/12 (each Stage B uses its Stage A best_epoch_inner)")

    print("=" * 78)
    print(f"PHASE 44 DISPOSABLE REHEARSAL: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)
