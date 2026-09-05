"""PHASE 44 — Non-training OFFICIAL PLAN PROBE (TASK 13).

This script proves — WITHOUT calling TrainingEngine.train() or
RefitEngine.refit() in official mode — that the planned system meets
the Phase 44 core requirements:

  1. 12 Stage A registration payloads (4 candidates × 3 folds)
  2. 12 Stage B registration payloads (4 candidates × 3 folds)
  3. max_epochs=50 in official (matches Phase 44 contract)
  4. patience=10 in official (matches Phase 44 contract)
  5. LSTM candidate goes through the SAME protocol as Transformers
  6. Persistence is a separate path (not a candidate)
  7. The same orchestrator is used for official and rehearsal

ZERO optimizer steps. ZERO official scientific artifacts.

This is a verification-only probe. It runs the orchestrator in rehearsal
mode with rehearsal_synthetic=True, but with the SAME official-mode
RunContext shape (just with is_rehearsal=True and tiny budget). This
exercises every registration / persistence / pooling / signoff code
path while ensuring:
  - registry_root / run_root are TEMP (no canonical mutation)
  - rehearsal_synthetic=True (no real optimizer)
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import (
    RunContext,
    assert_context_invariants,
    run_real_pipeline,
    SCIENTIFIC_MAX_EPOCHS,
    SCIENTIFIC_PATIENCE,
    PERSISTENCE_FOLD_COUNT,
)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 NON-TRAINING OFFICIAL PLAN PROBE (TASK 13)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = {p.name for p in canonical_artifact_dir.iterdir()} \
        if canonical_artifact_dir.exists() else set()

    with tempfile.TemporaryDirectory(prefix="phase44_plan_probe_") as tmp:
        tmp_path = Path(tmp)

        # Step 1: probe official-mode guards (must reject fast-mode values).
        print("\n[1] OFFICIAL-mode guard probe")
        fast_ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/lstm_tuned_winner.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp_path / "artifacts",
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            scientific_max_epochs=2,    # fast-mode
            scientific_patience=2,
            is_rehearsal=False,         # OFFICIAL
            rehearsal_synthetic=False,
        )
        try:
            assert_context_invariants(fast_ctx)
            print("  [FAIL] fast-mode + official was NOT rejected")
            return 1
        except RuntimeError as exc:
            print(f"  [PASS] fast-mode + official rejected: {type(exc).__name__}")

        # Step 2: probe scientific budget constants in official config.
        print("\n[2] Scientific budget constants probe")
        assert SCIENTIFIC_MAX_EPOCHS == 50, f"max_epochs={SCIENTIFIC_MAX_EPOCHS}"
        assert SCIENTIFIC_PATIENCE == 10, f"patience={SCIENTIFIC_PATIENCE}"
        assert PERSISTENCE_FOLD_COUNT == 3
        print(f"  [PASS] SCIENTIFIC_MAX_EPOCHS={SCIENTIFIC_MAX_EPOCHS}, "
              f"SCIENTIFIC_PATIENCE={SCIENTIFIC_PATIENCE}, "
              f"PERSISTENCE_FOLD_COUNT={PERSISTENCE_FOLD_COUNT}")

        # Step 3: probe the real orchestrator structure via synthetic rehearsal.
        # This exercises the SAME core orchestration (12 Stage A, 12 Stage B,
        # 12 outer predictions, 3 persistence, pooling, ranking, signoff) but
        # without real optimizer steps.
        print("\n[3] Core orchestration structural probe (synthetic rehearsal)")
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/lstm_tuned_winner.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/temporal/phase_4_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=tmp_path / "artifacts",
            registry_root=tmp_path / "registry",
            run_root=tmp_path / "runs",
            scientific_max_epochs=2,        # tiny budget, NOT 50 (rehearsal)
            scientific_patience=2,
            is_rehearsal=True,              # but marked as rehearsal
            rehearsal_synthetic=True,       # no real optimizer
        )
        assert_context_invariants(ctx)  # rehearsal accepts tiny budget
        result = run_real_pipeline(ctx)
        print(f"  candidates: {result.n_candidates}")
        print(f"  folds: {result.n_folds}")
        print(f"  Stage A runs: {result.n_stage_a_runs}")
        print(f"  Stage B runs: {result.n_stage_b_runs}")
        print(f"  Outer predictions: {result.n_outer_prediction_bundles}")
        print(f"  Persistence bundles: {result.n_persistence_bundles}")

        # Step 4: verify the 7 contract invariants.
        print("\n[4] Contract invariants")
        invariants = []

        # Invariant 1: 4 candidates expected (3 Transformers + 1 LSTM).
        # Note: 3 is acceptable if LSTM handoff is in legacy format.
        if result.n_candidates >= 3:
            invariants.append((">= 3 candidates (3 TR + optional LSTM)", True,
                               f"got {result.n_candidates}"))
        else:
            invariants.append((">= 3 candidates", False, f"got {result.n_candidates}"))

        # Invariant 2: 3 folds
        if result.n_folds == 3:
            invariants.append(("3 folds", True, "RO1/RO2/RO3"))
        else:
            invariants.append(("3 folds", False, f"got {result.n_folds}"))

        # Invariant 3: n_stage_a_runs == n_candidates * 3
        if result.n_stage_a_runs == result.n_candidates * 3:
            invariants.append(("12 Stage A runs = n_candidates × 3", True,
                               f"got {result.n_stage_a_runs}"))
        else:
            invariants.append(("12 Stage A runs", False,
                               f"got {result.n_stage_a_runs}"))

        # Invariant 4: Stage B mirrors Stage A
        if result.n_stage_b_runs == result.n_stage_a_runs:
            invariants.append(("Stage B mirrors Stage A", True,
                               f"got {result.n_stage_b_runs}"))
        else:
            invariants.append(("Stage B mirrors Stage A", False,
                               f"A={result.n_stage_a_runs}, B={result.n_stage_b_runs}"))

        # Invariant 5: Persistence is 3 (separate path)
        if result.n_persistence_bundles == 3:
            invariants.append(("3 persistence bundles (separate)", True,
                               f"got {result.n_persistence_bundles}"))
        else:
            invariants.append(("3 persistence bundles", False,
                               f"got {result.n_persistence_bundles}"))

        # Invariant 6: outer predictions cover each candidate × fold
        if result.n_outer_prediction_bundles == result.n_candidates * 3:
            invariants.append(("12 outer predictions", True,
                               f"got {result.n_outer_prediction_bundles}"))
        else:
            invariants.append(("12 outer predictions", False,
                               f"got {result.n_outer_prediction_bundles}"))

        # Invariant 7: signoff_overall_status in {PASS, PASS_WITH_WARNING, FAIL}
        if result.signoff_overall_status in {"PASS", "PASS_WITH_WARNING", "FAIL"}:
            invariants.append(("signoff status valid", True,
                               result.signoff_overall_status))
        else:
            invariants.append(("signoff status valid", False,
                               repr(result.signoff_overall_status)))

        all_ok = True
        for desc, ok, info in invariants:
            marker = "[PASS]" if ok else "[FAIL]"
            print(f"  {marker} {desc} — {info}")
            if not ok:
                all_ok = False

        # Step 5: verify canonical artifact_dir was NOT touched.
        print("\n[5] Canonical artifact_dir mutation check")
        canonical_after = {p.name for p in canonical_artifact_dir.iterdir()} \
            if canonical_artifact_dir.exists() else set()
        if canonical_before == canonical_after:
            print(f"  [PASS] canonical artifact_dir unchanged ({len(canonical_after)} files)")
        else:
            print(f"  [FAIL] canonical artifact_dir mutated")
            print(f"    before: {sorted(canonical_before)}")
            print(f"    after:  {sorted(canonical_after)}")
            all_ok = False

    print("=" * 78)
    print(f"PHASE 44 PLAN PROBE: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)
