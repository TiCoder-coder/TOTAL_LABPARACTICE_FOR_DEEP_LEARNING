"""TASK 7/8 — Prove synthetic official mock is unreachable AND real primitives are used.

This script:
  1. Monkeypatches the OLD synthetic functions in pipeline.py to RAISE.
  2. Asserts that --mode official planning succeeds without invoking them.
  3. Traces WHICH real primitives are invoked during a real run.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — SYNTHETIC MOCK UNREACHABILITY + REAL PATH PROBE (TASK 7/8)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = {p.name for p in canonical_artifact_dir.iterdir()} \
        if canonical_artifact_dir.exists() else set()

    print("\n[1] Locating the old synthetic functions in pipeline.py")
    from course_work.rolling_origin import pipeline as p44pipe
    synthetic_funcs = [
        "_synth_predict_for_official",
        "run_pipeline",
    ]
    found_funcs = {}
    for name in synthetic_funcs:
        if hasattr(p44pipe, name):
            obj = getattr(p44pipe, name)
            found_funcs[name] = obj
            print(f"  [FOUND] pipeline.{name} -> {obj!r:.80}")
        else:
            print(f"  [MISSING] pipeline.{name}")
    assert "run_pipeline" in found_funcs, "pipeline.run_pipeline must exist"
    assert "_synth_predict_for_official" in found_funcs, \
        "pipeline._synth_predict_for_official must exist"

    print("\n[2] Monkey-patching old synthetic functions to RAISE if invoked")
    def _explode(*args, **kwargs):
        raise RuntimeError(
            "OLD SYNTHETIC FUNCTION CALLED — this must NEVER happen in "
            "the real Phase 44 orchestrator"
        )
    for name in synthetic_funcs:
        setattr(p44pipe, name, _explode)
    print(f"  [PATCHED] pipeline.{synthetic_funcs[0]} -> raises RuntimeError")
    print(f"  [PATCHED] pipeline.{synthetic_funcs[1]} -> raises RuntimeError")

    print("\n[3] Running run_real_pipeline() with synthetic functions monkey-patched")
    from course_work.rolling_origin.real_run import (
        RunContext, run_real_pipeline, SCIENTIFIC_MAX_EPOCHS,
    )

    invocations = {
        "train_stage_a": 0,
        "refit_engine.refit": 0,
        "evaluate_stage_c": 0,
        "compute_persistence_bundle": 0,
        "compute_pooled_metrics": 0,
        "rank_transformers": 0,
    }

    import course_work.rolling_origin.stages as stages_mod
    import course_work.rolling_origin.refit_engine as refit_mod
    import course_work.rolling_origin.persistence as persist_mod
    import course_work.rolling_origin.pooling as pooling_mod
    import course_work.rolling_origin.ranking as ranking_mod
    from course_work.rolling_origin import real_run as real_run_mod

    orig_train_stage_a = stages_mod.train_stage_a
    orig_refit = refit_mod.RefitEngine.refit
    orig_evaluate_stage_c = stages_mod.evaluate_stage_c
    orig_persistence = persist_mod.compute_persistence_bundle
    orig_pooled = pooling_mod.compute_pooled_metrics
    orig_rank = ranking_mod.rank_transformers

    def counting_train_stage_a(*args, **kwargs):
        invocations["train_stage_a"] += 1
        return orig_train_stage_a(*args, **kwargs)

    def counting_refit(self, *args, **kwargs):
        invocations["refit_engine.refit"] += 1
        return orig_refit(self, *args, **kwargs)

    def counting_evaluate_stage_c(*args, **kwargs):
        invocations["evaluate_stage_c"] += 1
        return orig_evaluate_stage_c(*args, **kwargs)

    def counting_persistence(*args, **kwargs):
        invocations["compute_persistence_bundle"] += 1
        return orig_persistence(*args, **kwargs)

    def counting_pooled(*args, **kwargs):
        invocations["compute_pooled_metrics"] += 1
        return orig_pooled(*args, **kwargs)

    def counting_rank(*args, **kwargs):
        invocations["rank_transformers"] += 1
        return orig_rank(*args, **kwargs)

    stages_mod.train_stage_a = counting_train_stage_a
    refit_mod.RefitEngine.refit = counting_refit
    stages_mod.evaluate_stage_c = counting_evaluate_stage_c
    persist_mod.compute_persistence_bundle = counting_persistence
    pooling_mod.compute_pooled_metrics = counting_pooled
    ranking_mod.rank_transformers = counting_rank
    real_run_mod.train_stage_a = counting_train_stage_a
    real_run_mod.evaluate_stage_c = counting_evaluate_stage_c
    real_run_mod.compute_persistence_bundle = counting_persistence
    real_run_mod.compute_pooled_metrics = counting_pooled
    real_run_mod.rank_transformers = counting_rank

    with tempfile.TemporaryDirectory(prefix="phase44_mock_probe_") as tmp:
        ctx = RunContext(
            project_root=PROJECT_ROOT,
            transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
            lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
            phase_42_signoff_path=PROJECT_ROOT / "artifacts/candidate_synthesis/phase_42_signoff.json",
            phase_43_signoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase_43_signoff.json",
            artifact_dir=Path(tmp) / "artifacts",
            registry_root=Path(tmp) / "registry",
            run_root=Path(tmp) / "runs",
            scientific_max_epochs=2,
            scientific_patience=2,
            seed=42,
            is_rehearsal=True,
            rehearsal_synthetic=True,
        )
        try:
            result = run_real_pipeline(ctx)
        except RuntimeError as exc:
            if "OLD SYNTHETIC FUNCTION CALLED" in str(exc):
                print(f"  [FAIL] Synthetic mock WAS called: {exc}")
                return 1
            raise

    canonical_after = {p.name for p in canonical_artifact_dir.iterdir()} \
        if canonical_artifact_dir.exists() else set()
    assert canonical_before == canonical_after, \
        "canonical artifact_dir mutated by mock-probe run"

    print("\n[4] Real primitive invocations:")
    print(f"  {'primitive':<32} {'expected':<10} {'actual':<10} {'match':<6}")
    print("  " + "-" * 60)
    expected = {
        "train_stage_a": 12,            # 4 cand × 3 folds Stage A
        "refit_engine.refit": 12,       # 4 cand × 3 folds Stage B
        "evaluate_stage_c": 12,         # 4 cand × 3 folds Stage C
        "compute_persistence_bundle": 3, # 3 folds
        "compute_pooled_metrics": 5,    # 3 TR + 1 LSTM + 1 PERSISTENCE
        "rank_transformers": 1,         # once for the whole ranking
    }
    all_match = True
    for prim, exp in expected.items():
        act = invocations[prim]
        match = "PASS" if act == exp else "FAIL"
        if match == "FAIL":
            all_match = False
        print(f"  {prim:<32} {exp:<10} {act:<10} {match:<6}")

    print("\n[5] Result summary:")
    print(f"  exit_code: {result.exit_code}")
    print(f"  signoff_overall_status: {result.signoff_overall_status}")
    print(f"  n_candidates: {result.n_candidates}")
    print(f"  n_stage_a_runs: {result.n_stage_a_runs}")
    print(f"  n_stage_b_runs: {result.n_stage_b_runs}")
    print(f"  n_outer_prediction_bundles: {result.n_outer_prediction_bundles}")

    print("=" * 78)
    if all_match and result.exit_code == 0:
        print("[PASS] Real model path verified: synthetic mock unreachable, "
              "all real primitives invoked with correct counts.")
        return 0
    print("[FAIL] Real model path probe did not satisfy all invariants.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)
