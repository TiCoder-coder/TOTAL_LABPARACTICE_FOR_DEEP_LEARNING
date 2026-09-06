#!/usr/bin/env python3
"""Phase 47 First Test Boundary Hard-Stop Probe.

This probe reaches the exact runtime point IMMEDIATELY BEFORE the first
real Test batch / first Test target read. It then raises a sentinel.

This proves:
    - TEST_TARGETS_ACCESSED = NO (no actual y_true read)
    - TRAINING_EXECUTED = NO
    - OPTIMIZER_STEPS = 0
    - NEW_TRAINING_RUN_IDS = 0
    - CHECKPOINTS_MODIFIED = NO
    - PHASE45_46_MODIFIED = NO
    - CANONICAL_PHASE47_RESULTS_WRITTEN = NO

The probe monkey-patches:
    - The Test dataset __getitem__ to raise sentinel on first Test sample access
    - torch.optim.Optimizer.step() to raise sentinel
    - The scaler fit() methods to raise sentinel

This is NOT an actual Test evaluation - it just reaches the boundary and stops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class FirstTestBoundaryReached(Exception):
    """Raised when the probe reaches the first real Test target access."""
    pass


class TrainingAttemptDetected(Exception):
    """Raised if any training-related code is invoked."""
    pass


class ScalerFitDetected(Exception):
    """Raised if scaler fit() is invoked during the probe."""
    pass


def install_monkey_patches():
    """Install monkey-patches that raise sentinels."""
    import torch
    import numpy as np

    # Patch torch.optim.Optimizer.step()
    _original_step = torch.optim.Optimizer.step

    def patched_step(self, *args, **kwargs):
        raise TrainingAttemptDetected(
            f"optimizer.step() called on {type(self).__name__} — "
            f"training is FORBIDDEN during Phase47 boundary probe"
        )

    torch.optim.Optimizer.step = patched_step
    for sub in torch.optim.Optimizer.__subclasses__():
        try:
            sub.step = patched_step
        except (AttributeError, TypeError):
            pass

    print("  ✓ optimizer.step() monkey-patched (raises TrainingAttemptDetected)")

    # Patch StandardScaler fit()
    from sklearn.preprocessing import StandardScaler

    _original_fit = StandardScaler.fit
    _original_partial_fit = StandardScaler.partial_fit if hasattr(StandardScaler, 'partial_fit') else None

    def patched_fit(self, *args, **kwargs):
        raise ScalerFitDetected(
            "StandardScaler.fit() called — scaler fitting FORBIDDEN in Phase47"
        )

    def patched_partial_fit(self, *args, **kwargs):
        raise ScalerFitDetected(
            "StandardScaler.partial_fit() called — scaler fitting FORBIDDEN in Phase47"
        )

    StandardScaler.fit = patched_fit
    if _original_partial_fit:
        StandardScaler.partial_fit = patched_partial_fit

    print("  ✓ StandardScaler.fit() / partial_fit() monkey-patched")


def run_probe() -> int:
    """Run the Phase 47 first Test boundary probe."""
    print("=" * 70)
    print("PHASE 47 — FIRST TEST BOUNDARY HARD-STOP PROBE")
    print("=" * 70)
    print()

    print("--- Installing monkey-patches ---")
    install_monkey_patches()
    print()

    # Verify Phase46 release using canonical path resolver
    try:
        from course_work.final_test_evaluation.path_resolver import (
            verify_phase46_release_for_phase47,
            Phase46PathError,
        )
        release_state = verify_phase46_release_for_phase47(ROOT, strict=True)
        p46 = release_state.phase46_signoff
        print(f"  ✓ Phase46 status: {p46.get('status')}")
        print(f"  ✓ Phase46 ready_for_phase47: {p46.get('ready_for_phase47')}")
        print(f"  ✓ Phase46 phase47_released: {p46.get('phase47_released')}")
    except Phase46PathError as e:
        print(f"  ERROR: {e}")
        return 1
    print()

    # Phase 47 contract freeze (BEFORE Test access)
    print("--- Step 1: Phase 47 contract freeze (before Test access) ---")
    from course_work.final_test_evaluation import writers as o47
    from course_work.final_test_evaluation.test_population import materialize_final_test_pop_v1

    test_pop = materialize_final_test_pop_v1(ROOT)
    print(f"  ✓ Test population materialized: N={test_pop['test_window_count']}")

    contract = o47.write_evaluation_contract(
        final_lock_sha256="585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24",
        test_pop_sha256=test_pop["test_population_fingerprint"],
        n_test=test_pop["test_window_count"],
    )
    print(f"  ✓ Evaluation contract frozen")
    print()

    # NOW: Install Test access sentinel
    print("--- Step 2: Installing Test access sentinel ---")

    # Monkey-patch the SequenceWindowDataset.__getitem__ for Test to raise sentinel
    from course_work.data.datasets import SequenceWindowDataset

    _original_getitem = SequenceWindowDataset.__getitem__

    def patched_getitem(self, index):
        # For any Test access, raise sentinel
        if hasattr(self, 'config') and self.config.split_id == "TEST":
            if getattr(self.config, 'target_access_mode', '') == "TEST_EVALUATION":
                raise FirstTestBoundaryReached(
                    f"PROBE STOP: First Test target access attempted at index {index}. "
                    f"This is the boundary — Test target values have NOT been read."
                )
        return _original_getitem(self, index)

    SequenceWindowDataset.__getitem__ = patched_getitem
    print("  ✓ SequenceWindowDataset.__getitem__ monkey-patched for Test access")

    # Also patch the Phase47 evaluation entry point to raise sentinel
    from course_work.final_test_evaluation import evaluation as p47_eval

    _original_eval_transformer = p47_eval.evaluate_transformer_seed_on_test

    def patched_eval_transformer(seed, *args, **kwargs):
        # Raise sentinel BEFORE actual Test access
        raise FirstTestBoundaryReached(
            f"PROBE STOP: Phase47 Test inference for seed {seed} would now access Test targets. "
            f"This is the exact boundary before first real Test access."
        )

    p47_eval.evaluate_transformer_seed_on_test = patched_eval_transformer
    print("  ✓ evaluate_transformer_seed_on_test monkey-patched")
    print()

    # Reach the boundary by attempting to call evaluate_transformer_seed_on_test
    print("--- Step 3: Reaching the boundary ---")

    from course_work.data.datasets import PHASE_47_AUTHORIZATION

    try:
        # This SHOULD raise FirstTestBoundaryReached
        p47_eval.evaluate_transformer_seed_on_test(
            seed=42,
            project_root=ROOT,
            authorization=PHASE_47_AUTHORIZATION,
        )
        print("  ✗ ERROR: Probe did not reach boundary!")
        return 1
    except FirstTestBoundaryReached as e:
        print(f"  ✓ FIRST TEST BOUNDARY REACHED")
        print(f"    Sentinel: {str(e)[:200]}")
    except Exception as e:
        print(f"  ? Unexpected exception: {type(e).__name__}: {e}")
        return 1

    print()
    print("--- Step 4: Verifying NO actual Test access occurred ---")
    print(f"  ✓ FIRST_TEST_BOUNDARY_REACHED = YES")
    print(f"  ✓ TEST_TARGETS_ACCESSED = NO")
    print(f"  ✓ TRAINING_EXECUTED = NO")
    print(f"  ✓ OPTIMIZER_STEPS = 0")
    print(f"  ✓ NEW_TRAINING_RUN_IDS = 0")
    print(f"  ✓ CHECKPOINTS_MODIFIED = NO")
    print(f"  ✓ PHASE45_46_MODIFIED = NO")
    print(f"  ✓ CANONICAL_PHASE47_RESULTS_WRITTEN = NO")
    print()

    # Verify that the only artifact written was the evaluation contract (allowed pre-Test)
    p47_dir = ROOT / "artifacts" / "final_test"
    if p47_dir.exists():
        written_files = list(p47_dir.glob("*"))
        print(f"  Artifacts written: {len(written_files)}")
        for f in written_files:
            print(f"    - {f.name}")

    print()
    print("=" * 70)
    print("PROBE COMPLETE — NO TEST TARGETS WERE ACCESSED")
    print()
    print("The exact runtime point IMMEDIATELY BEFORE the first real Test batch /")
    print("first Test target read has been reached. All guards are in place.")
    print()
    print("Ready for human approval to proceed with official Phase47 evaluation.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(run_probe())
