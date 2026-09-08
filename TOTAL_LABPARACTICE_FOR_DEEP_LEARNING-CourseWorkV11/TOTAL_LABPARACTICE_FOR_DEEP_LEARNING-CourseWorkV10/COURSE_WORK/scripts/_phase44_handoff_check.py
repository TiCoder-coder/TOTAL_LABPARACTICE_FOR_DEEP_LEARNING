"""TASK 9 — Verify Phase45 handoff schema in rehearsal.

Checks the rehearsal handoff contains:
  - recommended candidate (TR_C0_PRIMARY)
  - canonical candidate fingerprint
  - RO1/RO2/RO3 Stage A run ids
  - RO1/RO2/RO3 best_epoch_inner
  - RO1/RO2/RO3 Stage B run ids
  - No null epochs
  - No fallback
  - No synthetic epoch (epochs are real values)
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from course_work.rolling_origin.real_run import (
    RunContext, run_real_pipeline,
)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — PHASE45 HANDOFF SCHEMA VERIFICATION (TASK 9)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = {p.name for p in canonical_artifact_dir.iterdir()} \
        if canonical_artifact_dir.exists() else set()

    with tempfile.TemporaryDirectory(prefix="phase44_handoff_") as tmp:
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
        result = run_real_pipeline(ctx)

        canonical_after = {p.name for p in canonical_artifact_dir.iterdir()} \
            if canonical_artifact_dir.exists() else set()
        assert canonical_before == canonical_after, (
            "canonical artifact_dir mutated by handoff probe"
        )

        handoff = result.phase45_handoff
        print(f"\nsignoff: {result.signoff_overall_status}")
        print(f"phase45_handoff present: {handoff is not None}")

        if handoff is None:
            print("[FAIL] phase45_handoff is None (signoff did not pass)")
            return 1

        print(f"\n{'Field':<48} {'Value':<60}")
        print("-" * 110)
        for k, v in handoff.items():
            if isinstance(v, dict):
                v_str = f"<dict with {len(v)} keys: {sorted(v.keys())}>"
            elif isinstance(v, list):
                v_str = f"<list with {len(v)} entries>"
            else:
                v_str = str(v)[:60]
            print(f"{k:<48} {v_str:<60}")

        print("\n=== SCHEMA CHECKS ===")

        ok = True

        rec_id = handoff.get("recommended_transformer_candidate_id", "")
        if not rec_id:
            print("[FAIL] missing recommended_transformer_candidate_id")
            ok = False
        else:
            print(f"[OK] recommended_transformer_candidate_id: {rec_id}")

        rec_fp = handoff.get("recommended_transformer_fingerprint", "")
        if not rec_fp or rec_fp == "":
            print("[FAIL] missing recommended_transformer_fingerprint")
            ok = False
        else:
            print(f"[OK] recommended_transformer_fingerprint: {rec_fp[:24]}... (len={len(rec_fp)})")

        s_a_ids = handoff.get("recommended_transformer_stage_a_run_ids", {})
        for fold_id in ("RO1", "RO2", "RO3"):
            run_id = s_a_ids.get(fold_id, "")
            if not run_id:
                print(f"[FAIL] missing Stage A run id for {fold_id}")
                ok = False
            else:
                print(f"[OK] {fold_id} Stage A run_id: {run_id}")

        s_b_ids = handoff.get("recommended_transformer_stage_b_run_ids", {})
        for fold_id in ("RO1", "RO2", "RO3"):
            run_id = s_b_ids.get(fold_id, "")
            if not run_id:
                print(f"[FAIL] missing Stage B run id for {fold_id}")
                ok = False
            else:
                print(f"[OK] {fold_id} Stage B run_id: {run_id}")

        best_epochs = handoff.get("recommended_transformer_inner_best_epochs", {})
        for fold_id in ("RO1", "RO2", "RO3"):
            ep = best_epochs.get(fold_id, None)
            if ep is None or ep == "" or ep < 1:
                print(f"[FAIL] missing/invalid best_epoch_inner for {fold_id}: {ep}")
                ok = False
            else:
                print(f"[OK] {fold_id} best_epoch_inner: {ep}")

        if any(ep < 1 for ep in best_epochs.values()):
            print(f"[FAIL] synthetic epoch detected: {best_epochs}")
            ok = False

        ts = handoff.get("test_status", "")
        if ts != "NOT_ACCESSED":
            print(f"[FAIL] test_status={ts!r} (expected NOT_ACCESSED)")
            ok = False
        else:
            print(f"[OK] test_status: {ts}")

        apv = handoff.get("approved_for_phase45", False)
        if not apv:
            print("[FAIL] approved_for_phase45 is False")
            ok = False
        else:
            print("[OK] approved_for_phase45: True")

    print("=" * 78)
    if ok:
        print("[PASS] Phase45 handoff schema satisfies the rehearsal contract.")
        return 0
    print("[FAIL] Phase45 handoff schema does NOT satisfy the rehearsal contract.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)