"""TASK 9 — Real-data pre-training harness (48 probes).

For all 4 candidates × 3 folds × 4 roles (inner_train, inner_val,
outer_train, outer_eval) = 48 loaders, verify each loader produces the
correct scientific input shape:

  x.ndim == 3
  x.shape[1] == candidate.lookback
  x.shape[2] == candidate.feature_count
  sample IDs exactly match the requested population
  no Test IDs

NO TrainingEngine.train call.
NO canonical mutation.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — REAL-DATA PRE-TRAINING HARNESS (TASK 9)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )

    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
        ROBASE_VERSION,
    )
    from course_work.rolling_origin.folds import build_rolling_folds
    from course_work.rolling_origin.real_run import build_real_canonical_base_dataset
    from course_work.rolling_origin.stages import load_fold_subset_loader

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)

    # Get all ROBASE IDs (no Test rows by construction).
    all_robase_ids = set(str(t) for t in (rtrn_ids + rval_ids))

    n_pass = 0
    n_fail = 0
    failures: list[str] = []

    for cand in candidates:
        base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
        L = cand.lookback_steps
        F = cand.config.get("data", {}).get("feature_count", 33)

        for fold in folds:
            for role, ids in [
                ("inner_train", list(fold.inner_train_ids)),
                ("inner_val", list(fold.inner_val_ids)),
                ("outer_train", list(fold.outer_train_ids)),
                ("outer_eval", list(fold.outer_eval_ids)),
            ]:
                loader, kept_indices = load_fold_subset_loader(
                    base_dataset=base,
                    target_ids=ids,
                    batch_size=32, shuffle=False, seed=42,
                )
                # Read one batch (drop_last=False so we still get a partial batch)
                batch = next(iter(loader))
                x = batch["x"]
                ok = (
                    x.ndim == 3
                    and x.shape[1] == L
                    and x.shape[2] == F
                    and int(x.shape[0]) > 0
                )
                # Check IDs match population
                kept_target_ids = {
                    str(base.window_records.iloc[i]["target_id"])
                    for i in kept_indices
                }
                expected_ids = set(str(t) for t in ids)
                id_match = kept_target_ids == expected_ids
                # Check no Test IDs
                no_test = kept_target_ids.issubset(all_robase_ids)

                if ok and id_match and no_test:
                    n_pass += 1
                    status = "PASS"
                else:
                    n_fail += 1
                    status = "FAIL"
                    failures.append(
                        f"  {cand.candidate_id} {fold.fold_id} {role}: "
                        f"shape={(x.ndim, x.shape[1] if x.ndim>=2 else -1, x.shape[2] if x.ndim>=2 else -1)}, "
                        f"id_match={id_match}, no_test={no_test}"
                    )
                print(
                    f"  [{status}] {cand.candidate_id:<22} {fold.fold_id} {role:<14}: "
                    f"x.shape={tuple(x.shape)}, n_ids={len(kept_target_ids)}, no_test={no_test}"
                )

    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    assert canonical_after == canonical_before, "canonical artifact_dir mutated"

    print()
    print(f"PROBE SUMMARY: {n_pass}/{n_pass + n_fail} PASS")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f)
        return 1
    print(f"[OK] canonical artifact_dir unchanged ({len(canonical_after)} files)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())