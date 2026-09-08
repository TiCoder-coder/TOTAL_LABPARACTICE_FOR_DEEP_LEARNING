"""TASK 4 — Verify target-ID -> dataset index mapping.

Prove that for each fold, the targets returned by the fold-specific loader
exactly match the requested target_ids (no fallback positional assumptions).

For each (candidate, fold) we verify:
  - first 3 inner_train IDs
  - last 3 inner_train IDs
  - first 3 inner_val IDs
  - first 3 outer_eval IDs

NO optimizer steps. NO canonical mutation.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — TARGET-ID → DATASET INDEX MAPPING (TASK 4)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )

    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
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

    focus = [c for c in candidates if c.candidate_id in ("TR_C0_PRIMARY", "TR_C2_ALT_LOOKBACK", "LSTM_TUNED_WINNER")]
    print(f"\nCANDIDATES UNDER TEST: {[c.candidate_id for c in focus]}")
    print(f"FOLDS: {[str(f.fold_id) for f in folds]}")

    failures: list[str] = []

    for cand in focus:
        base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
        wr = base.window_records
        tid_to_record_idx = {str(tid): i for i, tid in enumerate(wr["target_id"].tolist())}

        for fold in folds:
            for role, ids in [
                ("inner_train", list(fold.inner_train_ids)),
                ("inner_val", list(fold.inner_val_ids)),
                ("outer_eval", list(fold.outer_eval_ids)),
            ]:
                loader, kept_indices = load_fold_subset_loader(
                    base_dataset=base,
                    target_ids=ids,
                    batch_size=32,
                    shuffle=False,
                    seed=42,
                )
                kept_target_ids = [
                    str(wr.iloc[i]["target_id"]) for i in kept_indices
                ]
                expected = set(str(t) for t in ids)
                actual = set(kept_target_ids)
                missing = expected - actual
                extra = actual - expected
                if missing or extra:
                    failures.append(
                        f"  {cand.candidate_id} {fold.fold_id} {role}: "
                        f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
                    )
                probe = (
                    ids[:3] if role in ("inner_train", "inner_val", "outer_eval") else []
                )
                if role == "inner_train":
                    probe = ids[:3] + ids[-3:]  
                probe_kept = []
                for t in probe:
                    pos = tid_to_record_idx.get(str(t))
                    if pos in kept_indices:
                        probe_kept.append(str(t))
                if [str(t) for t in probe] != probe_kept:
                    failures.append(
                        f"  {cand.candidate_id} {fold.fold_id} {role}: "
                        f"probe target_ids mismatch. expected={probe} got={probe_kept}"
                    )

            print(
                f"  [OK] {cand.candidate_id} L={cand.lookback_steps} {fold.fold_id}: "
                f"inner_train={len(fold.inner_train_ids)}, "
                f"inner_val={len(fold.inner_val_ids)}, "
                f"outer_eval={len(fold.outer_eval_ids)}"
            )

    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    assert canonical_after == canonical_before, (
        "canonical artifact_dir was mutated by TASK 4 probe"
    )

    print()
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f)
        return 1
    else:
        print(f"[OK] all target-ID mappings verified")
    print(f"[OK] canonical artifact_dir unchanged ({len(canonical_after)} files)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())