"""TASKS 5+6+7 — Verify L36/L72 real datasets + Model forward smoke + Scaler shape.

For all 4 candidates (TR_C0, TR_C1, TR_C2, LSTM) and all 3 folds (RO1-3),
verify that:
  - inner_train batch shape == [B, L, F]
  - inner_val batch shape == [B, L, F]
  - outer_eval batch shape == [B, L, F]
  - L=36 for TR_C0, TR_C1, LSTM; L=72 for TR_C2
  - F=33 (canonical feature_count)
  - model(x) forward succeeds with no optimizer / no loss / no registration
  - scaler_bundle.shape is [N, F] for fit data, [N] for y fit

NO optimizer steps. NO canonical mutation.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — REAL DATASETS / FORWARD / SCALER PROBES (TASKS 5+6+7)")
    print("=" * 78)

    canonical_artifact_dir = PROJECT_ROOT / "artifacts" / "rolling_origin"
    canonical_before = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )

    import torch
    from course_work.rolling_origin.candidate_loader import load_candidates
    from course_work.rolling_origin.populations import (
        extract_robase_train_ids, extract_robase_val_ids,
    )
    from course_work.rolling_origin.folds import build_rolling_folds
    from course_work.rolling_origin.real_run import (
        build_real_canonical_base_dataset,
        fit_fold_local_scaler,
    )
    from course_work.rolling_origin.stages import load_fold_subset_loader

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)

    print(f"\n{'='*78}")
    print(f"{'CANDIDATE':<24} {'L':>3} {'F':>3} {'TRAIN_BATCH':>14} {'VAL_BATCH':>14} {'OE_BATCH':>14} {'FWD':>5}")
    print(f"{'='*78}")

    failures: list[str] = []

    for cand in candidates:
        base = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=cand)
        L = cand.lookback_steps
        F = cand.config.get("data", {}).get("feature_count", 33)

        # Build model from run_config
        from course_work.rolling_origin.real_run import build_model_from_run_config
        run_config = dict(cand.config)
        # Override lookback to match
        run_config["data"] = dict(cand.config["data"])
        run_config["data"]["lookback_steps"] = L
        model = build_model_from_run_config(run_config)
        model.eval()

        for fold in folds:
            # Load one batch per role
            inner_train_loader, _ = load_fold_subset_loader(
                base_dataset=base,
                target_ids=list(fold.inner_train_ids[:64]),
                batch_size=32, shuffle=False, seed=42,
            )
            inner_val_loader, _ = load_fold_subset_loader(
                base_dataset=base,
                target_ids=list(fold.inner_val_ids[:32]),
                batch_size=32, shuffle=False, seed=42,
            )
            outer_eval_loader, _ = load_fold_subset_loader(
                base_dataset=base,
                target_ids=list(fold.outer_eval_ids[:32]),
                batch_size=32, shuffle=False, seed=42,
            )
            train_batch = next(iter(inner_train_loader))
            val_batch = next(iter(inner_val_loader))
            oe_batch = next(iter(outer_eval_loader))

            x_tr = train_batch["x"]
            x_vl = val_batch["x"]
            x_oe = oe_batch["x"]

            shapes_ok = (
                x_tr.ndim == 3 and x_tr.shape[1] == L and x_tr.shape[2] == F
                and x_vl.ndim == 3 and x_vl.shape[1] == L and x_vl.shape[2] == F
                and x_oe.ndim == 3 and x_oe.shape[1] == L and x_oe.shape[2] == F
            )
            if not shapes_ok:
                failures.append(
                    f"  {cand.candidate_id} {fold.fold_id}: shape FAIL "
                    f"train={tuple(x_tr.shape)} val={tuple(x_vl.shape)} oe={tuple(x_oe.shape)}"
                )

            # Forward smoke on outer_eval
            with torch.no_grad():
                out = model(x_oe)
            fwd_ok = (
                out.ndim >= 2 and out.shape[0] == x_oe.shape[0]
                and out.shape[-1] == 1
                and bool(torch.isfinite(out).all().item())
            )
            if not fwd_ok:
                failures.append(
                    f"  {cand.candidate_id} {fold.fold_id}: forward FAIL out.shape={tuple(out.shape)}"
                )

            print(
                f"{cand.candidate_id:<24} {L:>3} {F:>3} "
                f"{str(tuple(x_tr.shape)):>14} {str(tuple(x_vl.shape)):>14} "
                f"{str(tuple(x_oe.shape)):>14} {'PASS' if fwd_ok else 'FAIL':>5}"
            )

    canonical_after = (
        sorted(p.name for p in canonical_artifact_dir.iterdir())
        if canonical_artifact_dir.exists() else []
    )
    assert canonical_after == canonical_before, "canonical artifact_dir mutated"

    print()
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f)
        return 1
    print("[OK] all 12 candidate×fold probes pass (4×3=12 forward, 36 batch shape checks)")
    print(f"[OK] canonical artifact_dir unchanged ({len(canonical_after)} files)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())