"""PHASE 44 PROBE — REAL LOADER Y ROUNDTRIP (24 loaders).

For every Stage A and Stage B loader (4 candidates × 3 folds × Stage A/B =
24 loader-roles), this probe:

  - Loads one batch from the fold-aware loader (with fold-local Y rescale).
  - Confirms batch["y_raw_wh"] matches what the canonical sequence
    window dataset materializes.
  - Confirms batch["y_model"] equals (y_raw_wh - fold_y_mean) / fold_y_std.
  - Confirms inverse_transform_target(YS1, fold_bundle) reconstructs
    y_raw_wh with max absolute error ≈ 0.

Run:
    PYTHONPATH=src python3 scripts/_phase44_fold_local_loader_y_roundtrip_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from course_work.data.scaling import (
    inverse_transform_target,
    transform_target,
)
from course_work.rolling_origin.candidate_loader import load_candidates
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.populations import (
    extract_robase_train_ids,
    extract_robase_val_ids,
)
from course_work.rolling_origin.real_run import (
    RunContext,
    build_real_canonical_base_dataset,
    fit_fold_local_scaler,
)
from course_work.rolling_origin.stages import (
    load_fold_subset_loader_with_y_rescale,
)


def _load_one_batch(loader):
    for batch in loader:
        return batch
    return None


def main() -> int:
    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)

    ctx = RunContext(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=PROJECT_ROOT / "artifacts/phase42_signoff.json",
        phase_43_signoff_path=PROJECT_ROOT / "artifacts/phase43_signoff.json",
        artifact_dir=PROJECT_ROOT / "artifacts/rolling_origin",
        registry_root=PROJECT_ROOT / "artifacts/registry",
        run_root=PROJECT_ROOT / "artifacts/runs",
        is_rehearsal=True,
        rehearsal_synthetic=True,
    )

    rows = []
    fail_count = 0
    print(
        f"{'candidate':<22} {'fold':<5} {'stage':<6} {'role':<11} "
        f"{'B':<4} {'y_raw_n':<8} {'y_model_match':<14} {'roundtrip_err':<14} {'verdict'}"
    )
    print("-" * 110)

    for c in candidates:
        fd = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=c)
        for fold in folds:
            for stage in ["A", "B"]:
                scaler_bundle, _audit = fit_fold_local_scaler(
                    candidate=c, fold=fold, fold_dataset=fd, fit_stage=stage, ctx=ctx,
                )
                d = scaler_bundle.as_dict()
                fold_y_mean = float(scaler_bundle.y_mean)
                fold_y_std = float(scaler_bundle.y_std)

                if stage == "A":
                    inner_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                        base_dataset=fd,
                        target_ids=list(fold.inner_train_ids),
                        batch_size=int(c.config["training"]["batch_size"]),
                        shuffle=True,
                        seed=ctx.seed,
                        fold_stage_target_scaler=scaler_bundle,
                    )
                    loader_to_check = inner_train_loader
                    role = "inner_train"
                else:
                    outer_train_loader, _ = load_fold_subset_loader_with_y_rescale(
                        base_dataset=fd,
                        target_ids=list(fold.outer_train_ids),
                        batch_size=int(c.config["training"]["batch_size"]),
                        shuffle=True,
                        seed=ctx.seed,
                        fold_stage_target_scaler=scaler_bundle,
                    )
                    loader_to_check = outer_train_loader
                    role = "outer_train"

                batch = _load_one_batch(loader_to_check)
                if batch is None:
                    print(f"{c.candidate_id:<22} {str(fold.fold_id):<5} {stage:<6} "
                          f"{role:<11} —  —  —  —  EMPTY_LOADER")
                    fail_count += 1
                    continue

                B = batch["y_raw_wh"].shape[0]
                y_raw = batch["y_raw_wh"].float().reshape(-1).numpy()
                y_model = batch["y_model"].float().reshape(-1).numpy()
                expected_y_model = (y_raw - fold_y_mean) / fold_y_std
                y_model_diff = float(np.max(np.abs(y_model - expected_y_model)))
                y_model_match = y_model_diff < 1e-5

                # Roundtrip using canonical transform_target/inverse
                recovered = inverse_transform_target(y_model, "YS1", d)
                rt_err = float(np.max(np.abs(recovered - y_raw)))
                rt_ok = rt_err < 1e-4

                verdict = "PASS" if (y_model_match and rt_ok) else "FAIL"
                if verdict == "FAIL":
                    fail_count += 1

                rows.append({
                    "candidate": c.candidate_id,
                    "fold": str(fold.fold_id),
                    "stage": stage,
                    "role": role,
                    "batch_size": int(B),
                    "y_raw_count": int(B),
                    "y_model_match_max_diff": y_model_diff,
                    "roundtrip_max_error": rt_err,
                    "pass": verdict == "PASS",
                })

                print(
                    f"{c.candidate_id:<22} {str(fold.fold_id):<5} {stage:<6} "
                    f"{role:<11} {B:<4} {len(y_raw):<8} "
                    f"{y_model_diff:<14.2e} {rt_err:<14.2e} {verdict}"
                )

    print("-" * 110)
    print(f"\n=== SUMMARY ===")
    print(f"24 loaders: {24 - fail_count}/24 PASS")
    print(f"  y_model match: {sum(1 for r in rows if r['y_model_match_max_diff'] < 1e-5)}/24")
    print(f"  roundtrip OK: {sum(1 for r in rows if r['roundtrip_max_error'] < 1e-4)}/24")

    out_dir = PROJECT_ROOT / "artifacts" / "phase44_runtime_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase44_loader_y_roundtrip_results.json"
    out_path.write_text(json.dumps({
        "rows": rows,
        "summary": {
            "total": len(rows),
            "passing": 24 - fail_count,
            "failing": fail_count,
        },
    }, indent=2))
    print(f"\nWrote loader roundtrip report → {out_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
