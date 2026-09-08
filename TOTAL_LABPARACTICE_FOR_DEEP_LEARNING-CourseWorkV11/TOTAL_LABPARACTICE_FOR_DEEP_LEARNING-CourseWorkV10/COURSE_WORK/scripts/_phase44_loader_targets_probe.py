"""Probe real loader y_raw_wh / y_model consistency for all 4 candidates × 3 folds.

For Stage A (inner_train / inner_val) and Stage B (outer_train / outer_eval):
1. Take a real batch.
2. Compare `batch["y_raw_wh"]` (raw Wh) and `batch["y_model"]` (scaled).
3. Recompute `y_model` using the fold-local Y scaler and assert equality/tolerance.
4. Inverse batch["y_model"] and recover y_raw_wh.
5. Assert roundtrip Wh ≈ original Wh.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from course_work.rolling_origin.candidate_loader import load_candidates
from course_work.rolling_origin.populations import (
    extract_robase_train_ids, extract_robase_val_ids,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.real_run import (
    fit_fold_local_scaler, build_real_canonical_base_dataset,
    load_fold_subset_loader, RunContext,
)
from course_work.data.scaling import transform_target, inverse_transform_target


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — REAL LOADER TARGET VERIFICATION")
    print("=" * 78)

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
        dataset_factory=lambda cand, fold: build_real_canonical_base_dataset(
            project_root=PROJECT_ROOT, candidate=cand,
        ),
    )

    results = []
    failures = []

    for c in candidates:
        for f in folds:
            fold_dataset = build_real_canonical_base_dataset(
                project_root=PROJECT_ROOT, candidate=c,
            )

            scaler_a, _ = fit_fold_local_scaler(
                candidate=c, fold=f, fold_dataset=fold_dataset,
                fit_stage="A", ctx=ctx,
            )
            d_a = scaler_a.as_dict()

            scaler_b, _ = fit_fold_local_scaler(
                candidate=c, fold=f, fold_dataset=fold_dataset,
                fit_stage="B", ctx=ctx,
            )
            d_b = scaler_b.as_dict()

            for role, target_ids in [
                ("inner_train", f.inner_train_ids),
                ("inner_val", f.inner_val_ids),
                ("outer_train", f.outer_train_ids),
                ("outer_eval", f.outer_eval_ids),
            ]:
                try:
                    loader, _ = load_fold_subset_loader(
                        base_dataset=fold_dataset,
                        target_ids=list(target_ids),
                        batch_size=8,
                        shuffle=False,
                    )
                    batch = next(iter(loader))
                    y_raw = batch["y_raw_wh"].cpu().numpy().reshape(-1)
                    y_model = batch["y_model"].cpu().numpy().reshape(-1)

                    if c.target_scaling_option == "YS1":
                        bundle = d_a if role.startswith("inner_") else d_b
                        y_model_computed = transform_target(y_raw.astype(np.float64), "YS1", bundle)
                        y_raw_recovered = inverse_transform_target(y_model, "YS1", bundle)
                    else:
                        y_model_computed = y_raw.copy()
                        y_raw_recovered = y_model.copy()

                    err_scale = float(np.max(np.abs(y_model - y_model_computed)))
                    err_roundtrip = float(np.max(np.abs(y_raw - y_raw_recovered)))

                    ok = (
                        err_scale < 1e-4
                        and err_roundtrip < 1e-3
                    )
                    row = {
                        "candidate": c.candidate_id,
                        "fold": str(f.fold_id),
                        "role": role,
                        "batch_size": len(y_raw),
                        "y_raw_range": (float(y_raw.min()), float(y_raw.max())),
                        "y_model_range": (float(y_model.min()), float(y_model.max())),
                        "err_scale": err_scale,
                        "err_roundtrip": err_roundtrip,
                        "PASS": ok,
                    }
                    results.append(row)
                    if not ok:
                        failures.append(row)
                except Exception as e:
                    failures.append({
                        "candidate": c.candidate_id,
                        "fold": str(f.fold_id),
                        "role": role,
                        "error": str(e),
                        "PASS": False,
                    })
                    print(f"  FAIL: {c.candidate_id} {f.fold_id} {role}: {e}")

    print(f"\n{'cand':<22} {'fold':<5} {'role':<12} {'bs':<3} {'scale_err':<12} {'roundtrip':<12} {'PASS':<6}")
    print("-" * 90)
    for r in results:
        fold_id = r['fold']
        print(f"{r['candidate']:<22} {fold_id:<5} {r['role']:<12} {r['batch_size']:<3} {r['err_scale']:<12.2e} {r['err_roundtrip']:<12.2e} {'YES' if r['PASS'] else 'NO':<6}")

    passed = sum(1 for r in results if r["PASS"])
    failed = len(results) - passed
    print(f"\n{passed}/{len(results)} loader-role probes PASS")
    if results:
        print(f"Max scale error: {max(r['err_scale'] for r in results):.2e}")
        print(f"Max roundtrip error: {max(r['err_roundtrip'] for r in results):.2e}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
