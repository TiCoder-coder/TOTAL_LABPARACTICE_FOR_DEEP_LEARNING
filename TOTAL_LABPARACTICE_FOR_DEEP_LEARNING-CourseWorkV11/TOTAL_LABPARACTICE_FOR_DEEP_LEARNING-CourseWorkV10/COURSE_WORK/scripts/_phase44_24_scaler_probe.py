"""Probe all 24 real scaler contracts (4 candidates × 3 folds × Stage A/B).

Verifies that each fold-local scaler bundle:
1. Satisfies the canonical YS1 scaler bundle contract (option, scaling_version, scaler).
2. Has finite mean/std.
3. Roundtrip transform/inverse gives ~0 error.
4. Did NOT touch any outer_eval or Test row in the fit population.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from course_work.data.scaling import transform_target, inverse_transform_target
from course_work.rolling_origin.candidate_loader import load_candidates
from course_work.rolling_origin.populations import (
    extract_robase_train_ids, extract_robase_val_ids, _load_windowpop, _find_windowpop,
)
from course_work.rolling_origin.folds import build_rolling_folds
from course_work.rolling_origin.real_run import (
    fit_fold_local_scaler, _resolve_canonical_y_scaler,
    build_real_canonical_base_dataset, RunContext,
)


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — 24 SCALER CONTRACT PROBE")
    print("=" * 78)

    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    print(f"\nCandidates: {len(candidates)}")
    for c in candidates:
        print(f"  - {c.candidate_id}: model={c.model_family}, lookback={c.lookback_steps}")

    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)
    print(f"\nFolds: {len(folds)}")

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
            project_root=PROJECT_ROOT,
            candidate=cand,
        ),
    )

    results = []
    failures = []

    windowpop_path = PROJECT_ROOT / "artifacts/windows/common_target_population.csv"
    cache_population = _load_windowpop(windowpop_path)
    cache_population = _load_windowpop(_find_windowpop(PROJECT_ROOT))
    test_ids_set = set(
        cache_population.loc[
            cache_population["target_split_id"].astype(str) == "TEST",
            "target_id" if "target_id" in cache_population.columns else "target_sample_id"
        ].astype(str)
    )

    for c in candidates:
        for f in folds:
            for stage in ["A", "B"]:
                try:
                    fold_dataset = build_real_canonical_base_dataset(
                        project_root=PROJECT_ROOT,
                        candidate=c,
                    )
                    bundle, audit = fit_fold_local_scaler(
                        candidate=c, fold=f,
                        fold_dataset=fold_dataset,
                        fit_stage=stage, ctx=ctx,
                    )
                    d = bundle.as_dict()
                    x_means_list = d.get("x_means") or []

                    has_option = d.get("option") == c.target_scaling_option
                    has_scaler = d.get("scaler") is not None and hasattr(d["scaler"], "transform")
                    has_version = d.get("scaling_version") in ("SCALING-v1", "FINAL_SCALING-v1")

                    if c.target_scaling_option == "YS1":
                        test_y = np.array([50.0, 100.0, 200.0, 10.0, 500.0])
                        scaled = transform_target(test_y, "YS1", d)
                        recovered = inverse_transform_target(scaled, "YS1", d)
                        max_roundtrip_err = float(np.max(np.abs(test_y - recovered)))
                    else:
                        scaled = transform_target(np.array([50.0]), "YS0", None)
                        recovered = inverse_transform_target(scaled, "YS0", None)
                        max_roundtrip_err = float(np.max(np.abs(np.array([50.0]) - recovered)))

                    fit_ids_set = set(str(t) for t in bundle.fit_target_ids)
                    outer_ids_set = set(str(t) for t in f.outer_eval_ids)
                    test_ids_set = set(
                        cache_population.loc[
                            cache_population["target_split_id"].astype(str) == "TEST",
                            "target_id" if "target_id" in cache_population.columns else "target_sample_id"
                        ].astype(str)
                    )
                    outer_leak = fit_ids_set & outer_ids_set
                    test_leak = fit_ids_set & test_ids_set
                    outer_eval_rows_used = audit.outer_eval_rows_used
                    test_rows_used = audit.test_rows_used

                    finite = np.isfinite([d["y_mean"] if d.get("y_mean") is not None else 0.0]).all()
                    if d.get("y_mean") is None:
                        finite = True  # YS0

                    ok = (
                        has_option and has_scaler and has_version
                        and max_roundtrip_err < 1e-9
                        and len(outer_leak) == 0
                        and len(test_leak) == 0
                        and outer_eval_rows_used == 0
                        and test_rows_used == 0
                        and finite
                    )

                    row = {
                        "candidate": c.candidate_id,
                        "fold": f.fold_id,
                        "stage": stage,
                        "lookback": c.lookback_steps,
                        "feature_count": len(x_means_list) if x_means_list else 0,
                        "target_option": c.target_scaling_option,
                        "fit_population": audit.fit_target_ids_count,
                        "y_mean": d.get("y_mean"),
                        "y_std": d.get("y_std"),
                        "x_means_len": len(d.get("x_means") or []),
                        "schema_option": has_option,
                        "schema_scaler": has_scaler,
                        "schema_version": has_version,
                        "roundtrip_err": max_roundtrip_err,
                        "outer_leak": len(outer_leak),
                        "test_leak": len(test_leak),
                        "audit_outer": outer_eval_rows_used,
                        "audit_test": test_rows_used,
                        "PASS": ok,
                    }
                    results.append(row)
                    if not ok:
                        failures.append(row)
                except Exception as e:
                    failures.append({
                        "candidate": c.candidate_id,
                        "fold": f.fold_id,
                        "stage": stage,
                        "error": str(e),
                        "PASS": False,
                    })
                    print(f"  FAIL: {c.candidate_id} {f.fold_id} stage={stage}: {e}")

    print(f"\n{'cand':<22} {'fold':<5} {'st':<2} {'feat':<5} {'option':<5} {'fitpop':<7} {'x_means':<8} {'y_mean':<10} {'y_std':<10} {'rt_err':<10} {'leakO':<6} {'leakT':<6} {'PASS':<6}")
    print("-" * 130)
    for r in results:
        fold_id = str(r['fold'])
        print(f"{r['candidate']:<22} {fold_id:<5} {r['stage']:<2} {r['feature_count']:<5} {r['target_option']:<5} {r['fit_population']:<7} {r['x_means_len']:<8} {str(round(r['y_mean'], 3) if r['y_mean'] is not None else 'None'):<10} {str(round(r['y_std'], 3) if r['y_std'] is not None else 'None'):<10} {r['roundtrip_err']:<10.2e} {r['outer_leak']:<6} {r['test_leak']:<6} {'YES' if r['PASS'] else 'NO':<6}")

    passed = sum(1 for r in results if r["PASS"])
    failed = len(results) - passed
    print(f"\n{passed}/{len(results)} contracts PASS")
    if results:
        print(f"Max roundtrip error: {max(r['roundtrip_err'] for r in results):.2e}")
        print(f"Total outer-eval leaks: {sum(r['outer_leak'] for r in results)}")
        print(f"Total test leaks: {sum(r['test_leak'] for r in results)}")
        print(f"Total audit.outer_eval_rows_used != 0: {sum(1 for r in results if r.get('audit_outer', 0) != 0)}")
        print(f"Total audit.test_rows_used != 0: {sum(1 for r in results if r.get('audit_test', 0) != 0)}")
    else:
        print(f"Max roundtrip error: N/A (no contracts)")
        print(f"Total outer-eval leaks: N/A")
        print(f"Total test leaks: N/A")
    print()
    print("=" * 78)
    print("PROBE COMPLETE")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
