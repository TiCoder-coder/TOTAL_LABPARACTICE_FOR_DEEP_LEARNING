"""PHASE 44 PROBE — TRUE FOLD-LOCAL Y SCALER CONTRACTS.

For all 4 candidates × 3 folds × Stage A/B = 24 fold-stage Y scalers, this
probe:

  - Fits a FoldLocalScalerBundle on the SCIENTIFICALLY CORRECT fold-specific
    allowed training population (inner_train for A, outer_train for B).
  - Verifies provenance by reporting fit_count, fit_min/max target timestamp,
    y_mean, y_std, outer_eval_rows_used, test_rows_used.
  - Verifies the bundle satisfies the canonical YS1 schema
    (option="YS1", valid scaling_version, fitted scaler object).
  - Roundtrips raw Wh → y_model → raw Wh with max error ~ 0.

Run:
    PYTHONPATH=src python3 scripts/_phase44_24_fold_local_y_scaler_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from course_work.data.scaling import (
    inverse_transform_target,
    load_validated_target_scaler,
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


def _has_test_leakage(fit_target_ids: list, test_ids: set[str]) -> bool:
    """True if any fit_target_id appears in test_ids."""
    return any(str(t) in test_ids for t in fit_target_ids)


def _has_outer_eval_leakage(fit_target_ids: list, outer_eval_ids: list) -> int:
    """Returns count of fit_target_ids that overlap with outer_eval."""
    outer = set(str(t) for t in outer_eval_ids)
    return sum(1 for t in fit_target_ids if str(t) in outer)


def main() -> int:
    candidates = load_candidates(
        project_root=PROJECT_ROOT,
        transformer_shortlist_path=PROJECT_ROOT / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=PROJECT_ROOT / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
    )
    rtrn_ids = extract_robase_train_ids(PROJECT_ROOT)
    rval_ids = extract_robase_val_ids(PROJECT_ROOT)
    folds = build_rolling_folds(rtrn_ids, rval_ids, k=3)

    test_df = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "windows" / "common_target_population.csv"
    )
    test_set = set(
        test_df[test_df["target_split_id"].astype(str).str.upper() == "TEST"]
        ["target_sample_id"].astype(str)
    )

    canonical = load_validated_target_scaler(PROJECT_ROOT)
    gmean = float(canonical["scaler"].mean_[0])
    gstd = float(canonical["scaler"].scale_[0])

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
        f"{'candidate':<22} {'fold':<5} {'stage':<6} {'fit_n':<7} "
        f"{'ym':<12} {'ys':<12} {'fit_min_ts':<22} {'fit_max_ts':<22} "
        f"{'outer':<7} {'test':<5} {'schema':<7} {'roundtrip':<9}"
    )
    print("-" * 145)

    for c in candidates:
        fd = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=c)
        wr = fd.window_records
        wr["target_id"] = wr["target_id"].astype(str)

        for fold in folds:
            for stage in ["A", "B"]:
                fit_ids = list(fold.inner_train_ids if stage == "A" else fold.outer_train_ids)
                bundle, audit = fit_fold_local_scaler(
                    candidate=c, fold=fold, fold_dataset=fd, fit_stage=stage, ctx=ctx,
                )
                d = bundle.as_dict()

                schema_ok = (
                    d["option"] == "YS1"
                    and d["scaling_version"] in {"SCALING-v1", "FINAL_SCALING-v1"}
                    and d["scaler"] is not None
                    and hasattr(d["scaler"], "transform")
                    and hasattr(d["scaler"], "inverse_transform")
                )

                rng = np.random.default_rng(0)
                raw = rng.normal(80, 100, size=20)
                scaled = transform_target(raw, "YS1", d)
                inv = inverse_transform_target(scaled, "YS1", d)
                rt_err = float(np.max(np.abs(inv - raw)))
                rt_ok = rt_err < 1e-5

                fit_id_set = set(str(t) for t in fit_ids)
                outer_overlap = _has_outer_eval_leakage(fit_ids, fold.outer_eval_ids)
                test_overlap = 1 if _has_test_leakage(fit_ids, test_set) else 0

                fit_sub = wr[wr["target_id"].isin(fit_id_set)]
                if "target_timestamp" in fit_sub.columns and len(fit_sub) > 0:
                    ts_min = str(fit_sub["target_timestamp"].min())
                    ts_max = str(fit_sub["target_timestamp"].max())
                else:
                    ts_min = "n/a"
                    ts_max = "n/a"

                row_pass = schema_ok and rt_ok and outer_overlap == 0 and test_overlap == 0
                if not row_pass:
                    fail_count += 1

                rows.append({
                    "candidate": c.candidate_id,
                    "fold": str(fold.fold_id),
                    "stage": stage,
                    "fit_count": audit.fit_target_ids_count,
                    "y_mean": bundle.y_mean,
                    "y_std": bundle.y_std,
                    "fit_min_ts": ts_min,
                    "fit_max_ts": ts_max,
                    "outer_eval_rows_used": outer_overlap,
                    "test_rows_used": test_overlap,
                    "schema_ok": schema_ok,
                    "roundtrip_max_error": rt_err,
                    "row_pass": row_pass,
                })

                print(
                    f"{c.candidate_id:<22} {str(fold.fold_id):<5} {stage:<6} "
                    f"{audit.fit_target_ids_count:<7} {bundle.y_mean:<12.6f} "
                    f"{bundle.y_std:<12.6f} {ts_min:<22} {ts_max:<22} "
                    f"{outer_overlap:<7} {test_overlap:<5} "
                    f"{'PASS' if schema_ok else 'FAIL':<7} "
                    f"{'%.1e' % rt_err if rt_ok else 'FAIL':<9}"
                )

    print("-" * 145)
    print(f"\nGlobal Phase9 YS1 reference: mean={gmean:.6f}, std={gstd:.6f}")
    print(f"\n=== SUMMARY ===")
    print(f"24 fold-stage Y scalers: {24 - fail_count}/24 PASS")
    print(f"  schema OK: {sum(1 for r in rows if r['schema_ok'])}/24")
    print(f"  roundtrip OK: {sum(1 for r in rows if r['roundtrip_max_error'] < 1e-5)}/24")
    print(f"  outer-eval leakage: {sum(r['outer_eval_rows_used'] for r in rows)} rows total")
    print(f"  test leakage: {sum(r['test_rows_used'] for r in rows)} rows total")

    print("\n=== PROVENANCE (fold-local vs global) ===")
    for r in rows:
        same_as_global = abs(r["y_mean"] - gmean) < 1e-6 and abs(r["y_std"] - gstd) < 1e-6
        if same_as_global:
            print(f"  WARN {r['candidate']}/{r['fold']}/{r['stage']}: matches global exactly")

    unique_keys = {(round(r["y_mean"], 4), round(r["y_std"], 4)) for r in rows}
    print(f"\nDistinct (y_mean, y_std) values across 24 contracts: {len(unique_keys)}")

    out_dir = PROJECT_ROOT / "artifacts" / "phase44_runtime_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase44_24_fold_local_y_scaler_provenance.json"
    out_path.write_text(json.dumps({
        "global_phase9_ys1_mean": gmean,
        "global_phase9_ys1_std": gstd,
        "rows": rows,
        "summary": {
            "total": len(rows),
            "passing": 24 - fail_count,
            "failing": fail_count,
        },
    }, indent=2, default=str))
    print(f"\nWrote provenance report → {out_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
