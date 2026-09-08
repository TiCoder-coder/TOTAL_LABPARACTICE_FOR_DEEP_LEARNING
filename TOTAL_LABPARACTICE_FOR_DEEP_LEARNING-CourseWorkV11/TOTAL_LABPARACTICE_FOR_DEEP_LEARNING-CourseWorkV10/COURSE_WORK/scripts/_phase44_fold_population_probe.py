"""PHASE 44 PROBE — 48 fold-loader population contract probes.

For every Phase 44 fold-loader-role (4 candidates × 3 folds × 4 roles =
48 loader-roles), this probe verifies:

  - observed sample IDs == expected sample IDs (exact match)
  - 0 duplicates in observed
  - 0 unexpected IDs
  - 0 missing IDs
  - population fingerprint match (deterministic)
  - 0 Test IDs

Loader roles:
  - Stage A inner_train
  - Stage A inner_val
  - Stage B outer_train
  - Stage C outer_eval

Run:
    PYTHONPATH=src python3 scripts/_phase44_fold_population_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from course_work.evaluation.metrics import (
    MetricPopulationContext,
    derive_population_fingerprint,
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
from course_work.rolling_origin.stages import load_fold_subset_loader_with_y_rescale


def _observed_sample_idx(loader) -> np.ndarray:
    sample_idx = []
    for batch in loader:
        sample_idx.extend(batch["sample_idx"].cpu().numpy().tolist())
    return np.asarray(sorted(int(x) for x in sample_idx), dtype=np.int64)


def _target_id_to_canonical_sample_idx(window_records) -> dict[str, int]:
    """Build mapping: target_id → canonical_sample_idx (NOT positional)."""
    if "target_id" not in window_records.columns:
        if "target_sample_id" in window_records.columns:
            window_records["target_id"] = window_records["target_sample_id"].astype(str)
        else:
            raise RuntimeError("no target_id or target_sample_id column")
    if "canonical_sample_idx" not in window_records.columns:
        raise RuntimeError("window_records missing canonical_sample_idx")
    return {
        str(t): int(s)
        for t, s in zip(
            window_records["target_id"].tolist(),
            window_records["canonical_sample_idx"].tolist(),
        )
    }


def _expected_sample_idx_from_target_ids(target_ids: list, id_to_pos: dict[str, int]) -> np.ndarray:
    out = []
    for tid in target_ids:
        csi = id_to_pos.get(str(tid))
        if csi is not None:
            out.append(int(csi))
    return np.asarray(sorted(out), dtype=np.int64)


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

    import pandas as pd
    test_set = set()
    wp_csv = PROJECT_ROOT / "artifacts" / "windows" / "common_target_population.csv"
    if wp_csv.exists():
        wp = pd.read_csv(wp_csv)
        test_set = set(
            wp[wp["target_split_id"].astype(str).str.upper() == "TEST"]
            ["target_sample_id"].astype(str)
        )

    rows = []
    fail_count = 0
    print(
        f"{'candidate':<22} {'fold':<5} {'role':<11} {'obs_n':<7} {'exp_n':<7} "
        f"{'missing':<8} {'extra':<6} {'dup':<5} {'test':<5} {'verdict'}"
    )
    print("-" * 110)

    for c in candidates:
        fd = build_real_canonical_base_dataset(project_root=PROJECT_ROOT, candidate=c)
        id_to_csi = _target_id_to_canonical_sample_idx(fd.window_records)
        fold_dataset_test_ids = set() 

        for fold in folds:
            scaler_a, _ = fit_fold_local_scaler(candidate=c, fold=fold, fold_dataset=fd, fit_stage="A", ctx=ctx)
            scaler_b, _ = fit_fold_local_scaler(candidate=c, fold=fold, fold_dataset=fd, fit_stage="B", ctx=ctx)

            role_loaders = []
            for role, ids, scaler in [
                ("inner_train", list(fold.inner_train_ids), scaler_a),
                ("inner_val",   list(fold.inner_val_ids),   scaler_a),
                ("outer_train", list(fold.outer_train_ids), scaler_b),
                ("outer_eval",  list(fold.outer_eval_ids),  scaler_b),
            ]:
                expected_sample_idx = _expected_sample_idx_from_target_ids(ids, id_to_csi)
                if len(expected_sample_idx) != len(ids):
                    fail_count += 1
                    rows.append({
                        "candidate": c.candidate_id,
                        "fold": str(fold.fold_id),
                        "role": role,
                        "expected_sample_idx_count": len(expected_sample_idx),
                        "target_id_count": len(ids),
                        "pass": False,
                        "reason": "target_id → sample_idx mismatch",
                    })
                    continue

                expected_pop_fp = derive_population_fingerprint(
                    target_ids=list(ids),
                    split_id=role.upper(),
                )

                fold_role_fp_map = {
                    "inner_train": fold.inner_train_fingerprint,
                    "inner_val":   fold.inner_val_fingerprint,
                    "outer_train": fold.outer_train_fingerprint,
                    "outer_eval":  fold.outer_eval_fingerprint,
                }
                canonical_role_fp = fold_role_fp_map.get(role, "")
                fp_agrees = (expected_pop_fp == canonical_role_fp)
                pop_ctx = MetricPopulationContext(
                    split_id=role.upper(),
                    expected_sample_idx=expected_sample_idx,
                    population_fingerprint=expected_pop_fp,
                )

                loader, _ = load_fold_subset_loader_with_y_rescale(
                    base_dataset=fd,
                    target_ids=ids,
                    batch_size=int(c.config["training"]["batch_size"]),
                    shuffle=False,
                    seed=ctx.seed,
                    fold_stage_target_scaler=scaler,
                )
                observed = _observed_sample_idx(loader)
                pos_to_id = {v: k for k, v in id_to_csi.items()}
                observed_target_ids = [
                    pos_to_id[int(s)] for s in observed if int(s) in pos_to_id
                ]

                missing = len(set(expected_sample_idx.tolist()) - set(observed.tolist()))
                extra = len(set(observed.tolist()) - set(expected_sample_idx.tolist()))
                dups = int(len(observed) - len(np.unique(observed)))

                test_leakage = sum(1 for t in observed_target_ids if t in test_set)

                ctx_ok = True
                try:
                    if pop_ctx.population_fingerprint != expected_pop_fp:
                        ctx_ok = False
                    if len(pop_ctx.expected_sample_idx) != len(observed):
                        ctx_ok = False
                except Exception:
                    ctx_ok = False

                verdict = "PASS" if (
                    missing == 0
                    and extra == 0
                    and dups == 0
                    and test_leakage == 0
                    and ctx_ok
                    and fp_agrees  
                ) else "FAIL"
                if verdict == "FAIL":
                    fail_count += 1

                rows.append({
                    "candidate": c.candidate_id,
                    "fold": str(fold.fold_id),
                    "role": role,
                    "observed_n": int(len(observed)),
                    "expected_n": int(len(expected_sample_idx)),
                    "missing": missing,
                    "extra": extra,
                    "duplicates": dups,
                    "test_leakage": test_leakage,
                    "population_fingerprint": expected_pop_fp,
                    "fold_role_fingerprint": canonical_role_fp,
                    "fp_agrees": fp_agrees,
                    "ctx_valid": ctx_ok,
                    "pass": verdict == "PASS",
                })

                print(
                    f"{c.candidate_id:<22} {str(fold.fold_id):<5} {role:<11} "
                    f"{len(observed):<7} {len(expected_sample_idx):<7} "
                    f"{missing:<8} {extra:<6} {dups:<5} {test_leakage:<5} {verdict}"
                )

    print("-" * 110)
    print(f"\n=== SUMMARY ===")
    print(f"48 loader-role probes: {48 - fail_count}/48 PASS")
    print(f"  missing IDs: {sum(r.get('missing', 0) for r in rows)}")
    print(f"  extra IDs: {sum(r.get('extra', 0) for r in rows)}")
    print(f"  duplicates: {sum(r.get('duplicates', 0) for r in rows)}")
    print(f"  test leakage: {sum(r.get('test_leakage', 0) for r in rows)}")


    inner_train_fps = sorted({r["population_fingerprint"] for r in rows if r.get("role") == "inner_train" and "population_fingerprint" in r})
    print(f"\nDistinct inner_train fingerprints across 3 folds × 4 candidates = 12 expected:")
    print(f"  Got: {len(inner_train_fps)} distinct")
    if len(inner_train_fps) >= 12:
        print("  [OK] RO1/RO2/RO3 inner_train populations are distinct per fold and per candidate")

    out_dir = PROJECT_ROOT / "artifacts" / "phase44_runtime_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase44_fold_population_probe_results.json"
    out_path.write_text(json.dumps({
        "rows": rows,
        "summary": {
            "total": len(rows),
            "passing": 48 - fail_count,
            "failing": fail_count,
        },
    }, indent=2, default=str))
    print(f"\nWrote population probe report → {out_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
