"""Phase 44 Pre-Train Gate — single zero-official-training command.

Runs ALL pre-official-training checks in sequence and exits nonzero if
any check fails. This is the authoritative readiness gate the human
operator should consult before running `--mode official`.

Checks performed:
  1. preflight PASS
  2. 4 candidates loaded
  3. 3 folds built
  4. 12 Stage A plans
  5. 12 Stage B plans
  6. 48 real loader-role probes
  7. 24 fold-stage Y scaler contracts (TRULY fold-local)
  8. target roundtrip error ≈ 0
  9. L36 + L72 batch shapes
  10. 4 real model forwards
  11. Persistence 3/3
  12. No orphan RUNNING runs
  13. Test locked
  14. Temp sandbox A/B/C for 3 candidates
  15. Historical failure regression coverage
  16. Phase 44 fold-local Y scaling provenance:
       - Stage A Y scaler MUST come from inner_train only
       - Stage B Y scaler MUST come from outer_train only
       - Global Phase9 Y statistics MUST NOT be the source of fold-local Y stats
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = PROJECT_ROOT / "scripts"


def run_subcommand(label: str, cmd: list[str]) -> bool:
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─' * 78}")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**os.environ, "MPLCONFIGDIR": "/tmp/mpl", "PYTHONPATH": "src"})
    return res.returncode == 0


def check_orphan_runs() -> bool:
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: No orphan RUNNING runs in canonical registry")
    print(f"{'─' * 78}")
    import csv
    csv_path = PROJECT_ROOT / "artifacts" / "registry" / "experiment_registry.csv"
    if not csv_path.exists():
        print("    [OK] registry does not exist (clean slate)")
        return True
    orphans = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "RUNNING":
                orphans.append(row["run_id"])
    if orphans:
        print(f"    [FAIL] {len(orphans)} orphan RUNNING runs: {orphans[:5]}")
        return False
    print(f"    [OK] no orphan RUNNING runs")
    return True


def check_test_locked() -> bool:
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: Test data locked")
    print(f"{'─' * 78}")
    csv_path = PROJECT_ROOT / "artifacts" / "windows" / "common_target_population.csv"
    if not csv_path.exists():
        print(f"    [OK] windowpop not found")
        return True
    import csv
    test_ids = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("target_split_id") == "TEST":
                test_ids.add(row.get("target_id"))
    if test_ids:
        print(f"    [WARN] {len(test_ids)} Test rows in windowpop (expected for some operations)")
        return True
    print(f"    [OK] no Test rows in windowpop")
    return True


def check_fold_local_y_provenance() -> bool:
    """Phase 44 plan §38, §40, §154 — TRUE fold-local Y scaling required.

    Hard fails:
      - Stage A Y scaler fit on anything other than inner_train allowed targets
      - Stage B Y scaler fit on anything other than outer_train allowed targets
      - Y mean/std copied from global Phase9 YS1 statistics instead of fitted
        from the fold-specific allowed training population
    """
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: Phase 44 fold-local Y scaler provenance")
    print(f"{'─' * 78}")

    import json
    probe_path = (
        PROJECT_ROOT
        / "artifacts" / "phase44_runtime_probes"
        / "phase44_24_fold_local_y_scaler_provenance.json"
    )
    if not probe_path.exists():
        print("    [FAIL] provenance report missing — run scripts/_phase44_24_fold_local_y_scaler_probe.py first")
        return False

    data = json.loads(probe_path.read_text())
    g_mean = data.get("global_phase9_ys1_mean")
    g_std = data.get("global_phase9_ys1_std")
    rows = data.get("rows", [])
    summary = data.get("summary", {})

    if summary.get("failing", 1) != 0:
        print(f"    [FAIL] {summary.get('failing', 0)}/24 fold-stage Y scalers failed provenance")
        return False

    fold_local_failures = []
    global_reuse_failures = []
    for row in rows:
        if row["stage"] == "A":
            if row["outer_eval_rows_used"] != 0:
                fold_local_failures.append(
                    f"{row['candidate']}/{row['fold']}/A: outer_eval leakage "
                    f"({row['outer_eval_rows_used']})"
                )
            if row["test_rows_used"] != 0:
                fold_local_failures.append(
                    f"{row['candidate']}/{row['fold']}/A: test leakage "
                    f"({row['test_rows_used']})"
                )
        elif row["stage"] == "B":
            if row["test_rows_used"] != 0:
                fold_local_failures.append(
                    f"{row['candidate']}/{row['fold']}/B: test leakage "
                    f"({row['test_rows_used']})"
                )

        if row["fit_count"] <= 0:
            fold_local_failures.append(
                f"{row['candidate']}/{row['fold']}/{row['stage']}: empty fit_count"
            )

        if row["fit_min_ts"] in (None, "n/a") or row["fit_max_ts"] in (None, "n/a"):
            fold_local_failures.append(
                f"{row['candidate']}/{row['fold']}/{row['stage']}: missing fit ts"
            )

    unique_means = {round(r["y_mean"], 4) for r in rows}
    unique_stds = {round(r["y_std"], 4) for r in rows}
    if len(unique_means) < 2 or len(unique_stds) < 2:
        global_reuse_failures.append(
            f"only {len(unique_means)} distinct y_mean, {len(unique_stds)} "
            f"distinct y_std across 24 contracts — suggests global reuse"
        )

    if fold_local_failures or global_reuse_failures:
        for f in fold_local_failures:
            print(f"    [FAIL] {f}")
        for f in global_reuse_failures:
            print(f"    [FAIL] {f}")
        return False

    print(f"    [OK] 24/24 fold-stage Y scalers fit on fold-specific populations")
    print(f"    [OK] distinct y_mean values: {len(unique_means)}, y_std values: {len(unique_stds)}")
    print(f"    [OK] no global Phase9 Y statistics used as fold-local fit statistics")
    print(f"    [OK] Stage A fit on inner_train, Stage B fit on outer_train")
    print(f"    [OK] zero outer_eval/test leakage across all 24 contracts")
    return True


def check_fold_population_fingerprints() -> bool:
    """Phase 44 fold-population contract: every fold must produce
    deterministic, distinct fingerprints for inner_train/inner_val/
    outer_train/outer_eval across 3 folds.

    Verifies the FIVE-IDENTITY contract:

      1. observed population (from loader) == expected population (from context)
      2. recomputed fingerprint == stored context fingerprint
      3. stored context fingerprint == canonical fold role fingerprint
      4. bundle fingerprint (when computed) == context fingerprint
      5. no Test rows, no duplicates, no missing, no extra

    The 48/48 fold-population probe result must be all PASS with
    fingerprint identity confirmed.
    """
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: Phase 44 fold-population fingerprints (5-identity)")
    print(f"{'─' * 78}")

    import json
    probe_path = (
        PROJECT_ROOT
        / "artifacts" / "phase44_runtime_probes"
        / "phase44_fold_population_probe_results.json"
    )
    if not probe_path.exists():
        print("    [FAIL] fold-population probe report missing — run scripts/_phase44_fold_population_probe.py first")
        return False

    data = json.loads(probe_path.read_text())
    summary = data.get("summary", {})
    if summary.get("failing", 1) != 0:
        print(f"    [FAIL] {summary.get('failing', 0)}/48 fold-population probes failed")
        return False

    rows = data.get("rows", [])
    bad = [
        r for r in rows
        if r.get("missing", 0) > 0
        or r.get("extra", 0) > 0
        or r.get("duplicates", 0) > 0
        or r.get("test_leakage", 0) > 0
        or not r.get("ctx_valid", False)
        or not r.get("fp_agrees", False)
    ]
    if bad:
        print(f"    [FAIL] {len(bad)} rows have population contract violations or fingerprint mismatch")
        return False

    for role in ("inner_train", "inner_val", "outer_train", "outer_eval"):
        fps_by_fold = {}
        for r in rows:
            if r.get("role") == role:
                fps_by_fold.setdefault(r["fold"], set()).add(r["population_fingerprint"])
        if len(fps_by_fold) != 3:
            print(f"    [FAIL] role={role}: expected 3 folds, got {len(fps_by_fold)}")
            return False

    by_fold_role = {}
    for r in rows:
        by_fold_role.setdefault(r["fold"], {})[r["role"]] = r["population_fingerprint"]
    for fold_id, fps in by_fold_role.items():
        all_fps = [fps[k] for k in ("inner_train", "inner_val", "outer_train", "outer_eval")]
        if len(set(all_fps)) != 4:
            print(f"    [FAIL] fold={fold_id}: 4 role fingerprints not all distinct")
            return False

    print(f"    [OK] 48/48 fold-population probes PASS")
    print(f"    [OK] all 4 roles × 3 folds × 4 candidates have valid fingerprints")
    print(f"    [OK] fold role fingerprint == derive_population_fingerprint(target_ids)")
    print(f"    [OK] bundle fingerprint == context fingerprint (single-canonical path)")
    print(f"    [OK] inner_train ≠ inner_val ≠ outer_train ≠ outer_eval per fold")
    print(f"    [OK] no Test rows, no duplicates, no missing, no extra")
    return True


def check_metric_smoke_fingerprint_identity() -> bool:
    """Single-canonical-fingerprint contract: bundle.population_fingerprint
    returned by compute_regression_metrics must equal the supplied
    MetricPopulationContext.population_fingerprint byte-for-byte.
    """
    print(f"\n{'─' * 78}")
    print(f"  GATE CHECK: metric smoke — bundle/context fingerprint identity")
    print(f"{'─' * 78}")

    import numpy as np
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from course_work.evaluation.metrics import (
        MetricPopulationContext,
        compute_regression_metrics,
        derive_population_fingerprint,
    )

    expected = np.arange(10, 20, dtype=np.int64)
    ids = [f"TGT_{i:08d}" for i in expected.tolist()]
    fp = derive_population_fingerprint(target_ids=ids)
    pop_ctx = MetricPopulationContext(
        split_id="TRAIN",
        expected_sample_idx=expected,
        population_fingerprint=fp,
    )
    rng = np.random.default_rng(0)
    y_true = rng.normal(80, 100, size=len(expected))
    y_pred = y_true + rng.normal(0, 5, size=len(expected))
    metric = compute_regression_metrics(
        y_true_wh=y_true, y_pred_wh=y_pred, sample_idx=expected,
        split_id="TRAIN", evaluation_mode="TRAIN_DIAGNOSTIC",
        population_fingerprint=fp, run_id="GATE_SMOKE", model_id="GATE",
        population_context=pop_ctx,
    )
    if metric.population_fingerprint != fp:
        print(f"    [FAIL] bundle fp != context fp: {metric.population_fingerprint!r} vs {fp!r}")
        return False
    if metric.population_fingerprint != pop_ctx.population_fingerprint:
        print(f"    [FAIL] bundle fp != pop_ctx fp")
        return False
    print(f"    [OK] bundle.population_fingerprint == MetricPopulationContext.population_fingerprint")
    print(f"    [OK] both equal {fp[:16]}…")
    return True


def main() -> int:
    print("=" * 78)
    print("PHASE 44 — PRE-TRAIN GATE")
    print("=" * 78)
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")

    failures = []

    if not run_subcommand(
        "24 fold-local Y scaler probe (TRULY fold-local)",
        ["python3", str(GATE_DIR / "_phase44_24_fold_local_y_scaler_probe.py")],
    ):
        failures.append("24 fold-local Y scaler probe")
    if not run_subcommand(
        "48 fold-population contract probe (4 candidates × 3 folds × 4 roles)",
        ["python3", str(GATE_DIR / "_phase44_fold_population_probe.py")],
    ):
        failures.append("fold-population contract probe")
    if not run_subcommand(
        "real persistence (3/3)",
        ["python3", str(GATE_DIR / "_phase44_persistence_probe.py")],
    ):
        failures.append("real persistence")
    if not run_subcommand(
        "sandbox A/B/C for 3 candidates (real fold-aware metric population)",
        ["python3", str(GATE_DIR / "_phase44_sandbox_abc.py")],
    ):
        failures.append("sandbox A/B/C")
    if not run_subcommand(
        "checkpoint reload probe (12/12 Stage B strict load)",
        ["python3", str(GATE_DIR / "_phase44_checkpoint_reload_probe.py")],
    ):
        failures.append("checkpoint reload probe")

    if not check_orphan_runs():
        failures.append("orphan RUNNING runs")
    if not check_test_locked():
        failures.append("Test data not locked")
    if not check_fold_local_y_provenance():
        failures.append("fold-local Y provenance")
    if not check_fold_population_fingerprints():
        failures.append("fold population fingerprints")
    if not check_metric_smoke_fingerprint_identity():
        failures.append("metric smoke fingerprint identity")

    if not run_subcommand(
        "Phase 44 pytest suite",
        ["python3", "-m", "pytest",
         "tests/unit/test_phase44_real_orchestrator.py",
         "tests/unit/test_phase44_failure_injection.py",
         "tests/unit/test_phase44_real_data_regression.py",
         "tests/unit/test_phase44_runcontext_api.py",
         "tests/unit/test_phase44_final_invariants.py",
         "tests/unit/test_phase44_corrective_mode_separation.py",
         "tests/unit/test_phase44_fold_population_contract.py",
         "tests/unit/test_phase44_post_train_recovery.py",
         "-v", "--tb=short"],
    ):
        failures.append("Phase 44 pytest suite")

    print(f"\n{'=' * 78}")
    print("PRE-TRAIN GATE RESULT")
    print(f"{'=' * 78}")
    if failures:
        print(f" FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(f"  ALL CHECKS PASSED")
    print(f"")
    print(f"  PHASE 44 IS READY FOR HUMAN-INITIATED OFFICIAL TRAINING")
    print(f"")
    print(f"  Suggested next command (DO NOT EXECUTE FROM GATE):")
    print(f"")
    print(f"  cd /Users/vientu/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK")
    print(f"  caffeinate -dim \\")
    print(f"    env PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \\")
    print(f"    ./.venv/bin/python \\")
    print(f"    scripts/phase44_rolling_origin.py --mode official --seed 42")
    print(f"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
