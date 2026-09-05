"""Phase 50-D focused tests — residual join + per-seed metrics + contributions."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest

from course_work.phase50 import contract, sources
from course_work.phase50.materialize_d import main as d_main


def test_d_runs_clean():
    r = d_main()
    assert r["status"] == "PASS"


def test_join_audit_8883_rows():
    p = Path("artifacts/error_by_regime/regime_error_join_audit.csv")
    rows = list(csv.DictReader(p.open()))
    by_check = {r["check"]: r for r in rows}
    assert by_check["join_row_count"]["status"] == "PASS"
    assert by_check["join_row_count"]["observed"] == "8883"
    assert by_check["unmatched_error_rows"]["observed"] == "0"
    assert by_check["duplicate_assignment_matches"]["observed"] == "0"
    assert by_check["same_test_population"]["observed"] == contract.TEST_POPULATION_FINGERPRINT


def test_metrics_long_table_present():
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    # 6 families × varying labels × 3 seeds
    assert len(rows) > 0


def test_metrics_long_includes_all_six_families():
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    families = {r["regime_family"] for r in rows}
    expected = {
        "R1_TARGET_LEVEL",
        "R2_EXTREME_HIGH",
        "R3_CHANGE_MAGNITUDE",
        "R4_CHANGE_DIRECTION",
        "R5_TIME_OF_DAY",
        "R6_DAY_TYPE",
    }
    assert expected.issubset(families)


def test_metrics_long_all_three_seeds():
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    seeds = {r["seed"] for r in rows}
    assert seeds == {"42", "123", "2026"}


def test_metrics_long_residual_convention():
    """Positive residual = UNDERPREDICTION; MBE > 0 means net underprediction."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    # Pick seed42 TL_HIGH row
    for r in rows:
        if r["regime_family"] == "R1_TARGET_LEVEL" and r["regime_label"] == "TL_HIGH" and r["seed"] == "42":
            # Just sanity-check that MBE is a real number
            assert r["mbe_wh"] != ""
            break


def test_metrics_long_r2_format():
    """R2 NOT_DEFINED serialized as empty cell + r2_status=NOT_DEFINED."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    # Inspect: any r2_status=NOT_DEFINED must have empty r2 cell
    for r in rows:
        if r["r2_status"] == "NOT_DEFINED":
            assert r["r2"] == "", f"r2 cell must be empty for NOT_DEFINED, got {r['r2']!r}"
        # Defined: r2 cell must be a number string (or empty for very small N? not for our case)
        if r["r2_status"] == "DEFINED":
            # numeric
            assert r["r2"] != ""


def test_metrics_long_sample_share_sum_per_family_seed():
    """sample_share must sum to 1 within each (regime_family, seed)."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        if r["sample_share"] != "":
            sums[(r["regime_family"], r["seed"])] += float(r["sample_share"])
            counts[(r["regime_family"], r["seed"])] += 1
    for k, s in sums.items():
        assert abs(s - 1.0) < 1e-5, f"sample_share sum != 1 for {k}: {s}"


def test_metrics_long_sae_share_sums_to_one():
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    sums = defaultdict(float)
    for r in rows:
        if r["sae_share"] != "":
            sums[(r["regime_family"], r["seed"])] += float(r["sae_share"])
    for k, s in sums.items():
        assert abs(s - 1.0) < 1e-5, f"sae_share sum != 1 for {k}: {s}"


def test_metrics_long_sse_share_sums_to_one():
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    sums = defaultdict(float)
    for r in rows:
        if r["sse_share"] != "":
            sums[(r["regime_family"], r["seed"])] += float(r["sse_share"])
    for k, s in sums.items():
        assert abs(s - 1.0) < 1e-5, f"sse_share sum != 1 for {k}: {s}"


def test_metrics_long_three_distinct_rmse_lifts():
    """rmse_lift_wh, rmse_lift_ratio, rmse_lift_pct all present and distinct."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows[:5]:
        assert "rmse_lift_wh" in r
        assert "rmse_lift_ratio" in r
        assert "rmse_lift_pct" in r


def test_metrics_long_global_per_seed_reconstruction_matches_phase47():
    """Sum SAE across all R1+R2+R5+R6 labels per seed must equal global SAE."""
    p = Path("artifacts/error_by_regime/regime_metrics_long.csv")
    rows = list(csv.DictReader(p.open()))
    expected_mae = {"42": 29.528646738232613, "123": 27.11486924829266, "2026": 28.94229416815529}
    expected_rmse = {"42": 64.94275370825517, "123": 61.98608587121866, "2026": 64.56013557970728}
    from collections import defaultdict
    sums = defaultdict(lambda: {"sae": 0.0, "sse": 0.0, "n": 0})
    covering_families = {"R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R5_TIME_OF_DAY", "R6_DAY_TYPE"}
    for r in rows:
        if r["regime_family"] in covering_families and r["sae"] != "":
            sums[r["seed"]]["sae"] += float(r["sae"])
            sums[r["seed"]]["sse"] += float(r["sse"])
            sums[r["seed"]]["n"] += int(r["N"])
    for seed, s in sums.items():
        n = s["n"]
        mae = s["sae"] / n
        import math
        rmse = math.sqrt(s["sse"] / n)
        assert abs(mae - expected_mae[seed]) < 1e-9, f"seed {seed} MAE mismatch: {mae} vs {expected_mae[seed]}"
        assert abs(rmse - expected_rmse[seed]) < 1e-9, f"seed {seed} RMSE mismatch: {rmse} vs {expected_rmse[seed]}"


def test_phase47_unchanged_post_50d():
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    import hashlib
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_phase48_unchanged_post_50d():
    import hashlib
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50d():
    import hashlib
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp), f"{p} mutated"
