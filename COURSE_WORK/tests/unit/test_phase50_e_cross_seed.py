"""Phase 50-E focused tests — cross-seed aggregation + contrasts + prevalence + rank stability."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import pytest

from course_work.analysis.error_regime_analysis import contract, sources
from course_work.analysis.error_regime_analysis.materialize_e import main as e_main


def test_e_runs_clean():
    r = e_main()
    assert r["status"] == "PASS"


def test_cross_seed_summary_present():
    p = Path("artifacts/error_by_regime/regime_cross_seed_summary.csv")
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    # 18 keys (family, label) combinations x multiple metrics
    assert len(rows) > 0


def test_cross_seed_summary_ddof_one():
    """Sample SD must use ddof=1."""
    # Manually compute for one metric and verify
    p = Path("artifacts/error_by_regime/regime_cross_seed_summary.csv")
    rows = list(csv.DictReader(p.open()))
    # Pick one (family, label, metric) row and check sd
    sample = next(r for r in rows if r["metric"] == "mae_wh")
    vals = [
        float(sample["seed_42_value"]),
        float(sample["seed_123_value"]),
        float(sample["seed_2026_value"]),
    ]
    m = sum(vals) / len(vals)
    var_ddof1 = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    expected_sd = math.sqrt(var_ddof1)
    actual_sd = float(sample["sd_ddof1"])
    assert abs(actual_sd - expected_sd) < 1e-9


def test_cross_seed_summary_no_best_seed():
    """No 'best' seed selection fields."""
    p = Path("artifacts/error_by_regime/regime_cross_seed_summary.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        assert "best_seed" not in r
        assert "winner" not in r


def test_rmse_lift_three_forms_present():
    p = Path("artifacts/error_by_regime/regime_rmse_lift.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows[:5]:
        for k in ("rmse_lift_wh", "rmse_lift_ratio", "rmse_lift_pct"):
            assert k in r
            assert r[k] != ""


def test_rmse_lift_invariant_ratio_equals_pct_over_100():
    """rmse_lift_pct = 100 * rmse_lift_ratio (within rounding)."""
    p = Path("artifacts/error_by_regime/regime_rmse_lift.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows[:10]:
        ratio = float(r["rmse_lift_ratio"])
        pct = float(r["rmse_lift_pct"])
        # Both written with 6-decimal precision; invariant holds to ~6 decimal points
        assert abs(pct - 100 * ratio) < 1e-3


def test_pairwise_contrasts_predeclared_only():
    """Only 6 predeclared contrasts are emitted (R5 uses hardest/easiest, not pairwise)."""
    p = Path("artifacts/error_by_regime/regime_pairwise_contrasts.csv")
    rows = list(csv.DictReader(p.open()))
    expected = {
        "TL_HIGH vs TL_LOW",
        "TL_HIGH vs TL_MID",
        "EXTREME_HIGH vs NON_EXTREME",
        "CHANGE_RAPID vs CHANGE_NORMAL",
        "DIR_UP vs DIR_DOWN",
        "DAY_WEEKEND vs DAY_WEEKDAY",
    }
    actual = {r["contrast_id"] for r in rows}
    assert actual == expected


def test_pairwise_contrasts_per_seed():
    """Each contrast has 3 seed rows."""
    p = Path("artifacts/error_by_regime/regime_pairwise_contrasts.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    counts = defaultdict(int)
    for r in rows:
        counts[r["contrast_id"]] += 1
    for cid, c in counts.items():
        assert c == 3, f"contrast {cid} has {c} rows, expected 3"


def test_train_vs_test_prevalence_present():
    p = Path("artifacts/error_by_regime/regime_train_vs_test_prevalence.csv")
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    # 18 labels (3 + 2 + 3 + 4 + 4 + 2)
    assert len(rows) == 18


def test_train_vs_test_prevalence_fractions_sum_to_one():
    p = Path("artifacts/error_by_regime/regime_train_vs_test_prevalence.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    train_sums = defaultdict(float)
    test_sums = defaultdict(float)
    for r in rows:
        train_sums[r["regime_family"]] += float(r["fraction_train"])
        test_sums[r["regime_family"]] += float(r["fraction_test"])
    for fam in train_sums:
        assert abs(train_sums[fam] - 1.0) < 1e-9, f"train {fam}: {train_sums[fam]}"
        assert abs(test_sums[fam] - 1.0) < 1e-9, f"test {fam}: {test_sums[fam]}"


def test_rank_stability_present():
    p = Path("artifacts/error_by_regime/regime_rank_stability.csv")
    assert p.exists()


def test_rank_stability_three_seeds_per_row():
    p = Path("artifacts/error_by_regime/regime_rank_stability.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows[:5]:
        assert int(r["rank_seed_42"]) >= 1
        assert int(r["rank_seed_123"]) >= 1
        assert int(r["rank_seed_2026"]) >= 1


def test_cross_seed_no_3n_pooling():
    """Cross-seed summary uses per-seed SD (mean of 3 values); NOT a single SD over 8883 rows."""
    p = Path("artifacts/error_by_regime/regime_cross_seed_summary.csv")
    rows = list(csv.DictReader(p.open()))
    # Spot check: for a single metric, n_seeds = 3
    sample = rows[0]
    assert int(sample["n_seeds"]) == 3


def test_phase47_unchanged_post_50e():
    import hashlib
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_phase48_unchanged_post_50e():
    import hashlib
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50e():
    import hashlib
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)
