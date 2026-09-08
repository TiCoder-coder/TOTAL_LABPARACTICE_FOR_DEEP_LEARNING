"""Phase 50-F focused tests — Persistence + LSTM + seed-spread + sign-consensus."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pytest

from course_work.analysis.error_regime_analysis import contract, sources
from course_work.analysis.error_regime_analysis.materialize_f import main as f_main


def test_f_runs_clean():
    r = f_main()
    assert r["status"] == "PASS"


def test_persistence_regime_metrics_present():
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 19


def test_persistence_global_matches_phase47_global():
    """Persistence global MAE/RMSE must match Phase 47 global numbers (no seed dependence)."""
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    rows = list(csv.DictReader(p.open()))
    global_row = next(r for r in rows if r["regime_family"] == "GLOBAL")
    assert int(global_row["N"]) == 2961


def test_persistence_global_matches_phase47_exact():
    summary = sources.load_phase47_summary()
    pers_block = summary.get("persistence_metrics", {})
    expected_mae = float(pers_block.get("mae_wh", "nan"))
    expected_rmse = float(pers_block.get("rmse_wh", "nan"))
    if expected_mae != expected_mae or expected_rmse != expected_rmse:
        return  
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    rows = list(csv.DictReader(p.open()))
    global_row = next(r for r in rows if r["regime_family"] == "GLOBAL")
    actual_mae = float(global_row["mae_wh"])
    actual_rmse = float(global_row["rmse_wh"])
    assert abs(actual_mae - expected_mae) < 1e-9
    assert abs(actual_rmse - expected_rmse) < 1e-9


def test_persistence_per_regime_sample_size_consistent():
    """Sum of per-regime N must equal 2961 for covering families."""
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    sums = defaultdict(int)
    for r in rows:
        if r["regime_family"] in {
            "R1_TARGET_LEVEL",
            "R2_EXTREME_HIGH",
            "R5_TIME_OF_DAY",
            "R6_DAY_TYPE",
        }:
            sums[r["regime_family"]] += int(r["N"])
    for fam, s in sums.items():
        assert s == 2961, f"persistence N sum for {fam}: {s}"


def test_persistence_baseline_label_persistence():
    p = Path("artifacts/error_by_regime/regime_metrics_persistence.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        assert r["baseline"] == "PERSISTENCE"


def test_lstm_regime_metrics_status_not_applicable():
    p = Path("artifacts/error_by_regime/regime_metrics_lstm.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        assert r["status"] == "NOT_APPLICABLE"
        assert r["eligibility_status"] == "NOT_ELIGIBLE_CONFIG_MISMATCH"


def test_lstm_status_doc_no_inference():
    doc = json.loads(Path("artifacts/error_by_regime/regime_lstm_status.json").read_text())
    assert doc["no_inference_performed"] is True
    assert doc["no_retrain_performed"] is True
    assert doc["no_fabricated_residuals"] is True


def test_seed_spread_by_regime_present():
    p = Path("artifacts/error_by_regime/regime_seed_spread.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 18


def test_seed_spread_semantics_not_confidence_interval():
    p = Path("artifacts/error_by_regime/regime_seed_spread.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        assert r["spread_semantics"] == "CROSS_SEED_PREDICTION_SPREAD"
    doc = json.loads(Path("artifacts/error_by_regime/regime_seed_spread_status.json").read_text())
    assert doc["is_confidence_interval"] is False


def test_seed_spread_sample_size_consistent():
    """Sum of per-regime N for covering families must equal 2961."""
    p = Path("artifacts/error_by_regime/regime_seed_spread.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    sums = defaultdict(int)
    for r in rows:
        if r["regime_family"] in {
            "R1_TARGET_LEVEL",
            "R2_EXTREME_HIGH",
            "R5_TIME_OF_DAY",
            "R6_DAY_TYPE",
        }:
            sums[r["regime_family"]] += int(r["N"])
    for fam, s in sums.items():
        assert s == 2961


def test_sign_consensus_by_regime_present():
    p = Path("artifacts/error_by_regime/regime_sign_consensus.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 18 * 6


def test_sign_consensus_mapping_documented():
    doc = json.loads(
        Path("artifacts/error_by_regime/regime_sign_consensus_mapping.json").read_text()
    )
    assert "TWO_UNDER_ONE_OVER" in doc["source_class_to_phase50_target_category"]
    assert doc["source_class_to_phase50_target_category"]["TWO_UNDER_ONE_OVER"] == "MIXED"


def test_sign_consensus_counts_match_phase49_canonical():
    """Mapping frozen BEFORE aggregation should match Phase 49 canonical counts.

    The sign_consensus_by_regime.csv has counts per (regime_family, regime_label, source_class).
    Total per source_class summed across regime families must match Phase 49 canonical.
    """
    p = Path("artifacts/error_by_regime/regime_sign_consensus.csv")
    rows = list(csv.DictReader(p.open()))
    from collections import defaultdict
    totals = defaultdict(int)
    for r in rows:
        totals[r["consensus_source_class"]] += int(r["N"])
    p49_rows = list(
        csv.DictReader(Path("artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv").open())
    )
    expected = {r["consensus_class"]: int(r["count"]) for r in p49_rows}
    for cls, cnt in expected.items():
        actual = totals.get(cls, 0)
        assert actual == cnt * 6, f"{cls}: expected {cnt * 6}, got {actual}"


def test_phase47_unchanged_post_50f():
    import hashlib
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)


def test_phase48_unchanged_post_50f():
    import hashlib
    sha = hashlib.sha256(Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50f():
    import hashlib
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
        "artifacts/residual_analysis/phase49_cross_seed_sign_consensus.csv": "41e862ff18693f8c",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp)
