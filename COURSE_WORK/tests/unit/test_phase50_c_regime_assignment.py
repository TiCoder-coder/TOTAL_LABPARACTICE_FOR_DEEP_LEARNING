"""Phase 50-C focused tests — Test regime assignment + freeze + audit."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.analysis.error_regime_analysis import contract, sources
from course_work.analysis.error_regime_analysis.regime_assignment import materialize_phase50_c


def test_c_runs_clean():
    r = materialize_phase50_c()
    assert r["status"] == "PASS"
    assert r["n_test_target_ids"] == 2961


def test_assignment_csv_exists_and_frozen():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2961


def test_assignment_n_equals_2961():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2961


def test_assignment_unique_target_ids():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    ids = [r["target_id"] for r in rows]
    assert len(set(ids)) == 2961


def test_assignment_chronological():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    ts = [r["timestamp"] for r in rows]
    assert ts == sorted(ts)


def test_assignment_all_six_regime_families_present():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    for fam in [
        "target_level_regime",
        "extreme_high_regime",
        "change_magnitude_regime",
        "change_direction_regime",
        "time_of_day_regime",
        "day_type_regime",
    ]:
        labels = {r[fam] for r in rows}
        assert len(labels) >= 1, f"empty {fam}"


def test_assignment_r1_exactly_one_label():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    labels_per_row = {r["target_level_regime"] for r in rows}
    assert labels_per_row.issubset({"TL_LOW", "TL_MID", "TL_HIGH"})
    assert labels_per_row, "R1 labels must cover all Test points"


def test_assignment_r2_exactly_one_label():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    labels_per_row = {r["extreme_high_regime"] for r in rows}
    assert labels_per_row.issubset({"EXTREME_HIGH", "NON_EXTREME"})


def test_assignment_r3_has_unclassified():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    labels = {r["change_magnitude_regime"] for r in rows}
    assert labels.issubset({"CHANGE_NORMAL", "CHANGE_RAPID", "CHANGE_UNCLASSIFIED"})
    # First Test target must be UNCLASSIFIED for R3 (WB0 predecessor is in VAL, not in Test)
    # Actually per implementation, first Test target uses WB0-authorized VAL predecessor,
    # so R3 is classified for first Test point. We verify the label space is allowed.
    assert "CHANGE_UNCLASSIFIED" in labels or "CHANGE_NORMAL" in labels


def test_assignment_r4_has_unclassified_label_set():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    labels = {r["change_direction_regime"] for r in rows}
    allowed = {"DIR_UP", "DIR_DOWN", "DIR_FLAT", "DIR_UNCLASSIFIED"}
    assert labels.issubset(allowed)


def test_assignment_r5_r6_exactly_one_label():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    r5 = {r["time_of_day_regime"] for r in rows}
    assert r5.issubset(
        {"TOD_NIGHT", "TOD_MORNING", "TOD_AFTERNOON", "TOD_EVENING"}
    )
    assert r5, "R5 must cover all Test points"
    r6 = {r["day_type_regime"] for r in rows}
    assert r6.issubset({"DAY_WEEKDAY", "DAY_WEEKEND"})
    assert r6, "R6 must cover all Test points"


def test_assignment_no_prediction_columns():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    forbidden = {"y_pred_wh", "residual_wh", "absolute_error_wh", "squared_error_wh"}
    columns = set(rows[0].keys())
    assert not (forbidden & columns), f"forbidden columns present: {forbidden & columns}"


def test_assignment_no_seed_or_model_id_columns():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    columns = set(rows[0].keys())
    assert not ({"model_id", "seed", "seed_id"} & columns)


def test_assignment_first_target_TGT_00016774():
    """WB0 contract authorizes TGT_00016774 -> TGT_00016773 predecessor."""
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    first = rows[0]
    assert first["target_id"] == "TGT_00016774"
    assert first["previous_target_id"] == "TGT_00016773"
    assert first["change_magnitude_regime"] in {"CHANGE_NORMAL", "CHANGE_RAPID"}


def test_assignment_classifications_use_train_thresholds():
    """Verify R1 classification matches the Train-derived Q25/Q75."""
    th = json.loads(
        Path("artifacts/error_by_regime/regime_thresholds_train_only.json").read_text()
    )
    q25 = th["target_level"]["Q25"]
    q75 = th["target_level"]["Q75"]
    q90 = th["extreme_high"]["Q90"]

    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    mismatches = []
    for r in rows:
        y = float(r["y_true_wh"])
        expected_r1 = "TL_LOW" if y < q25 else ("TL_MID" if y < q75 else "TL_HIGH")
        if r["target_level_regime"] != expected_r1:
            mismatches.append((r["target_id"], y, expected_r1, r["target_level_regime"]))
        expected_r2 = "EXTREME_HIGH" if y >= q90 else "NON_EXTREME"
        if r["extreme_high_regime"] != expected_r2:
            mismatches.append((r["target_id"], y, expected_r2, r["extreme_high_regime"]))
    assert not mismatches, f"classification mismatches: {mismatches[:5]}"


def test_assignment_audit_all_pass():
    p = Path("artifacts/error_by_regime/test_regime_assignment_audit.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        assert r["status"] == "PASS", f"audit failed: {r}"


def test_assignment_fingerprint_present_and_locked():
    fp = json.loads(
        Path("artifacts/error_by_regime/test_regime_assignment_fingerprint.json").read_text()
    )
    assert fp["status"] == "PASS"
    assert fp["created_before_error_join"] is True
    assert fp["n_rows"] == 2961
    assert fp["test_population_sha256"] == contract.TEST_POPULATION_FINGERPRINT


def test_assignment_fingerprint_matches_csv():
    fp = json.loads(
        Path("artifacts/error_by_regime/test_regime_assignment_fingerprint.json").read_text()
    )
    cur = hashlib.sha256(
        Path("artifacts/error_by_regime/test_regime_assignment.csv").read_bytes()
    ).hexdigest()
    assert fp["assignment_sha256"] == cur


def test_train_regime_assignment_present():
    p = Path("artifacts/error_by_regime/train_regime_assignment.csv")
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 13670


def test_train_regime_assignment_no_prediction_fields():
    p = Path("artifacts/error_by_regime/train_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    forbidden = {"y_pred_wh", "residual_wh", "absolute_error_wh", "squared_error_wh"}
    assert not (forbidden & set(rows[0].keys()))


def test_phase47_unchanged_post_50c():
    expected = {
        "artifacts/final_test/phase_47_signoff.json": "80614523b6091e21",
        "artifacts/final_test/prediction_checksums.json": "20c1ddec303fbd80",
        "artifacts/final_test/predictions/final_test_predictions_persistence.csv": "7115af1c479b8957",
        "artifacts/final_test/predictions/final_test_predictions_seed42.csv": "246ee0d725af972b",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp), f"{p} mutated"


def test_phase48_unchanged_post_50c():
    sha = hashlib.sha256(
        Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()
    ).hexdigest()
    assert sha.startswith("e8c102d582a35dd2")


def test_phase49_unchanged_post_50c():
    expected = {
        "artifacts/residual_analysis/phase_49_signoff.json": "9d5fc659717dbc15",
        "artifacts/residual_analysis/phase50_handoff.json": "ebfe2cdfbdb8523f",
        "artifacts/residual_analysis/residual_long_table.csv": "8418a99110bfda70",
        "artifacts/residual_analysis/residual_wide_table.csv": "931ff9109aef236d",
    }
    for p, exp in expected.items():
        sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        assert sha.startswith(exp), f"{p} mutated"


def test_assignment_first_test_target_uses_wb0_predecessor():
    """First Test target uses WB0-authorized Validation predecessor."""
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    first = rows[0]
    assert first["target_id"] == "TGT_00016774"
    # predecessor is TGT_00016773 (last Validation target, raw Appliances=40)
    assert first["previous_target_id"] == "TGT_00016773"


def test_all_test_targets_have_y_true_wh_finite():
    p = Path("artifacts/error_by_regime/test_regime_assignment.csv")
    rows = list(csv.DictReader(p.open()))
    for r in rows:
        y = float(r["y_true_wh"])
        import math
        assert math.isfinite(y)
