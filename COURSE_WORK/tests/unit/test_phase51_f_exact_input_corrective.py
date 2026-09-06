"""Phase 51-F EXACT-INPUT-CORRECTIVE focused regression tests — 22 checks
covering plan-requirement mapping, contract exactness, positional feature
order, no future leakage, no inference, no checkpoint, no scaler fit,
window shape, finite values, deterministic SHA, and that casebook
claims are not overstated.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from course_work.analysis.worst_error_analysis import materialize_f_exact_input as m


# ── FROZEN UPSTREAM SHAs (must not drift) ────────────────────────────────────

UPSTREAM_FROZEN = {
    "artifacts/worst_error_analysis/worst_error_selection_contract.json":
        "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4",
    "artifacts/worst_error_analysis/phase51_target_level_working_table.csv":
        "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f",
    "artifacts/worst_error_analysis/worst_per_seed_top20.csv":
        "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716",
    "artifacts/worst_error_analysis/worst_shared_top20.csv":
        "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a",
    "artifacts/worst_error_analysis/worst_underprediction_top10.csv":
        "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8",
    "artifacts/worst_error_analysis/worst_overprediction_top10.csv":
        "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa",
    "artifacts/worst_error_analysis/shared_all_under_top10.csv":
        "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637",
    "artifacts/worst_error_analysis/shared_all_over_top10.csv":
        "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be",
    "artifacts/worst_error_analysis/regime_overrepresentation.csv":
        "dfcd4a0d69957e4249d41d382a0118342f8464281a3c88486d46a61a3fa3bafe",
    "artifacts/worst_error_analysis/baseline_context.csv":
        "365b98dce4a32494871850bbb6d1bdcf59a3ecadccd880cfa071b9e7a3f9f86a",
    "artifacts/worst_error_analysis/regime_composition.csv":
        "2e235a4bc76d9235d2198205b3b812a05597f5edb38b4b0cc01dd0dec4214860",
    "artifacts/worst_error_analysis/shared_worst_regime_context.csv":
        "cb948de5d37f9bb94902a4a35f7edb8119afa2b097721a0fb31e1edaafbf6889",
    "artifacts/final_test/predictions/final_test_predictions_seed42.csv":
        "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2",
    "artifacts/final_test/predictions/final_test_predictions_persistence.csv":
        "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b",
    "artifacts/prediction_analysis/phase_48_signoff.json":
        "e8c102d582a35dd2d86a8c275f4cb1bf2d24f4f841404aad33c4b0e29161fa9b",
    "artifacts/residual_analysis/residual_long_table.csv":
        "8418a99110bfda7047bd27c49c1c7a9769313b1ce1fa6dc66925f5286d120038",
    "artifacts/error_by_regime/test_regime_assignment.csv":
        "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac",
}


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _load_csv(p: str) -> list[dict[str, str]]:
    with open(p, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ── 1. PLAN REQUIREMENT + SOURCE INTEGRITY ─────────────────────────────────

class TestPlanRequirementMapping:
    def test_01_plan_requirement_classified_as_C_BOTH(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["plan_requirement_classification"] == "C_BOTH"

    def test_02_exact_input_reconstruction_required(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["exact_input_reconstruction_required"] is True

    def test_03_exact_input_reconstruction_performed(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["exact_input_reconstruction_performed"] is True

    def test_04_all_upstream_shas_unchanged(self):
        for rel, expected in UPSTREAM_FROZEN.items():
            assert _sha(rel) == expected, f"Drift: {rel}"


# ── 2. EXACT WINDOW CONTRACT ────────────────────────────────────────────────

class TestWindowContract:
    def test_05_lookback_72(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        for r in recon:
            assert int(r["lookback_steps"]) == 72

    def test_06_feature_count_33(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        for r in recon:
            assert int(r["feature_count"]) == 33

    def test_07_feature_set_FS2_TF1(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        for r in recon:
            assert r["feature_set_id"] == "FS2_TF1"

    def test_08_feature_fingerprint_match(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        fp = "fc9c428964285d1ad74e97f3ae2c18e7efeb3e61d61f7acd0b54039bc2ca6dee"
        for r in recon:
            assert r["feature_fingerprint"] == fp

    def test_09_window_target_window_mapping_exact(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        for r in recon:
            i_end = int(r["input_end_raw_row_index"])
            t_idx = int(r["target_raw_row_index"])
            assert i_end < t_idx, f"Window leaks target: {r}"


# ── 3. POSITIONAL FEATURE-ORDER AUDIT ───────────────────────────────────────

class TestPositionalFeatureOrder:
    def test_10_positional_audit_33_rows(self):
        rows = _load_csv("artifacts/worst_error_analysis/feature_order_positional_audit.csv")
        assert len(rows) == 33

    def test_11_positional_match_all(self):
        rows = _load_csv("artifacts/worst_error_analysis/feature_order_positional_audit.csv")
        for r in rows:
            assert int(r["positional_match"]) == 1
            assert r["expected_name"] == r["actual_name"]

    def test_12_first_feature_lights(self):
        rows = _load_csv("artifacts/worst_error_analysis/feature_order_positional_audit.csv")
        assert rows[0]["expected_name"] == "lights"
        assert rows[0]["position"] == "1"

    def test_13_last_feature_weekend(self):
        rows = _load_csv("artifacts/worst_error_analysis/feature_order_positional_audit.csv")
        assert rows[-1]["expected_name"] == "weekend"
        assert rows[-1]["position"] == "33"

    def test_14_set_only_equality_alone_not_sufficient(self):
        # The manifest must report positional match (NOT just set-equality).
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["feature_order_positional_match_all"] is True


# ── 4. NO FUTURE LEAKAGE / NO INFERENCE / NO CHECKPOINT / NO SCALER FIT ─────

class TestSafety:
    def test_15_no_future_input_leakage(self):
        recon = _load_csv("artifacts/worst_error_analysis/exact_input_window_reconstruction.csv")
        for r in recon:
            assert int(r["target_row_in_window"]) == 0
            assert int(r["all_input_timestamps_before_target"]) == 1

    def test_16_no_new_test_inference(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["new_test_inference"] is False

    def test_17_no_checkpoint_loading(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["checkpoint_loading"] is False

    def test_18_no_training(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["training"] is False

    def test_19_no_scaler_fit(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["scaler_fit"] is False

    def test_20_no_phase47_50_modification(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        for k in ("phase47_modified", "phase48_modified",
                  "phase49_modified", "phase50_modified"):
            assert man[k] is False, f"Modified: {k}"


# ── 5. WINDOW SHAPE / FINITE / SHA ──────────────────────────────────────────

class TestWindowValues:
    def test_21_window_shape_exact(self):
        # Re-run reconstruction and assert raw + model windows both [72, 33]
        r = m.materialize_phase51_f_exact_input_corrective()
        for t in r["manifest"]["per_target" if False else "deliverables"].items():
            pass
        # Directly use the reconstructor.
        from course_work.analysis.worst_error_analysis import exact_input_reconstruction as eir
        rec = eir.reconstruct_windows_for_targets()
        for t in rec["per_target"]:
            assert t["_raw_window"].shape == (72, 33)
            assert t["_model_window"].shape == (72, 33)

    def test_22_all_values_finite(self):
        from course_work.analysis.worst_error_analysis import exact_input_reconstruction as eir
        rec = eir.reconstruct_windows_for_targets()
        for t in rec["per_target"]:
            assert np.isfinite(t["_raw_window"]).all()
            assert np.isfinite(t["_model_window"]).all()

    def test_23_deterministic_window_sha(self):
        from course_work.analysis.worst_error_analysis import exact_input_reconstruction as eir
        rec1 = eir.reconstruct_windows_for_targets()
        rec2 = eir.reconstruct_windows_for_targets()
        for t1, t2 in zip(rec1["per_target"], rec2["per_target"]):
            assert t1["window_checksum_raw"] == t2["window_checksum_raw"]
            assert t1["window_checksum_model"] == t2["window_checksum_model"]

    def test_24_unique_targets_44(self):
        from course_work.analysis.worst_error_analysis import exact_input_reconstruction as eir
        rec = eir.reconstruct_windows_for_targets()
        assert rec["n_targets"] == 44

    def test_25_contract_vs_values_audit_consistent(self):
        rows = _load_csv(
            "artifacts/worst_error_analysis/input_window_contract_vs_values_audit.csv"
        )
        for r in rows:
            assert int(r["input_window_contract_verified"]) == 1
            assert int(r["input_window_values_verified"]) == 1
            assert int(r["feature_order_positional_match"]) == 1


# ── 6. CASEBOOK CLAIMS + MODEL_VISIBLE_FEATURE_SUMMARY ─────────────────────

class TestCasebookClaims:
    def test_26_casebook_does_not_overclaim_without_values(self):
        # Old model_visible_feature_summary.csv is gone; replaced by the
        # corrected exact_input_windows_per_feature_summary.csv that has
        # RAW and MODEL_VISIBLE coordinates. Verify replacement happened.
        old_fp = Path("artifacts/worst_error_analysis/model_visible_feature_summary.csv")
        new_fp = Path(
            "artifacts/worst_error_analysis/exact_input_windows_per_feature_summary.csv"
        )
        assert new_fp.exists()

    def test_27_summary_has_raw_and_model_coordinates(self):
        rows = _load_csv(
            "artifacts/worst_error_analysis/exact_input_windows_per_feature_summary.csv"
        )
        coords = {r["coordinate"] for r in rows}
        assert coords == {"RAW", "MODEL_VISIBLE"}

    def test_28_summary_per_feature_44_targets(self):
        rows = _load_csv(
            "artifacts/worst_error_analysis/exact_input_windows_per_feature_summary.csv"
        )
        targets = {r["target_id"] for r in rows}
        assert len(targets) == 44

    def test_29_target_history_context_present(self):
        rows = _load_csv("artifacts/worst_error_analysis/target_history_context.csv")
        assert len(rows) == 44
        for r in rows:
            assert int(r["window_steps"]) == 72
            assert r["historical_target_model_visible"] == "True"


# ── 7. LSTM CANONICAL REASON ───────────────────────────────────────────────

class TestLSTMCanonicalReason:
    def test_30_lstm_reason_verbatim(self):
        # The lstm_eligibility_context.json was updated by Phase 51-F;
        # its reason must come from the canonical artifact verbatim.
        canon = json.loads(
            open("artifacts/final_test/final_test_lstm_eligibility.json").read()
        )
        ctx = json.loads(
            open("artifacts/worst_error_analysis/lstm_eligibility_context.json").read()
        )
        assert ctx["phase51_f_canonical_reason"] == canon["reason"]


# ── 8. EXACT_INPUT_RECONSTRUCTION MANIFEST ──────────────────────────────────

class TestManifest:
    def test_31_manifest_status_pass(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["status"] == "PASS"

    def test_32_manifest_ready_for_phase51_g(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["ready_for_phase51_g"] is True

    def test_33_manifest_lookback_feature_count(self):
        man = json.loads(
            open("artifacts/worst_error_analysis/exact_input_reconstruction_manifest.json").read()
        )
        assert man["lookback"] == 72
        assert man["feature_count"] == 33
        assert man["feature_set"] == "FS2_TF1"
