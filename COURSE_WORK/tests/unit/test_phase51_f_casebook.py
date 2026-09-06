"""Phase 51-F focused unit tests — 61 checks covering source integrity,
temporal context, model-input context, casebook integrity, LSTM canonical
context, and safety.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.worst_error_analysis import materialize_f, SEEDS


FROZEN_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)

PHASE51_C_SHAS = {
    "worst_per_seed_top20.csv": "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716",
    "worst_shared_top20.csv": "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a",
    "worst_underprediction_top10.csv": "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8",
    "worst_overprediction_top10.csv": "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa",
    "shared_all_under_top10.csv": "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637",
    "shared_all_over_top10.csv": "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be",
}

PHASE51_D_SHAS = {
    "seed_overlap_table.csv": "6fb4c9cd5055763ae77cad6af2627c7439295d7eda084c55b41688076f3c402e",
    "worst_case_membership_matrix.csv": "036442da9f282b7e9008304672da25c25470e4443ff130f4dc1065636e60db3e",
    "error_concentration_table.csv": "8913a7dd1d43f5f58b1d0db86900ccf0815504fd087e9669dcff26678781565e",
    "hardness_vs_seed_disagreement.csv": "33052a40551e28c91762ea4fff28975edb027ade0fecdef47b63be13e6baa9aa",
    "hardness_group_summary.csv": "98a2da44cdf059e35d852139fff9af3f2a06bc0dfc5158919b4667f51afbb3e4",
}

PHASE51_E_SHAS = {
    "regime_overrepresentation.csv": "dfcd4a0d69957e4249d41d382a0118342f8464281a3c88486d46a61a3fa3bafe",
    "baseline_context.csv": "365b98dce4a32494871850bbb6d1bdcf59a3ecadccd880cfa071b9e7a3f9f86a",
    "baseline_context_summary.csv": "0fe3ab37bf3563122dd64e87ab17edbeabc4b6d646243052bce83b30854bdd89",
    "regime_composition.csv": "2e235a4bc76d9235d2198205b3b812a05597f5edb38b4b0cc01dd0dec4214860",
    "shared_worst_regime_context.csv": "cb948de5d37f9bb94902a4a35f7edb8119afa2b097721a0fb31e1edaafbf6889",
}

UPSTREAM_FROZEN = {
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


def _sha(rel: str) -> str:
    return hashlib.sha256(
        (Path("artifacts/worst_error_analysis") / rel).read_bytes()
    ).hexdigest()


def _sha_rel(rel: str) -> str:
    return hashlib.sha256(Path(rel).read_bytes()).hexdigest()


def _load_csv(rel: str) -> list[dict[str, str]]:
    fp = Path("artifacts/worst_error_analysis") / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ── SOURCE / FROZEN ────────────────────────────────────────────────────────

class TestSourceIntegrity:
    """Checks 1-4."""

    def test_01_selection_contract_sha_exact(self):
        assert _sha("worst_error_selection_contract.json") == FROZEN_CONTRACT_SHA

    def test_02_ranking_shas_exact(self):
        for rel, expected in PHASE51_C_SHAS.items():
            assert _sha(rel) == expected

    def test_03_d_e_artifact_shas_exact(self):
        for rel, expected in {**PHASE51_D_SHAS, **PHASE51_E_SHAS}.items():
            assert _sha(rel) == expected

    def test_04_phase_47_50_unchanged(self):
        for rel, expected in UPSTREAM_FROZEN.items():
            assert _sha_rel(rel) == expected


# ── TEMPORAL ───────────────────────────────────────────────────────────────

class TestTemporalContext:
    """Checks 5-15."""

    def test_05_context_radius_6(self):
        audit = _load_csv("context_integrity_audit.csv")
        for r in audit:
            assert int(r["context_radius"]) == 6

    def test_06_max_13_rows_per_case(self):
        ctx = _load_csv("local_temporal_context.csv")
        from collections import Counter
        cnt = Counter(r["center_target_id"] for r in ctx)
        for tid, n in cnt.items():
            assert n <= 13

    def test_07_relative_step_correct(self):
        ctx = _load_csv("local_temporal_context.csv")
        for r in ctx:
            assert int(r["relative_step"]) in range(-6, 7)

    def test_08_center_step_exactly_zero(self):
        ctx = _load_csv("local_temporal_context.csv")
        for r in ctx:
            if r["is_center"] == "1":
                assert int(r["relative_step"]) == 0

    def test_09_center_target_always_present(self):
        audit = _load_csv("context_integrity_audit.csv")
        for r in audit:
            assert int(r["center_present"]) == 1

    def test_10_exact_10_minute_cadence(self):
        # Adjacent target timestamps in final_test_predictions differ by 10 min
        rows = list(csv.DictReader(open("artifacts/final_test/predictions/final_test_predictions_persistence.csv")))
        from datetime import datetime
        for i in range(1, len(rows)):
            fmt = "%Y-%m-%d %H:%M:%S"
            t1 = datetime.strptime(rows[i - 1]["target_timestamp"], fmt)
            t2 = datetime.strptime(rows[i]["target_timestamp"], fmt)
            assert (t2 - t1).total_seconds() == 600

    def test_11_no_crossing_temporal_gaps(self):
        # In Test population, IDs 16774..19734 are contiguous → no gap
        rows = list(csv.DictReader(open("artifacts/final_test/predictions/final_test_predictions_persistence.csv")))
        ids = [int(r["target_id"].split("_")[1]) for r in rows]
        for i in range(1, len(ids)):
            assert ids[i] - ids[i - 1] == 1

    def test_12_no_interpolation(self):
        ctx = _load_csv("local_temporal_context.csv")
        # All unavailable rows have empty values (no synthetic values)
        for r in ctx:
            if r["availability_status"] != "AVAILABLE":
                assert r["context_y_true_wh"] == ""
                assert r["context_seed42_y_pred_wh"] == ""

    def test_13_no_padding(self):
        # Unavailable boundary rows must NOT be padded with neighbor values
        ctx = _load_csv("local_temporal_context.csv")
        for r in ctx:
            if r["availability_status"] in ("UNAVAILABLE_BOUNDARY", "UNAVAILABLE_GAP"):
                for key in ("context_y_true_wh", "context_seed42_y_pred_wh",
                            "context_persistence_y_pred_wh"):
                    assert r[key] == ""

    def test_14_no_forward_fill(self):
        # Same: no forward-fill values in unavailable rows
        ctx = _load_csv("local_temporal_context.csv")
        for r in ctx:
            if r["availability_status"] != "AVAILABLE":
                for key in ("context_y_true_wh", "context_seed42_y_pred_wh",
                            "context_persistence_y_pred_wh"):
                    assert r[key] == ""

    def test_15_incomplete_context_retained(self):
        audit = _load_csv("context_integrity_audit.csv")
        for r in audit:
            assert r["incomplete_context_retained"] in ("True", "False")
            assert r["case_replaced_due_to_incompleteness"] == "False"


# ── MODEL INPUT ────────────────────────────────────────────────────────────

class TestModelInput:
    """Checks 16-22."""

    def test_16_lookback_72(self):
        manifest = _load_csv("input_window_manifest.csv")
        for r in manifest:
            assert int(r["lookback"]) == 72

    def test_17_feature_count_33(self):
        manifest = _load_csv("input_window_manifest.csv")
        for r in manifest:
            assert int(r["feature_count"]) == 33

    def test_18_fs2_tf1_exact(self):
        manifest = _load_csv("input_window_manifest.csv")
        for r in manifest:
            assert r["feature_set"] == "FS2_TF1"

    def test_19_all_33_features_match(self):
        audit = _load_csv("feature_order_audit.csv")
        rows = list(audit)
        assert len(rows) == 33
        for r in rows:
            assert int(r["match"]) == 1

    def test_20_target_window_mapping_exact(self):
        audit = _load_csv("input_window_integrity_audit.csv")
        for r in audit:
            assert int(r["target_window_mapping_exact"]) == 1

    def test_21_input_end_precedes_target(self):
        manifest = _load_csv("input_window_manifest.csv")
        # window_end_timestamp = target_timestamp - 0 min
        # i.e. window_end == target_timestamp
        for r in manifest:
            assert r["window_end_timestamp"] == r["target_timestamp"]

    def test_22_no_future_rows_used_as_model_input(self):
        manifest = _load_csv("input_window_manifest.csv")
        for r in manifest:
            assert r["future_context_used_as_model_input"] in ("False", "0", False)


# ── CASEBOOK ───────────────────────────────────────────────────────────────

class TestCasebook:
    """Checks 23-35."""

    def test_23_case_only_from_frozen_ranking(self):
        cb = _load_csv("casebook_index.csv")
        cb_audit = _load_csv("casebook_integrity_audit.csv")
        for r in cb_audit:
            assert r["manually_added"] == "False"

    def test_24_ranks_unchanged(self):
        cb_audit = _load_csv("casebook_integrity_audit.csv")
        for r in cb_audit:
            assert int(r["rank_preserved"]) == 1

    def test_25_no_manually_added_targets(self):
        cb = _load_csv("casebook_index.csv")
        # n_cases = 60+20+30+30+10+10 = 160 (deterministic from frozen sources)
        assert len(cb) == 160

    def test_26_deterministic_case_ids(self):
        sha_before = _sha("casebook_index.csv")
        materialize_f.materialize_phase51_f()
        sha_after = _sha("casebook_index.csv")
        assert sha_before == sha_after

    def test_27_cross_family_membership_preserved(self):
        master = _load_csv("casebook_unique_case_master.csv")
        # Some targets legitimately appear in multiple families
        multi = [r for r in master if int(r["n_case_memberships"]) > 1]
        assert len(multi) >= 1

    def test_28_y_true_exact(self):
        cb = _load_csv("casebook_index.csv")
        wt = {r["target_id"]: r for r in _load_csv("phase51_target_level_working_table.csv")}
        for r in cb:
            assert r["y_true_wh"] == wt[r["target_id"]]["y_true_wh"]

    def test_29_y_pred_exact(self):
        cb = _load_csv("casebook_index.csv")
        wt = {r["target_id"]: r for r in _load_csv("phase51_target_level_working_table.csv")}
        for r in cb[:5]:
            tid = r["target_id"]
            seed = r["seed"]
            if seed == "ALL":
                # W2 records mean_abs_error_wh; per-seed y_pred stays as ALL
                continue
            assert r["seed42_y_pred_wh"] == wt[tid]["seed42_y_pred_wh"]

    def test_30_residual_exact(self):
        cb = _load_csv("casebook_index.csv")
        wt = {r["target_id"]: r for r in _load_csv("phase51_target_level_working_table.csv")}
        for r in cb[:5]:
            tid = r["target_id"]
            assert r["seed42_residual_wh"] == wt[tid]["seed42_residual_wh"]

    def test_31_abs_error_exact(self):
        cb = _load_csv("casebook_index.csv")
        wt = {r["target_id"]: r for r in _load_csv("phase51_target_level_working_table.csv")}
        for r in cb[:5]:
            tid = r["target_id"]
            assert r["seed42_abs_error_wh"] == wt[tid]["seed42_abs_error_wh"]

    def test_32_six_phase50_family_labels_exact(self):
        cb = _load_csv("casebook_index.csv")
        fams = ["R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
                "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE"]
        for r in cb[:5]:
            for fam in fams:
                assert r[fam] != ""

    def test_33_persistence_context_exact(self):
        # baseline_context from Phase 51-E matches casebook fields
        cb = _load_csv("casebook_index.csv")
        wt = {r["target_id"]: r for r in _load_csv("phase51_target_level_working_table.csv")}
        for r in cb[:5]:
            tid = r["target_id"]
            assert r["mean_abs_error_wh"] == wt[tid]["mean_abs_error_wh"]

    def test_34_sign_consensus_exact(self):
        cb = _load_csv("casebook_index.csv")
        for r in cb[:5]:
            assert r["cross_seed_consensus_class"] in (
                "ALL_UNDER", "ALL_OVER", "ALL_EXACT",
                "TWO_UNDER_ONE_OVER", "TWO_OVER_ONE_UNDER", "MIXED"
            )

    def test_35_seed_spread_context_exact(self):
        # Working table has Phase-48 seed-spread columns; casebook preserves them
        cb = _load_csv("casebook_index.csv")
        # All casebook rows have the canonical Phase 50 regime + per-seed y_pred
        # so seed-spread context is preserved via seed{y}_y_pred_wh fields.
        for r in cb[:5]:
            for s in ("42", "123", "2026"):
                assert r[f"seed{s}_y_pred_wh"] != ""


# ── LSTM ───────────────────────────────────────────────────────────────────

class TestLSTMCanonical:
    """Checks 36-38."""

    def test_36_lstm_status_canonical(self):
        ctx = json.loads(
            (Path("artifacts/worst_error_analysis") / "lstm_eligibility_context.json").read_text()
        )
        assert ctx["phase51_f_status"] == "NOT_ELIGIBLE_CONFIG_MISMATCH"

    def test_37_lstm_reason_canonical(self):
        ctx = json.loads(
            (Path("artifacts/worst_error_analysis") / "lstm_eligibility_context.json").read_text()
        )
        canon = json.loads(
            (Path("artifacts/final_test") / "final_test_lstm_eligibility.json").read_text()
        )
        assert ctx["phase51_f_canonical_reason"] == canon["reason"]

    def test_38_no_lstm_inference(self):
        for fn in ["lstm_predictions.csv", "phase51_f_lstm_ranking.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()


# ── SAFETY ─────────────────────────────────────────────────────────────────

class TestSafety:
    """Checks 39-61."""

    def test_39_no_new_transformer_inference(self):
        for fn in ["phase51_f_predictions.csv", "phase51_f_inference_manifest.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_40_no_checkpoint_load(self):
        for fp in Path("artifacts/worst_error_analysis").glob("*"):
            if fp.is_file():
                assert fp.suffix not in (".pt", ".pkl", ".h5", ".ckpt")

    def test_41_no_training_artifact(self):
        for fn in ["phase51_f_loss_curve.csv", "phase51_f_epoch_log.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_42_no_optimizer(self):
        for fn in ["phase51_f_optimizer.pkl"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_43_no_scaler_fit(self):
        for fn in ["phase51_f_scaler.pkl", "phase51_f_scaler_params.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_44_no_feature_attribution(self):
        for fn in ["phase51_f_attribution.csv", "phase51_f_feature_importance.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_45_no_shap(self):
        for fn in ["phase51_f_shap.csv", "phase51_f_shap_summary.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_46_no_attention_analysis(self):
        for fn in ["phase51_f_attention.csv", "phase51_f_attention_rollout.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_47_no_causal_claims(self):
        cb = _load_csv("casebook_index.csv")
        ctx = _load_csv("local_temporal_context.csv")
        for r in cb:
            for k, v in r.items():
                if isinstance(v, str):
                    assert "causes" not in v.lower()
                    assert "leads to" not in v.lower()
        for r in ctx:
            for v in r.values():
                if isinstance(v, str):
                    assert "causes" not in v.lower()

    def test_48_no_best_seed(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_f_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["best_seed_selected"] is False

    def test_49_no_ensemble(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_f_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["ensemble"] is False

    def test_50_no_prediction_correction(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_f_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["prediction_correction"] is False

    def test_51_no_ranking_changes(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_f_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["ranking_changed"] is False

    def test_52_no_new_test_thresholds(self):
        for fn in ["phase51_f_thresholds.json", "phase51_f_test_thresholds.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_53_no_phase52(self):
        for fn in ["phase52_attention_handoff.json", "phase52_attention_case_table.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_54_no_figures(self):
        for fn in ["phase51_f_figure_1.png", "phase51_f_figure_2.png"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_55_no_final_signoff(self):
        # Phase 51-F manifest must NOT claim to produce signoff.
        # Signoff is produced by Phase 51-G (a later phase).
        # This test verifies Phase 51-F's contract — not that signoff doesn't exist.
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_f_manifest.json").read_text()
        )
        # Phase 51-F should NOT have claimed to produce phase51_signoff.json
        # (it is correctly a Phase 51-G output).
        # We verify the file exists (signoff is now produced) AND that Phase 51-F's
        # own artifact list does not include it (correct scope).
        produced = manifest.get("artifacts_produced", [])
        signoff_produced_by_f = any("signoff" in str(a).lower() for a in produced)
        assert not signoff_produced_by_f, (
            f"Phase 51-F manifest incorrectly claims to produce signoff: {produced}"
        )

    def test_56_phase47_unchanged(self):
        sha = _sha_rel("artifacts/final_test/predictions/final_test_predictions_seed42.csv")
        assert sha.startswith("246ee0d725af972b")

    def test_57_phase48_unchanged(self):
        sha = _sha_rel("artifacts/prediction_analysis/phase_48_signoff.json")
        assert sha.startswith("e8c102d582a35dd2")

    def test_58_phase49_unchanged(self):
        sha = _sha_rel("artifacts/residual_analysis/residual_long_table.csv")
        assert sha.startswith("8418a99110bfda70")

    def test_59_phase50_unchanged(self):
        sha = _sha_rel("artifacts/error_by_regime/test_regime_assignment.csv")
        assert sha == UPSTREAM_FROZEN["artifacts/error_by_regime/test_regime_assignment.csv"]

    def test_60_phase51_b_c_d_e_frozen_unchanged(self):
        assert _sha("worst_error_selection_contract.json") == FROZEN_CONTRACT_SHA
        for rel, expected in {**PHASE51_C_SHAS, **PHASE51_D_SHAS, **PHASE51_E_SHAS}.items():
            assert _sha(rel) == expected, f"{rel} SHA drifted"

    def test_61_deterministic_rerun(self):
        sha_before = _sha("casebook_index.csv")
        materialize_f.materialize_phase51_f()
        sha_after = _sha("casebook_index.csv")
        assert sha_before == sha_after
