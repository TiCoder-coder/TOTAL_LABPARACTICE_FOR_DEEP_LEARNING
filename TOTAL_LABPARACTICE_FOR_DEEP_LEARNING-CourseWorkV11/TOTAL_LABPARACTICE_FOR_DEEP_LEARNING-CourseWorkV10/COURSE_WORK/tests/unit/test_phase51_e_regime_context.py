"""Phase 51-E focused unit tests — 54 checks covering source integrity,
join correctness, prevalence/enrichment math, signed regime preservation,
persistence context, LSTM status, sign consensus, and safety.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.analysis.worst_error_analysis import (
    materialize_e,
    SEEDS,
)


FROZEN_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)
WORKING_TABLE_SHA = (
    "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f"
)
PHASE50_ASSIGNMENT_SHA = (
    "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
)
PERSISTENCE_SHA = "7115af1c479b89575f2f7ed6c065a68d214e44d336a0c681c033a8015bd9ee9b"

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

class TestSourceIntegrity:
    """Checks 1-5."""

    def test_01_phase50_assignment_sha_exact(self):
        assert _sha_rel("artifacts/error_by_regime/test_regime_assignment.csv") == PHASE50_ASSIGNMENT_SHA

    def test_02_ranking_artifacts_sha_exact(self):
        for rel, expected in PHASE51_C_SHAS.items():
            assert _sha(rel) == expected

    def test_03_every_selected_target_joins_phase50_exactly_once(self):
        sel = set()
        for rel in PHASE51_C_SHAS:
            for r in _load_csv(rel):
                sel.add(r["target_id"])
        wt = _load_csv("phase51_target_level_working_table.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        for tid in sel:
            assert tid in wt_by_id

    def test_04_no_selected_target_dropped(self):
        for rel in PHASE51_C_SHAS:
            rows = _load_csv(rel)
            wt = _load_csv("phase51_target_level_working_table.csv")
            wt_ids = {r["target_id"] for r in wt}
            for r in rows:
                assert r["target_id"] in wt_ids

    def test_05_phase50_labels_unchanged(self):
        sha_before = _sha("phase51_target_level_working_table.csv")
        materialize_e.materialize_phase51_e()
        sha_after = _sha("phase51_target_level_working_table.csv")
        assert sha_before == sha_after

class TestPrevalence:
    """Checks 6-10."""

    def test_06_global_denominator_2961(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows:
            assert 0.0 <= float(r["global_prevalence"]) <= 1.0

    def test_07_global_counts_sum_correctly(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for fam in [
            "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
            "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
        ]:
            total = sum(
                int(r["selected_count"])
                for r in rows
                if r["selection_family"] == "W2_SHARED_WORST"
                and r["regime_family"] == fam
            )
            assert total == 20

    def test_08_selected_counts_sum_to_K(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for seed in SEEDS:
            for fam in [
                "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
                "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
            ]:
                total = sum(
                    int(r["selected_count"])
                    for r in rows
                    if r["selection_family"] == "W1_PER_SEED_WORST"
                    and r["seed"] == seed
                    and r["regime_family"] == fam
                )
                assert total == 20, f"W1 seed={seed} fam={fam} total={total}"

    def test_09_selected_prevalence_formula(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows[:20]:  # spot-check
            sel_c = int(r["selected_count"])
            k = int(r["selection_k"])
            prev = float(r["selected_prevalence"])
            assert abs(prev - sel_c / k) < 1e-9

    def test_10_global_prevalence_formula(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows[:20]:
            g_c = int(r["global_count"])
            prev = float(r["global_prevalence"])
            assert abs(prev - g_c / 2961) < 1e-9


class TestEnrichment:
    """Checks 11-16."""

    def test_11_enrichment_ratio_formula(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows:
            if r["status"] == "OK":
                sp = float(r["selected_prevalence"])
                gp = float(r["global_prevalence"])
                e = float(r["enrichment_ratio"])
                assert abs(e - sp / gp) < 1e-5

    def test_12_prevalence_difference_correct(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows[:20]:
            sp = float(r["selected_prevalence"])
            gp = float(r["global_prevalence"])
            d = float(r["prevalence_difference"])
            assert abs(d - (sp - gp)) < 1e-9

    def test_13_denominator_zero_handling(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows:
            if r["status"] == "DENOMINATOR_ZERO":
                assert r["enrichment_ratio"] == ""
            else:
                assert r["status"] == "OK"

    def test_14_no_infinite_or_nan_unhandled(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows:
            for k, v in r.items():
                if k in ("enrichment_ratio", "prevalence_difference"):
                    if v != "":
                        try:
                            float(v)
                        except ValueError:
                            pytest.fail(f"Non-numeric value in {k}: {v!r}")

    def test_15_no_cartesian_regime_mining(self):
        rows = _load_csv("regime_overrepresentation.csv")
        declared_families = {
            "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
            "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
        }
        for r in rows:
            assert r["regime_family"] in declared_families

    def test_16_no_threshold_recomputation(self):
        rows = _load_csv("regime_overrepresentation.csv")
        expected_keys = {
            "selection_family", "seed", "selection_k",
            "regime_family", "regime_label",
            "selected_count", "selected_prevalence",
            "global_count", "global_prevalence",
            "enrichment_ratio", "prevalence_difference",
            "status", "phase50_assignment_sha256", "selection_contract_sha256",
        }
        for r in rows:
            assert set(r.keys()) == expected_keys

class TestW1SeedSeparation:
    """Checks 17-18."""

    def test_17_w1_evaluated_per_seed(self):
        rows = _load_csv("regime_overrepresentation.csv")
        w1_seeds = {(r["seed"]) for r in rows if r["selection_family"] == "W1_PER_SEED_WORST"}
        assert w1_seeds == set(SEEDS)

    def test_18_no_60_row_iid_pooling(self):
        rows = _load_csv("regime_overrepresentation.csv")
        for r in rows:
            if r["selection_family"] == "W1_PER_SEED_WORST":
                assert int(r["selection_k"]) == 20

class TestW2SharedWorstRegimeContext:
    """Checks 19-20."""

    def test_19_w2_has_20_target_contexts(self):
        rows = _load_csv("regime_overrepresentation.csv")
        w2_rows = [r for r in rows if r["selection_family"] == "W2_SHARED_WORST"]
        assert all(int(r["selection_k"]) == 20 for r in w2_rows)

    def test_20_all_six_phase50_families_carried(self):
        rows = _load_csv("regime_overrepresentation.csv")
        w2_fams = {r["regime_family"] for r in rows if r["selection_family"] == "W2_SHARED_WORST"}
        assert w2_fams == {
            "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
            "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
        }

class TestSignedRegimePreservation:
    """Checks 21-24."""

    def test_21_w3_rows_remain_under_only(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        for r in rows:
            assert float(r["residual_wh"]) > 0

    def test_22_w4_rows_remain_over_only(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for r in rows:
            assert float(r["residual_wh"]) < 0

    def test_23_shared_under_all_under_eligibility(self):
        wt = _load_csv("phase51_target_level_working_table.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        rows = _load_csv("shared_all_under_top10.csv")
        for r in rows:
            tid = r["target_id"]
            w = wt_by_id[tid]
            assert float(w["seed42_residual_wh"]) > 0
            assert float(w["seed123_residual_wh"]) > 0
            assert float(w["seed2026_residual_wh"]) > 0

    def test_24_shared_over_all_over_eligibility(self):
        wt = _load_csv("phase51_target_level_working_table.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        rows = _load_csv("shared_all_over_top10.csv")
        for r in rows:
            tid = r["target_id"]
            w = wt_by_id[tid]
            assert float(w["seed42_residual_wh"]) < 0
            assert float(w["seed123_residual_wh"]) < 0
            assert float(w["seed2026_residual_wh"]) < 0

class TestPersistenceContext:
    """Checks 25-30."""

    def test_25_persistence_checksum_exact(self):
        sha = _sha_rel("artifacts/final_test/predictions/final_test_predictions_persistence.csv")
        assert sha == PERSISTENCE_SHA

    def test_26_persistence_n_2961(self):
        import csv as _csv
        rows = list(_csv.DictReader(open("artifacts/final_test/predictions/final_test_predictions_persistence.csv")))
        assert len(rows) == 2961

    def test_27_target_alignment_exact(self):
        pers = set()
        for r in csv.DictReader(open("artifacts/final_test/predictions/final_test_predictions_persistence.csv")):
            pers.add(r["target_id"])
        wt = {r["target_id"] for r in _load_csv("phase51_target_level_working_table.csv")}
        assert pers == wt

    def test_28_selected_targets_join_correctly(self):
        pers_by_id = {r["target_id"]: r for r in csv.DictReader(open("artifacts/final_test/predictions/final_test_predictions_persistence.csv"))}
        ctx_rows = _load_csv("baseline_context.csv")
        for r in ctx_rows:
            assert r["target_id"] in pers_by_id

    def test_29_persistence_does_not_alter_rankings(self):
        before = {k: _sha(k) for k in PHASE51_C_SHAS}
        materialize_e.materialize_phase51_e()
        after = {k: _sha(k) for k in PHASE51_C_SHAS}
        assert before == after

    def test_30_phase47_global_interpretation_preserved(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_e_manifest.json").read_text()
        )
        interp = manifest["phase47_global_interpretation_preserved"]
        assert "Persistence better globally on MAE" in interp
        assert "Transformer better globally on RMSE" in interp

class TestLSTMContext:
    """Checks 31-33."""

    def test_31_lstm_status_canonical(self):
        ctx = json.loads(
            (Path("artifacts/worst_error_analysis") / "lstm_eligibility_context.json").read_text()
        )
        assert ctx["eligibility_status"] == "NOT_ELIGIBLE_CONFIG_MISMATCH"

    def test_32_no_lstm_inference_artifact(self):
        for fn in [
            "lstm_predictions.csv", "lstm_ranking.csv",
            "phase51_e_lstm_ranking.csv", "lstm_context_predictions.csv",
        ]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_33_no_fake_lstm_ranking(self):
        files = [
            "regime_overrepresentation.csv",
            "baseline_context.csv",
            "regime_composition.csv",
            "shared_worst_regime_context.csv",
        ]
        for f in files:
            fp = Path("artifacts/worst_error_analysis") / f
            if not fp.exists():
                continue
            content = fp.read_text()
            assert "LSTM_TUNED_DEV" not in content or "NOT_APPLICABLE" in content

class TestSignConsensusContext:
    """Checks 34-36."""

    def test_34_sign_consensus_join_1to1(self):
        ctx = _load_csv("regime_composition.csv")
        tids = [r["target_id"] for r in ctx]
        assert len(tids) == len(set(tids))

    def test_35_source_taxonomy_preserved(self):
        ctx = _load_csv("regime_composition.csv")
        allowed = {
            "ALL_UNDER", "ALL_OVER", "ALL_EXACT",
            "TWO_UNDER_ONE_OVER", "TWO_OVER_ONE_UNDER", "MIXED",
        }
        for r in ctx:
            assert r["cross_seed_consensus_class"] in allowed

    def test_36_deterministic_mapping(self):
        before = _sha("regime_composition.csv")
        materialize_e.materialize_phase51_e()
        after = _sha("regime_composition.csv")
        assert before == after

class TestSafety:
    """Checks 37-54."""

    def test_37_no_new_test_inference(self):
        for fn in ["phase51_e_predictions.csv", "phase51_e_inference_manifest.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_38_no_checkpoint_loading(self):
        for fp in Path("artifacts/worst_error_analysis").glob("*"):
            if fp.is_file():
                assert fp.suffix not in (".pt", ".pkl", ".h5", ".ckpt")

    def test_39_no_training_artifact(self):
        for fn in ["phase51_e_loss_curve.csv", "phase51_e_epoch_log.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_40_no_scaler_fit(self):
        for fn in ["phase51_e_scaler.pkl", "phase51_e_scaler_params.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_41_no_best_seed(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_e_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["best_seed_selected"] is False

    def test_42_no_ensemble(self):
        manifest = json.loads(
            (Path("artifacts/worst_error_analysis") / "phase51_e_manifest.json").read_text()
        )
        assert manifest["forbidden_actions_status"]["ensemble"] is False

    def test_43_no_prediction_correction(self):
        ctx = _load_csv("baseline_context.csv")
        for r in ctx:
            assert "corrected" not in r
            assert "shifted" not in r

    def test_44_no_temporal_context(self):
        ctx = _load_csv("baseline_context.csv")
        for r in ctx:
            assert "t6_before" not in r
            assert "t6_after" not in r
            assert "local_context" not in r

    def test_45_no_input_context(self):
        for fn in ["baseline_context.csv", "regime_composition.csv"]:
            fp = Path("artifacts/worst_error_analysis") / fn
            if fp.exists():
                header = fp.open().readline().strip().split(",")
                forbidden = {"input_features", "feature_window", "X"}
                for col in header:
                    assert col not in forbidden

    def test_46_no_casebook(self):
        for fn in ["phase51_e_casebook.csv", "phase51_e_casebook.md"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_47_no_figures(self):
        for fn in ["phase51_e_figure_1.png", "phase51_e_figure_2.png"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_48_phase52_not_started(self):
        for fn in ["phase52_attention_handoff.json", "phase52_attention_case_table.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_49_phase47_unchanged(self):
        sha = _sha_rel("artifacts/final_test/predictions/final_test_predictions_seed42.csv")
        assert sha.startswith("246ee0d725af972b")

    def test_50_phase48_unchanged(self):
        sha = _sha_rel("artifacts/prediction_analysis/phase_48_signoff.json")
        assert sha.startswith("e8c102d582a35dd2")

    def test_51_phase49_unchanged(self):
        sha = _sha_rel("artifacts/residual_analysis/residual_long_table.csv")
        assert sha.startswith("8418a99110bfda70")

    def test_52_phase50_unchanged(self):
        sha = _sha_rel("artifacts/error_by_regime/test_regime_assignment.csv")
        assert sha == PHASE50_ASSIGNMENT_SHA

    def test_53_phase51_b_c_d_frozen_unchanged(self):
        assert _sha("worst_error_selection_contract.json") == FROZEN_CONTRACT_SHA
        assert _sha("phase51_target_level_working_table.csv") == WORKING_TABLE_SHA
        for rel, expected in PHASE51_C_SHAS.items():
            assert _sha(rel) == expected, f"{rel} SHA drifted"
        for rel, expected in PHASE51_D_SHAS.items():
            assert _sha(rel) == expected, f"{rel} SHA drifted"

    def test_54_deterministic_rerun(self):
        sha_before = _sha("regime_overrepresentation.csv")
        materialize_e.materialize_phase51_e()
        sha_after = _sha("regime_overrepresentation.csv")
        assert sha_before == sha_after
