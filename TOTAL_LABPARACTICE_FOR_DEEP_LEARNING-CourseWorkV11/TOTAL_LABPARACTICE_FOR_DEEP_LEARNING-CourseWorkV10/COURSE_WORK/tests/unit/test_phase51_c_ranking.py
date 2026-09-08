"""Phase 51-C focused unit tests.

45 checks verifying ranking correctness, frozen contract compliance,
deterministic ordering, sign-eligibility, and safety invariants.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.analysis.worst_error_analysis import (
    SEEDS,
    K_ABS_PER_SEED,
    K_SHARED,
    K_UNDER_PER_SEED,
    K_OVER_PER_SEED,
    K_SHARED_SIGNED,
    materialize_c,
    ranking,
    signed_ranking,
)


def _load_csv(rel: str) -> list[dict[str, str]]:
    fp = Path("artifacts/worst_error_analysis") / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(s: str) -> float:
    return float(s)

class TestFrozenContractGate:
    """Check 1-2: selection contract SHA + working table exist."""

    def test_01_frozen_contract_sha(self):
        fp = Path("artifacts/worst_error_analysis/worst_error_selection_contract.json")
        sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert sha == "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"

    def test_02_working_table_present(self):
        fp = Path("artifacts/worst_error_analysis/phase51_target_level_working_table.csv")
        assert fp.exists()


class TestW1PerSeedWorst:
    """Checks 3-8: W1 (per-seed Top 20 absolute_error_wh)."""

    def test_03_w1_total_rows_60(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        assert len(rows) == 60  

    def test_04_w1_20_per_seed(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            assert len(seed_rows) == K_ABS_PER_SEED

    def test_05_w1_rank_1_to_20(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            seed_rows = sorted([r for r in rows if r["seed"] == seed],
                               key=lambda r: int(r["rank"]))
            ranks = [int(r["rank"]) for r in seed_rows]
            assert ranks == list(range(1, 21))

    def test_06_w1_abs_error_desc(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            seed_rows = sorted([r for r in rows if r["seed"] == seed],
                               key=lambda r: int(r["rank"]))
            aes = [_to_float(r["absolute_error_wh"]) for r in seed_rows]
            for i in range(1, len(aes)):
                assert aes[i] <= aes[i - 1]

    def test_07_w1_tie_break_correct(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            seed_rows = sorted([r for r in rows if r["seed"] == seed],
                               key=lambda r: int(r["rank"]))
            for i in range(1, len(seed_rows)):
                ae_prev = _to_float(seed_rows[i - 1]["absolute_error_wh"])
                ae_curr = _to_float(seed_rows[i]["absolute_error_wh"])
                if ae_curr == ae_prev:
                    assert seed_rows[i]["target_id"] > seed_rows[i - 1]["target_id"]
            assert seed_rows[0]["tie_break"] == "target_id ASC"

    def test_08_w1_no_duplicates_per_seed(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            tids = [r["target_id"] for r in seed_rows]
            assert len(tids) == len(set(tids))


class TestW2SharedWorst:
    """Checks 9-12: W2 (shared Top 20 mean_abs_error)."""

    def test_09_w2_exactly_20_rows(self):
        rows = _load_csv("worst_shared_top20.csv")
        assert len(rows) == K_SHARED

    def test_10_w2_mean_formula(self):
        wt_fp = Path("artifacts/worst_error_analysis/phase51_target_level_working_table.csv")
        with wt_fp.open() as f:
            wt = {r["target_id"]: r for r in csv.DictReader(f)}

        rows = _load_csv("worst_shared_top20.csv")
        for r in rows:
            tid = r["target_id"]
            w = wt[tid]
            expected = (
                _to_float(w["seed42_abs_error_wh"])
                + _to_float(w["seed123_abs_error_wh"])
                + _to_float(w["seed2026_abs_error_wh"])
            ) / 3.0
            actual = _to_float(r["mean_abs_error_wh"])
            assert abs(expected - actual) < 1e-4

    def test_11_w2_desc_ranking(self):
        rows = _load_csv("worst_shared_top20.csv")
        aes = [_to_float(r["mean_abs_error_wh"]) for r in rows]
        for i in range(1, len(aes)):
            assert aes[i] <= aes[i - 1]

    def test_12_w2_no_duplicates(self):
        rows = _load_csv("worst_shared_top20.csv")
        tids = [r["target_id"] for r in rows]
        assert len(tids) == len(set(tids))


class TestW3UnderpredictionWorst:
    """Checks 13-16: W3 (residual > 0 only)."""

    def test_13_w3_residual_strictly_positive(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        for r in rows:
            assert _to_float(r["residual_wh"]) > 0

    def test_14_w3_max_10_per_seed(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            assert len(seed_rows) == K_UNDER_PER_SEED

    def test_15_w3_correct_ordering(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        for seed in SEEDS:
            seed_rows = sorted([r for r in rows if r["seed"] == seed],
                               key=lambda r: int(r["rank"]))
            aes = [_to_float(r["absolute_error_wh"]) for r in seed_rows]
            for i in range(1, len(aes)):
                assert aes[i] <= aes[i - 1]

    def test_16_w3_exact_zero_excluded(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        for r in rows:
            assert _to_float(r["residual_wh"]) != 0


class TestW4OverpredictionWorst:
    """Checks 17-20: W4 (residual < 0 only)."""

    def test_17_w4_residual_strictly_negative(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for r in rows:
            assert _to_float(r["residual_wh"]) < 0

    def test_18_w4_max_10_per_seed(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            assert len(seed_rows) == K_OVER_PER_SEED

    def test_19_w4_correct_ordering(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for seed in SEEDS:
            seed_rows = sorted([r for r in rows if r["seed"] == seed],
                               key=lambda r: int(r["rank"]))
            aes = [_to_float(r["absolute_error_wh"]) for r in seed_rows]
            for i in range(1, len(aes)):
                assert aes[i] <= aes[i - 1]

    def test_20_w4_exact_zero_excluded(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for r in rows:
            assert _to_float(r["residual_wh"]) != 0

class TestW3SharedAllUnder:
    """Checks 21-22: W3_SH (residual > 0 for ALL 3 seeds)."""

    def test_21_w3sh_all_3_residuals_positive(self):
        rows = _load_csv("shared_all_under_top10.csv")
        for r in rows:
            assert _to_float(r["seed42_residual_wh"]) > 0
            assert _to_float(r["seed123_residual_wh"]) > 0
            assert _to_float(r["seed2026_residual_wh"]) > 0

    def test_22_w3sh_correct_topk(self):
        rows = _load_csv("shared_all_under_top10.csv")
        assert len(rows) <= K_SHARED_SIGNED
        aes = [_to_float(r["mean_abs_error_wh"]) for r in rows]
        for i in range(1, len(aes)):
            assert aes[i] <= aes[i - 1]

class TestW4SharedAllOver:
    """Checks 23-24: W4_SH (residual < 0 for ALL 3 seeds)."""

    def test_23_w4sh_all_3_residuals_negative(self):
        rows = _load_csv("shared_all_over_top10.csv")
        for r in rows:
            assert _to_float(r["seed42_residual_wh"]) < 0
            assert _to_float(r["seed123_residual_wh"]) < 0
            assert _to_float(r["seed2026_residual_wh"]) < 0

    def test_24_w4sh_correct_topk(self):
        rows = _load_csv("shared_all_over_top10.csv")
        assert len(rows) <= K_SHARED_SIGNED
        aes = [_to_float(r["mean_abs_error_wh"]) for r in rows]
        for i in range(1, len(aes)):
            assert aes[i] <= aes[i - 1]

class TestSafetyAndReproducibility:
    """Checks 25-45: deterministic, no upstream modification, no Phase 52."""

    def test_25_frozen_contract_sha_exact(self):
        contract_sha = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_error_selection_contract.json").read_bytes()
        ).hexdigest()
        assert contract_sha == "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"

    def test_26_working_table_sha_stable(self):
        wt_sha_before = hashlib.sha256(
            Path("artifacts/worst_error_analysis/phase51_target_level_working_table.csv").read_bytes()
        ).hexdigest()
        materialize_c.materialize_phase51_c()
        wt_sha_after = hashlib.sha256(
            Path("artifacts/worst_error_analysis/phase51_target_level_working_table.csv").read_bytes()
        ).hexdigest()
        assert wt_sha_before == wt_sha_after

    def test_27_no_random_tiebreak(self):
        materialize_c.materialize_phase51_c()
        sha1 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        materialize_c.materialize_phase51_c()
        sha2 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        assert sha1 == sha2

    def test_28_no_manual_selected_flag(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        forbidden = {"selected", "is_top_k", "manual_pick", "interesting"}
        for r in rows:
            assert not forbidden.intersection(r.keys())

    def test_29_no_best_seed_in_w1(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for r in rows:
            assert "best_seed" not in r
            assert "winner" not in r

    def test_30_no_ensemble_in_w2(self):
        rows = _load_csv("worst_shared_top20.csv")
        for r in rows:
            assert "ensemble" not in r
            assert "aggregated_score" not in r

    def test_31_no_new_inference_artifact(self):
        for fn in ["phase51_inference_manifest.csv", "phase51_predictions.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_32_no_checkpoint_loading(self):
        for fp in Path("artifacts/worst_error_analysis").glob("*"):
            if fp.is_file():
                assert fp.suffix not in (".pt", ".pkl", ".h5", ".ckpt")

    def test_33_no_training_artifact(self):
        for fn in ["phase51_loss_curve.csv", "phase51_epoch_log.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_34_no_scaler_fit(self):
        for fn in ["phase51_scaler.pkl", "phase51_scaler_params.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_35_no_prediction_correction(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for r in rows:
            assert "corrected_y_pred" not in r
            assert "shifted_y_pred" not in r
            assert "calibrated_y_pred" not in r

    def test_36_phase47_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/final_test/predictions/final_test_predictions_seed42.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("246ee0d725af972b")

    def test_37_phase48_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()
        ).hexdigest()
        assert sha.startswith("e8c102d582a35dd2")

    def test_38_phase49_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/residual_analysis/residual_long_table.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("8418a99110bfda70")

    def test_39_phase50_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/error_by_regime/test_regime_assignment.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("e90553cfc747a3f1")

    def test_40_phase52_not_started(self):
        for fn in ["phase52_attention_handoff.json", "phase52_attention_case_table.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_41_deterministic_rerun_identical(self):
        materialize_c.materialize_phase51_c()
        sha_w1 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        sha_w2 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_shared_top20.csv").read_bytes()
        ).hexdigest()
        materialize_c.materialize_phase51_c()
        sha_w1_rerun = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        sha_w2_rerun = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_shared_top20.csv").read_bytes()
        ).hexdigest()
        assert sha_w1 == sha_w1_rerun
        assert sha_w2 == sha_w2_rerun

    def test_42_no_casebook_created(self):
        from course_work.analysis.worst_error_analysis import materialize_c
        materialize_c.materialize_phase51_c()
        manifest = json.loads(
            Path("artifacts/worst_error_analysis/worst_error_ranking_manifest.json").read_text()
        )
        artifacts = manifest.get("artifacts", {})
        for key in artifacts:
            assert "casebook" not in key.lower()

    def test_43_no_figures_created(self):
        for fn in ["phase51_figure_1.png", "phase51_figure_2.png", "phase51_figures.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_44_no_overlap_jaccard_artifact(self):
        from course_work.analysis.worst_error_analysis import materialize_c
        import csv
        before = set((Path("artifacts/worst_error_analysis")).glob("*"))
        materialize_c.materialize_phase51_c()
        manifest = json.loads(
            Path("artifacts/worst_error_analysis/worst_error_ranking_manifest.json").read_text()
        )
        artifacts = manifest.get("artifacts", {})
        for key in artifacts:
            assert "overlap" not in key.lower()
            assert "membership" not in key.lower()

    def test_45_no_error_concentration_artifact(self):
        from course_work.analysis.worst_error_analysis import materialize_c
        manifest = json.loads(
            Path("artifacts/worst_error_analysis/worst_error_ranking_manifest.json").read_text()
        )
        artifacts = manifest.get("artifacts", {})
        for key in artifacts:
            assert "concentration" not in key.lower()
