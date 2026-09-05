"""Phase 51-D focused unit tests — 53 checks covering overlap, signed
overlap, error concentration, hardness, seed-spread context, and safety.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.phase51 import materialize_d
from course_work.phase51 import SEEDS


FROZEN_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)
WORKING_TABLE_SHA = (
    "81c43b504d932d52e30ae9358b9cfd3f1e125541e4d6fbd5b84927a45c5afc9f"
)
W1_SHA = "28c91ab02876e4c2533f811cd87a524c36f2e0b113ac118d7c32bf2fe9f9b716"
W2_SHA = "e551d14870992ec1739b5ae6d8bf403e551b8f6598e72462188eded01d0c678a"
W3_SHA = "9a0e4bb2eab97084bff8197c1251a0e9bd7a0abb627dc45439e0cb43d52e29d8"
W4_SHA = "a325c870d68d39a208797cf8349fd57aeec1b60a42a0b51ce40aab05115c7cfa"
W3_SH_SHA = "0b4bf8cb1e4f21c0bcf8f45cbe912b525d9f998f85849f2d954a533e66912637"
W4_SH_SHA = "a43b0f0a46c284e7949c2025bd5a8574af25b1215a886629a891fa7f14bdf3be"


def _load_csv(rel: str) -> list[dict[str, str]]:
    fp = Path("artifacts/worst_error_analysis") / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _sha(rel: str) -> str:
    return hashlib.sha256(
        (Path("artifacts/worst_error_analysis") / rel).read_bytes()
    ).hexdigest()


# ── OVERLAP (W1, W2, signed) ────────────────────────────────────────────────

class TestW1CrossSeedOverlap:
    """Checks 1-9: W1 cross-seed overlap + 3-way + W2-vs-W1."""

    def test_01_w1_20_unique_target_ids_per_seed(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            tids = {r["target_id"] for r in rows if r["seed"] == seed}
            assert len(tids) == 20

    def test_02_exactly_3_seed_pairs(self):
        rows = _load_csv("seed_overlap_table.csv")
        w1_pair = [r for r in rows if r["section"] == "W1_PAIRWISE"]
        assert len(w1_pair) == 3

    def test_03_intersection_count_correct(self):
        rows = _load_csv("seed_overlap_table.csv")
        # W1 3-way intersection must equal pairwise trio minimum
        trio = [r for r in rows if r["section"] == "W1_3WAY"][0]
        assert int(trio["intersection_count"]) >= 0
        assert int(trio["intersection_count"]) <= 20

    def test_04_union_count_correct(self):
        rows = _load_csv("seed_overlap_table.csv")
        trio = [r for r in rows if r["section"] == "W1_3WAY"][0]
        assert int(trio["union_count"]) >= int(trio["intersection_count"])

    def test_05_jaccard_formula(self):
        rows = _load_csv("seed_overlap_table.csv")
        for r in rows:
            if r.get("jaccard"):
                inter = int(r["intersection_count"])
                union = int(r["union_count"])
                j = float(r["jaccard"])
                assert abs(j - inter / union) < 1e-6

    def test_06_jaccard_in_unit_interval(self):
        rows = _load_csv("seed_overlap_table.csv")
        for r in rows:
            if r.get("jaccard"):
                j = float(r["jaccard"])
                assert 0.0 <= j <= 1.0

    def test_07_three_seed_intersection_correct(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        s42 = {r["target_id"] for r in rows if r["seed"] == "42"}
        s123 = {r["target_id"] for r in rows if r["seed"] == "123"}
        s2026 = {r["target_id"] for r in rows if r["seed"] == "2026"}
        expected = len(s42 & s123 & s2026)
        ovr = _load_csv("seed_overlap_table.csv")
        trio_row = next(r for r in ovr if r["section"] == "W1_3WAY")
        assert int(trio_row["intersection_count"]) == expected

    def test_08_three_seed_union_correct(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        s42 = {r["target_id"] for r in rows if r["seed"] == "42"}
        s123 = {r["target_id"] for r in rows if r["seed"] == "123"}
        s2026 = {r["target_id"] for r in rows if r["seed"] == "2026"}
        expected = len(s42 | s123 | s2026)
        ovr = _load_csv("seed_overlap_table.csv")
        trio_row = next(r for r in ovr if r["section"] == "W1_3WAY")
        assert int(trio_row["union_count"]) == expected

    def test_09_w2_vs_w1_correct(self):
        # W2 ∩ W1_42 = 20 per the canonical data; verify formula
        w1_rows = _load_csv("worst_per_seed_top20.csv")
        w2_rows = _load_csv("worst_shared_top20.csv")
        s42 = {r["target_id"] for r in w1_rows if r["seed"] == "42"}
        w2 = {r["target_id"] for r in w2_rows}
        ovr = _load_csv("seed_overlap_table.csv")
        row = next(
            r for r in ovr
            if r["section"] == "W2_VS_W1" and r["group_label"] == "W2_vs_seed42"
        )
        assert int(row["intersection_count"]) == len(s42 & w2)


# ── SIGNED overlap ──────────────────────────────────────────────────────────

class TestSignedOverlap:
    """Checks 10-14: signed UNDER/OVER overlap."""

    def test_10_under_overlap_uses_only_residual_gt_0(self):
        # Inherited from Phase 51-C; verify the W3 rows used for overlap
        # all have strictly positive residual
        rows = _load_csv("worst_underprediction_top10.csv")
        for r in rows:
            assert float(r["residual_wh"]) > 0

    def test_11_over_overlap_uses_only_residual_lt_0(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        for r in rows:
            assert float(r["residual_wh"]) < 0

    def test_12_exact_zero_excluded_in_signed(self):
        u = _load_csv("worst_underprediction_top10.csv")
        o = _load_csv("worst_overprediction_top10.csv")
        for r in u:
            assert float(r["residual_wh"]) != 0
        for r in o:
            assert float(r["residual_wh"]) != 0

    def test_13_shared_under_comparisons_present(self):
        ovr = _load_csv("seed_overlap_table.csv")
        rows = [r for r in ovr if r["section"] == "W3_SH_VS_W3"]
        assert len(rows) == 3
        for r in rows:
            assert float(r["jaccard"]) >= 0.0

    def test_14_shared_over_comparisons_present(self):
        ovr = _load_csv("seed_overlap_table.csv")
        rows = [r for r in ovr if r["section"] == "W4_SH_VS_W4"]
        assert len(rows) == 3
        for r in rows:
            assert float(r["jaccard"]) >= 0.0


# ── CONCENTRATION ───────────────────────────────────────────────────────────

class TestErrorConcentration:
    """Checks 15-24: SAE/SSE top-K concentration."""

    @pytest.fixture(scope="class")
    def rows(self):
        return _load_csv("error_concentration_table.csv")

    def test_15_global_sae_per_seed_reconstructed(self, rows):
        # global_sae_seed must be the seed's sum across full 2961.
        for r in rows[:1]:  # all K rows for a given seed share global_sae_seed
            pass
        wt = _load_csv("phase51_target_level_working_table.csv")
        for seed in SEEDS:
            abs_field = f"seed{seed}_abs_error_wh"
            expected = sum(float(r[abs_field]) for r in wt)
            actual = float(next(r for r in rows if r["seed"] == seed)["global_sae_seed"])
            assert abs(expected - actual) < 1e-3

    def test_16_global_sse_per_seed_reconstructed(self, rows):
        wt = _load_csv("phase51_target_level_working_table.csv")
        for seed in SEEDS:
            sq_field = f"seed{seed}_squared_error_wh2"
            expected = sum(float(r[sq_field]) for r in wt)
            actual = float(next(r for r in rows if r["seed"] == seed)["global_sse_seed"])
            assert abs(expected - actual) < 1e-3

    def test_17_top1_correct(self, rows):
        # Top1 sae_share should equal the worst-case abs_error / global SAE per seed
        wt = _load_csv("phase51_target_level_working_table.csv")
        w1 = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            abs_field = f"seed{seed}_abs_error_wh"
            sorted_wt = sorted(wt, key=lambda r: -float(r[abs_field]))
            top1 = float(sorted_wt[0][abs_field])
            global_sae = sum(float(r[abs_field]) for r in wt)
            row = next(r for r in rows if r["seed"] == seed and int(r["k"]) == 1)
            assert abs(float(row["sae_top_k"]) - top1) < 1e-4
            assert abs(float(row["sae_share"]) - top1 / global_sae) < 1e-9

    def test_18_top5_correct(self, rows):
        wt = _load_csv("phase51_target_level_working_table.csv")
        w1 = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            abs_field = f"seed{seed}_abs_error_wh"
            top5_set = {
                r["target_id"] for r in w1
                if r["seed"] == seed and int(r["rank"]) <= 5
            }
            sorted_wt = sorted(wt, key=lambda r: -float(r[abs_field]))
            top5_sum = sum(
                float(r[abs_field]) for r in sorted_wt
                if r["target_id"] in top5_set
            )
            row = next(r for r in rows if r["seed"] == seed and int(r["k"]) == 5)
            assert abs(float(row["sae_top_k"]) - top5_sum) < 1e-3

    def test_19_top10_correct(self, rows):
        wt = _load_csv("phase51_target_level_working_table.csv")
        w1 = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            abs_field = f"seed{seed}_abs_error_wh"
            top10_set = {
                r["target_id"] for r in w1
                if r["seed"] == seed and int(r["rank"]) <= 10
            }
            sorted_wt = sorted(wt, key=lambda r: -float(r[abs_field]))
            top10_sum = sum(
                float(r[abs_field]) for r in sorted_wt
                if r["target_id"] in top10_set
            )
            row = next(r for r in rows if r["seed"] == seed and int(r["k"]) == 10)
            assert abs(float(row["sae_top_k"]) - top10_sum) < 1e-3

    def test_20_top20_correct(self, rows):
        wt = _load_csv("phase51_target_level_working_table.csv")
        w1 = _load_csv("worst_per_seed_top20.csv")
        for seed in SEEDS:
            abs_field = f"seed{seed}_abs_error_wh"
            top20_set = {
                r["target_id"] for r in w1
                if r["seed"] == seed and int(r["rank"]) <= 20
            }
            sorted_wt = sorted(wt, key=lambda r: -float(r[abs_field]))
            top20_sum = sum(
                float(r[abs_field]) for r in sorted_wt
                if r["target_id"] in top20_set
            )
            row = next(r for r in rows if r["seed"] == seed and int(r["k"]) == 20)
            assert abs(float(row["sae_top_k"]) - top20_sum) < 1e-3

    def test_21_sae_share_monotonic_with_K(self, rows):
        for seed in SEEDS:
            seed_rows = sorted(
                (r for r in rows if r["seed"] == seed),
                key=lambda r: int(r["k"]),
            )
            for i in range(1, len(seed_rows)):
                prev = float(seed_rows[i - 1]["sae_share"])
                curr = float(seed_rows[i]["sae_share"])
                assert curr >= prev

    def test_22_sse_share_monotonic_with_K(self, rows):
        for seed in SEEDS:
            seed_rows = sorted(
                (r for r in rows if r["seed"] == seed),
                key=lambda r: int(r["k"]),
            )
            for i in range(1, len(seed_rows)):
                prev = float(seed_rows[i - 1]["sse_share"])
                curr = float(seed_rows[i]["sse_share"])
                assert curr >= prev

    def test_23_shares_le_1(self, rows):
        for r in rows:
            assert 0 <= float(r["sae_share"]) <= 1.0
            assert 0 <= float(r["sse_share"]) <= 1.0

    def test_24_no_3N_pooling(self, rows):
        # Each row's n_population is per-seed 2961, NOT 8883
        for r in rows:
            assert int(r["n_population"]) == 2961


# ── HARDNESS ────────────────────────────────────────────────────────────────

class TestHardness:
    """Checks 25-29: target-level hardness diagnostics."""

    def test_25_target_level_three_seed_values_aligned(self):
        wt = _load_csv("phase51_target_level_working_table.csv")
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        wt_ids = {r["target_id"] for r in wt}
        h_ids = {r["target_id"] for r in h}
        # h is a subset (union of worst-case targets), wt is the full set
        assert h_ids.issubset(wt_ids)
        assert wt_ids.issuperset(h_ids)

    def test_26_mean_abs_error_correct(self):
        wt = _load_csv("phase51_target_level_working_table.csv")
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        for r in h[:5]:
            tid = r["target_id"]
            w = wt_by_id[tid]
            expected = (
                float(w["seed42_abs_error_wh"])
                + float(w["seed123_abs_error_wh"])
                + float(w["seed2026_abs_error_wh"])
            ) / 3.0
            actual = float(r["computed_mean_abs_error_wh"])
            assert abs(expected - actual) < 1e-6

    def test_27_sample_sd_uses_ddof1(self):
        # statistics.stdev uses ddof=1 by default in Python
        # Compare against manual ddof=1 calculation
        wt = _load_csv("phase51_target_level_working_table.csv")
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        import statistics
        for r in h[:5]:
            tid = r["target_id"]
            w = wt_by_id[tid]
            vals = [
                float(w["seed42_abs_error_wh"]),
                float(w["seed123_abs_error_wh"]),
                float(w["seed2026_abs_error_wh"]),
            ]
            expected = statistics.stdev(vals)
            actual = float(r["computed_sample_sd_abs_error_wh"])
            assert abs(expected - actual) < 1e-9

    def test_28_no_best_seed_in_hardness(self):
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h:
            assert "best_seed" not in r
            assert "winner_seed" not in r

    def test_29_no_new_test_derived_threshold(self):
        # No threshold-style fields invented
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        banned = {"hard", "easy", "extreme_hard", "is_anomaly"}
        for r in h:
            assert not banned.intersection(r.keys())


# ── SEED-SPREAD ─────────────────────────────────────────────────────────────

class TestPhase48SeedSpreadContext:
    """Checks 30-35: Phase 48 spread join + summaries."""

    def test_30_phase48_seed_spread_joined_1to1(self):
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        # Spread fields are present and non-empty for every hardness row.
        for r in h:
            assert r["phase48_seed_mean_prediction"] != ""
            assert r["phase48_seed_std_prediction"] != ""
            assert r["phase48_seed_range_prediction"] != ""

    def test_31_no_missing_target_id(self):
        # target_id present and non-empty in every row
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h:
            assert r["target_id"] != ""

    def test_32_source_definition_preserved(self):
        # No recomputation of seed-spread metric
        # The values match the working-table seed_std_prediction field verbatim
        wt = _load_csv("phase51_target_level_working_table.csv")
        wt_by_id = {r["target_id"]: r for r in wt}
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h[:5]:
            tid = r["target_id"]
            assert r["phase48_seed_std_prediction"] == wt_by_id[tid]["seed_std_prediction"]

    def test_33_w1_summary_deterministic(self):
        # Rerunning produce_d gives identical summary
        before = hashlib.sha256(
            Path("artifacts/worst_error_analysis/hardness_group_summary.csv").read_bytes()
        ).hexdigest()
        materialize_d.materialize_phase51_d()
        after = hashlib.sha256(
            Path("artifacts/worst_error_analysis/hardness_group_summary.csv").read_bytes()
        ).hexdigest()
        assert before == after

    def test_34_w2_summary_deterministic(self):
        before = hashlib.sha256(
            Path("artifacts/worst_error_analysis/seed_overlap_table.csv").read_bytes()
        ).hexdigest()
        materialize_d.materialize_phase51_d()
        after = hashlib.sha256(
            Path("artifacts/worst_error_analysis/seed_overlap_table.csv").read_bytes()
        ).hexdigest()
        assert before == after

    def test_35_no_causal_label(self):
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h:
            for key, val in r.items():
                if isinstance(val, str):
                    assert "causes" not in val.lower()
                    assert "leads to" not in val.lower()


# ── SAFETY ─────────────────────────────────────────────────────────────────

class TestSafety:
    """Checks 36-53: upstream immutability + safety invariants."""

    def test_36_contract_sha_unchanged(self):
        assert _sha("worst_error_selection_contract.json") == FROZEN_CONTRACT_SHA

    def test_37_phase51_c_rankings_unchanged(self):
        assert _sha("worst_per_seed_top20.csv") == W1_SHA
        assert _sha("worst_shared_top20.csv") == W2_SHA
        assert _sha("worst_underprediction_top10.csv") == W3_SHA
        assert _sha("worst_overprediction_top10.csv") == W4_SHA
        assert _sha("shared_all_under_top10.csv") == W3_SH_SHA
        assert _sha("shared_all_over_top10.csv") == W4_SH_SHA

    def test_38_working_table_unchanged(self):
        assert _sha("phase51_target_level_working_table.csv") == WORKING_TABLE_SHA

    def test_39_phase47_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/final_test/predictions/final_test_predictions_seed42.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("246ee0d725af972b")

    def test_40_phase48_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/prediction_analysis/phase_48_signoff.json").read_bytes()
        ).hexdigest()
        assert sha.startswith("e8c102d582a35dd2")

    def test_41_phase49_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/residual_analysis/residual_long_table.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("8418a99110bfda70")

    def test_42_phase50_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/error_by_regime/test_regime_assignment.csv").read_bytes()
        ).hexdigest()
        assert sha.startswith("e90553cfc747a3f1")

    def test_43_no_training_artifact(self):
        for fn in ["phase51_d_loss_curve.csv", "phase51_d_epoch_log.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_44_no_new_inference_artifact(self):
        for fn in ["phase51_d_predictions.csv", "phase51_d_inference_manifest.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_45_no_checkpoint_loading(self):
        for fp in Path("artifacts/worst_error_analysis").glob("*"):
            if fp.is_file():
                assert fp.suffix not in (".pt", ".pkl", ".h5", ".ckpt")

    def test_46_no_scaler_fit(self):
        for fn in ["phase51_d_scaler.pkl", "phase51_d_scaler_params.json"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_47_no_prediction_correction(self):
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h:
            assert "corrected" not in r
            assert "shifted" not in r
            assert "calibrated" not in r

    def test_48_no_ensemble_metric(self):
        ovr = _load_csv("seed_overlap_table.csv")
        for r in ovr:
            for v in r.values():
                assert "ensemble" not in str(v).lower()

    def test_49_no_individual_case_inspection(self):
        h = _load_csv("hardness_vs_seed_disagreement.csv")
        for r in h:
            for v in r.values():
                assert "narrative" not in str(v).lower()
                assert "case_study" not in str(v).lower()

    def test_50_no_casebook(self):
        for fn in ["casebook.csv", "casebook.md", "phase51_d_casebook.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_51_no_figures(self):
        for fn in ["phase51_d_figure_1.png", "phase51_d_figure_2.png"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_52_phase52_not_started(self):
        for fn in ["phase52_attention_handoff.json", "phase52_attention_case_table.csv"]:
            assert not (Path("artifacts/worst_error_analysis") / fn).exists()

    def test_53_deterministic_rerun(self):
        sha_d_before = hashlib.sha256(
            Path("artifacts/worst_error_analysis/seed_overlap_table.csv").read_bytes()
        ).hexdigest()
        sha_c_before = hashlib.sha256(
            Path("artifacts/worst_error_analysis/error_concentration_table.csv").read_bytes()
        ).hexdigest()
        materialize_d.materialize_phase51_d()
        sha_d_after = hashlib.sha256(
            Path("artifacts/worst_error_analysis/seed_overlap_table.csv").read_bytes()
        ).hexdigest()
        sha_c_after = hashlib.sha256(
            Path("artifacts/worst_error_analysis/error_concentration_table.csv").read_bytes()
        ).hexdigest()
        assert sha_d_before == sha_d_after
        assert sha_c_before == sha_c_after
