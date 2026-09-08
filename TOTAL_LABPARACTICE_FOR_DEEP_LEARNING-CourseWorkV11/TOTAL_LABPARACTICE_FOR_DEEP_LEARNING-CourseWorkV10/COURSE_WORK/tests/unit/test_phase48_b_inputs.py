"""Focused Phase 48-B unit tests.

Verifies the safety invariants and acceptance criteria for the Phase 48-B slice:

* All 3 transformer seed bundles verified with SHA-256
* Population fingerprint match across bundles
* N_TEST = 2961
* Target IDs / timestamps / y_true identical across all 4 bundles
* 0 duplicates
* 0 non-finite predictions
* Chronological order
* seed_std uses ddof=1 (sample SD)
* Lag sign convention frozen in contract BEFORE any lag computation
* Persistence bundle seed field normalized to 'PERSISTENCE' in long table
* Phase 47 source bundles byte-identical to baseline (no mutation)
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
F47 = REPO / "artifacts" / "final_test"
P48 = REPO / "artifacts" / "prediction_analysis"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def checksum_registry() -> dict:
    return json.loads((F47 / "prediction_checksums.json").read_text())


@pytest.fixture(scope="module")
def pop_manifest() -> dict:
    return json.loads((F47 / "final_test_population_manifest.json").read_text())


@pytest.fixture(scope="module")
def phase48_manifest() -> dict:
    return json.loads((P48 / "prediction_analysis_manifest.json").read_text())


@pytest.fixture(scope="module")
def phase48_contract() -> dict:
    return json.loads((P48 / "prediction_analysis_contract.json").read_text())


@pytest.fixture(scope="module")
def preflight_rows() -> list[dict]:
    with (P48 / "phase48_preflight_audit.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def alignment_rows() -> list[dict]:
    with (P48 / "prediction_alignment_audit.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def source_verification_rows() -> list[dict]:
    with (P48 / "prediction_source_verification.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def wide_rows() -> list[dict]:
    with (P48 / "prediction_wide_table.csv").open() as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def long_rows() -> list[dict]:
    with (P48 / "prediction_long_table.csv").open() as fh:
        return list(csv.DictReader(fh))


class TestSourceBundleIntegrity:
    def test_three_transformer_bundles_verified(self, source_verification_rows):
        transformer_rows = [r for r in source_verification_rows if r["source_id"].startswith("TRANSFORMER_SEED")]
        assert len(transformer_rows) == 3
        assert all(r["status"] == "PASS" for r in transformer_rows)

    def test_persistence_bundle_verified(self, source_verification_rows):
        persistence_rows = [r for r in source_verification_rows if r["source_id"] == "PERSISTENCE"]
        assert len(persistence_rows) == 1
        assert persistence_rows[0]["status"] == "PASS"

    def test_all_checksums_match_registry(self, checksum_registry, source_verification_rows):
        for row in source_verification_rows:
            sid = row["source_id"]
            if sid.startswith("TRANSFORMER_SEED"):
                seed = int(sid.replace("TRANSFORMER_SEED", ""))
                expected = checksum_registry["predictions"][f"seed_{seed}"]["sha256"]
            else:
                expected = checksum_registry["predictions"]["persistence"]["sha256"]
            assert row["observed_sha256"] == expected

    def test_row_count_2961_per_bundle(self, source_verification_rows):
        for row in source_verification_rows:
            assert int(row["row_count"]) == 2961

    def test_population_fingerprint_consistent(self, source_verification_rows, pop_manifest):
        pop_sha = pop_manifest["target_ids_sha256"]
        for row in source_verification_rows:
            assert row["population_sha256"] == pop_sha


class TestAlignmentAudit:
    def test_alignment_all_pass(self, alignment_rows):
        assert all(r["status"] == "PASS" for r in alignment_rows)

    def test_n_test_2961(self, alignment_rows):
        n_row = next(r for r in alignment_rows if r["check"] == "same_N_rows")
        assert n_row["seed42"] == "2961"
        assert n_row["seed123"] == "2961"
        assert n_row["seed2026"] == "2961"
        assert n_row["persistence"] == "2961"
        assert n_row["expected"] == "2961"

    def test_same_target_ids(self, alignment_rows):
        row = next(r for r in alignment_rows if r["check"] == "same_target_ids")
        assert row["status"] == "PASS"

    def test_same_timestamps(self, alignment_rows):
        row = next(r for r in alignment_rows if r["check"] == "same_target_timestamps")
        assert row["status"] == "PASS"

    def test_same_y_true(self, alignment_rows):
        row = next(r for r in alignment_rows if r["check"] == "same_y_true_wh")
        assert row["status"] == "PASS"


class TestNoDuplicatesAndFinite:
    def test_no_duplicate_target_ids_in_wide_table(self, wide_rows):
        ids = [r["target_id"] for r in wide_rows]
        assert len(ids) == len(set(ids))
        assert len(ids) == 2961

    def test_all_predictions_finite_in_wide_table(self, wide_rows):
        for r in wide_rows:
            for col in ("y_pred_seed42", "y_pred_seed123", "y_pred_seed2026",
                        "seed_mean_prediction", "seed_std_prediction",
                        "seed_min_prediction", "seed_max_prediction"):
                v = float(r[col])
                assert v == v and v not in (float("inf"), float("-inf"))

    def test_chronological_order(self, wide_rows):
        ts = [r["target_timestamp"] for r in wide_rows]
        assert ts == sorted(ts)


class TestSampleSDWithDDoF1:
    def test_seed_std_uses_ddof_1(self, wide_rows):
        for r in wide_rows[:50]:
            triple = [
                float(r["y_pred_seed42"]),
                float(r["y_pred_seed123"]),
                float(r["y_pred_seed2026"]),
            ]
            expected_std = statistics.stdev(triple)
            observed_std = float(r["seed_std_prediction"])
            assert abs(expected_std - observed_std) < 1e-9

    def test_seed_range_is_max_minus_min(self, wide_rows):
        for r in wide_rows[:50]:
            triple = [
                float(r["y_pred_seed42"]),
                float(r["y_pred_seed123"]),
                float(r["y_pred_seed2026"]),
            ]
            expected_range = max(triple) - min(triple)
            observed_range = float(r["seed_range_prediction"])
            assert abs(expected_range - observed_range) < 1e-9


class TestLagConventionFrozen:
    def test_contract_has_lag_sign_convention(self, phase48_contract):
        assert "lag_sign_convention" in phase48_contract
        assert "lag_k_positive" in phase48_contract["lag_sign_convention"]

    def test_contract_freezes_lag_range(self, phase48_contract):
        assert phase48_contract["lag_range_steps"] == list(range(-6, 7))

    def test_contract_freezes_seed_std_ddof(self, phase48_contract):
        assert phase48_contract["seed_std_ddof"] == 1

    def test_preflight_has_lag_convention_check(self, preflight_rows):
        row = next((r for r in preflight_rows if r["check"] == "lag_sign_convention_frozen"), None)
        assert row is not None
        assert row["status"] == "PASS"


class TestForbiddenActions:
    def test_no_inference_path(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_inference_path_invoked")
        assert row["status"] == "PASS"

    def test_no_training_path(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_training_path_invoked")
        assert row["status"] == "PASS"

    def test_no_checkpoint_loading(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_checkpoint_loading_for_new_predictions")
        assert row["status"] == "PASS"

    def test_no_scaler_fitting(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_scaler_fitting")
        assert row["status"] == "PASS"

    def test_no_best_seed_selection(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_best_seed_selection")
        assert row["status"] == "PASS"

    def test_no_ensemble_metric(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "no_ensemble_metric")
        assert row["status"] == "PASS"

    def test_no_source_mutation(self, preflight_rows):
        row = next(r for r in preflight_rows if r["check"] == "source_files_readonly")
        assert row["status"] == "PASS"

    def test_contract_forbids_all_actions(self, phase48_contract):
        forbidden = phase48_contract["forbidden_actions"]
        assert all(forbidden.values()), f"Forbidden actions not all True: {forbidden}"


class TestPersistenceNormalization:
    def test_long_table_persistence_label(self, long_rows):
        pers_rows = [r for r in long_rows if r["seed"] == "PERSISTENCE"]
        assert len(pers_rows) == 2961

    def test_long_table_no_empty_seed(self, long_rows):
        bad = [r for r in long_rows if r["seed"] == ""]
        assert len(bad) == 0

    def test_long_table_3_transformer_seeds(self, long_rows):
        for seed in ("42", "123", "2026"):
            seed_rows = [r for r in long_rows if r["seed"] == seed]
            assert len(seed_rows) == 2961


class TestPhase47SourcesUnchanged:
    def test_seed42_csv_byte_identical(self):
        baseline = Path("/tmp/p47_seed42.csv")
        if not baseline.exists():
            pytest.skip("baseline snapshot not present")
        assert _sha(baseline) == _sha(F47 / "predictions" / "final_test_predictions_seed42.csv")

    def test_seed123_csv_byte_identical(self):
        baseline = Path("/tmp/p47_seed123.csv")
        if not baseline.exists():
            pytest.skip("baseline snapshot not present")
        assert _sha(baseline) == _sha(F47 / "predictions" / "final_test_predictions_seed123.csv")

    def test_seed2026_csv_byte_identical(self):
        baseline = Path("/tmp/p47_seed2026.csv")
        if not baseline.exists():
            pytest.skip("baseline snapshot not present")
        assert _sha(baseline) == _sha(F47 / "predictions" / "final_test_predictions_seed2026.csv")

    def test_persistence_csv_byte_identical(self):
        baseline = Path("/tmp/p47_persistence.csv")
        if not baseline.exists():
            pytest.skip("baseline snapshot not present")
        assert _sha(baseline) == _sha(F47 / "predictions" / "final_test_predictions_persistence.csv")
