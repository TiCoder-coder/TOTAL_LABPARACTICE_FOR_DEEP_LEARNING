"""Phase 51-B focused unit tests.

40 checks verifying Phase 51-B governance, contract freeze, source verification,
alignment audit, working-table structure, and safety invariants.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from course_work.phase51 import (
    SEEDS,
    N_TEST,
    TEST_POPULATION_FINGERPRINT,
    RESIDUAL_CONVENTION,
    POSITIVE_RESIDUAL_SEMANTICS,
    NEGATIVE_RESIDUAL_SEMANTICS,
    ZERO_POLICY,
    CADENCE_MINUTES,
    LOCAL_CONTEXT_RADIUS,
    K_ABS_PER_SEED,
    K_SHARED,
    K_UNDER_PER_SEED,
    K_OVER_PER_SEED,
    K_SHARED_SIGNED,
    PRIMARY_RANKING_METRIC,
    TIE_BREAK,
    RANKING_FAMILIES,
)
from course_work.phase51 import alignment
from course_work.phase51 import materialize_b
from course_work.phase51 import sources


# ── Governance / contract constants ───────────────────────────────────────────

class TestFrozenContractConstants:
    """Checks 1-8: contract constants match frozen values."""

    def test_01_seeds(self):
        assert SEEDS == ("42", "123", "2026")

    def test_02_n_test(self):
        assert N_TEST == 2961

    def test_03_residual_long_rows(self):
        # 3 seeds × 2961 = 8883
        assert N_TEST * len(SEEDS) == 8883

    def test_04_test_population_fingerprint(self):
        assert TEST_POPULATION_FINGERPRINT == (
            "d7dbc0b3cde772dc3f62c181f0a09a4a7a5b0e7c78e567361dbfde089578da87"
        )

    def test_05_residual_convention(self):
        assert RESIDUAL_CONVENTION == "y_true - y_pred"

    def test_06_residual_sign_semantics(self):
        assert POSITIVE_RESIDUAL_SEMANTICS == "UNDERPREDICTION"
        assert NEGATIVE_RESIDUAL_SEMANTICS == "OVERPREDICTION"
        assert ZERO_POLICY == "EXACT_ZERO"


# ── K values and contract freeze ──────────────────────────────────────────────

class TestSelectionContractConstants:
    """Checks 9-13: K values match frozen Human-approved values."""

    def test_07_k_per_seed(self):
        assert K_ABS_PER_SEED == 20

    def test_08_k_shared(self):
        assert K_SHARED == 20

    def test_09_k_signed(self):
        assert K_UNDER_PER_SEED == 10
        assert K_OVER_PER_SEED == 10
        assert K_SHARED_SIGNED == 10

    def test_10_context_radius(self):
        assert LOCAL_CONTEXT_RADIUS == 6

    def test_11_primary_ranking(self):
        assert PRIMARY_RANKING_METRIC == "absolute_error_wh"
        assert TIE_BREAK == "target_id ASC"

    def test_12_ranking_families_registered(self):
        expected = {"W1", "W2", "W3", "W4", "W3_SH", "W4_SH"}
        assert set(RANKING_FAMILIES.keys()) == expected

    def test_13_cadence(self):
        assert CADENCE_MINUTES == 10


# ── Source verification ───────────────────────────────────────────────────────

class TestSourceVerification:
    """Checks 14-19: source SHA verification."""

    def test_14_all_sources_verify(self):
        results = sources.verify_all_sources()
        assert all(results.values())
        assert len(results) == 8

    def test_15_phase47_seed42(self):
        fp = Path("artifacts/final_test/predictions/final_test_predictions_seed42.csv")
        expected = "246ee0d725af972bd621ce9cf4dbc550d8c02ec7c9dc1214b373807c99bf73f2"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected

    def test_16_phase47_seed123(self):
        fp = Path("artifacts/final_test/predictions/final_test_predictions_seed123.csv")
        expected = "1bb55c445ffe132d3cfdc22e09439d77c76a2defd2f918bfb8f47029f666a08b"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected

    def test_17_phase47_seed2026(self):
        fp = Path("artifacts/final_test/predictions/final_test_predictions_seed2026.csv")
        expected = "bfb575357dd6a8e3a23fd08230c69f8e3d3297582ea73c16cf006601fcce79d8"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected

    def test_18_phase50_assignment_sha(self):
        fp = Path("artifacts/error_by_regime/test_regime_assignment.csv")
        expected = "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected

    def test_19_phase50_threshold_sha(self):
        fp = Path("artifacts/error_by_regime/regime_thresholds_train_only.json")
        expected = "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected


# ── Alignment audit ──────────────────────────────────────────────────────────

class TestAlignmentAudit:
    """Checks 20-30: alignment audit verifies target/source integrity."""

    def test_20_alignment_runs(self):
        result = alignment.audit_alignment()
        assert result.n_test == 2961
        assert result.n_residual_long == 8883
        assert result.n_seeds == 3

    def test_21_seeds_match(self):
        result = alignment.audit_alignment()
        assert set(result.seeds) == set(SEEDS)

    def test_22_residual_rows_per_seed(self):
        result = alignment.audit_alignment()
        for seed in SEEDS:
            assert result.residual_rows_per_seed[seed] == 2961

    def test_23_target_ids_unique_per_seed(self):
        result = alignment.audit_alignment()
        for seed in SEEDS:
            assert result.target_ids_unique_per_seed[seed] == 2961

    def test_24_timestamps_exact(self):
        result = alignment.audit_alignment()
        assert result.timestamps_exact is True

    def test_25_y_true_exact(self):
        result = alignment.audit_alignment()
        assert result.y_true_exact is True

    def test_26_residual_convention_verified(self):
        result = alignment.audit_alignment()
        assert result.residual_convention_verified is True

    def test_27_abs_error_verified(self):
        result = alignment.audit_alignment()
        assert result.abs_error_verified is True

    def test_28_squared_error_verified(self):
        result = alignment.audit_alignment()
        assert result.squared_error_verified is True

    def test_29_residual_sign_verified(self):
        result = alignment.audit_alignment()
        assert result.residual_sign_verified is True

    def test_30_phase50_join_lossless(self):
        result = alignment.audit_alignment()
        assert result.phase50_assignment_join_lossless is True


# ── Phase 50 regime labels (inherited) ───────────────────────────────────────

class TestRegimeLabelsInherited:
    """Check 31: regime labels unchanged from Phase 50."""

    def test_31_phase50_regime_labels_unchanged(self):
        result = alignment.audit_alignment()
        assert result.phase50_regime_labels_unchanged is True


# ── Working table structure ──────────────────────────────────────────────────

class TestWorkingTableStructure:
    """Checks 32-34: working table structure compliance."""

    def test_32_working_table_n_rows(self):
        rows = alignment.build_target_level_working_table()
        assert len(rows) == 2961

    def test_33_working_table_no_rank_columns(self):
        rows = alignment.build_target_level_working_table()
        forbidden = {"rank", "selected", "worst_case", "is_top_k", "top_k"}
        for row in rows:
            assert not forbidden.intersection(row.keys())

    def test_34_working_table_has_required_fields(self):
        rows = alignment.build_target_level_working_table()
        required = {
            "target_id", "target_timestamp", "y_true_wh",
            "seed42_y_pred_wh", "seed42_residual_wh", "seed42_abs_error_wh",
            "seed123_y_pred_wh", "seed123_residual_wh", "seed123_abs_error_wh",
            "seed2026_y_pred_wh", "seed2026_residual_wh", "seed2026_abs_error_wh",
            "mean_abs_error_wh", "seed_abs_error_std_wh",
            "R1_TARGET_LEVEL", "R2_EXTREME_HIGH", "R3_CHANGE_MAGNITUDE",
            "R4_CHANGE_DIRECTION", "R5_TIME_OF_DAY", "R6_DAY_TYPE",
            "seed_mean_prediction", "seed_std_prediction", "seed_range_prediction",
        }
        for row in rows:
            assert required.issubset(row.keys())


# ── Materialize_b safety invariants ──────────────────────────────────────────

class TestMaterializeBSafety:
    """Checks 35-40: materialize_b produces artifacts without ranking."""

    def test_35_materialize_b_runs(self):
        result = materialize_b.materialize_phase51_b()
        assert result["status"] == "PASS"
        assert result["n_defects"] == 0

    def test_36_selection_contract_frozen(self):
        contract_path = (
            Path("artifacts/worst_error_analysis") / "worst_error_selection_contract.json"
        )
        assert contract_path.exists()
        payload = json.loads(contract_path.read_text())
        assert payload["contract_frozen"] is True
        assert payload["selection_executed"] is False
        assert payload["worst_error_ranking_executed"] is False
        assert payload["individual_case_inspection"] is False
        assert payload["phase52_authorized"] is False

    def test_37_selection_contract_no_ranking(self):
        contract_path = (
            Path("artifacts/worst_error_analysis") / "worst_error_selection_contract.json"
        )
        payload = json.loads(contract_path.read_text())
        # No actual rank columns or selected fields.
        # Allow the literal word "rank" in metadata keys like "ranking_families"
        # but forbid the specific rank-related fields.
        forbidden_field_names = {
            "rank", "selected", "worst_case", "is_top_k", "top_k",
            "rank_column", "rank_per_seed", "rank_overall", "case_selected",
        }
        for key in payload.keys():
            assert key not in forbidden_field_names, f"forbidden field in contract: {key}"
        # Check no nested forbidden keys
        def _check_nested(obj, path="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert k not in forbidden_field_names, (
                        f"forbidden field at {path}.{k}"
                    )
                    _check_nested(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _check_nested(v, f"{path}[{i}]")
        _check_nested(payload)

    def test_38_fingerprint_deterministic(self):
        contract_path = (
            Path("artifacts/worst_error_analysis") / "worst_error_selection_contract.json"
        )
        contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        fp_path = (
            Path("artifacts/worst_error_analysis") / "selection_contract_fingerprint.json"
        )
        payload = json.loads(fp_path.read_text())
        assert payload["selection_contract_sha256"] == contract_sha

    def test_39_working_table_no_ranking(self):
        # Phase 51-B builds the prerequisite working table. It must NOT
        # itself create ranking CSVs (Phase 51-C), casebook CSVs
        # (Phase 51-F), or D/E diagnostics. Verify by introspecting the
        # Phase 51-B manifest rather than checking directory state
        # (since later phases legitimately produce them).
        result = materialize_b.materialize_phase51_b()
        # The Phase 51-B manifest is the source-of-truth for what 51-B produced.
        manifest_fp = Path("artifacts/worst_error_analysis/worst_error_analysis_manifest.json")
        if manifest_fp.exists():
            import json as _json
            manifest = _json.loads(manifest_fp.read_text())
            forbidden_substrings = (
                "worst_per_seed_top20",
                "worst_shared_top20",
                "worst_underprediction_top10",
                "worst_overprediction_top10",
                "shared_all_under_top10",
                "shared_all_over_top10",
                "seed_overlap",
                "worst_case_membership",
                "error_concentration",
                "hardness_vs_seed",
                "hardness_group_summary",
                "regime_overrepresentation",
                "baseline_context",
                "regime_composition",
                "shared_worst_regime_context",
                "casebook",
                "local_temporal_context",
                "input_window_manifest",
                "feature_order_audit",
                "input_window_integrity_audit",
                "casebook_integrity_audit",
                "context_integrity_audit",
            )
            deliverables = manifest.get("deliverables", {}) or manifest.get("artifacts", {})
            for key in deliverables:
                for sub in forbidden_substrings:
                    assert sub not in key, (
                        f"Phase 51-B manifest must not list {key} (contains {sub!r})"
                    )

    def test_40_no_phase50_modification(self):
        # Phase 50 artifacts must be unchanged
        fp = Path("artifacts/error_by_regime/test_regime_assignment.csv")
        expected = "e90553cfc747a3f15e0e9ec9e6868ae497e7ade797dc14a81999e416b74219ac"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected
        fp = Path("artifacts/error_by_regime/regime_thresholds_train_only.json")
        expected = "2fe9ad4f873e3b3e42013fe3b2d630e377e4e568120769bd2234a76d6974b109"
        assert hashlib.sha256(fp.read_bytes()).hexdigest() == expected
