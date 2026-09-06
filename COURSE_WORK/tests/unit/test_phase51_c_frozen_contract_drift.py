"""Phase 51-C frozen-contract conformance regression tests.

Regression coverage for the contract-drift defect that introduced
``target_timestamp`` as an additional tie-break tier. The frozen
contract specifies tie_break = ``target_id ASC`` (no timestamp).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from course_work.analysis.worst_error_analysis import (
    materialize_c,
)
from course_work.analysis.worst_error_analysis.frozen_sort_key_compliance import (
    FORBIDDEN_SORT_KEYS,
    PERMITTED_SORT_KEYS,
    audit_no_forbidden_sort_keys,
    verify_frozen_sort_keys_used,
)


FROZEN_CONTRACT_SHA = (
    "ec798326cb03586e85ce7ba09d53be03016a234fe15e1ba5fb4b3fbf0eb967d4"
)


def _load_csv(rel: str) -> list[dict[str, str]]:
    fp = Path("artifacts/worst_error_analysis") / rel
    with fp.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestFrozenContractSortKeyCompliance:
    """W1/W2/W3/W4/W3_SH/W4_SH use ONLY frozen-contract permitted keys."""

    def test_01_permitted_sort_keys_constant(self):
        assert "absolute_error_wh" in PERMITTED_SORT_KEYS
        assert "mean_abs_error_wh" in PERMITTED_SORT_KEYS
        assert "target_id" in PERMITTED_SORT_KEYS

    def test_02_target_timestamp_is_forbidden(self):
        assert "target_timestamp" in FORBIDDEN_SORT_KEYS

    def test_03_audit_module_classifies_timestamp(self):
        findings = audit_no_forbidden_sort_keys()
        # Aggregate total: zero forbidden across both ranking modules
        all_forbidden = [f for r in findings.values() for f in r["forbidden"]]
        assert "target_timestamp" not in all_forbidden

    def test_04_compliance_function_passes(self):
        assert verify_frozen_sort_keys_used() is True

    def test_05_w1_tie_break_target_id_only(self):
        rows = _load_csv("worst_per_seed_top20.csv")
        # All row records must declare tie_break = target_id ASC (not timestamp).
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_06_w2_tie_break_target_id_only(self):
        rows = _load_csv("worst_shared_top20.csv")
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_07_w3_tie_break_target_id_only(self):
        rows = _load_csv("worst_underprediction_top10.csv")
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_08_w4_tie_break_target_id_only(self):
        rows = _load_csv("worst_overprediction_top10.csv")
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_09_w3_shared_tie_break_target_id_only(self):
        rows = _load_csv("shared_all_under_top10.csv")
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_10_w4_shared_tie_break_target_id_only(self):
        rows = _load_csv("shared_all_over_top10.csv")
        bad = [r for r in rows if r["tie_break"] != "target_id ASC"]
        assert bad == []

    def test_11_no_target_timestamp_tie_break_anywhere(self):
        # Grep all Phase 51-C outputs to ensure no tie_break field mentions timestamp.
        for rel in [
            "worst_per_seed_top20.csv",
            "worst_shared_top20.csv",
            "worst_underprediction_top10.csv",
            "worst_overprediction_top10.csv",
            "shared_all_under_top10.csv",
            "shared_all_over_top10.csv",
        ]:
            for row in _load_csv(rel):
                assert "timestamp" not in row["tie_break"].lower()


class TestNoManualTopKModification:
    """Verify Top-K membership is contract-driven, not manually edited."""

    def test_12_w1_deterministic_rerun(self):
        materialize_c.materialize_phase51_c()
        sha1 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        materialize_c.materialize_phase51_c()
        sha2 = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_per_seed_top20.csv").read_bytes()
        ).hexdigest()
        assert sha1 == sha2

    def test_13_no_manual_selected_flag_anywhere(self):
        forbidden_columns = {"selected", "manual_pick", "interesting", "case_note"}
        for rel in [
            "worst_per_seed_top20.csv",
            "worst_shared_top20.csv",
            "worst_underprediction_top10.csv",
            "worst_overprediction_top10.csv",
            "shared_all_under_top10.csv",
            "shared_all_over_top10.csv",
        ]:
            for row in _load_csv(rel):
                assert not forbidden_columns.intersection(row.keys())

    def test_14_frozen_contract_sha_unchanged(self):
        sha = hashlib.sha256(
            Path("artifacts/worst_error_analysis/worst_error_selection_contract.json").read_bytes()
        ).hexdigest()
        assert sha == FROZEN_CONTRACT_SHA

    def test_15_frozen_contract_tie_break_literal(self):
        contract = json.loads(
            Path("artifacts/worst_error_analysis/worst_error_selection_contract.json").read_text()
        )
        assert contract["tie_break"] == "target_id ASC"
        assert "timestamp" not in contract["tie_break"].lower()
