import json
from pathlib import Path

import pytest

from practice_2_2.r1_contract import (
    EXPECTED_LABELS,
    build_pilot_manifest,
    build_r1_verification,
    build_review_template,
    compute_review_agreement,
    load_r1_contract,
    validate_pilot_manifest,
    validate_review,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data/final/data_clean_balanced"


def _complete_review(reviewer_id, pilot):
    return {
        "schema_version": 1,
        "reviewer_id": reviewer_id,
        "contract_version": "r1_label_contract_v1",
        "pilot_sha256": pilot["pilot_sha256"],
        "decisions": [
            {
                "sample_id": sample["sample_id"],
                "decision": "accept",
                "proposed_label": sample["original_label"],
                "reason_code": "ACCEPT_CLEAR_SINGLE_PRODUCT",
                "policy_case_resolved": True,
            }
            for sample in pilot["samples"]
        ],
    }


def test_contract_defines_complete_rules_for_all_labels():
    contract = load_r1_contract()
    assert set(contract["classes"]) == EXPECTED_LABELS
    for rules in contract["classes"].values():
        assert rules["definition"]
        assert rules["positive_evidence"]
        assert rules["negative_evidence"]
        assert rules["borderline_cases"]


def test_pilot_is_deterministic_balanced_original_only_and_hash_verified():
    contract = load_r1_contract()
    first = build_pilot_manifest(DATASET_ROOT, contract)
    second = build_pilot_manifest(DATASET_ROOT, contract)
    assert first == second
    result = validate_pilot_manifest(first, DATASET_ROOT, contract)
    assert result["sample_count"] == 100
    assert set(result["counts_by_label"].values()) == {10}
    assert result["generated_images_included"] is False
    assert all("_aug" not in sample["relative_path"] for sample in first["samples"])


def test_review_template_is_incomplete_and_cannot_pass_validation():
    contract = load_r1_contract()
    pilot = build_pilot_manifest(DATASET_ROOT, contract)
    template = build_review_template(pilot, contract)
    with pytest.raises(RuntimeError, match="reviewer identity"):
        validate_review(template, pilot, contract)


def test_review_validation_enforces_decision_and_reason_semantics():
    contract = load_r1_contract()
    pilot = build_pilot_manifest(DATASET_ROOT, contract)
    review = _complete_review("reviewer_a", pilot)
    review["decisions"][0]["reason_code"] = "QUARANTINE_TEXT_ONLY"
    with pytest.raises(RuntimeError, match="reason code does not match"):
        validate_review(review, pilot, contract)


def test_identical_independent_reviews_pass_predeclared_thresholds():
    contract = load_r1_contract()
    pilot = build_pilot_manifest(DATASET_ROOT, contract)
    left = _complete_review("reviewer_a", pilot)
    right = _complete_review("reviewer_b", pilot)
    agreement = compute_review_agreement(left, right, pilot, contract)
    assert agreement["exact_decision_agreement"] == 1.0
    assert agreement["cohen_kappa"] == 1.0
    assert set(agreement["per_label_agreement"].values()) == {1.0}
    result = build_r1_verification(
        pilot, DATASET_ROOT, [left, right], contract
    )
    assert result["status"] == "passed"
    assert result["gate_passed"] is True


def test_current_r1_gate_stays_pending_without_real_reviews():
    contract = load_r1_contract()
    pilot = json.loads((ROOT / "docs/R1_PILOT_SAMPLE.json").read_text())
    result = build_r1_verification(pilot, DATASET_ROOT, contract=contract)
    assert result["status"] == "pending_independent_review"
    assert result["gate_passed"] is False
    assert result["test_loader_constructed"] is False
    assert result["test_evaluated"] is False
    assert result["source_images_mutated"] is False


def test_r1_implementation_has_no_training_test_or_delete_operations():
    source = (ROOT / "src/practice_2_2/r1_contract.py").read_text()
    assert "torch" not in source
    assert "DataLoader" not in source
    assert "optimizer" not in source
    assert ".unlink(" not in source
    assert ".remove(" not in source
