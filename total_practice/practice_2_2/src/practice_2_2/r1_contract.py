from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import get_practice_2_2_root
from .resources import file_sha256


CONTRACT_RELATIVE_PATH = Path("configs/r1_label_contract.json")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_LABELS = {
    "body_wash",
    "face_mask",
    "facial_cleanser",
    "lipstick",
    "moisturizer",
    "perfume",
    "serum",
    "shampoo",
    "sunscreen",
    "toner",
}


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_r1_contract(contract_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(contract_path or root / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != 1:
        raise RuntimeError("Unsupported R1 contract schema")
    if set(contract.get("classes", {})) != EXPECTED_LABELS:
        raise RuntimeError("R1 contract must define exactly ten target labels")
    if set(contract.get("allowed_decisions", [])) != {
        "accept",
        "quarantine",
        "relabel",
    }:
        raise RuntimeError("R1 contract decisions are invalid")
    policy = contract.get("review_policy", {})
    if policy.get("required_reviewer_count") != 2:
        raise RuntimeError("R1 pilot requires exactly two reviewers")
    if not 0 <= policy.get("minimum_exact_decision_agreement", -1) <= 1:
        raise RuntimeError("R1 exact agreement threshold is invalid")
    if not -1 <= policy.get("minimum_cohen_kappa", -2) <= 1:
        raise RuntimeError("R1 kappa threshold is invalid")
    if not 0 <= policy.get("minimum_per_label_agreement", -1) <= 1:
        raise RuntimeError("R1 per-label agreement threshold is invalid")
    reason_codes = contract.get("reason_codes", {})
    if not reason_codes or any(not code.strip() for code in reason_codes):
        raise RuntimeError("R1 reason codes are missing")
    for label, rules in contract["classes"].items():
        required = {
            "display_name",
            "definition",
            "positive_evidence",
            "negative_evidence",
            "borderline_cases",
            "confusable_labels",
        }
        if not required.issubset(rules):
            raise RuntimeError(f"R1 class contract is incomplete: {label}")
        if not rules["positive_evidence"] or not rules["negative_evidence"]:
            raise RuntimeError(f"R1 class evidence is incomplete: {label}")
        if not rules["borderline_cases"]:
            raise RuntimeError(f"R1 borderline cases are incomplete: {label}")
        if not set(rules["confusable_labels"]).issubset(EXPECTED_LABELS - {label}):
            raise RuntimeError(f"R1 confusable labels are invalid: {label}")
    return contract


def build_pilot_manifest(
    dataset_root: Path,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(contract or load_r1_contract())
    dataset_root = Path(dataset_root).expanduser().resolve()
    policy = contract["review_policy"]
    seed = int(policy["pilot_seed"])
    per_label = int(policy["samples_per_label"])
    samples = []
    for label in sorted(EXPECTED_LABELS):
        class_root = dataset_root / label
        candidates = [
            path
            for path in class_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and "_aug" not in path.stem
        ]
        ranked = sorted(
            candidates,
            key=lambda path: hashlib.sha256(
                f"{seed}:{path.relative_to(dataset_root).as_posix()}".encode()
            ).hexdigest(),
        )
        if len(ranked) < per_label:
            raise RuntimeError(f"Insufficient original images for R1 pilot: {label}")
        for path in ranked[:per_label]:
            relative = path.relative_to(dataset_root).as_posix()
            samples.append(
                {
                    "sample_id": hashlib.sha256(relative.encode()).hexdigest()[:20],
                    "relative_path": relative,
                    "original_label": label,
                    "file_sha256": file_sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "pilot_id": "r1_semantic_pilot_s42_v1",
        "contract_version": contract["contract_version"],
        "selection_method": "sha256_ranked_original_only",
        "seed": seed,
        "samples_per_label": per_label,
        "sample_count": len(samples),
        "split_blinded": True,
        "model_output_blinded": True,
        "samples": samples,
    }
    manifest["pilot_sha256"] = _json_sha256(manifest)
    return manifest


def validate_pilot_manifest(
    pilot: Mapping[str, Any],
    dataset_root: Path,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(contract or load_r1_contract())
    expected = build_pilot_manifest(dataset_root, contract)
    if dict(pilot) != expected:
        raise RuntimeError("R1 pilot manifest is not deterministic or has changed")
    counts = Counter(sample["original_label"] for sample in pilot["samples"])
    return {
        "pilot_sha256": pilot["pilot_sha256"],
        "sample_count": pilot["sample_count"],
        "counts_by_label": dict(sorted(counts.items())),
        "all_files_verified": True,
        "generated_images_included": False,
        "split_blinded": pilot["split_blinded"],
        "model_output_blinded": pilot["model_output_blinded"],
    }


def validate_review(
    review: Mapping[str, Any],
    pilot: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    contract = dict(contract or load_r1_contract())
    reviewer_id = str(review.get("reviewer_id", "")).strip()
    if not reviewer_id or reviewer_id.lower() in {"pending", "unknown", "anonymous"}:
        raise RuntimeError("R1 reviewer identity is invalid")
    if review.get("contract_version") != contract["contract_version"]:
        raise RuntimeError("R1 review contract version mismatch")
    if review.get("pilot_sha256") != pilot["pilot_sha256"]:
        raise RuntimeError("R1 review pilot hash mismatch")
    expected = {sample["sample_id"]: sample for sample in pilot["samples"]}
    decisions = review.get("decisions", [])
    if len(decisions) != len(expected):
        raise RuntimeError("R1 review must cover every pilot sample exactly once")
    by_sample: dict[str, dict[str, Any]] = {}
    reason_codes = set(contract["reason_codes"])
    for item in decisions:
        sample_id = item.get("sample_id")
        if sample_id not in expected or sample_id in by_sample:
            raise RuntimeError("R1 review contains an unknown or duplicate sample")
        decision = item.get("decision")
        proposed = item.get("proposed_label")
        original = expected[sample_id]["original_label"]
        if decision not in contract["allowed_decisions"]:
            raise RuntimeError("R1 review decision is invalid")
        if item.get("reason_code") not in reason_codes:
            raise RuntimeError("R1 review reason code is invalid")
        if not str(item["reason_code"]).startswith(f"{decision.upper()}_"):
            raise RuntimeError("R1 reason code does not match the decision")
        if not isinstance(item.get("policy_case_resolved"), bool):
            raise RuntimeError("R1 policy resolution field must be boolean")
        if decision == "accept" and proposed != original:
            raise RuntimeError("R1 accept decision must preserve the original label")
        if decision == "relabel" and (
            proposed not in EXPECTED_LABELS or proposed == original
        ):
            raise RuntimeError("R1 relabel decision must choose another target label")
        if decision == "quarantine" and proposed is not None:
            raise RuntimeError("R1 quarantine decision must not assign a label")
        by_sample[sample_id] = dict(item)
    return by_sample


def build_review_template(
    pilot: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(contract or load_r1_contract())
    return {
        "schema_version": 1,
        "reviewer_id": "",
        "contract_version": contract["contract_version"],
        "pilot_sha256": pilot["pilot_sha256"],
        "decisions": [
            {
                "sample_id": sample["sample_id"],
                "decision": None,
                "proposed_label": None,
                "reason_code": None,
                "policy_case_resolved": None,
            }
            for sample in pilot["samples"]
        ],
    }


def _agreement_category(item: Mapping[str, Any]) -> str:
    return f"{item['decision']}:{item.get('proposed_label') or 'none'}"


def _cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("R1 agreement inputs are invalid")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[category] * right_counts[category] for category in categories
    ) / (len(left) ** 2)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def compute_review_agreement(
    left_review: Mapping[str, Any],
    right_review: Mapping[str, Any],
    pilot: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(contract or load_r1_contract())
    left = validate_review(left_review, pilot, contract)
    right = validate_review(right_review, pilot, contract)
    ordered = [sample["sample_id"] for sample in pilot["samples"]]
    left_categories = [_agreement_category(left[sample_id]) for sample_id in ordered]
    right_categories = [_agreement_category(right[sample_id]) for sample_id in ordered]
    exact = sum(a == b for a, b in zip(left_categories, right_categories)) / len(ordered)
    per_label = {}
    for label in sorted(EXPECTED_LABELS):
        ids = [
            sample["sample_id"]
            for sample in pilot["samples"]
            if sample["original_label"] == label
        ]
        per_label[label] = sum(
            _agreement_category(left[sample_id])
            == _agreement_category(right[sample_id])
            for sample_id in ids
        ) / len(ids)
    unresolved = sorted(
        sample_id
        for sample_id in ordered
        if not left[sample_id]["policy_case_resolved"]
        or not right[sample_id]["policy_case_resolved"]
    )
    conflicts = sorted(
        sample_id
        for sample_id in ordered
        if _agreement_category(left[sample_id])
        != _agreement_category(right[sample_id])
    )
    return {
        "reviewer_ids": [left_review["reviewer_id"], right_review["reviewer_id"]],
        "exact_decision_agreement": exact,
        "cohen_kappa": _cohen_kappa(left_categories, right_categories),
        "per_label_agreement": per_label,
        "unresolved_policy_case_ids": unresolved,
        "conflict_sample_ids": conflicts,
    }


def build_r1_verification(
    pilot: Mapping[str, Any],
    dataset_root: Path,
    reviews: Sequence[Mapping[str, Any]] = (),
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(contract or load_r1_contract())
    pilot_result = validate_pilot_manifest(pilot, dataset_root, contract)
    policy = contract["review_policy"]
    result = {
        "schema_version": 1,
        "phase": "R1",
        "contract_version": contract["contract_version"],
        "contract_status": contract["status"],
        "label_count": len(contract["classes"]),
        "decision_set": contract["allowed_decisions"],
        "reason_code_count": len(contract["reason_codes"]),
        "pilot": pilot_result,
        "review_count": len(reviews),
        "required_reviewer_count": policy["required_reviewer_count"],
        "minimum_exact_decision_agreement": policy[
            "minimum_exact_decision_agreement"
        ],
        "minimum_cohen_kappa": policy["minimum_cohen_kappa"],
        "minimum_per_label_agreement": policy["minimum_per_label_agreement"],
        "test_loader_constructed": False,
        "test_evaluated": False,
        "source_images_mutated": False,
    }
    if len(reviews) < policy["required_reviewer_count"]:
        result["status"] = "pending_independent_review"
        result["gate_passed"] = False
        return result
    if len(reviews) != policy["required_reviewer_count"]:
        raise RuntimeError("R1 verification requires exactly two reviews")
    if reviews[0].get("reviewer_id") == reviews[1].get("reviewer_id"):
        raise RuntimeError("R1 reviewers must be independent")
    agreement = compute_review_agreement(reviews[0], reviews[1], pilot, contract)
    thresholds_passed = (
        agreement["exact_decision_agreement"]
        >= policy["minimum_exact_decision_agreement"]
        and agreement["cohen_kappa"] >= policy["minimum_cohen_kappa"]
        and min(agreement["per_label_agreement"].values())
        >= policy["minimum_per_label_agreement"]
        and (
            not policy["require_zero_unresolved_policy_cases"]
            or not agreement["unresolved_policy_case_ids"]
        )
    )
    result["agreement"] = agreement
    result["status"] = "passed" if thresholds_passed else "failed_review_thresholds"
    result["gate_passed"] = thresholds_passed
    return result


def write_json_record(value: Mapping[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R1 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n")
    return output_path
