from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import get_practice_2_2_root


POLICY_RELATIVE_PATH = Path("configs/r4_split_policy.json")
EXPECTED_SPLITS = ("Train", "Validation", "Test")


def load_r4_policy(policy_path: Path | None = None) -> dict[str, Any]:
    root = get_practice_2_2_root()
    path = Path(policy_path or root / POLICY_RELATIVE_PATH).expanduser().resolve()
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise RuntimeError("Unsupported R4 split policy schema")
    if policy.get("seed") != 42:
        raise RuntimeError("R4 split must use seed 42")
    if tuple(policy.get("splits", {})) != EXPECTED_SPLITS:
        raise RuntimeError("R4 split names or order are invalid")
    if abs(sum(policy["splits"].values()) - 1.0) > 1e-12:
        raise RuntimeError("R4 split ratios must sum to one")
    if policy.get("generated_assets_excluded") is not True:
        raise RuntimeError("R4 must exclude generated assets")
    if policy.get("component_splitting_allowed") is not False:
        raise RuntimeError("R4 must prohibit component splitting")
    if policy.get("predecessor_gate_required") is not True:
        raise RuntimeError("R4 must require its predecessor gate")
    if policy.get("blocked_manifest_use_for_model") is not False:
        raise RuntimeError("Blocked R4 manifests must prohibit model use")
    if policy.get("automatic_training_allowed") is not False:
        raise RuntimeError("R4 must prohibit automatic training")
    if policy.get("automatic_test_evaluation_allowed") is not False:
        raise RuntimeError("R4 must prohibit automatic Test evaluation")
    return policy


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_component_splits(
    components: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    policy = dict(policy or load_r4_policy())
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    member_ids = set()
    component_ids = set()
    for component in components:
        component_id = str(component["component_id"])
        members = list(component["member_asset_ids"])
        labels = list(component["labels"])
        if component_id in component_ids or not members or len(labels) != 1:
            raise RuntimeError("R4 received an invalid R3 component")
        if any(asset_id in member_ids for asset_id in members):
            raise RuntimeError("R4 received overlapping R3 components")
        component_ids.add(component_id)
        member_ids.update(members)
        by_label[str(labels[0])].append(component)
    assignments = {}
    split_names = tuple(policy["splits"])
    for label, label_components in sorted(by_label.items()):
        if len(label_components) < len(split_names):
            raise RuntimeError(f"R4 class has too few components: {label}")
        total_images = sum(int(item["member_count"]) for item in label_components)
        total_groups = len(label_components)
        target_images = {
            split: policy["splits"][split] * total_images for split in split_names
        }
        target_groups = {
            split: policy["splits"][split] * total_groups for split in split_names
        }
        image_counts = {split: 0 for split in split_names}
        group_counts = {split: 0 for split in split_names}
        ordered = sorted(
            label_components,
            key=lambda item: (
                -int(item["member_count"]),
                _seed_rank(policy["seed"], str(item["component_id"])),
                str(item["component_id"]),
            ),
        )
        for component in ordered:
            size = int(component["member_count"])
            selected = min(
                split_names,
                key=lambda split: (
                    ((image_counts[split] + size) / target_images[split]) ** 2
                    + ((group_counts[split] + 1) / target_groups[split]) ** 2,
                    _seed_rank(
                        policy["seed"], f"{component['component_id']}:{split}"
                    ),
                    split,
                ),
            )
            assignments[str(component["component_id"])] = selected
            image_counts[selected] += size
            group_counts[selected] += 1
        if any(image_counts[split] == 0 for split in split_names):
            raise RuntimeError(f"R4 failed class coverage: {label}")
    return assignments


def build_asset_manifest(
    components: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
    lineage_authorized: bool,
) -> list[dict[str, Any]]:
    originals = {
        str(record["asset_id"]): record
        for record in inventory
        if not record["is_generated"] and record["image_error"] is None
    }
    component_members = {
        str(asset_id)
        for component in components
        for asset_id in component["member_asset_ids"]
    }
    if component_members != set(originals):
        raise RuntimeError("R4 component membership does not match R2 originals")
    rows = []
    for component in components:
        component_id = str(component["component_id"])
        split = assignments[component_id]
        for asset_id in component["member_asset_ids"]:
            record = originals[str(asset_id)]
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "relative_path": record["relative_path"],
                    "class_name": record["class_name"],
                    "component_id": component_id,
                    "split": split,
                    "raw_sha256": record["raw_sha256"],
                    "decoded_pixel_sha256": record["decoded_pixel_sha256"],
                    "source_group": record["source_group"],
                    "product_id": record["product_id"],
                    "is_generated": False,
                    "eligibility": (
                        "accepted_for_model"
                        if lineage_authorized
                        else "provisional_pending_predecessor_gates"
                    ),
                    "use_for_model": lineage_authorized,
                }
            )
    return sorted(rows, key=lambda item: item["asset_id"])


def build_split_summary(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r4_policy())
    labels = sorted({str(row["class_name"]) for row in rows})
    split_names = tuple(policy["splits"])
    total_images = len(rows)
    total_groups = len({row["component_id"] for row in rows})
    split_inventory = []
    class_inventory = []
    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        split_inventory.append(
            {
                "split": split,
                "images": len(split_rows),
                "image_ratio": len(split_rows) / total_images,
                "image_ratio_deviation": abs(
                    len(split_rows) / total_images - policy["splits"][split]
                ),
                "components": len({row["component_id"] for row in split_rows}),
                "component_ratio": len(
                    {row["component_id"] for row in split_rows}
                )
                / total_groups,
            }
        )
    for label in labels:
        label_rows = [row for row in rows if row["class_name"] == label]
        label_groups = {row["component_id"] for row in label_rows}
        counts = {}
        for split in split_names:
            selected = [row for row in label_rows if row["split"] == split]
            selected_groups = {row["component_id"] for row in selected}
            counts[split] = {
                "images": len(selected),
                "image_ratio": len(selected) / len(label_rows),
                "image_ratio_deviation": abs(
                    len(selected) / len(label_rows) - policy["splits"][split]
                ),
                "components": len(selected_groups),
                "component_ratio": len(selected_groups) / len(label_groups),
                "component_ratio_deviation": abs(
                    len(selected_groups) / len(label_groups)
                    - policy["splits"][split]
                ),
            }
        class_inventory.append(
            {
                "class_name": label,
                "images": len(label_rows),
                "components": len(label_groups),
                "splits": counts,
            }
        )
    return {
        "schema_version": 1,
        "lineage": policy["policy_version"],
        "seed": policy["seed"],
        "target_ratios": policy["splits"],
        "images": total_images,
        "components": total_groups,
        "split_inventory": split_inventory,
        "class_inventory": class_inventory,
        "maximum_overall_image_ratio_deviation": max(
            item["image_ratio_deviation"] for item in split_inventory
        ),
        "maximum_per_class_image_ratio_deviation": max(
            values["image_ratio_deviation"]
            for item in class_inventory
            for values in item["splits"].values()
        ),
        "maximum_per_class_component_ratio_deviation": max(
            values["component_ratio_deviation"]
            for item in class_inventory
            for values in item["splits"].values()
        ),
        "every_class_in_every_split": all(
            values["images"] > 0 and values["components"] > 0
            for item in class_inventory
            for values in item["splits"].values()
        ),
    }


def _identity_violations(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value not in {None, ""}:
            groups[str(value)].append(row)
    violations = []
    for value, members in sorted(groups.items()):
        splits = sorted({str(member["split"]) for member in members})
        if len(splits) <= 1:
            continue
        violations.append(
            {
                "identity_key": key,
                "identity_value_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "splits": splits,
                "asset_ids": sorted(str(member["asset_id"]) for member in members),
            }
        )
    return violations


def build_leakage_report(
    rows: Sequence[Mapping[str, Any]],
    accepted_edges: Sequence[Mapping[str, Any]],
    review_edges: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_asset = {str(row["asset_id"]): row for row in rows}
    component_violations = []
    component_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        component_splits[str(row["component_id"])].add(str(row["split"]))
    for component_id, splits in sorted(component_splits.items()):
        if len(splits) > 1:
            component_violations.append(
                {"component_id": component_id, "splits": sorted(splits)}
            )
    identity = {
        key: _identity_violations(rows, key)
        for key in (
            "product_id",
            "raw_sha256",
            "decoded_pixel_sha256",
            "source_group",
        )
    }
    confirmed_visual_violations = []
    for edge in accepted_edges:
        left = by_asset[str(edge["left_asset_id"])]
        right = by_asset[str(edge["right_asset_id"])]
        if left["split"] != right["split"]:
            confirmed_visual_violations.append(
                {
                    "edge_id": edge["edge_id"],
                    "left_split": left["split"],
                    "right_split": right["split"],
                }
            )
    cross_split_review = []
    for edge in review_edges:
        left = by_asset[str(edge["left_asset_id"])]
        right = by_asset[str(edge["right_asset_id"])]
        if left["split"] == right["split"]:
            continue
        completed = dict(edge)
        completed["left_split"] = left["split"]
        completed["right_split"] = right["split"]
        cross_split_review.append(completed)
    cross_split_review.sort(key=lambda item: item["edge_id"])
    strong_violation_count = (
        len(component_violations)
        + len(confirmed_visual_violations)
        + sum(len(items) for items in identity.values())
    )
    report = {
        "schema_version": 1,
        "component_cross_split_violation_count": len(component_violations),
        "component_cross_split_violations": component_violations,
        "identity_cross_split_violation_counts": {
            key: len(items) for key, items in identity.items()
        },
        "identity_cross_split_violations": identity,
        "confirmed_visual_cross_split_violation_count": len(
            confirmed_visual_violations
        ),
        "confirmed_visual_cross_split_violations": confirmed_visual_violations,
        "strong_leakage_violation_count": strong_violation_count,
        "review_only_cross_split_pair_count": len(cross_split_review),
        "review_only_cross_split_same_label_count": sum(
            edge["left_label"] == edge["right_label"]
            for edge in cross_split_review
        ),
        "review_only_cross_split_cross_label_count": sum(
            edge["left_label"] != edge["right_label"]
            for edge in cross_split_review
        ),
        "all_review_only_cross_split_pairs_documented": True,
    }
    return report, cross_split_review


def build_split_fingerprints(
    rows: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset_payload = [
        {
            key: row[key]
            for key in (
                "asset_id",
                "relative_path",
                "class_name",
                "component_id",
                "raw_sha256",
                "decoded_pixel_sha256",
                "source_group",
                "product_id",
            )
        }
        for row in sorted(rows, key=lambda item: item["asset_id"])
    ]
    split_payload = [
        {"asset_id": row["asset_id"], "split": row["split"]}
        for row in sorted(rows, key=lambda item: item["asset_id"])
    ]
    component_payload = [
        {
            "component_id": component["component_id"],
            "member_asset_ids": sorted(component["member_asset_ids"]),
        }
        for component in sorted(components, key=lambda item: item["component_id"])
    ]
    test_payload = [
        {
            "asset_id": row["asset_id"],
            "component_id": row["component_id"],
            "relative_path": row["relative_path"],
        }
        for row in sorted(rows, key=lambda item: item["asset_id"])
        if row["split"] == "Test"
    ]
    return {
        "schema_version": 1,
        "dataset_fingerprint_sha256": canonical_json_sha256(dataset_payload),
        "component_fingerprint_sha256": canonical_json_sha256(component_payload),
        "split_fingerprint_sha256": canonical_json_sha256(split_payload),
        "test_partition_fingerprint_sha256": canonical_json_sha256(test_payload),
    }


def build_r4_report(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    leakage: Mapping[str, Any],
    r3_report: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r4_policy())
    blocked_reasons = []
    if not r3_report.get("gate_passed"):
        blocked_reasons.append("R3_GATE_NOT_PASSED")
    if r3_report.get("product_provenance_coverage") != 1.0:
        blocked_reasons.append("UNIQUE_PRODUCT_DISTRIBUTION_NOT_PROVEN")
    if leakage["strong_leakage_violation_count"]:
        blocked_reasons.append("CONFIRMED_GROUP_LEAKAGE_PRESENT")
    if leakage["review_only_cross_split_pair_count"]:
        blocked_reasons.append("REVIEW_ONLY_CROSS_SPLIT_EDGES_PENDING")
    if not summary["every_class_in_every_split"]:
        blocked_reasons.append("CLASS_COVERAGE_FAILED")
    if summary["maximum_overall_image_ratio_deviation"] > policy[
        "maximum_overall_image_ratio_deviation"
    ]:
        blocked_reasons.append("OVERALL_IMAGE_RATIO_DEVIATION_EXCEEDED")
    if summary["maximum_per_class_image_ratio_deviation"] > policy[
        "maximum_per_class_image_ratio_deviation"
    ]:
        blocked_reasons.append("CLASS_IMAGE_RATIO_DEVIATION_EXCEEDED")
    if summary["maximum_per_class_component_ratio_deviation"] > policy[
        "maximum_per_class_group_ratio_deviation"
    ]:
        blocked_reasons.append("CLASS_COMPONENT_RATIO_DEVIATION_EXCEEDED")
    asset_ids = [str(row["asset_id"]) for row in rows]
    if len(asset_ids) != len(set(asset_ids)):
        blocked_reasons.append("ASSET_ASSIGNMENT_NOT_UNIQUE")
    return {
        "schema_version": 1,
        "phase": "R4",
        "lineage": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "seed": policy["seed"],
        "manifest_kind": "authorized" if not blocked_reasons else "candidate",
        "assets": len(rows),
        "unique_assets": len(set(asset_ids)),
        "components": summary["components"],
        "generated_assets_included": sum(bool(row["is_generated"]) for row in rows),
        "classes": len(summary["class_inventory"]),
        "every_class_in_every_split": summary["every_class_in_every_split"],
        "maximum_overall_image_ratio_deviation": summary[
            "maximum_overall_image_ratio_deviation"
        ],
        "maximum_per_class_image_ratio_deviation": summary[
            "maximum_per_class_image_ratio_deviation"
        ],
        "maximum_per_class_component_ratio_deviation": summary[
            "maximum_per_class_component_ratio_deviation"
        ],
        "strong_leakage_violation_count": leakage[
            "strong_leakage_violation_count"
        ],
        "review_only_cross_split_pair_count": leakage[
            "review_only_cross_split_pair_count"
        ],
        "review_only_cross_split_pairs_documented": leakage[
            "all_review_only_cross_split_pairs_documented"
        ],
        "model_training_authorized": not blocked_reasons,
        "test_partition_cryptographically_sealed": True,
        "test_access_allowed": False,
        "test_access_count": 0,
        "test_evaluation_count": 0,
        "historical_metric_comparison_allowed": False,
        "source_images_mutated": False,
        "canonical_notebook_mutated": False,
        "successor_phase_executed": False,
    }


def build_test_guard(
    report: Mapping[str, Any], fingerprints: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lineage": report["lineage"],
        "guard_state": (
            "sealed_authorized_for_development"
            if report["gate_passed"]
            else "sealed_candidate_blocked"
        ),
        "lineage_authorized": report["gate_passed"],
        "model_training_authorized": report["model_training_authorized"],
        "test_partition_cryptographically_sealed": True,
        "split_fingerprint_sha256": fingerprints["split_fingerprint_sha256"],
        "test_partition_fingerprint_sha256": fingerprints[
            "test_partition_fingerprint_sha256"
        ],
        "test_access_allowed": False,
        "test_access_count": 0,
        "test_evaluation_count": 0,
        "authorization_token_sha256": None,
        "repeat_evaluation_allowed": False,
        "maintenance_override_used": False,
    }


def authorize_test_access(
    guard_path: Path, split_fingerprint_sha256: str, authorization_token: str | None
) -> bool:
    guard = json.loads(Path(guard_path).read_text())
    if guard.get("split_fingerprint_sha256") != split_fingerprint_sha256:
        raise RuntimeError("R4 Test guard split fingerprint mismatch")
    if guard.get("lineage_authorized") is not True:
        raise RuntimeError("R4 lineage is not authorized")
    if guard.get("test_access_allowed") is not True:
        raise RuntimeError("R4 Test access is sealed")
    expected = guard.get("authorization_token_sha256")
    actual = (
        hashlib.sha256(authorization_token.encode()).hexdigest()
        if authorization_token is not None
        else None
    )
    if expected is None or actual != expected:
        raise RuntimeError("R4 Test authorization token is invalid")
    if guard.get("test_access_count") != 0:
        raise RuntimeError("R4 Test has already been accessed")
    return True


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R4 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path


def write_manifest_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R4 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path
