from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .paths import get_practice_2_2_root
from .r0_authority import directory_digest
from .r1_contract import EXPECTED_LABELS, load_r1_contract
from .r2_provenance import load_r2_policy
from .resources import file_sha256


AUGMENTED_PATTERN = re.compile(r"_aug\d+$", flags=re.IGNORECASE)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _source_group(path: Path) -> str:
    return AUGMENTED_PATTERN.sub("", path.stem)


def _dhash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _image_metrics(path: Path, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            rgb_image = image.convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
            gray = rgb.astype(np.float32).mean(axis=2)
            center = gray[1:-1, 1:-1]
            laplacian = (
                gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
                - 4 * center
            )
            spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(
                np.int16
            )
            pixel_payload = (
                rgb_image.tobytes() + f"{rgb_image.width}x{rgb_image.height}".encode()
            )
            metrics = {
                "width": int(rgb_image.width),
                "height": int(rgb_image.height),
                "mode": image.mode,
                "brightness": float(gray.mean() / 255.0),
                "contrast": float(gray.std() / 255.0),
                "laplacian_variance": float(laplacian.var()),
                "content_ratio": float((spread > 20).mean()),
                "decoded_pixel_sha256": hashlib.sha256(pixel_payload).hexdigest(),
                "dhash64": f"{_dhash(rgb_image):016x}",
                "image_error": None,
            }
    except Exception as error:
        return {
            "width": None,
            "height": None,
            "mode": None,
            "brightness": None,
            "contrast": None,
            "laplacian_variance": None,
            "content_ratio": None,
            "decoded_pixel_sha256": None,
            "dhash64": None,
            "image_error": repr(error),
            "quality_flags": ["INVALID_IMAGE"],
        }
    aspect = metrics["width"] / metrics["height"]
    flags = []
    if metrics["width"] < thresholds["minimum_width"]:
        flags.append("WIDTH_BELOW_MINIMUM")
    if metrics["height"] < thresholds["minimum_height"]:
        flags.append("HEIGHT_BELOW_MINIMUM")
    if not thresholds["minimum_aspect_ratio"] <= aspect <= thresholds[
        "maximum_aspect_ratio"
    ]:
        flags.append("EXTREME_ASPECT_RATIO")
    if metrics["brightness"] < thresholds["minimum_brightness"]:
        flags.append("EXTREMELY_DARK")
    if metrics["brightness"] > thresholds["maximum_brightness"]:
        flags.append("EXTREMELY_BRIGHT")
    if metrics["contrast"] < thresholds["minimum_contrast"]:
        flags.append("LOW_CONTRAST")
    if metrics["laplacian_variance"] < thresholds["minimum_laplacian_variance"]:
        flags.append("POTENTIALLY_BLURRY")
    if metrics["content_ratio"] < thresholds["minimum_content_ratio"]:
        flags.append("LOW_COLOR_CONTENT")
    metrics["quality_flags"] = flags
    return metrics


def _load_provenance(metadata_root: Path | None) -> dict[str, list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if metadata_root is None or not Path(metadata_root).is_dir():
        return by_hash
    for path in sorted(Path(metadata_root).glob("*.json")):
        record = json.loads(path.read_text())
        by_hash[str(record.get("raw_sha256"))].append(record)
    return by_hash


def inventory_legacy_dataset(
    dataset_root: Path,
    metadata_root: Path | None = None,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r2_policy())
    dataset_root = Path(dataset_root).expanduser().resolve()
    provenance = _load_provenance(metadata_root)
    records = []
    classes = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    if set(classes) != EXPECTED_LABELS:
        raise RuntimeError("R2 dataset labels do not match the label contract")
    for class_name in classes:
        for path in sorted((dataset_root / class_name).iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = path.relative_to(dataset_root).as_posix()
            raw_sha256 = file_sha256(path)
            sidecars = provenance.get(raw_sha256, [])
            matching = [
                record for record in sidecars if record.get("class_name") == class_name
            ]
            record = {
                "asset_id": hashlib.sha256(relative.encode()).hexdigest(),
                "relative_path": relative,
                "class_name": class_name,
                "raw_sha256": raw_sha256,
                "source_group": f"{class_name}/{_source_group(path)}",
                "is_generated": AUGMENTED_PATTERN.search(path.stem) is not None,
                "provenance_status": "complete" if len(matching) == 1 else "legacy_missing",
                "product_id": matching[0]["product_id"] if len(matching) == 1 else None,
                "provenance_asset_id": matching[0]["asset_id"] if len(matching) == 1 else None,
            }
            record.update(_image_metrics(path, policy["quality_thresholds"]))
            records.append(record)
    return records


def _load_r1_pilot_ids(pilot_path: Path | None) -> set[str]:
    if pilot_path is None or not Path(pilot_path).is_file():
        return set()
    pilot = json.loads(Path(pilot_path).read_text())
    return {sample["relative_path"] for sample in pilot.get("samples", [])}


def _load_historical_candidates(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    values = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("relative_path") or row.get("filepath") or row.get("path")
            if value:
                values.add(str(value))
    return values


def cross_label_visual_pairs(
    records: Sequence[Mapping[str, Any]],
    maximum_dhash_distance: int,
) -> list[dict[str, Any]]:
    originals = [
        record
        for record in records
        if not record["is_generated"] and record["dhash64"] is not None
    ]
    pairs = []
    for left_index, left in enumerate(originals):
        left_hash = int(left["dhash64"], 16)
        for right in originals[left_index + 1 :]:
            if left["class_name"] == right["class_name"]:
                continue
            exact = left["decoded_pixel_sha256"] == right["decoded_pixel_sha256"]
            distance = (left_hash ^ int(right["dhash64"], 16)).bit_count()
            if not exact and distance > maximum_dhash_distance:
                continue
            pairs.append(
                {
                    "left_relative_path": left["relative_path"],
                    "left_class_name": left["class_name"],
                    "right_relative_path": right["relative_path"],
                    "right_class_name": right["class_name"],
                    "decoded_pixel_exact": exact,
                    "dhash_distance": distance,
                    "review_status": "pending",
                }
            )
    pairs.sort(
        key=lambda item: (
            not item["decoded_pixel_exact"],
            item["dhash_distance"],
            item["left_relative_path"],
            item["right_relative_path"],
        )
    )
    return pairs


def build_semantic_review_queue(
    records: Sequence[Mapping[str, Any]],
    visual_pairs: Sequence[Mapping[str, Any]],
    r1_pilot_path: Path | None = None,
    historical_candidates_path: Path | None = None,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy = dict(policy or load_r2_policy())
    pilot_paths = _load_r1_pilot_ids(r1_pilot_path)
    historical_paths = _load_historical_candidates(
        Path(historical_candidates_path)
        if historical_candidates_path is not None
        else Path(policy["historical_review_candidates_path"])
    )
    pair_reasons: dict[str, set[str]] = defaultdict(set)
    for pair in visual_pairs:
        reason = (
            "CROSS_LABEL_EXACT_PIXEL_MATCH"
            if pair["decoded_pixel_exact"]
            else "CROSS_LABEL_DHASH_NEIGHBOR"
        )
        pair_reasons[pair["left_relative_path"]].add(reason)
        pair_reasons[pair["right_relative_path"]].add(reason)
    confusion_paths = set()
    for label in policy["confusion_priority_labels"]:
        candidates = sorted(
            (
                record
                for record in records
                if record["class_name"] == label and not record["is_generated"]
            ),
            key=lambda record: hashlib.sha256(
                f"42:{record['relative_path']}".encode()
            ).hexdigest(),
        )
        confusion_paths.update(
            record["relative_path"]
            for record in candidates[: policy["confusion_oversample_per_label"]]
        )
    queue = []
    for record in records:
        if record["is_generated"]:
            continue
        relative = record["relative_path"]
        reasons = {"FULL_SEMANTIC_AUDIT_REQUIRED"}
        reasons.update(pair_reasons.get(relative, set()))
        reasons.update(record["quality_flags"])
        if relative in pilot_paths:
            reasons.add("R1_PILOT_SAMPLE")
        if relative in historical_paths:
            reasons.add("HISTORICAL_REVIEW_CANDIDATE")
        if relative in confusion_paths:
            reasons.add("CONFUSION_GROUP_OVERSAMPLE")
        if "INVALID_IMAGE" in reasons or "CROSS_LABEL_EXACT_PIXEL_MATCH" in reasons:
            priority = "P0"
        elif "CROSS_LABEL_DHASH_NEIGHBOR" in reasons or "R1_PILOT_SAMPLE" in reasons:
            priority = "P1"
        elif len(reasons) > 1:
            priority = "P2"
        else:
            priority = "P3"
        queue.append(
            {
                "asset_id": record["asset_id"],
                "relative_path": relative,
                "raw_sha256": record["raw_sha256"],
                "original_label": record["class_name"],
                "product_id": record["product_id"],
                "provenance_status": record["provenance_status"],
                "priority": priority,
                "review_reasons": sorted(reasons),
                "review_status": "pending",
                "decision": None,
                "proposed_label": None,
                "reason_code": None,
                "reviewer_id": None,
                "reviewed_at_utc": None,
                "policy_case_resolved": None,
            }
        )
    queue.sort(key=lambda item: (item["priority"], item["original_label"], item["relative_path"]))
    return queue


def validate_semantic_decisions(
    queue: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = load_r1_contract()
    expected = {item["asset_id"]: item for item in queue}
    if len(decisions) != len(expected):
        raise RuntimeError("R2 decisions must cover every original asset")
    seen = set()
    counts = Counter()
    unresolved = []
    for decision in decisions:
        asset_id = decision.get("asset_id")
        if asset_id not in expected or asset_id in seen:
            raise RuntimeError("R2 decision contains an unknown or duplicate asset")
        seen.add(asset_id)
        action = decision.get("decision")
        proposed = decision.get("proposed_label")
        original = expected[asset_id]["original_label"]
        reason = str(decision.get("reason_code", ""))
        if action not in contract["allowed_decisions"]:
            raise RuntimeError("R2 semantic decision is invalid")
        if reason not in contract["reason_codes"] or not reason.startswith(
            f"{action.upper()}_"
        ):
            raise RuntimeError("R2 semantic reason code is invalid")
        if action == "accept" and proposed != original:
            raise RuntimeError("R2 accept decision must preserve the original label")
        if action == "relabel" and (
            proposed not in EXPECTED_LABELS or proposed == original
        ):
            raise RuntimeError("R2 relabel decision is invalid")
        if action == "quarantine" and proposed is not None:
            raise RuntimeError("R2 quarantine decision must not assign a label")
        if not str(decision.get("reviewer_id", "")).strip():
            raise RuntimeError("R2 decision reviewer is missing")
        if not str(decision.get("reviewed_at_utc", "")).strip():
            raise RuntimeError("R2 decision timestamp is missing")
        if decision.get("policy_case_resolved") is not True:
            unresolved.append(asset_id)
        counts[action] += 1
    return {
        "decision_count": len(decisions),
        "decision_counts": dict(sorted(counts.items())),
        "unresolved_policy_case_ids": sorted(unresolved),
    }


def build_r2_report(
    dataset_root: Path,
    records: Sequence[Mapping[str, Any]],
    visual_pairs: Sequence[Mapping[str, Any]],
    queue: Sequence[Mapping[str, Any]],
    r0_verification_path: Path,
    r1_verification_path: Path,
    historical_candidates_path: Path,
    decisions: Sequence[Mapping[str, Any]] = (),
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_r2_policy())
    r0 = json.loads(Path(r0_verification_path).read_text())
    r1 = json.loads(Path(r1_verification_path).read_text())
    originals = [record for record in records if not record["is_generated"]]
    class_inventory = []
    for label in sorted(EXPECTED_LABELS):
        class_records = [record for record in records if record["class_name"] == label]
        class_originals = [record for record in class_records if not record["is_generated"]]
        product_ids = {
            record["product_id"] for record in class_originals if record["product_id"]
        }
        class_inventory.append(
            {
                "class_name": label,
                "files": len(class_records),
                "original_files": len(class_originals),
                "generated_files": sum(record["is_generated"] for record in class_records),
                "decoded_pixel_groups": len(
                    {
                        record["decoded_pixel_sha256"]
                        for record in class_originals
                        if record["decoded_pixel_sha256"]
                    }
                ),
                "source_groups": len({record["source_group"] for record in class_records}),
                "provenance_complete": sum(
                    record["provenance_status"] == "complete"
                    for record in class_originals
                ),
                "unique_products": len(product_ids) if product_ids else None,
                "minimum_unique_products_required": policy[
                    "minimum_unique_products_per_class"
                ],
            }
        )
    review_counts = Counter(item["priority"] for item in queue)
    quality_counts = Counter(
        flag for record in originals for flag in record["quality_flags"]
    )
    provenance_complete = sum(
        record["provenance_status"] == "complete" for record in originals
    )
    decision_result = None
    if decisions:
        decision_result = validate_semantic_decisions(queue, decisions)
    review_complete = bool(decision_result) and decision_result["decision_count"] == len(
        originals
    )
    unique_product_gate = all(
        item["unique_products"] is not None
        and item["unique_products"] >= item["minimum_unique_products_required"]
        for item in class_inventory
    )
    blocked_reasons = []
    if r0.get("status") != "ready_for_manual_authorization":
        blocked_reasons.append("R0_AUTHORITY_NOT_READY")
    if not r1.get("gate_passed"):
        blocked_reasons.append("R1_CONTRACT_REVIEW_NOT_PASSED")
    if not Path(historical_candidates_path).is_file():
        blocked_reasons.append("HISTORICAL_729_CANDIDATES_NOT_RESTORED")
    if provenance_complete != len(originals):
        blocked_reasons.append("PRODUCT_PROVENANCE_INCOMPLETE")
    if not review_complete:
        blocked_reasons.append("FULL_SEMANTIC_REVIEW_INCOMPLETE")
    if decision_result and decision_result["unresolved_policy_case_ids"]:
        blocked_reasons.append("UNRESOLVED_POLICY_CASES")
    if not unique_product_gate:
        blocked_reasons.append("UNIQUE_PRODUCT_MINIMUM_NOT_PROVEN")
    dataset_record = directory_digest(dataset_root)
    return {
        "schema_version": 1,
        "phase": "R2",
        "policy_version": policy["policy_version"],
        "status": "passed" if not blocked_reasons else "blocked",
        "gate_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "dataset": dataset_record,
        "files": len(records),
        "original_files": len(originals),
        "generated_files_excluded": sum(record["is_generated"] for record in records),
        "invalid_images": sum(record["image_error"] is not None for record in records),
        "provenance_complete_files": provenance_complete,
        "provenance_coverage": provenance_complete / len(originals),
        "historical_review_candidates_restored": Path(
            historical_candidates_path
        ).is_file(),
        "semantic_review_queue_count": len(queue),
        "semantic_review_priority_counts": dict(sorted(review_counts.items())),
        "semantic_decision_result": decision_result,
        "quality_flag_counts": dict(sorted(quality_counts.items())),
        "cross_label_visual_pair_count": len(visual_pairs),
        "cross_label_exact_pixel_pair_count": sum(
            pair["decoded_pixel_exact"] for pair in visual_pairs
        ),
        "class_inventory": class_inventory,
        "minimum_unique_products_per_class": policy[
            "minimum_unique_products_per_class"
        ],
        "source_images_mutated": False,
        "automatic_deletion_performed": False,
        "test_loader_constructed": False,
        "test_evaluated": False,
    }


def write_json(value: Any, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    root = get_practice_2_2_root().resolve()
    if root not in output_path.parents:
        raise RuntimeError("R2 output must remain inside Practice 2.2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return output_path
