"""Deterministic Train/Validation/Holdout split infrastructure for Practice 3 v2."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

from .config import DATASET_NAME
from .experiment_protocol_v2 import (
    PROTOCOL_VERSION,
    SPLIT_SEED,
    atomic_write_json,
    sha256_payload,
)


EXPECTED_SOURCE_COUNTS = {"train": 8530, "validation": 1066}
EXPECTED_SPLIT_COUNTS = {"train": 7676, "validation": 960, "holdout": 960}
EXPECTED_CLASS_COUNTS = {
    "train": {"0": 3838, "1": 3838},
    "validation": {"0": 480, "1": 480},
    "holdout": {"0": 480, "1": 480},
}
ALGORITHM = "python_random_mt19937_per_label_grouped_exact_v1"


def load_development_pool() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load official Train and Validation only; official Test is never requested."""
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for split_name, expected_count in EXPECTED_SOURCE_COUNTS.items():
        split = load_dataset(DATASET_NAME, split=split_name)
        if len(split) != expected_count:
            raise ValueError(f"{split_name} count {len(split)} != {expected_count}")
        if not {"text", "label"}.issubset(split.column_names):
            raise ValueError(f"{split_name} lacks text/label columns")
        fingerprints[split_name] = str(getattr(split, "_fingerprint", "NOT_AVAILABLE"))
        for index, row in enumerate(split):
            label = int(row["label"])
            text = row["text"]
            if label not in (0, 1) or not isinstance(text, str):
                raise ValueError(f"Invalid development record {split_name}:{index}")
            records.append({
                "source_id": f"{split_name}:{index}",
                "original_split": split_name,
                "original_index": index,
                "label": label,
                "text": text,
            })
    if len(records) != 9596:
        raise ValueError(f"Development pool count {len(records)} != 9596")
    return records, fingerprints


def _content_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["source_id"]):
        line = f"{record['source_id']}\t{record['label']}\t{record['text']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _ids_hash(source_ids: Iterable[str]) -> str:
    canonical = "\n".join(sorted(source_ids)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_records(records: list[dict[str, Any]]) -> None:
    source_ids = [record["source_id"] for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Development source IDs must be unique")
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    for record in records:
        if int(record["label"]) not in (0, 1):
            raise ValueError("Labels must be in {0,1}")
        labels_by_text[record["text"]].add(int(record["label"]))
    conflicts = [text for text, labels in labels_by_text.items() if len(labels) > 1]
    if conflicts:
        raise ValueError("Exact duplicate text has conflicting labels")


def _allocate_label_groups(
    records: list[dict[str, Any]],
    label: int,
    targets: dict[str, int],
    seed: int,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if int(record["label"]) == label:
            groups[record["text"]].append(record["source_id"])
    ordered_groups = [
        sorted(source_ids)
        for _, source_ids in sorted(groups.items(), key=lambda item: item[0])
    ]
    random.Random(seed).shuffle(ordered_groups)
    allocation = {name: [] for name in ("holdout", "validation", "train")}
    remaining = {
        "holdout": targets["holdout"],
        "validation": targets["validation"],
        "train": targets["train"],
    }
    for group in ordered_groups:
        eligible = [name for name in ("holdout", "validation", "train") if len(group) <= remaining[name]]
        if not eligible:
            raise ValueError("Grouped duplicate allocation cannot preserve exact locked counts")
        destination = eligible[0]
        allocation[destination].extend(group)
        remaining[destination] -= len(group)
    if any(remaining.values()):
        raise ValueError(f"Split targets were not filled exactly: {remaining}")
    return allocation


def create_split_manifest(
    records: list[dict[str, Any]],
    upstream_fingerprints: dict[str, str],
    *,
    split_counts: dict[str, int] | None = None,
    class_counts: dict[str, dict[str, int]] | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Create a content-free split manifest while keeping duplicate texts grouped."""
    _validate_records(records)
    split_counts = split_counts or EXPECTED_SPLIT_COUNTS
    class_counts = class_counts or EXPECTED_CLASS_COUNTS
    assignments = {name: [] for name in ("train", "validation", "holdout")}
    for label in (0, 1):
        targets = {name: int(class_counts[name][str(label)]) for name in assignments}
        per_label = _allocate_label_groups(records, label, targets, seed)
        for name in assignments:
            assignments[name].extend(per_label[name])
    assignments = {name: sorted(ids) for name, ids in assignments.items()}

    id_to_record = {record["source_id"]: record for record in records}
    all_sets = {name: set(ids) for name, ids in assignments.items()}
    overlap = {
        "train_validation": len(all_sets["train"] & all_sets["validation"]),
        "train_holdout": len(all_sets["train"] & all_sets["holdout"]),
        "validation_holdout": len(all_sets["validation"] & all_sets["holdout"]),
    }
    observed_counts = {name: len(ids) for name, ids in assignments.items()}
    observed_classes = {
        name: {
            str(label): sum(int(id_to_record[source_id]["label"]) == label for source_id in ids)
            for label in (0, 1)
        }
        for name, ids in assignments.items()
    }
    text_sets = {
        name: {id_to_record[source_id]["text"] for source_id in ids}
        for name, ids in assignments.items()
    }
    text_overlap = {
        "train_validation": len(text_sets["train"] & text_sets["validation"]),
        "train_holdout": len(text_sets["train"] & text_sets["holdout"]),
        "validation_holdout": len(text_sets["validation"] & text_sets["holdout"]),
    }
    duplicate_count = sum(
        count - 1 for count in Counter(record["text"] for record in records).values()
        if count > 1
    )
    checks = {
        "development_count": len(records) == sum(split_counts.values()),
        "split_counts_exact": observed_counts == split_counts,
        "class_counts_exact": observed_classes == class_counts,
        "source_id_overlap_zero": all(value == 0 for value in overlap.values()),
        "exact_text_cross_split_overlap_zero": all(value == 0 for value in text_overlap.values()),
        "all_source_ids_assigned_once": (
            set().union(*all_sets.values()) == set(id_to_record)
            and sum(len(items) for items in all_sets.values()) == len(id_to_record)
        ),
        "official_test_excluded": all(
            not source_id.startswith("test:") for ids in assignments.values() for source_id in ids
        ),
        "holdout_raw_content_sealed": True,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "LOCKED",
        "dataset": DATASET_NAME,
        "source_splits": dict(EXPECTED_SOURCE_COUNTS),
        "excluded_splits": {"test": 1066},
        "split_seed": seed,
        "algorithm": ALGORITHM,
        "upstream_fingerprints": upstream_fingerprints,
        "development_pool_hash": _content_hash(records),
        "development_pool_count": len(records),
        "split_counts": observed_counts,
        "class_counts": observed_classes,
        "source_ids": assignments,
        "train_source_ids_hash": _ids_hash(assignments["train"]),
        "validation_source_ids_hash": _ids_hash(assignments["validation"]),
        "holdout_source_ids_hash": _ids_hash(assignments["holdout"]),
        "within_development_exact_duplicate_count": duplicate_count,
        "source_id_overlap": overlap,
        "exact_text_cross_split_overlap": text_overlap,
        "validation_checks": checks,
        "holdout_content_in_manifest": False,
        "official_test_loaded": False,
    }
    if not all(checks.values()):
        raise ValueError(f"Dataset protocol validation failed: {checks}")
    payload["split_manifest_hash"] = sha256_payload(payload)
    return payload


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    supplied = manifest.get("split_manifest_hash")
    body = {key: value for key, value in manifest.items() if key != "split_manifest_hash"}
    checks = (
        manifest.get("protocol_version") == PROTOCOL_VERSION,
        manifest.get("status") == "LOCKED",
        manifest.get("split_counts") == EXPECTED_SPLIT_COUNTS,
        manifest.get("class_counts") == EXPECTED_CLASS_COUNTS,
        manifest.get("official_test_loaded") is False,
        manifest.get("holdout_content_in_manifest") is False,
        all(manifest.get("validation_checks", {}).values()),
        supplied == sha256_payload(body),
    )
    if not all(checks):
        raise ValueError("Dataset split manifest is invalid")


def save_split_manifest(path: Any, manifest: dict[str, Any]) -> None:
    validate_split_manifest(manifest)
    atomic_write_json(path, manifest)


def _select_source_ids(source_dataset: Dataset, source_ids: list[str], split: str) -> Dataset:
    prefix = f"{split}:"
    selected_ids = [source_id for source_id in source_ids if source_id.startswith(prefix)]
    indices = [int(source_id.removeprefix(prefix)) for source_id in selected_ids]
    subset = source_dataset.select(indices)
    return subset.add_column("source_id", selected_ids)


def materialize_train_validation_only(
    manifest: dict[str, Any],
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    """Materialize v2 Train/Validation IDs only; never index v2 Holdout IDs."""
    validate_split_manifest(manifest)
    source_datasets = {
        split: load_dataset(DATASET_NAME, split=split)
        for split in ("train", "validation")
    }
    materialized: dict[str, Dataset] = {}
    for target in ("train", "validation"):
        source_ids = manifest["source_ids"][target]
        pieces = [
            _select_source_ids(source_datasets[source], source_ids, source)
            for source in ("train", "validation")
            if any(item.startswith(f"{source}:") for item in source_ids)
        ]
        materialized[target] = concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
        if set(materialized[target]["source_id"]) != set(source_ids):
            raise RuntimeError(f"{target} source IDs do not match the locked manifest")
    observed_counts = {name: len(split) for name, split in materialized.items()}
    observed_classes = {
        name: {str(label): list(split["label"]).count(label) for label in (0, 1)}
        for name, split in materialized.items()
    }
    checks = {
        "train_count": observed_counts["train"] == EXPECTED_SPLIT_COUNTS["train"],
        "validation_count": observed_counts["validation"] == EXPECTED_SPLIT_COUNTS["validation"],
        "train_classes": observed_classes["train"] == EXPECTED_CLASS_COUNTS["train"],
        "validation_classes": observed_classes["validation"] == EXPECTED_CLASS_COUNTS["validation"],
        "official_test_excluded": True,
        "holdout_not_indexed": True,
        "holdout_not_materialized": True,
    }
    if not all(value is True for value in checks.values()):
        raise RuntimeError(f"Train/Validation materialization failed: {checks}")
    return materialized, {
        "counts": observed_counts,
        "class_counts": observed_classes,
        "checks": checks,
    }


def materialize_holdout_only(
    manifest: dict[str, Any],
) -> tuple[Dataset, dict[str, Any]]:
    """Materialize v2 Holdout IDs only for final evaluation."""
    validate_split_manifest(manifest)
    source_datasets = {
        split: load_dataset(DATASET_NAME, split=split)
        for split in ("train", "validation")
    }
    target = "holdout"
    source_ids = manifest["source_ids"][target]
    pieces = [
        _select_source_ids(source_datasets[source], source_ids, source)
        for source in ("train", "validation")
        if any(item.startswith(f"{source}:") for item in source_ids)
    ]
    holdout_dataset = concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    if set(holdout_dataset["source_id"]) != set(source_ids):
        raise RuntimeError(f"{target} source IDs do not match the locked manifest")
        
    observed_count = len(holdout_dataset)
    observed_classes = {str(label): list(holdout_dataset["label"]).count(label) for label in (0, 1)}
    
    checks = {
        "holdout_count": observed_count == EXPECTED_SPLIT_COUNTS["holdout"],
        "holdout_classes": observed_classes == EXPECTED_CLASS_COUNTS["holdout"],
        "official_test_excluded": True,
        "train_not_materialized": True,
        "validation_not_materialized": True,
    }
    if not all(value is True for value in checks.values()):
        raise RuntimeError(f"Holdout materialization failed: {checks}")
    return holdout_dataset, {
        "counts": {target: observed_count},
        "class_counts": {target: observed_classes},
        "checks": checks,
    }
