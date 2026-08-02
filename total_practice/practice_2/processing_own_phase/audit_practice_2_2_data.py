"""Reproducible, read-only data audit for the local Practice 2.2 dataset."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


AUGMENTED_PATTERN = re.compile(r"_aug\d+$", flags=re.IGNORECASE)


def is_generated_augmentation(path):
    return AUGMENTED_PATTERN.search(Path(path).stem) is not None


def source_stem(path):
    return AUGMENTED_PATTERN.sub("", Path(path).stem)


def inventory_dataset(dataset_root):
    root = Path(dataset_root)
    rows = []
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    for label, class_name in enumerate(classes):
        for path in sorted((root / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append(
                    {
                        "path": str(path),
                        "class_name": class_name,
                        "label": label,
                        "group": f"{label:02d}/{source_stem(path)}",
                        "is_augmented_file": is_generated_augmentation(path),
                    }
                )
    return pd.DataFrame(rows)


def assign_group_splits(records, seed=42, val_ratio=0.15, test_ratio=0.15):
    result = records.copy()
    split_by_group = {}
    rng = np.random.default_rng(seed)
    for label in sorted(result["label"].unique()):
        groups = np.array(
            sorted(result.loc[result["label"] == label, "group"].unique()),
            dtype=object,
        )
        rng.shuffle(groups)
        n_validation = round(len(groups) * val_ratio)
        n_test = round(len(groups) * test_ratio)
        n_train = len(groups) - n_validation - n_test
        split_by_group.update({group: "Train" for group in groups[:n_train]})
        split_by_group.update(
            {
                group: "Validation"
                for group in groups[n_train:n_train + n_validation]
            }
        )
        split_by_group.update(
            {group: "Test" for group in groups[n_train + n_validation:]}
        )
    result["split"] = result["group"].map(split_by_group)
    result["used_current_policy"] = (
        (result["split"] == "Train") | ~result["is_augmented_file"]
    )
    result["used_original_only_policy"] = ~result["is_augmented_file"]
    return result


def decoded_pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = rgb.tobytes() + str(rgb.size).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def difference_hash(path, hash_size=8):
    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _duplicate_summary(records):
    hash_groups = defaultdict(list)
    for row in records.itertuples():
        hash_groups[row.pixel_hash].append(row)

    duplicate_groups = [rows for rows in hash_groups.values() if len(rows) > 1]
    cross_split = []
    cross_class = []
    for rows in duplicate_groups:
        splits = sorted({row.split for row in rows})
        classes = sorted({row.class_name for row in rows})
        payload = {
            "splits": splits,
            "classes": classes,
            "paths": [row.path for row in rows],
        }
        if len(splits) > 1:
            cross_split.append(payload)
        if len(classes) > 1:
            cross_class.append(payload)
    return {
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_files": sum(len(rows) for rows in duplicate_groups),
        "cross_split_exact_duplicate_groups": len(cross_split),
        "cross_class_exact_duplicate_groups": len(cross_class),
        "cross_split_examples": cross_split[:20],
        "cross_class_examples": cross_class[:20],
    }


def _offline_augmentation_similarity(records):
    by_group = defaultdict(list)
    for row in records.itertuples():
        by_group[row.group].append(row)
    distances = []
    missing_sources = 0
    for rows in by_group.values():
        originals = [row for row in rows if not row.is_augmented_file]
        generated = [row for row in rows if row.is_augmented_file]
        if generated and not originals:
            missing_sources += len(generated)
            continue
        if not generated:
            continue
        original_hash = originals[0].dhash
        distances.extend((original_hash ^ row.dhash).bit_count() for row in generated)
    if not distances:
        return {
            "pairs": 0,
            "missing_source_files": missing_sources,
        }
    array = np.asarray(distances, dtype=np.float64)
    return {
        "pairs": len(distances),
        "missing_source_files": missing_sources,
        "dhash_distance_mean": float(array.mean()),
        "dhash_distance_median": float(np.median(array)),
        "dhash_distance_min": int(array.min()),
        "dhash_distance_max": int(array.max()),
        "pairs_distance_le_5": int((array <= 5).sum()),
        "pairs_distance_le_10": int((array <= 10).sum()),
    }


def _pixel_mae(left_path, right_path):
    with Image.open(left_path) as left_image:
        left = np.asarray(left_image.convert("RGB"), dtype=np.int16)
    with Image.open(right_path) as right_image:
        right = np.asarray(right_image.convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        return float("inf")
    return float(np.abs(left - right).mean())


def _perceptual_cross_boundary_summary(
    records, max_distance=5, near_identical_pixel_mae=5.0
):
    """Screen for visually similar files crossing split or class boundaries.

    Offline variants from the same source group are intentionally kept together
    and skipped. dHash candidates require visual review and are never deleted
    automatically by this audit.
    """
    rows = list(records.itertuples())
    cross_split = []
    cross_class = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left.group == right.group:
                continue
            split_crosses = left.split != right.split
            class_crosses = left.class_name != right.class_name
            if not split_crosses and not class_crosses:
                continue
            distance = (int(left.dhash) ^ int(right.dhash)).bit_count()
            if distance > max_distance:
                continue
            pixel_mae = _pixel_mae(left.path, right.path)
            payload = {
                "distance": distance,
                "pixel_mae": pixel_mae,
                "near_identical": pixel_mae <= near_identical_pixel_mae,
                "left_split": left.split,
                "right_split": right.split,
                "left_class": left.class_name,
                "right_class": right.class_name,
                "left_path": left.path,
                "right_path": right.path,
            }
            if split_crosses:
                cross_split.append(payload)
            if class_crosses:
                cross_class.append(payload)
    sort_key = lambda item: (not item["near_identical"], item["pixel_mae"], item["distance"])
    cross_split.sort(key=sort_key)
    cross_class.sort(key=sort_key)
    review_thresholds = (1.0, 2.0, 3.0, near_identical_pixel_mae)
    return {
        "hash": "dHash-64",
        "max_hamming_distance": max_distance,
        "near_identical_pixel_mae_threshold": near_identical_pixel_mae,
        "cross_split_candidate_pairs": len(cross_split),
        "cross_class_candidate_pairs": len(cross_class),
        "cross_split_near_identical_pairs": sum(
            item["near_identical"] for item in cross_split
        ),
        "cross_class_near_identical_pairs": sum(
            item["near_identical"] for item in cross_class
        ),
        "cross_split_pairs_by_pixel_mae": {
            str(threshold): sum(
                item["pixel_mae"] <= threshold for item in cross_split
            )
            for threshold in review_thresholds
        },
        "cross_class_pairs_by_pixel_mae": {
            str(threshold): sum(
                item["pixel_mae"] <= threshold for item in cross_class
            )
            for threshold in review_thresholds
        },
        "cross_split_examples": cross_split[:20],
        "cross_class_examples": cross_class[:20],
        "interpretation": (
            "dHash candidates require visual review. pixel_mae <= threshold is a "
            "stronger near-identical signal, but remains an audit flag rather than "
            "an automatic deletion decision."
        ),
    }


def run_audit(dataset_root):
    records = assign_group_splits(inventory_dataset(dataset_root))
    sizes = defaultdict(int)
    modes = defaultdict(int)
    corrupt = []
    pixel_hashes = []
    difference_hashes = []
    for path in records["path"]:
        try:
            with Image.open(path) as image:
                image.load()
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
            pixel_hashes.append(decoded_pixel_hash(path))
            difference_hashes.append(difference_hash(path))
        except Exception as error:
            corrupt.append({"path": path, "error": repr(error)})
            pixel_hashes.append(f"CORRUPT:{path}")
            difference_hashes.append(0)
    records["pixel_hash"] = pixel_hashes
    records["dhash"] = difference_hashes

    class_inventory = (
        records.groupby("class_name")
        .agg(
            files=("path", "size"),
            source_groups=("group", "nunique"),
            original_files=("is_augmented_file", lambda values: int((~values).sum())),
            generated_files=("is_augmented_file", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    split_inventory = []
    for split in ("Train", "Validation", "Test"):
        frame = records.loc[records["split"] == split]
        current = frame.loc[frame["used_current_policy"]]
        originals = frame.loc[frame["used_original_only_policy"]]
        split_inventory.append(
            {
                "split": split,
                "source_groups": int(frame["group"].nunique()),
                "current_policy_files": len(current),
                "current_policy_generated_files": int(
                    current["is_augmented_file"].sum()
                ),
                "original_only_files": len(originals),
            }
        )

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "files": len(records),
        "classes": int(records["class_name"].nunique()),
        "source_groups": int(records["group"].nunique()),
        "generated_files": int(records["is_augmented_file"].sum()),
        "class_inventory": class_inventory,
        "split_inventory": split_inventory,
        "image_sizes": dict(sorted(sizes.items(), key=lambda item: -item[1])),
        "image_modes": dict(modes),
        "corrupt_files": corrupt,
        "duplicates": _duplicate_summary(records),
        "perceptual_near_duplicates": _perceptual_cross_boundary_summary(records),
        "offline_augmentation_similarity": _offline_augmentation_similarity(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.dataset_root)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
