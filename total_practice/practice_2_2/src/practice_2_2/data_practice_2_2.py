"""Duplicate-aware data preparation for the local Practice 2.2 dataset."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .audit_practice_2_2_data import (
    _pixel_mae,
    decoded_pixel_hash,
    difference_hash,
    inventory_dataset,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path, chunk_size=1024 * 1024):
    """Return a content hash for exact-duplicate and dataset verification."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(rows, columns):
    """Hash canonical, path-independent rows using sorted compact JSON."""
    canonical_rows = (
        rows.loc[:, columns]
        .sort_values(columns[0], kind="stable")
        .to_dict(orient="records")
    )
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_artifact_fingerprints(manifest_path, summary_path):
    """Verify persisted dataset and split fingerprints without loading images."""
    manifest = pd.read_csv(manifest_path)
    summary = json.loads(Path(summary_path).read_text())
    dataset_fingerprint = _stable_fingerprint(
        manifest,
        ["relative_path", "class_name", "file_hash"],
    )
    split_fingerprint = _stable_fingerprint(
        manifest,
        [
            "relative_path",
            "class_name",
            "group",
            "duplicate_cluster",
            "split",
            "is_augmented_file",
            "quarantined",
            "file_hash",
        ],
    )
    if dataset_fingerprint != summary["dataset_fingerprint_sha256"]:
        raise RuntimeError("Dataset fingerprint does not match the canonical summary")
    if split_fingerprint != summary["split_fingerprint_sha256"]:
        raise RuntimeError("Split fingerprint does not match the canonical summary")
    return {
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "split_fingerprint_sha256": split_fingerprint,
    }


def get_practice_2_2_transforms(image_size=224, augment_strength="strong"):
    """Return lazy Train and deterministic Validation/Test transforms.

    Split membership is read from the manifest before either transform is
    attached.  TorchVision applies these transforms only in ``__getitem__``.

    ``augment_strength`` controls the magnitude of random augmentation:

    - ``"base"`` keeps the original light recipe.
    - ``"e4_moderate"`` changes only the Train recipe for the controlled E4.
    - ``"strong"`` adds RandAugment + rotation + affine, which is recommended
      for fine-grained classification on small datasets such as
      ``data_clean_balanced`` (~3.2k images).
    """
    if augment_strength == "base":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.10,
                    scale=(0.02, 0.10),
                    ratio=(0.5, 2.0),
                    value="random",
                ),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    elif augment_strength == "e4_moderate":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.05,
                ),
                transforms.RandomRotation(
                    degrees=8,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.20,
                    scale=(0.02, 0.10),
                    ratio=(0.5, 2.0),
                    value="random",
                ),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    elif augment_strength == "strong":
        train_transform = transforms.Compose(
            [
                # Slightly wider crop window than base to avoid learning
                # absolute pose cues.
                transforms.RandomResizedCrop(image_size, scale=(0.60, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.30, 0.30, 0.30, 0.05)],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR)],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1))],
                    p=0.4,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
            ]
        )
    else:
        raise ValueError(f"Unknown augment_strength: {augment_strength!r}")
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_split_manifest(manifest_path):
    """Load and validate the persisted duplicate-aware split manifest."""
    manifest = pd.read_csv(manifest_path)
    required = {
        "relative_path",
        "class_name",
        "label",
        "duplicate_cluster",
        "is_augmented_file",
        "quarantined",
        "split",
        "use_for_model",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    used = manifest.loc[manifest["use_for_model"]]
    if used.groupby("duplicate_cluster")["split"].nunique().max() != 1:
        raise RuntimeError("A duplicate cluster crosses split boundaries")
    if used["is_augmented_file"].any() or used["quarantined"].any():
        raise RuntimeError("Offline-generated or quarantined files entered a split")
    return manifest


def _dataset_subset_from_manifest(dataset_root, split_rows, transform):
    """Build one independent ImageFolder view and select manifest members."""
    root = Path(dataset_root).resolve()
    dataset_view = datasets.ImageFolder(root=str(root), transform=transform)
    path_to_index = {
        str(Path(path).resolve().relative_to(root)): index
        for index, (path, _) in enumerate(dataset_view.samples)
    }
    requested_paths = split_rows["relative_path"].tolist()
    missing = sorted(set(requested_paths).difference(path_to_index))
    if missing:
        raise FileNotFoundError(f"Manifest files are missing from the dataset: {missing[:5]}")
    indices = [path_to_index[path] for path in requested_paths]

    expected_mapping = (
        split_rows[["class_name", "label"]]
        .drop_duplicates()
        .set_index("class_name")["label"]
        .astype(int)
        .to_dict()
    )
    if dataset_view.class_to_idx != expected_mapping:
        raise RuntimeError(
            "ImageFolder class mapping does not match the split manifest: "
            f"{dataset_view.class_to_idx} != {expected_mapping}"
        )
    return Subset(dataset_view, indices)


def create_train_validation_datasets(
    dataset_root,
    manifest_path,
    image_size=224,
    augment_strength="strong",
):
    """Create Train/Validation only; no Test dataset is constructed here."""
    manifest = load_split_manifest(manifest_path)
    used = manifest.loc[manifest["use_for_model"]]
    train_rows = used.loc[used["split"] == "Train"]
    validation_rows = used.loc[used["split"] == "Validation"]
    if train_rows.empty or validation_rows.empty:
        raise RuntimeError("Both Train and Validation must be non-empty")

    train_transform, eval_transform = get_practice_2_2_transforms(
        image_size, augment_strength=augment_strength
    )
    train_dataset = _dataset_subset_from_manifest(
        dataset_root,
        train_rows,
        train_transform,
    )
    validation_dataset = _dataset_subset_from_manifest(
        dataset_root,
        validation_rows,
        eval_transform,
    )
    if train_dataset.dataset is validation_dataset.dataset:
        raise RuntimeError("Train and Validation must use independent dataset views")
    return train_dataset, validation_dataset


def create_test_dataset(
    dataset_root,
    manifest_path,
    image_size=224,
    *,
    allow_test=False,
):
    """Construct Test only after an explicit final-evaluation authorization."""
    if not allow_test:
        raise PermissionError(
            "Test is locked during experiment selection; pass allow_test=True only "
            "for the one-time final evaluation."
        )
    manifest = load_split_manifest(manifest_path)
    test_rows = manifest.loc[
        manifest["use_for_model"] & (manifest["split"] == "Test")
    ]
    _, eval_transform = get_practice_2_2_transforms(image_size)
    return _dataset_subset_from_manifest(dataset_root, test_rows, eval_transform)


def make_train_validation_loaders(
    train_dataset,
    validation_dataset,
    batch_size=32,
    num_workers=0,
    seed=42,
):
    """Create selection-stage loaders; intentionally returns no Test loader."""
    generator = torch.Generator().manual_seed(seed)
    persistent_workers = num_workers > 0
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader


def compute_train_class_weights(train_dataset):
    """Return inverse-frequency class weights fitted from Train labels only."""
    labels = [train_dataset.dataset.targets[index] for index in train_dataset.indices]
    counts = np.bincount(labels, minlength=len(train_dataset.dataset.classes))
    if (counts == 0).any():
        raise RuntimeError("Cannot compute class weights: Train is missing a class")
    weights = len(labels) / (len(counts) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _compute_image_signatures(records):
    result = records.copy().reset_index(drop=True)
    result["record_id"] = np.arange(len(result), dtype=np.int64)
    result["file_hash"] = [file_sha256(path) for path in result["path"]]
    result["pixel_hash"] = [decoded_pixel_hash(path) for path in result["path"]]
    result["dhash"] = [difference_hash(path) for path in result["path"]]
    return result


def _build_duplicate_clusters(
    records,
    max_dhash_distance=5,
    max_pixel_mae=3.0,
):
    result = records.copy().reset_index(drop=True)
    disjoint_set = DisjointSet(len(result))
    ambiguous_pairs = []
    perceptual_pairs = []

    for indices in result.groupby("group").indices.values():
        indices = list(indices)
        for index in indices[1:]:
            disjoint_set.union(indices[0], index)

    exact_pairs = []
    for indices in result.groupby("file_hash").indices.values():
        indices = list(indices)
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1:]:
                pair = (left, right, 0, 0.0, "exact_file")
                exact_pairs.append(pair)
                if result.at[left, "label"] == result.at[right, "label"]:
                    disjoint_set.union(left, right)
                else:
                    ambiguous_pairs.append(pair)

    # Decoded-pixel equality catches byte-different encodings of the same image.
    for indices in result.groupby("pixel_hash").indices.values():
        indices = list(indices)
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1:]:
                if result.at[left, "file_hash"] == result.at[right, "file_hash"]:
                    continue
                pair = (left, right, 0, 0.0, "exact_pixel")
                exact_pairs.append(pair)
                if result.at[left, "label"] == result.at[right, "label"]:
                    disjoint_set.union(left, right)
                else:
                    ambiguous_pairs.append(pair)

    rows = list(result.itertuples())
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left.group == right.group:
                continue
            distance = (int(left.dhash) ^ int(right.dhash)).bit_count()
            if distance > max_dhash_distance:
                continue
            pixel_mae = _pixel_mae(left.path, right.path)
            if pixel_mae > max_pixel_mae:
                continue
            pair = (
                int(left.record_id),
                int(right.record_id),
                distance,
                pixel_mae,
                "perceptual",
            )
            perceptual_pairs.append(pair)
            if left.label == right.label:
                disjoint_set.union(left.record_id, right.record_id)
            else:
                ambiguous_pairs.append(pair)

    components = defaultdict(list)
    for record_id in result["record_id"]:
        components[disjoint_set.find(int(record_id))].append(int(record_id))

    ambiguous_roots = set()
    for left, right, *_ in ambiguous_pairs:
        ambiguous_roots.add(disjoint_set.find(left))
        ambiguous_roots.add(disjoint_set.find(right))

    component_name = {}
    for root, members in components.items():
        label = int(result.at[min(members), "label"])
        component_name[root] = f"{label:02d}/cluster_{min(members):05d}"

    result["duplicate_cluster"] = [
        component_name[disjoint_set.find(int(record_id))]
        for record_id in result["record_id"]
    ]
    result["quarantined"] = [
        disjoint_set.find(int(record_id)) in ambiguous_roots
        for record_id in result["record_id"]
    ]
    perceptual_cluster_roots = {
        disjoint_set.find(left) for left, right, *_ in perceptual_pairs
    } | {
        disjoint_set.find(right) for left, right, *_ in perceptual_pairs
    }
    result["has_perceptual_duplicate"] = [
        disjoint_set.find(int(record_id)) in perceptual_cluster_roots
        for record_id in result["record_id"]
    ]
    return result, exact_pairs, perceptual_pairs, ambiguous_pairs


def _assign_cluster_splits(records, seed=42, val_ratio=0.15, test_ratio=0.15):
    result = records.copy()
    split_by_cluster = {}
    rng = np.random.default_rng(seed)
    eligible = result.loc[~result["quarantined"] & ~result["is_augmented_file"]]

    for label in sorted(eligible["label"].unique()):
        clusters = np.array(
            sorted(
                eligible.loc[eligible["label"] == label, "duplicate_cluster"].unique()
            ),
            dtype=object,
        )
        rng.shuffle(clusters)
        n_validation = round(len(clusters) * val_ratio)
        n_test = round(len(clusters) * test_ratio)
        n_train = len(clusters) - n_validation - n_test
        if min(n_train, n_validation, n_test) <= 0:
            raise RuntimeError(f"Class {label} cannot be split into three non-empty sets")
        split_by_cluster.update(
            {cluster: "Train" for cluster in clusters[:n_train]}
        )
        split_by_cluster.update(
            {
                cluster: "Validation"
                for cluster in clusters[n_train:n_train + n_validation]
            }
        )
        split_by_cluster.update(
            {cluster: "Test" for cluster in clusters[n_train + n_validation:]}
        )

    result["split"] = result["duplicate_cluster"].map(split_by_cluster)
    result.loc[result["quarantined"], "split"] = "Quarantine"
    result["use_for_model"] = (
        ~result["quarantined"] & ~result["is_augmented_file"]
    )
    return result


def validate_manifest(records, perceptual_pairs):
    used = records.loc[records["use_for_model"]]
    if used.empty:
        raise RuntimeError("The split manifest contains no usable images")
    if used["split"].isna().any():
        raise RuntimeError("A usable image has no split assignment")
    if used.groupby("duplicate_cluster")["split"].nunique().max() != 1:
        raise RuntimeError("A duplicate cluster crosses split boundaries")
    if records.loc[records["is_augmented_file"], "use_for_model"].any():
        raise RuntimeError("An offline-generated file is marked for model use")
    if set(used["split"].unique()) != {"Train", "Validation", "Test"}:
        raise RuntimeError("Train, Validation and Test must all be present")
    if used.groupby(["label", "split"]).size().unstack(fill_value=0).min().min() <= 0:
        raise RuntimeError("Every class must be represented in every split")

    model_splits = {"Train", "Validation", "Test"}
    assigned = records.loc[records["split"].isin(model_splits)]
    split_paths = {
        split: set(assigned.loc[assigned["split"] == split, "relative_path"])
        for split in model_splits
    }
    if not split_paths["Train"].isdisjoint(split_paths["Validation"]):
        raise RuntimeError("Train and Validation paths overlap")
    if not split_paths["Train"].isdisjoint(split_paths["Test"]):
        raise RuntimeError("Train and Test paths overlap")
    if not split_paths["Validation"].isdisjoint(split_paths["Test"]):
        raise RuntimeError("Validation and Test paths overlap")
    if assigned.groupby("duplicate_cluster")["split"].nunique().max() != 1:
        raise RuntimeError("A duplicate cluster crosses split boundaries")
    if assigned.groupby("file_hash")["split"].nunique().max() != 1:
        raise RuntimeError("An exact file hash crosses split boundaries")
    if assigned.groupby("group")["split"].nunique().max() != 1:
        raise RuntimeError("An original/generated family crosses split boundaries")
    generated_groups = set(
        records.loc[records["is_augmented_file"], "group"]
    )
    original_groups = set(
        records.loc[~records["is_augmented_file"], "group"]
    )
    missing_sources = sorted(generated_groups.difference(original_groups))
    if missing_sources:
        raise RuntimeError(
            "Generated derivatives are missing their source originals: "
            f"{missing_sources[:5]}"
        )

    indexed = records.set_index("record_id")
    violations = []
    for left, right, distance, pixel_mae, source in perceptual_pairs:
        left_row = indexed.loc[left]
        right_row = indexed.loc[right]
        if left_row["quarantined"] or right_row["quarantined"]:
            continue
        if left_row["split"] != right_row["split"]:
            violations.append(
                {
                    "left": left_row["path"],
                    "right": right_row["path"],
                    "distance": distance,
                    "pixel_mae": pixel_mae,
                    "source": source,
                }
            )
    if violations:
        raise RuntimeError(f"Perceptual duplicate pairs cross splits: {violations[:5]}")
    return True


def build_duplicate_aware_manifest(
    dataset_root,
    seed=42,
    val_ratio=0.15,
    test_ratio=0.15,
    max_dhash_distance=5,
    max_pixel_mae=3.0,
):
    root = Path(dataset_root).resolve()
    records = _compute_image_signatures(inventory_dataset(root))
    records, exact_pairs, perceptual_pairs, ambiguous_pairs = _build_duplicate_clusters(
        records,
        max_dhash_distance=max_dhash_distance,
        max_pixel_mae=max_pixel_mae,
    )
    records = _assign_cluster_splits(
        records,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    records["relative_path"] = [
        str(Path(path).resolve().relative_to(root)) for path in records["path"]
    ]
    records["image_path"] = records["relative_path"]
    records["source_group"] = records["group"]
    records["duplicate_cluster_id"] = records["duplicate_cluster"]
    records["is_generated"] = records["is_augmented_file"]
    validate_manifest(records, perceptual_pairs)

    dataset_fingerprint = _stable_fingerprint(
        records,
        ["relative_path", "class_name", "file_hash"],
    )
    split_fingerprint = _stable_fingerprint(
        records,
        [
            "relative_path",
            "class_name",
            "group",
            "duplicate_cluster",
            "split",
            "is_augmented_file",
            "quarantined",
            "file_hash",
        ],
    )

    exact_group_sizes = records.groupby("file_hash").size()
    exact_duplicate_groups = int((exact_group_sizes > 1).sum())
    exact_duplicate_files = int(exact_group_sizes[exact_group_sizes > 1].sum())
    perceptual_duplicate_clusters = int(
        records.loc[records["has_perceptual_duplicate"], "duplicate_cluster"].nunique()
    )

    split_summary = (
        records.loc[records["use_for_model"]]
        .groupby("split")
        .agg(
            files=("record_id", "size"),
            duplicate_clusters=("duplicate_cluster", "nunique"),
            classes=("label", "nunique"),
        )
        .reindex(["Train", "Validation", "Test"])
        .reset_index()
        .to_dict(orient="records")
    )
    assigned_split_summary = (
        records.loc[records["split"].isin(["Train", "Validation", "Test"])]
        .groupby("split")
        .agg(
            files=("record_id", "size"),
            source_groups=("source_group", "nunique"),
            duplicate_clusters=("duplicate_cluster_id", "nunique"),
            generated_files=("is_generated", "sum"),
            classes=("label", "nunique"),
        )
        .reindex(["Train", "Validation", "Test"])
        .reset_index()
        .to_dict(orient="records")
    )
    class_split_counts = (
        records.loc[records["use_for_model"]]
        .groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Train", "Validation", "Test"], fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "dataset_root": str(root),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "max_dhash_distance": max_dhash_distance,
        "max_pixel_mae": max_pixel_mae,
        "files_discovered": len(records),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_files": exact_duplicate_files,
        "offline_generated_files_excluded": int(
            records["is_augmented_file"].sum()
        ),
        "quarantined_files": int(records["quarantined"].sum()),
        "cross_class_suspicious_images_quarantined": int(
            records.loc[records["quarantined"], "record_id"].nunique()
        ),
        "perceptual_duplicate_clusters": perceptual_duplicate_clusters,
        "perceptual_pairs_merged_or_quarantined": len(perceptual_pairs),
        "cross_class_ambiguous_pairs": len(ambiguous_pairs),
        "usable_original_files": int(records["use_for_model"].sum()),
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "split_fingerprint_sha256": split_fingerprint,
        "fingerprint_schema_version": 1,
        "split_summary": split_summary,
        "assigned_split_summary_including_generated": assigned_split_summary,
        "class_split_counts": class_split_counts,
        "leakage_assertions": {
            "train_validation_test_disjoint": True,
            "duplicate_clusters_do_not_cross_splits": True,
            "generated_families_do_not_cross_splits": True,
            "exact_hashes_do_not_cross_splits": True,
            "test_loader_constructed": False,
        },
        "test_loader_constructed": False,
        "test_evaluated": False,
    }
    return records, summary


def write_split_artifacts(records, summary, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_columns = [
        "record_id",
        "image_path",
        "relative_path",
        "class_name",
        "label",
        "source_group",
        "group",
        "duplicate_cluster_id",
        "duplicate_cluster",
        "is_generated",
        "is_augmented_file",
        "has_perceptual_duplicate",
        "quarantined",
        "split",
        "use_for_model",
        "file_hash",
        "pixel_hash",
        "dhash",
    ]
    records[manifest_columns].to_csv(output / "split_manifest.csv", index=False)
    records.loc[records["quarantined"], manifest_columns].to_csv(
        output / "quarantined_images.csv", index=False
    )
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records, summary = build_duplicate_aware_manifest(
        args.dataset_root,
        seed=args.seed,
    )
    write_split_artifacts(records, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
