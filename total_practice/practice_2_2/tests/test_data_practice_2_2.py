"""Tests for duplicate-aware Practice 2.2 data preparation."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms

from practice_2_2.data_practice_2_2 import (
    build_duplicate_aware_manifest,
    compute_train_class_weights,
    create_test_dataset,
    create_train_validation_datasets,
    get_practice_2_2_transforms,
    make_train_validation_loaders,
    validate_manifest,
    verify_artifact_fingerprints,
    write_split_artifacts,
)


def _write_jpeg(path, pixels, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path, quality=quality)


def _make_dataset(root):
    rng = np.random.default_rng(123)
    for class_name in ("alpha", "beta"):
        for index in range(12):
            pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            _write_jpeg(root / class_name / f"{class_name}_{index:03d}.jpg", pixels)

    duplicate = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    _write_jpeg(root / "alpha" / "alpha_duplicate_a.jpg", duplicate, quality=95)
    _write_jpeg(root / "alpha" / "alpha_duplicate_b.jpg", duplicate, quality=95)

    ambiguous = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    _write_jpeg(root / "alpha" / "alpha_ambiguous.jpg", ambiguous, quality=95)
    _write_jpeg(root / "beta" / "beta_ambiguous.jpg", ambiguous, quality=95)

    generated = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    _write_jpeg(root / "alpha" / "alpha_source.jpg", generated)
    _write_jpeg(root / "alpha" / "alpha_source_aug001.jpg", generated)


def test_duplicate_aware_manifest_is_deterministic_and_leakage_safe(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    first, first_summary = build_duplicate_aware_manifest(dataset_root, seed=42)
    second, second_summary = build_duplicate_aware_manifest(dataset_root, seed=42)

    first_view = first[["relative_path", "duplicate_cluster", "split", "use_for_model"]]
    second_view = second[["relative_path", "duplicate_cluster", "split", "use_for_model"]]
    assert first_view.equals(second_view)
    assert first_summary == second_summary
    assert first_summary["dataset_fingerprint_sha256"]
    assert first_summary["split_fingerprint_sha256"]
    assert validate_manifest(first, [])

    required_canonical_columns = {
        "image_path",
        "class_name",
        "source_group",
        "duplicate_cluster_id",
        "split",
        "is_generated",
        "file_hash",
    }
    assert required_canonical_columns.issubset(first.columns)

    duplicate_rows = first.loc[
        first["relative_path"].isin(
            ["alpha/alpha_duplicate_a.jpg", "alpha/alpha_duplicate_b.jpg"]
        )
    ]
    assert duplicate_rows["duplicate_cluster"].nunique() == 1
    assert duplicate_rows["split"].nunique() == 1

    ambiguous_rows = first.loc[
        first["relative_path"].isin(
            ["alpha/alpha_ambiguous.jpg", "beta/beta_ambiguous.jpg"]
        )
    ]
    assert ambiguous_rows["quarantined"].all()
    assert set(ambiguous_rows["split"]) == {"Quarantine"}

    generated_row = first.loc[
        first["relative_path"] == "alpha/alpha_source_aug001.jpg"
    ].iloc[0]
    assert bool(generated_row["is_augmented_file"])
    assert not bool(generated_row["use_for_model"])
    source_row = first.loc[
        first["relative_path"] == "alpha/alpha_source.jpg"
    ].iloc[0]
    assert generated_row["source_group"] == source_row["source_group"]
    assert generated_row["duplicate_cluster_id"] == source_row["duplicate_cluster_id"]
    assert generated_row["split"] == source_row["split"]
    assigned = first.loc[first["split"].isin(["Train", "Validation", "Test"])]
    assert assigned.groupby("duplicate_cluster_id")["split"].nunique().max() == 1
    assert assigned.groupby("source_group")["split"].nunique().max() == 1
    assert assigned.groupby("file_hash")["split"].nunique().max() == 1
    assert first_summary["test_loader_constructed"] is False
    assert first_summary["test_evaluated"] is False


def test_every_class_is_present_in_all_model_splits(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    records, _ = build_duplicate_aware_manifest(dataset_root, seed=42)
    used = records.loc[records["use_for_model"]]
    class_split_counts = used.groupby(["class_name", "split"]).size().unstack()
    assert (class_split_counts[["Train", "Validation", "Test"]] > 0).all().all()


def test_train_validation_views_and_transforms_are_isolated(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "split"
    _make_dataset(dataset_root)
    records, summary = build_duplicate_aware_manifest(dataset_root, seed=42)
    write_split_artifacts(records, summary, output_dir)

    train_dataset, validation_dataset = create_train_validation_datasets(
        dataset_root,
        output_dir / "split_manifest.csv",
        image_size=32,
    )
    used = records.loc[records["use_for_model"]]
    assert len(train_dataset) == int((used["split"] == "Train").sum())
    assert len(validation_dataset) == int((used["split"] == "Validation").sum())
    assert train_dataset.dataset is not validation_dataset.dataset

    def transform_types(transform):
        result = {type(transform)}
        for child in getattr(transform, "transforms", []):
            result.update(transform_types(child))
        return result

    train_types = transform_types(train_dataset.dataset.transform)
    validation_types = {
        type(item) for item in validation_dataset.dataset.transform.transforms
    }
    assert transforms.RandomResizedCrop in train_types
    assert transforms.RandomHorizontalFlip in train_types
    assert transforms.ColorJitter in train_types
    assert transforms.RandomErasing in train_types
    assert transforms.CenterCrop in validation_types
    assert not train_types.intersection(
        {transforms.CenterCrop}
    )
    assert not validation_types.intersection(
        {
            transforms.RandomResizedCrop,
            transforms.RandomHorizontalFlip,
            transforms.ColorJitter,
            transforms.RandomErasing,
        }
    )

    train_loader, validation_loader = make_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=4,
        num_workers=0,
    )
    assert train_loader.dataset is train_dataset
    assert validation_loader.dataset is validation_dataset
    weights = compute_train_class_weights(train_dataset)
    assert weights.shape == (2,)
    assert weights.isfinite().all()


def test_eval_transform_is_shared_by_definition_but_dataset_views_are_not():
    _, first_eval = get_practice_2_2_transforms(image_size=32)
    _, second_eval = get_practice_2_2_transforms(image_size=32)
    assert repr(first_eval) == repr(second_eval)
    assert first_eval is not second_eval


def test_test_dataset_is_locked_by_default(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "split"
    _make_dataset(dataset_root)
    records, summary = build_duplicate_aware_manifest(dataset_root, seed=42)
    write_split_artifacts(records, summary, output_dir)
    with pytest.raises(PermissionError, match="Test is locked"):
        create_test_dataset(dataset_root, output_dir / "split_manifest.csv")


def test_persisted_fingerprints_detect_manifest_changes(tmp_path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "split"
    _make_dataset(dataset_root)
    records, summary = build_duplicate_aware_manifest(dataset_root, seed=42)
    write_split_artifacts(records, summary, output_dir)

    verified = verify_artifact_fingerprints(
        output_dir / "split_manifest.csv",
        output_dir / "split_summary.json",
    )
    assert verified["dataset_fingerprint_sha256"] == summary["dataset_fingerprint_sha256"]
    assert verified["split_fingerprint_sha256"] == summary["split_fingerprint_sha256"]

    manifest = pd.read_csv(output_dir / "split_manifest.csv")
    manifest.loc[0, "split"] = "Validation"
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    with pytest.raises(RuntimeError, match="Split fingerprint"):
        verify_artifact_fingerprints(
            output_dir / "split_manifest.csv",
            output_dir / "split_summary.json",
        )
