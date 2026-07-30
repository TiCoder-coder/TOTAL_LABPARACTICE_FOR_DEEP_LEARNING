"""Test data pipeline."""

import pytest
from processing_own_phase.data import load_datasets, make_dataloaders, validate_dataset
from configs import DATA_DIR, CONFIG


@pytest.fixture(scope="module")
def dataset_splits():
    """Load the deterministic default split once for the data tests."""
    return load_datasets(str(DATA_DIR), seed=CONFIG["seed"])


def test_expected_split_sizes(dataset_splits):
    train_subset, val_subset, test_dataset = dataset_splits

    assert len(train_subset) == 45_000
    assert len(val_subset) == 5_000
    assert len(test_dataset) == 10_000


def test_train_validation_indices_are_disjoint(dataset_splits):
    train_subset, val_subset, _ = dataset_splits

    assert set(train_subset.indices).isdisjoint(val_subset.indices)


def test_train_validation_use_separate_dataset_views(dataset_splits):
    train_subset, val_subset, _ = dataset_splits

    assert train_subset.dataset is not val_subset.dataset


def test_train_transform_contains_random_augmentation(dataset_splits):
    train_subset, _, _ = dataset_splits
    train_transform = repr(train_subset.dataset.transform)

    assert "RandomCrop" in train_transform
    assert "RandomHorizontalFlip" in train_transform
    assert "ColorJitter" in train_transform


def test_validation_transform_has_no_random_augmentation(dataset_splits):
    _, val_subset, _ = dataset_splits
    val_transform = repr(val_subset.dataset.transform)

    assert "CenterCrop" in val_transform
    assert "RandomCrop" not in val_transform
    assert "RandomHorizontalFlip" not in val_transform
    assert "ColorJitter" not in val_transform


def test_validation_and_test_transforms_match(dataset_splits):
    _, val_subset, test_dataset = dataset_splits

    assert repr(val_subset.dataset.transform) == repr(test_dataset.transform)


def test_make_dataloaders_settings(dataset_splits):
    """Test that dataloaders respect the requested batch size and settings."""
    train_subset, val_subset, test_dataset = dataset_splits
    train_loader, val_loader, test_loader = make_dataloaders(
        train_subset, val_subset, test_dataset, batch_size=32, num_workers=0
    )
    
    # Batch size
    assert train_loader.batch_size == 32
    assert val_loader.batch_size == 32
    
    # Drop last for train, not for val
    assert train_loader.drop_last is True
    assert val_loader.drop_last is False
    
    # Check shape
    images, labels = next(iter(train_loader))
    assert images.shape == (32, 3, CONFIG["image_size"], CONFIG["image_size"])


def test_validate_dataset(dataset_splits):
    """Test the validation function."""
    _, val_subset, _ = dataset_splits
    # Should pass normally
    assert validate_dataset(val_subset, expected_classes=10) is True
    
    # Test empty
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset([])
