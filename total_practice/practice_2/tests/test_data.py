"""Test data pipeline."""

import pytest
import torch
from processing_own_phase.data import load_datasets, make_dataloaders, validate_dataset
from configs import DATA_DIR, CONFIG


def test_load_datasets_no_leakage():
    """Test that train and val sets have different transforms and no index overlap."""
    train_subset, val_subset, test_dataset = load_datasets(str(DATA_DIR), val_ratio=0.1)
    
    assert len(train_subset) + len(val_subset) == 50000
    assert len(test_dataset) == 10000
    
    # Train subset uses augmented transform
    train_tfm_str = str(train_subset.dataset.transform)
    assert "RandomHorizontalFlip" in train_tfm_str
    assert "RandomCrop" in train_tfm_str
    assert "ColorJitter" in train_tfm_str
    
    # Val subset uses clean transform
    val_tfm_str = str(val_subset.dataset.transform)
    assert "RandomHorizontalFlip" not in val_tfm_str
    assert "CenterCrop" in val_tfm_str
    
    # Check no data leakage
    train_indices = set(train_subset.indices)
    val_indices = set(val_subset.indices)
    assert len(train_indices.intersection(val_indices)) == 0


def test_make_dataloaders_settings():
    """Test that dataloaders respect the requested batch size and settings."""
    train_subset, val_subset, test_dataset = load_datasets(str(DATA_DIR), val_ratio=0.1)
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


def test_validate_dataset():
    """Test the validation function."""
    _, val_subset, _ = load_datasets(str(DATA_DIR), val_ratio=0.1)
    # Should pass normally
    assert validate_dataset(val_subset, expected_classes=10) is True
    
    # Test empty
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset([])
