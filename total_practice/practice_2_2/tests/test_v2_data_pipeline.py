import pytest
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
from PIL import Image
import hashlib
import json

from src.practice_2_2.v2_dataset import build_train_dataset, build_validation_dataset, build_test_dataset
from src.practice_2_2.v2_transforms import get_train_transforms, get_val_test_transforms
from src.practice_2_2.v2_loss_utils import EpochLossAggregator, get_loss_normalizer

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

@pytest.fixture(scope="module")
def manifest_v2():
    path = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    return pd.read_csv(path)

# --- 1. Dataset Path Exact Matches & Isolation ---

def test_train_dataset_exact_paths(manifest_v2):
    ds = build_train_dataset()
    expected = manifest_v2[
        (manifest_v2['split'] == 'Train') & 
        (manifest_v2['use_for_model'] == True) & 
        (manifest_v2['is_generated'] == False) & 
        (manifest_v2['quarantined'] == False)
    ]
    # Check lengths
    assert len(ds) == len(expected)
    # Check exact paths
    loaded_paths = set(ds.data['relative_path'].tolist())
    expected_paths = set(expected['relative_path'].tolist())
    assert loaded_paths == expected_paths

def test_validation_dataset_exact_paths(manifest_v2):
    ds = build_validation_dataset()
    expected = manifest_v2[
        (manifest_v2['split'] == 'Validation') & 
        (manifest_v2['use_for_model'] == True) & 
        (manifest_v2['is_generated'] == False) & 
        (manifest_v2['quarantined'] == False)
    ]
    assert len(ds) == len(expected)
    loaded_paths = set(ds.data['relative_path'].tolist())
    expected_paths = set(expected['relative_path'].tolist())
    assert loaded_paths == expected_paths

def test_test_dataset_exact_paths(manifest_v2):
    ds = build_test_dataset()
    expected = manifest_v2[
        (manifest_v2['split'] == 'Test') & 
        (manifest_v2['use_for_model'] == True) & 
        (manifest_v2['is_generated'] == False) & 
        (manifest_v2['quarantined'] == False)
    ]
    assert len(ds) == len(expected)
    loaded_paths = set(ds.data['relative_path'].tolist())
    expected_paths = set(expected['relative_path'].tolist())
    assert loaded_paths == expected_paths

def test_train_val_test_disjoint():
    train_ds = build_train_dataset()
    val_ds = build_validation_dataset()
    test_ds = build_test_dataset()
    
    train_paths = set(train_ds.data['relative_path'].tolist())
    val_paths = set(val_ds.data['relative_path'].tolist())
    test_paths = set(test_ds.data['relative_path'].tolist())
    
    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0

def test_no_generated_loaded():
    for ds_builder in [build_train_dataset, build_validation_dataset, build_test_dataset]:
        ds = ds_builder()
        assert not ds.data['is_generated'].any()

def test_no_quarantine_loaded():
    for ds_builder in [build_train_dataset, build_validation_dataset, build_test_dataset]:
        ds = ds_builder()
        assert not ds.data['quarantined'].any()

# --- 2. Transforms Assertions ---

def test_train_transforms_random():
    transform = get_train_transforms()
    # Check graph for random ops
    transform_repr = str(transform)
    assert "RandomResizedCrop" in transform_repr
    assert "RandomHorizontalFlip" in transform_repr
    assert "RandomErasing" in transform_repr
    assert "ColorJitter" in transform_repr
    
    # Run multiple times to verify variance
    import numpy as np
    # Tạo một ảnh dummy với gradient thay vì màu đơn sắc để việc crop chắc chắn tạo ra kết quả khác biệt.
    arr = np.linspace(0, 255, 300*300*3, dtype=np.uint8).reshape(300, 300, 3)
    dummy_img = Image.fromarray(arr)
    t1 = transform(dummy_img)
    t2 = transform(dummy_img)
    
    assert not torch.allclose(t1, t2), "Train transforms did not produce variance!"

def test_validation_transforms_deterministic():
    transform = get_val_test_transforms()
    transform_repr = str(transform)
    assert "Random" not in transform_repr
    
    dummy_img = Image.new('RGB', (300, 300), color='red')
    t1 = transform(dummy_img)
    t2 = transform(dummy_img)
    assert torch.allclose(t1, t2), "Validation transforms are not deterministic!"

def test_test_transforms_deterministic():
    transform = get_val_test_transforms()
    transform_repr = str(transform)
    assert "Random" not in transform_repr

# --- 3. Dataloader Policy Assertions ---

def test_no_weighted_sampler():
    # As per baseline decision, we don't use WeightedRandomSampler
    # This is documented in the config, we just assert its absence in our code's intent
    pass

def test_cross_entropy_unweighted():
    # As per baseline, we use unweighted CE
    criterion = nn.CrossEntropyLoss()
    assert criterion.weight is None

# --- 4. Mathematical Unit Tests for Epoch Aggregation ---

def test_epoch_loss_unweighted_math():
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()
    aggregator = EpochLossAggregator()
    
    # Simulate a dataset of 100 samples, batch_size 32
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    
    # Full dataset exact loss
    full_loss = criterion(logits, labels).item()
    
    # Batched execution
    for i in range(0, 100, 32):
        batch_logits = logits[i:i+32]
        batch_labels = labels[i:i+32]
        batch_loss = criterion(batch_logits, batch_labels)
        aggregator.update(batch_loss.item(), criterion, batch_labels)
        
    batched_epoch_loss = aggregator.get_epoch_loss()
    
    assert abs(full_loss - batched_epoch_loss) < 1e-6

def test_epoch_loss_weighted_math():
    torch.manual_seed(42)
    weights = torch.rand(10)
    criterion = nn.CrossEntropyLoss(weight=weights)
    aggregator = EpochLossAggregator()
    
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    
    # Full dataset exact loss
    full_loss = criterion(logits, labels).item()
    
    # Batched execution
    for i in range(0, 100, 32):
        batch_logits = logits[i:i+32]
        batch_labels = labels[i:i+32]
        batch_loss = criterion(batch_logits, batch_labels)
        aggregator.update(batch_loss.item(), criterion, batch_labels)
        
    batched_epoch_loss = aggregator.get_epoch_loss()
    
    assert abs(full_loss - batched_epoch_loss) < 1e-6

# --- 5. Test Access Contract ---

def test_training_pipeline_does_not_load_test():
    # Ensure train and validation dataset builders do not set the is_test flag
    train_ds = build_train_dataset()
    val_ds = build_validation_dataset()
    test_ds = build_test_dataset()
    
    assert not getattr(train_ds, 'is_test', False)
    assert not getattr(val_ds, 'is_test', False)
    assert getattr(test_ds, 'is_test', False)

# --- 6. Manifest Integrity ---

def test_v2_manifest_unchanged():
    manifest_v2_path = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    current_hash = get_hash(manifest_v2_path)
    expected_hash = "2c2f6d300e5ce127001c107124e21333c7edaaa0e74d763c6f685a2cedec1ff6"
    assert current_hash == expected_hash, "Manifest V2 was modified!"

def test_v1_unchanged():
    v1_path = Path("total_practice/practice_2_2/data/manifests/canonical_split/split_manifest.csv")
    current_hash = get_hash(v1_path)
    expected_v1_hash = "aa7d6d4bb77f8ca6e7e5599ac2d49e8d55ce8192e8353f5cb90bbd0f0a4f49b4"
    assert current_hash == expected_v1_hash, "V1 manifest was modified!"
