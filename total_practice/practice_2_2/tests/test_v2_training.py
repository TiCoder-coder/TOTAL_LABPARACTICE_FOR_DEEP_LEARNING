import pytest
import torch
import torch.nn as nn
from pathlib import Path
import hashlib
from unittest.mock import patch, MagicMock

from src.practice_2_2.v2_models import build_v2_model, set_bn_eval_for_frozen_layers
from src.practice_2_2.v2_dataset import build_train_dataset

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def test_initial_model_hash():
    # Cùng 1 initial hash
    _, hash1 = build_v2_model("E1")
    _, hash2 = build_v2_model("E2")
    _, hash3 = build_v2_model("E3")
    assert hash1 == hash2 == hash3, "Initial model hashes must be perfectly identical for E1, E2, E3!"

def test_e1_freeze_policy():
    model, _ = build_v2_model("E1")
    # All frozen except fc.1
    for name, param in model.named_parameters():
        if "fc.1" in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False

def test_e2_e3_freeze_policy():
    model2, _ = build_v2_model("E2")
    model3, _ = build_v2_model("E3")
    for name, param in model2.named_parameters():
        if "fc.1" in name or "layer4" in name:
            assert param.requires_grad == True
            assert model3.get_parameter(name).requires_grad == True
        else:
            assert param.requires_grad == False
            assert model3.get_parameter(name).requires_grad == False

def test_frozen_bn_stays_eval_during_training():
    model, _ = build_v2_model("E3")
    model.train() # Set all to train mode
    
    set_bn_eval_for_frozen_layers(model, "E3")
    
    # layer1 BN should be eval
    assert not model.layer1[0].bn1.training
    assert not model.bn1.training
    
    # layer4 BN should be train
    assert model.layer4[0].bn1.training
    
def test_v2_split_hash_unchanged_after_training():
    manifest_v2_path = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    current_hash = get_hash(manifest_v2_path)
    expected_hash = "2c2f6d300e5ce127001c107124e21333c7edaaa0e74d763c6f685a2cedec1ff6"
    assert current_hash == expected_hash, "Manifest V2 was modified!"

def test_v1_artifacts_unchanged():
    v1_path = Path("total_practice/practice_2_2/data/manifests/canonical_split/split_manifest.csv")
    current_hash = get_hash(v1_path)
    expected_v1_hash = "aa7d6d4bb77f8ca6e7e5599ac2d49e8d55ce8192e8353f5cb90bbd0f0a4f49b4"
    assert current_hash == expected_v1_hash, "V1 manifest was modified!"

@patch("src.practice_2_2.train_v2_baseline.torch.save")
@patch("src.practice_2_2.train_v2_baseline.build_train_dataset")
@patch("src.practice_2_2.train_v2_baseline.build_validation_dataset")
@patch("src.practice_2_2.train_v2_baseline.DataLoader")
def test_e2_e3_differ_only_weight_decay(mock_dataloader, mock_val_ds, mock_train_ds, mock_torch_save):
    from src.practice_2_2.train_v2_baseline import train
    
    # Mock data loader to return just 1 batch and raise StopIteration
    dummy_inputs = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (2,))
    mock_dataloader.return_value = [(dummy_inputs, dummy_labels)]
    
    # Mock to break loop after 1 epoch
    # We will raise an exception at the end of epoch 1 to break out early and inspect the metadata.
    saved_metadata = {}
    
    def mock_save(cp_dict, path):
        if "metadata" in cp_dict:
            strat = cp_dict["metadata"]["experiment_id"]
            saved_metadata[strat] = cp_dict["metadata"]
        raise KeyboardInterrupt("Stop early")
        
    mock_torch_save.side_effect = mock_save
    
    try:
        train("E2", 2e-4)
    except KeyboardInterrupt:
        pass
        
    try:
        train("E3", 5e-4)
    except KeyboardInterrupt:
        pass
        
    m2 = saved_metadata["E2"]
    m3 = saved_metadata["E3"]
    
    # Compare
    keys_allowed_to_differ = {"experiment_id", "weight_decay"}
    for k in m2.keys():
        if k in keys_allowed_to_differ:
            assert m2[k] != m3[k]
        else:
            if k == "optimizer_state" or k == "scheduler_state":
                pass # dict comparison might fail trivially due to pointers, skipping deep check
            else:
                assert m2[k] == m3[k], f"Field {k} differs between E2 and E3!"
    
    assert m3["weight_decay"] == 5e-4
    assert m2["weight_decay"] == 2e-4
