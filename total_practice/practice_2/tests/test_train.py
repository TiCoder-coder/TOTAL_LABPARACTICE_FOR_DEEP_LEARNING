"""Test training pipeline."""

import os
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from processing_own_phase.train import (
    train_one_epoch, 
    evaluate, 
    CheckpointManager, 
    get_optimizer, 
    get_scheduler
)


def test_train_one_epoch():
    """Test train_one_epoch runs and updates weights."""
    model = nn.Linear(10, 2)
    device = torch.device("cpu")
    
    # Dummy data
    X = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Save initial weights to check if they change
    initial_weight = model.weight.clone()
    
    # Run one epoch
    loss, acc = train_one_epoch(
        model, loader, criterion, optimizer, device, grad_clip=1.0, scaler=None
    )
    
    assert loss > 0
    assert 0 <= acc <= 100
    assert not torch.allclose(model.weight, initial_weight), "Weights did not update!"


def test_checkpoint_manager():
    """Test saving and loading checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
        config = {"batch_size": 32, "epochs": 5}
        
        # Save
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            best_metric=0.5,
            config=config,
            is_best=True
        )
        
        assert os.path.exists(manager.latest_path)
        assert os.path.exists(manager.best_path)
        
        # Load into new objects
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        
        loaded_epoch, loaded_metric = manager.load_checkpoint(
            manager.latest_path, new_model, new_optimizer, new_scheduler
        )
        
        assert loaded_epoch == 3
        assert loaded_metric == 0.5
        
        # Weights should match
        assert torch.allclose(model.weight, new_model.weight)


def test_get_optimizer_and_scheduler():
    """Test the factory functions."""
    model = nn.Linear(10, 2)
    
    opt_adamw = get_optimizer(model, {"optimizer": "AdamW", "learning_rate": 0.1})
    assert isinstance(opt_adamw, torch.optim.AdamW)
    assert opt_adamw.param_groups[0]["lr"] == 0.1
    
    opt_sgd = get_optimizer(model, {"optimizer": "SGD", "learning_rate": 0.01})
    assert isinstance(opt_sgd, torch.optim.SGD)
    
    sched_cosine = get_scheduler(opt_sgd, {"scheduler": "CosineAnnealingLR", "epochs": 20})
    assert isinstance(sched_cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
