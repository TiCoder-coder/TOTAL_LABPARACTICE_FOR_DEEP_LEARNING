import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

from src.practice_2_2.v2_dataset import build_train_dataset, build_validation_dataset
from src.practice_2_2.v2_transforms import get_train_transforms, get_val_test_transforms
from src.practice_2_2.v2_loss_utils import EpochLossAggregator
from src.practice_2_2.v2_models import build_v2_model, set_bn_eval_for_frozen_layers, get_or_create_initial_state_dict

def train(strategy: str, weight_decay: float):
    # Set seed
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Paths
    config_path = "total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/v2_baseline_config.json"
    save_dir = Path(f"total_practice/practice_2_2/artifacts/experiments/v2_baseline_s42_e1_e2_v1/{strategy}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Dataset & DataLoader (Train & Validation ONLY)
    train_ds = build_train_dataset(transforms=get_train_transforms())
    val_ds = build_validation_dataset(transforms=get_val_test_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # Model
    initial_sd_path = get_or_create_initial_state_dict(save_dir.parent)
    model, initial_hash = build_v2_model(strategy, state_dict_path=initial_sd_path)
    
    # STRICT HASH CHECK
    EXPECTED_HASH = "4e7500733de1a2f3262489d172646dcc5c730ec8ffe0e82f00a9533a1b9368fd"
    if initial_hash != EXPECTED_HASH:
        raise ValueError(f"INITIAL HASH MISMATCH! Expected {EXPECTED_HASH}, got {initial_hash}")
        
    model = model.to(device)
    
    # Setup Optimizer
    if strategy == "E1":
        # Only head is trainable
        optimizer = optim.AdamW(model.fc[1].parameters(), lr=1e-3, weight_decay=weight_decay)
    elif strategy in ["E2", "E3"]:
        # Head and Layer4
        optimizer = optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc[1].parameters(), 'lr': 1e-3}
        ], weight_decay=weight_decay)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_patience = 3
    early_stop_counter = 0
    
    log_path = save_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_Macro_F1,Head_LR,Backbone_LR\n")
        
    for epoch in range(1, 16):
        # ----------------- TRAIN -----------------
        model.train()
        set_bn_eval_for_frozen_layers(model, strategy)
        
        train_agg = EpochLossAggregator()
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_agg.update(loss.item(), criterion, labels)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = train_agg.get_epoch_loss()
        train_acc = correct_train / total_train
        
        # ----------------- VALIDATION -----------------
        model.eval()
        val_agg = EpochLossAggregator()
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_agg.update(loss.item(), criterion, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_agg.get_epoch_loss()
        val_acc = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # ----------------- SCHEDULER & LOGGING -----------------
        scheduler.step(val_loss)
        
        head_lr = optimizer.param_groups[-1]['lr']
        backbone_lr = optimizer.param_groups[0]['lr'] if strategy in ["E2", "E3"] else 0.0
        
        print(f"Epoch {epoch:02d} | TL: {train_loss:.4f} TA: {train_acc:.4f} | VL: {val_loss:.4f} VA: {val_acc:.4f} F1: {val_f1:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_acc},{val_loss},{val_acc},{val_f1},{head_lr},{backbone_lr}\n")
            
        # ----------------- CHECKPOINTING -----------------
        metadata = {
            "run_id": "v2_baseline_s42_e1_e2_v1",
            "experiment_id": strategy,
            "epoch": epoch,
            "weight_decay": weight_decay,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "seed": 42,
            "dataset_fingerprint": config["dataset_fingerprint"],
            "split_fingerprint": config["split_fingerprint"],
            "manifest_hash": config["manifest_sha256"],
            "config_hash": "N/A", # Will be constant
            "best_validation_accuracy": best_val_acc,
            "best_validation_loss": best_val_loss,
            "initial_model_hash": initial_hash,
            "test_evaluated": False,
        }
        
        latest_cp = {"model_state": model.state_dict(), "metadata": metadata}
        torch.save(latest_cp, save_dir / "latest.pt")
        
        is_best = False
        if val_acc > best_val_acc:
            is_best = True
        elif val_acc == best_val_acc and val_loss < best_val_loss:
            is_best = True
            
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            metadata["best_validation_accuracy"] = best_val_acc
            metadata["best_validation_loss"] = best_val_loss
            best_cp = {"model_state": model.state_dict(), "metadata": metadata}
            torch.save(best_cp, save_dir / "best.pt")
            early_stop_counter = 0
            print(f" -> Best Checkpoint Saved!")
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, choices=["E1", "E2", "E3"])
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    args = parser.parse_args()
    train(args.strategy, args.weight_decay)
