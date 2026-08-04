import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score

from src.practice_2_2.v2_dataset import build_validation_dataset
from src.practice_2_2.v2_transforms import get_val_test_transforms
from src.practice_2_2.v2_loss_utils import EpochLossAggregator
from src.practice_2_2.v2_models import build_v2_model

def evaluate_reloaded(strategy: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    save_dir = Path(f"total_practice/practice_2_2/artifacts/experiments/v2_baseline_s42_e1_e2_v1/{strategy}")
    cp_path = save_dir / "best.pt"
    
    if not cp_path.exists():
        print(f"Checkpoint not found at {cp_path}")
        return
        
    checkpoint = torch.load(cp_path)
    metadata = checkpoint['metadata']
    
    print(f"Reloading model {strategy} from Epoch {metadata['epoch']}")
    print(f"Metadata recorded best Val Acc: {metadata['best_validation_accuracy']:.4f}, Val Loss: {metadata['best_validation_loss']:.4f}")
    
    # Load val dataset
    val_ds = build_validation_dataset(transforms=get_val_test_transforms())
    # Ensure config batch_size
    import json
    with open("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/v2_baseline_config.json") as f:
        config = json.load(f)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    model, _ = build_v2_model(strategy)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
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
    
    print(f"[RELOAD VERIFICATION] {strategy}")
    print(f"Acc: {val_acc:.4f} (Expected: {metadata['best_validation_accuracy']:.4f})")
    print(f"Loss: {val_loss:.4f} (Expected: {metadata['best_validation_loss']:.4f})")
    print(f"Macro F1: {val_f1:.4f}")
    
    # Assert tolerance
    assert abs(val_acc - metadata['best_validation_accuracy']) < 1e-4, "Accuracy mismatch!"
    assert abs(val_loss - metadata['best_validation_loss']) < 1e-4, "Loss mismatch!"
    
    print("Verification Passed!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True, choices=["E1", "E2", "E3"])
    args = parser.parse_args()
    evaluate_reloaded(args.strategy)
