import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from sklearn.metrics import f1_score
from tqdm import tqdm
from train_v2_6_baseline import ProductDataset, get_transforms, EpochLossAggregator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["E1_visualsafe_head_only", "E2_visualsafe_layer4"])
    args = parser.parse_args()
    
    base_dir = "total_practice/practice_2_2"
    manifest_path = os.path.join(base_dir, "data/manifests/v2_visual_group_stratified_s42/split_manifest.csv")
    df = pd.read_csv(manifest_path)
    df_usable = df[df['use_for_model'] == True]
    val_df = df_usable[df_usable['split'] == 'Validation']
    
    _, val_transform = get_transforms()
    root_img_dir = os.path.join(base_dir, "data/final/data_clean_balanced")
    val_dataset = ProductDataset(val_df, root_img_dir, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 10)
    )
    
    out_dir = os.path.join(base_dir, "artifacts/experiments/v2_visualsafe_baseline_s42_v1", args.exp)
    best_path = os.path.join(out_dir, "best.pt")
    
    state_dict = torch.load(best_path)
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(weight=None)
    
    val_agg = EpochLossAggregator()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Reload Verification {args.exp}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_agg.add_batch(loss.item(), inputs.size(0))
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = val_agg.get_epoch_loss()
    val_acc = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Reload Verification for {args.exp}:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Val F1: {val_f1:.4f}")
    
    # Check against saved metrics
    with open(os.path.join(out_dir, "best_metrics.json"), 'r') as f:
        metrics = json.load(f)
        
    assert abs(val_loss - metrics['val_loss']) < 1e-4, f"Loss mismatch: {val_loss} vs {metrics['val_loss']}"
    assert abs(val_acc - metrics['val_acc']) < 1e-4, f"Acc mismatch: {val_acc} vs {metrics['val_acc']}"
    assert abs(val_f1 - metrics['val_f1']) < 1e-4, f"F1 mismatch: {val_f1} vs {metrics['val_f1']}"
    
    print("Verification Passed! Saved metrics match exact reload metrics.")

if __name__ == "__main__":
    main()
