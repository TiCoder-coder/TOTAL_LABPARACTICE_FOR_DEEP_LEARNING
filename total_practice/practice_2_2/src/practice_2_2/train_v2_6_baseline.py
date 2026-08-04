import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import hashlib
import json
from sklearn.metrics import f1_score
from tqdm import tqdm

# Epoch Loss Aggregator V2
class EpochLossAggregator:
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0
        
    def add_batch(self, batch_loss, batch_size):
        # We assume batch_loss is the *mean* loss returned by nn.CrossEntropyLoss(reduction='mean')
        self.total_loss += batch_loss * batch_size
        self.total_samples += batch_size
        
    def get_epoch_loss(self):
        if self.total_samples == 0: return 0.0
        return self.total_loss / self.total_samples

def get_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def get_deterministic_tensor_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        tensor = state_dict[k]
        h.update(k.encode('utf-8'))
        tensor_np = tensor.cpu().numpy().copy(order='C')
        h.update(tensor_np.tobytes())
    return h.hexdigest()

def verify_data(df, manifest_path):
    # Verify exact lineage id from path
    assert "v2_visual_group_stratified_s42" in manifest_path, "MUST USE NEW MANIFEST LINEAGE"
    assert "v2_stratified_group_s42" not in manifest_path, "REJECT OLD MANIFEST"
    
    # 1. Dataset fingerprint
    dataset_hash = get_hash(df[['image_path', 'label', 'split', 'visual_group_id', 'is_generated']])
    assert dataset_hash == "d20079086a97a863f91803d1ca798007d25b2542dfff9b0270968de2ff057531"
    
    # 2. Split fingerprint
    split_hash = get_hash(df[['image_path', 'split']])
    assert split_hash == "bb83c2172dc40bfd1ad4d56ea25bc61f3752acf2efdc9d5767a211727015d461"
    
    # 3. Manifest SHA-256
    manifest_hash = get_hash(df)
    assert manifest_hash == "d94a9ae7b3aefa54bff7f5f24af075afe95f71a0fec8922bd04be46ff6952b67", f"Manifest hash mismatch: {manifest_hash}"
    
    df_usable = df[df['use_for_model'] == True]
    
    # Verify counts
    train_df = df_usable[df_usable['split'] == 'Train']
    val_df = df_usable[df_usable['split'] == 'Validation']
    test_df = df_usable[df_usable['split'] == 'Test']
    
    assert len(train_df) == 2025
    assert len(val_df) == 434
    assert len(test_df) == 435
    
    assert df_usable['is_generated'].sum() == 0
    
    # Leakage assertions
    cross_split_vg = df_usable.groupby('visual_group_id')['split'].nunique()
    assert (cross_split_vg > 1).sum() == 0
    
    cross_split_eg = df_usable.groupby('effective_group_id')['split'].nunique()
    assert (cross_split_eg > 1).sum() == 0

    cross_split_dup = df_usable.groupby('duplicate_cluster_id')['split'].nunique()
    assert (cross_split_dup > 1).sum() == 0
    
    cross_split_src = df_usable.groupby('source_group')['split'].nunique()
    assert (cross_split_src > 1).sum() == 0

    # Read candidates to verify confirmed leakage is 0
    candidates_path = "total_practice/practice_2_2/artifacts/NEAR_DUPLICATE_REVIEW_REQUIRED.csv"
    if os.path.exists(candidates_path):
        cand_df = pd.read_csv(candidates_path)
        def is_confirmed(r):
            return (r['phash_distance'] == 0 and r['secondary_similarity'] >= 0.95) or \
                   (r['phash_distance'] <= 2 and r['dhash_distance'] <= 2 and r['secondary_similarity'] >= 0.95)
        if len(cand_df) > 0:
            confirmed = cand_df.apply(is_confirmed, axis=1)
            assert confirmed.sum() == 0
            
    print("Pre-train verification gate passed!")

class ProductDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = int(row['label'])
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def apply_freeze_policy(model, experiment_id):
    if experiment_id == "E1_visualsafe_head_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif experiment_id == "E2_visualsafe_layer4":
        for param in model.parameters():
            param.requires_grad = False
        
        # Trainable layer4 and classifier
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
            
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["E1_visualsafe_head_only", "E2_visualsafe_layer4"])
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    base_dir = "total_practice/practice_2_2"
    manifest_path = os.path.join(base_dir, "data/manifests/v2_visual_group_stratified_s42/split_manifest.csv")
    df = pd.read_csv(manifest_path)
    
    verify_data(df, manifest_path)
    
    df_usable = df[df['use_for_model'] == True]
    train_df = df_usable[df_usable['split'] == 'Train']
    val_df = df_usable[df_usable['split'] == 'Validation']
    
    train_transform, val_transform = get_transforms()
    
    root_img_dir = os.path.join(base_dir, "data/final/data_clean_balanced")
    train_dataset = ProductDataset(train_df, root_img_dir, train_transform)
    val_dataset = ProductDataset(val_df, root_img_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 10)
    )
    
    init_state_path = os.path.join(base_dir, "artifacts/experiments/v2_visualsafe_baseline_s42_v1/initial_state.pt")
    state_dict = torch.load(init_state_path)
    model.load_state_dict(state_dict)
    
    # Verify deterministic tensor hash
    t_hash = get_deterministic_tensor_hash(state_dict)
    assert t_hash == "4e7500733de1a2f3262489d172646dcc5c730ec8ffe0e82f00a9533a1b9368fd"
    
    apply_freeze_policy(model, args.exp)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    if args.exp == "E1_visualsafe_head_only":
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=2e-4)
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ], weight_decay=2e-4)
        
    criterion = nn.CrossEntropyLoss(weight=None)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode='min')
    
    out_dir = os.path.join(base_dir, "artifacts/experiments/v2_visualsafe_baseline_s42_v1", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    
    best_val_acc = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 3
    
    history = []
    
    for epoch in range(15):
        model.train()
        
        # Apply BN freeze
        if args.exp == "E1_visualsafe_head_only":
            model.apply(set_bn_eval)
        elif args.exp == "E2_visualsafe_layer4":
            model.apply(set_bn_eval)
            # Unfreeze layer4 BN
            for module in model.layer4.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    
        train_agg = EpochLossAggregator()
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_agg.add_batch(loss.item(), inputs.size(0))
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += inputs.size(0)
            
        train_loss = train_agg.get_epoch_loss()
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_agg = EpochLossAggregator()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
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
        
        scheduler.step(val_loss)
        
        lr_head = optimizer.param_groups[-1]['lr']
        lr_backbone = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 1 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        print(f"LR Head: {lr_head}, LR Backbone: {lr_backbone}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
        
        # Save latest
        torch.save(model.state_dict(), os.path.join(out_dir, "latest.pt"))
        
        # Determine best (primary: Val Acc, secondary: Val Loss)
        is_best = False
        if val_acc > best_val_acc:
            is_best = True
        elif val_acc == best_val_acc:
            if val_loss < best_val_loss:
                is_best = True
                
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            
            with open(os.path.join(out_dir, "best_metrics.json"), 'w') as f:
                json.dump({
                    "run_id": "v2_visualsafe_baseline_s42_v1",
                    "experiment_id": args.exp,
                    "epoch": epoch + 1,
                    "seed": 42,
                    "dataset_fingerprint": "d20079086a97a863f91803d1ca798007d25b2542dfff9b0270968de2ff057531",
                    "split_fingerprint": "bb83c2172dc40bfd1ad4d56ea25bc61f3752acf2efdc9d5767a211727015d461",
                    "manifest_hash": "d94a9ae7b3aefa54bff7f5f24af075afe95f71a0fec8922bd04be46ff6952b67",
                    "initial_tensor_hash": "4e7500733de1a2f3262489d172646dcc5c730ec8ffe0e82f00a9533a1b9368fd",
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "test_evaluated": False
                }, f, indent=4)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print("Early stopping triggered!")
            break

    with open(os.path.join(out_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=4)
        
    print("Training finished!")

if __name__ == "__main__":
    main()
