import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import json

class CosmeticsDatasetV2(Dataset):
    def __init__(self, manifest_path: str, dataset_root: str, split: str, transforms=None):
        """
        Khởi tạo V2 Dataset. 
        split: 'Train', 'Validation', hoặc 'Test'
        Chỉ load is_generated=False, use_for_model=True, quarantined=False.
        """
        self.dataset_root = Path(dataset_root)
        self.transforms = transforms
        self.split = split
        
        # Guard: Chặn không cho vô tình sử dụng đối với môi trường train
        if split == "Test":
            # Just a soft marker for downstream guarding if needed
            self.is_test = True
        else:
            self.is_test = False
            
        df = pd.read_csv(manifest_path)
        self.data = df[
            (df['split'] == split) &
            (df['use_for_model'] == True) &
            (df['is_generated'] == False) &
            (df['quarantined'] == False)
        ].copy()
        
        # Hardcode class names sorted to ensure deterministic label mapping
        self.classes = sorted(self.data['class_name'].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.dataset_root / row['relative_path']
        label_idx = self.class_to_idx[row['class_name']]
        
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        if self.transforms:
            image = self.transforms(image)
            
        return image, label_idx

def _get_default_paths():
    manifest_path = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    dataset_root = Path("total_practice/practice_2_2/data/final/data_clean_balanced")
    return str(manifest_path), str(dataset_root)

def build_train_dataset(transforms=None) -> CosmeticsDatasetV2:
    manifest_path, dataset_root = _get_default_paths()
    ds = CosmeticsDatasetV2(manifest_path, dataset_root, split="Train", transforms=transforms)
    assert len(ds) == 2027, f"Train dataset loaded {len(ds)} samples, expected 2027."
    return ds

def build_validation_dataset(transforms=None) -> CosmeticsDatasetV2:
    manifest_path, dataset_root = _get_default_paths()
    ds = CosmeticsDatasetV2(manifest_path, dataset_root, split="Validation", transforms=transforms)
    assert len(ds) == 433, f"Validation dataset loaded {len(ds)} samples, expected 433."
    return ds

def build_test_dataset(transforms=None) -> CosmeticsDatasetV2:
    manifest_path, dataset_root = _get_default_paths()
    ds = CosmeticsDatasetV2(manifest_path, dataset_root, split="Test", transforms=transforms)
    assert len(ds) == 434, f"Test dataset loaded {len(ds)} samples, expected 434."
    return ds
