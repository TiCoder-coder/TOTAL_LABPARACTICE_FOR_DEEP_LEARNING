import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import sys

def calculate_brightness_contrast(img_tensor):
    # tensor: C x H x W in [0.0, 1.0]
    gray = img_tensor.mean(dim=0)
    brightness = gray.mean().item()
    contrast = gray.std().item()
    return brightness, contrast

def main():
    manifest_path = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    dataset_root = Path("total_practice/practice_2_2/data/final/data_clean_balanced")
    
    df_raw = pd.read_csv(manifest_path)
    
    # Validation/Test checks
    val_test_df = df_raw[df_raw['split'].isin(['Validation', 'Test'])]
    val_test_paths = set(val_test_df['relative_path'].tolist())
    
    # Load Train only
    df = df_raw[
        (df_raw['split'] == 'Train') & 
        (df_raw['use_for_model'] == True) & 
        (df_raw['is_generated'] == False) & 
        (df_raw['quarantined'] == False)
    ].copy()
    
    train_paths = set(df['relative_path'].tolist())
    
    # Assertions
    expected_train_df = df_raw[
        (df_raw['split'] == 'Train') & 
        (df_raw['use_for_model'] == True) & 
        (df_raw['is_generated'] == False) & 
        (df_raw['quarantined'] == False)
    ]
    expected_train_paths = set(expected_train_df['relative_path'].tolist())
    
    assert len(df) == 2027, f"Expected 2027 Train samples, got {len(df)}"
    assert train_paths == expected_train_paths, "Loaded Train paths do not match expected Train paths from manifest!"
    assert len(train_paths.intersection(val_test_paths)) == 0, "Leakage detected: Train paths intersect with Val/Test paths!"
    
    # 2. Train Class Distribution
    class_counts = df['class_name'].value_counts()
    classes = class_counts.index.tolist()
    counts = class_counts.values
    
    min_cnt = counts.min()
    max_cnt = counts.max()
    mean_cnt = counts.mean()
    std_cnt = counts.std()
    cv = std_cnt / mean_cnt
    max_min_ratio = max_cnt / min_cnt
    
    if max_min_ratio < 1.1:
        imbalance = "BALANCED"
    elif max_min_ratio < 1.5:
        imbalance = "SLIGHT"
    elif max_min_ratio < 2.0:
        imbalance = "MODERATE"
    else:
        imbalance = "SEVERE"
        
    class_dist = []
    for cls, cnt in class_counts.items():
        class_dist.append({
            "class": cls,
            "samples": int(cnt),
            "percentage": float(cnt / len(df) * 100)
        })
        
    # 3. Effective Diversity
    diversity = []
    for cls in classes:
        cls_df = df[df['class_name'] == cls]
        images = len(cls_df)
        groups = cls_df['effective_group_id'].nunique()
        largest_group = int(cls_df.groupby('effective_group_id').size().max())
        diversity.append({
            "class": cls,
            "images": int(images),
            "effective_groups": int(groups),
            "images_per_group": float(images / groups),
            "largest_group": largest_group,
            "diversity_ratio": float(groups / images)
        })
        
    # 5 & 6 & 7. Image Statistics & Normalization & Review Candidates
    print("Processing images...")
    
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    
    widths = []
    heights = []
    brightnesses = []
    contrasts = []
    
    review_candidates = []
    
    for idx, row in df.iterrows():
        file_path = dataset_root / row['relative_path']
        if not file_path.exists():
            continue
            
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                widths.append(w)
                heights.append(h)
                
                # Heuristics for review
                aspect = w / h
                reason = []
                if aspect < 0.33 or aspect > 3.0:
                    reason.append(f"Extreme aspect ratio ({aspect:.2f})")
                if w < 100 or h < 100:
                    reason.append(f"Very small resolution ({w}x{h})")
                
                tensor = TF.to_tensor(img) # C x H x W in [0,1]
                
                b, c = calculate_brightness_contrast(tensor)
                brightnesses.append(b)
                contrasts.append(c)
                
                if b < 0.15:
                    reason.append(f"Extremely dark (brightness={b:.2f})")
                elif b > 0.85:
                    reason.append(f"Extremely bright/washed out (brightness={b:.2f})")
                if c < 0.05:
                    reason.append(f"Extremely low contrast (c={c:.2f})")
                    
                if reason:
                    review_candidates.append({
                        "relative_path": row['relative_path'],
                        "label": row['class_name'],
                        "reason": " | ".join(reason),
                        "confidence": "High"
                    })
                
                # Global statistics
                pixels = w * h
                total_pixels += pixels
                pixel_sum += tensor.sum(dim=[1,2]).to(torch.float64)
                pixel_sq_sum += (tensor ** 2).sum(dim=[1,2]).to(torch.float64)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    # Calculate RGB mean and std
    mean_rgb = (pixel_sum / total_pixels).tolist()
    std_rgb = torch.sqrt((pixel_sq_sum / total_pixels) - (pixel_sum / total_pixels)**2).tolist()
    
    # Class weights candidates
    total = len(df)
    num_classes = len(classes)
    inv_freq = {cls: total / (num_classes * count) for cls, count in class_counts.items()}
    sum_inv_freq = sum(inv_freq.values())
    norm_inv_freq = {cls: (w / sum_inv_freq * num_classes) for cls, w in inv_freq.items()}
    
    # V1 weight info
    v1_weights_bug_note = (
        "V1 calculated loss as: loss.item() * inputs.size(0) inside train loop. "
        "But loss.item() from CrossEntropyLoss with weights is a weighted mean, "
        "meaning it does NOT correspond directly to the batch size unweighted average. "
        "Multiplying by batch_size causes the running epoch loss to be mathematically incorrect "
        "(scaling by a mismatched denominator)."
    )
    
    results = {
        "train_distribution": {
            "total_samples": int(total),
            "min_count": int(min_cnt),
            "max_count": int(max_cnt),
            "mean_count": float(mean_cnt),
            "std_count": float(std_cnt),
            "cv": float(cv),
            "max_min_ratio": float(max_min_ratio),
            "imbalance_assessment": imbalance,
            "classes": class_dist
        },
        "effective_diversity": diversity,
        "image_statistics": {
            "mean_width": float(np.mean(widths)),
            "mean_height": float(np.mean(heights)),
            "mean_brightness": float(np.mean(brightnesses)),
            "mean_contrast": float(np.mean(contrasts))
        },
        "rgb_statistics": {
            "train_mean": mean_rgb,
            "train_std": std_rgb,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225]
        },
        "normalization_decision": "KEEP_IMAGENET_NORMALIZATION", # We keep it for now as requested
        "class_weights_candidates": {
            "inverse_frequency": inv_freq,
            "normalized_inverse_frequency": norm_inv_freq
        },
        "v1_bug_note": v1_weights_bug_note,
        "review_candidates_count": len(review_candidates)
    }
    
    # Save results
    out_dir = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42")
    with open(out_dir / "eda_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    df_review = pd.DataFrame(review_candidates)
    if not df_review.empty:
        df_review.to_csv(out_dir / "TRAIN_REVIEW_REQUIRED.csv", index=False)
    else:
        # Create empty CSV with headers
        pd.DataFrame(columns=["filepath", "label", "reason", "confidence"]).to_csv(out_dir / "TRAIN_REVIEW_REQUIRED.csv", index=False)
        
    print(f"EDA completed. Processed {len(df)} images.")
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
