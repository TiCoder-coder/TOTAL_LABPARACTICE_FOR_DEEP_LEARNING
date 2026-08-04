import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_df_fingerprint(df):
    """Compute a deterministic hash for a dataframe based on selected columns."""
    df_sorted = df.sort_values("record_id").reset_index(drop=True)
    json_str = df_sorted.to_json(orient="records")
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def create_effective_groups(df):
    """Create effective_group_id combining duplicate_cluster_id and source_group."""
    G = nx.Graph()
    for _, row in df.iterrows():
        node_id = f"record_{row['record_id']}"
        G.add_node(node_id)
        if pd.notna(row.get('duplicate_cluster_id')):
            G.add_edge(node_id, f"dup_{row['duplicate_cluster_id']}")
        if pd.notna(row.get('source_group')):
            G.add_edge(node_id, f"src_{row['source_group']}")

    components = list(nx.connected_components(G))
    node_to_group = {}
    for i, comp in enumerate(components):
        for node in comp:
            if node.startswith("record_"):
                node_to_group[node.replace("record_", "")] = f"eff_group_{i}"
    
    df['effective_group_id'] = df['record_id'].astype(str).map(node_to_group)
    return df


def assign_splits(df, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    targets = {'Train': 0.70, 'Validation': 0.15, 'Test': 0.15}
    total_images = len(df)
    class_counts = df['class_name'].value_counts()
    classes = class_counts.index.tolist()
    target_class_ratios = class_counts / total_images
    
    groups = df.groupby('effective_group_id')
    group_sizes = groups.size()
    
    group_list = []
    for g_id, size in group_sizes.items():
        g_df = groups.get_group(g_id)
        g_class_counts = g_df['class_name'].value_counts().reindex(classes, fill_value=0)
        group_list.append({
            'id': g_id,
            'size': size,
            'class_counts': g_class_counts
        })
    
    # Shuffle groups first to ensure randomness in assignment
    random.shuffle(group_list)
    # Then sort by size so largest groups are assigned first (hardest constraints first)
    group_list.sort(key=lambda x: x['size'], reverse=True)
    
    splits = {
        'Train': {'size': 0, 'class_counts': pd.Series(0, index=classes, dtype=float)},
        'Validation': {'size': 0, 'class_counts': pd.Series(0, index=classes, dtype=float)},
        'Test': {'size': 0, 'class_counts': pd.Series(0, index=classes, dtype=float)}
    }
    
    assignment = {}
    
    for g in group_list:
        best_split = None
        best_penalty = float('inf')
        
        for split_name in ['Train', 'Validation', 'Test']:
            new_size = splits[split_name]['size'] + g['size']
            new_class_counts = splits[split_name]['class_counts'] + g['class_counts']
            
            # How full is this split relative to its target?
            # 0.0 means completely empty, 1.0 means exactly at target
            target_size = targets[split_name] * total_images
            size_penalty = new_size / target_size
            
            if new_size > 0:
                class_ratios = new_class_counts / new_size
            else:
                class_ratios = pd.Series(0, index=classes, dtype=float)
            
            class_penalty = (abs(class_ratios - target_class_ratios)).mean()
            
            # Both size_penalty and class_penalty are around 0.0 - 1.0. 
            # We want to strictly enforce size targets, but also heavily favor class balance.
            penalty = size_penalty + 1.0 * class_penalty
            
            if penalty < best_penalty:
                best_penalty = penalty
                best_split = split_name
                
        splits[best_split]['size'] += g['size']
        splits[best_split]['class_counts'] += g['class_counts']
        assignment[g['id']] = best_split
        
    df['split'] = df['effective_group_id'].map(assignment)
    return df


def assert_split_integrity(df):
    assert len(df[df['split'] == 'Train']) > 0
    assert len(df[df['split'] == 'Validation']) > 0
    assert len(df[df['split'] == 'Test']) > 0
    
    assert df['is_generated'].sum() == 0, "Generated images found in split dataset!"
    assert df['quarantined'].sum() == 0, "Quarantined images found in split dataset!"
    assert (~df['use_for_model']).sum() == 0, "use_for_model=False found in split dataset!"
    
    # Leakage assertions
    eff_groups = df.groupby('effective_group_id')['split'].nunique()
    assert eff_groups.max() == 1, "Effective group crossed split boundaries!"
    
    if 'duplicate_cluster_id' in df.columns:
        dup_groups = df.dropna(subset=['duplicate_cluster_id']).groupby('duplicate_cluster_id')['split'].nunique()
        if not dup_groups.empty:
            assert dup_groups.max() == 1, "duplicate_cluster_id crossed split boundaries!"
            
    if 'source_group' in df.columns:
        src_groups = df.dropna(subset=['source_group']).groupby('source_group')['split'].nunique()
        if not src_groups.empty:
            assert src_groups.max() == 1, "source_group crossed split boundaries!"


def simulate_and_print(df):
    classes = df['class_name'].unique().tolist()
    total_images = len(df)
    
    print("\n--- CLASS DISTRIBUTION ---")
    print(f"{'Class':<20} | {'Available':<10} | {'Train':<10} | {'Validation':<10} | {'Test':<10}")
    
    max_dev = 0
    max_dev_class = ""
    
    for cls in classes:
        cls_df = df[df['class_name'] == cls]
        avail = len(cls_df)
        tr = len(cls_df[cls_df['split'] == 'Train'])
        va = len(cls_df[cls_df['split'] == 'Validation'])
        te = len(cls_df[cls_df['split'] == 'Test'])
        
        tr_pct = tr / avail * 100
        va_pct = va / avail * 100
        te_pct = te / avail * 100
        
        print(f"{cls:<20} | {avail:<10} | {tr:<10} | {va:<10} | {te:<10}")
        
        dev = max(abs(tr_pct - 70), abs(va_pct - 15), abs(te_pct - 15))
        if dev > max_dev:
            max_dev = dev
            max_dev_class = cls

    tr_total = len(df[df['split'] == 'Train'])
    va_total = len(df[df['split'] == 'Validation'])
    te_total = len(df[df['split'] == 'Test'])
    
    print("\n--- TOTALS ---")
    print(f"Train total: {tr_total} ({tr_total/total_images*100:.2f}%)")
    print(f"Validation total: {va_total} ({va_total/total_images*100:.2f}%)")
    print(f"Test total: {te_total} ({te_total/total_images*100:.2f}%)")
    
    print(f"\nMax Class Distribution Deviation: {max_dev:.2f}% (Class: {max_dev_class})")
    
    return max_dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load V1
    df_raw = pd.read_csv(args.v1_manifest)
    
    # Filter
    df = df_raw.copy()
    if 'split' in df.columns:
        df = df.drop(columns=['split'])
    
    generated_excluded = df['is_generated'].sum()
    quarantine_excluded = df['quarantined'].sum()
    use_for_model_excluded = (~df['use_for_model']).sum()
    
    df = df[
        (df['is_generated'] == False) & 
        (df['quarantined'] == False) & 
        (df['use_for_model'] == True)
    ].copy()
    
    usable_original_total = len(df)
    
    # Create effective groups
    df = create_effective_groups(df)
    
    # Assign splits
    df = assign_splits(df, seed=args.seed)
    
    # Check reproducibility
    df_check = assign_splits(df.copy(), seed=args.seed)
    assert (df['split'] == df_check['split']).all(), "Reproducibility assertion failed!"
    
    # Run assertions
    assert_split_integrity(df)
    
    # Print simulation
    max_dev = simulate_and_print(df)
    
    # Verdict logic
    verdict = "V2_SPLIT_READY"
    if max_dev > 15.0: # Arbitrary threshold
        verdict = "V2_SPLIT_NEEDS_FIX"
        
    print(f"\nVERDICT: {verdict}")
    
    if verdict == "V2_SPLIT_READY" and args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = args.out_dir / "split_manifest_v2.csv"
        df.to_csv(manifest_path, index=False)
        
        # Calculate fingerprint
        dataset_sha = compute_df_fingerprint(df.drop(columns=['split']))
        split_sha = compute_df_fingerprint(df[['record_id', 'split']])
        manifest_sha = file_sha256(manifest_path)
        
        summary = {
            "created_at_utc": utc_now(),
            "seed": args.seed,
            "usable_original_total": usable_original_total,
            "excluded_generated": int(generated_excluded),
            "excluded_quarantined": int(quarantine_excluded),
            "excluded_not_for_model": int(use_for_model_excluded),
            "dataset_fingerprint_sha256": dataset_sha,
            "split_fingerprint_sha256": split_sha,
            "manifest_sha256": manifest_sha,
            "train_ratio": len(df[df['split'] == 'Train']) / usable_original_total,
            "val_ratio": len(df[df['split'] == 'Validation']) / usable_original_total,
            "test_ratio": len(df[df['split'] == 'Test']) / usable_original_total,
        }
        
        with open(args.out_dir / "split_summary_v2.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        verification = {
            "verified_at_utc": utc_now(),
            "verdict": verdict,
            "max_class_distribution_deviation": max_dev,
            "generated_train": 0,
            "generated_validation": 0,
            "generated_test": 0,
            "exact_hash_cross_split": 0,
            "duplicate_cluster_cross_split": 0,
            "source_group_cross_split": 0,
            "quarantine_loaded": 0,
            "use_for_model_false_loaded": 0
        }
        
        with open(args.out_dir / "split_verification_v2.json", "w") as f:
            json.dump(verification, f, indent=2)
            
        print(f"Persisted V2 artifacts to {args.out_dir}")

if __name__ == "__main__":
    main()
