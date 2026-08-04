import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from hashlib import sha256

def get_hash(df):
    return sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def get_file_hash(filepath):
    h = sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    base_dir = "total_practice/practice_2_2"
    v2_manifest_path = os.path.join(base_dir, "data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    candidates_path = os.path.join(base_dir, "artifacts/ALL_NEAR_DUPLICATES.csv")
    
    df_v2 = pd.read_csv(v2_manifest_path)
    df_cand = pd.read_csv(candidates_path)
    
    # Define confirmed edges
    # Policy: 
    # A. pHash distance == 0 AND SSIM >= 0.95
    # B. pHash distance <= 2 AND dHash distance <= 2 AND SSIM >= 0.95
    
    confirmed_edges = []
    cross_label_conflicts = set()
    
    for idx, row in df_cand.iterrows():
        p_dist = row['phash_distance']
        d_dist = row['dhash_distance']
        ssim = row['secondary_similarity']
        
        is_confirmed = False
        if p_dist == 0 and ssim >= 0.95:
            is_confirmed = True
        elif p_dist <= 2 and d_dist <= 2 and ssim >= 0.95:
            is_confirmed = True
            
        if is_confirmed:
            # We map image_path to absolute in df_cand, but in df_v2 it's relative path 'image_path'
            path_a = row['image_a'].split('data_clean_balanced/')[-1]
            path_b = row['image_b'].split('data_clean_balanced/')[-1]
            confirmed_edges.append((path_a, path_b))
            
            if row['label_a'] != row['label_b']:
                cross_label_conflicts.add(path_a)
                cross_label_conflicts.add(path_b)
                
    # Also save cross-label to a separate CSV
    if len(cross_label_conflicts) > 0:
        cross_df = df_v2[df_v2['image_path'].isin(cross_label_conflicts)].copy()
        cross_df.to_csv(os.path.join(base_dir, "artifacts/V2_CROSS_LABEL_REVIEW_REQUIRED.csv"), index=False)
        print(f"Saved {len(cross_df)} images to V2_CROSS_LABEL_REVIEW_REQUIRED.csv")
    
    # Build Graph
    G = nx.Graph()
    # Add nodes
    for _, row in df_v2.iterrows():
        G.add_node(row['image_path'], 
                   label=row['label'], 
                   duplicate_cluster_id=row['duplicate_cluster_id'],
                   source_group=row['source_group'],
                   effective_group_id=row['effective_group_id'])
        
    # Add edges based on metadata
    metadata_cols = ['duplicate_cluster_id', 'source_group', 'effective_group_id']
    for col in metadata_cols:
        grouped = df_v2.groupby(col)['image_path'].apply(list)
        for group_images in grouped:
            if len(group_images) > 1:
                first = group_images[0]
                for i in range(1, len(group_images)):
                    G.add_edge(first, group_images[i])
                    
    # Add confirmed visual edges
    for u, v in confirmed_edges:
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v)
            
    # Find connected components
    components = list(nx.connected_components(G))
    
    visual_group_map = {}
    for i, comp in enumerate(components):
        v_id = f"vg_{i:05d}"
        for node in comp:
            visual_group_map[node] = v_id
            
    df_v2['visual_group_id'] = df_v2['image_path'].map(visual_group_map)
    df_v2['review_required'] = df_v2['image_path'].isin(cross_label_conflicts)
    
    # the user asked to mark these use_for_model = False but DO NOT DELETE
    # however, we also want to preserve existing use_for_model values from df_v2 
    # (e.g., if there were others already False).
    df_v2['use_for_model'] = ~df_v2['review_required'] & df_v2['use_for_model']
    
    # Greedy Assignment Algorithm
    usable_df = df_v2[df_v2['use_for_model'] == True].copy()
    
    # Calculate group stats
    group_stats = usable_df.groupby('visual_group_id').agg(
        size=('image_path', 'count'),
        label=('label', lambda x: x.iloc[0]) 
    ).reset_index()
    
    # Sort groups by size descending to pack largest first
    group_stats = group_stats.sort_values('size', ascending=False)
    largest_group_size = group_stats['size'].max()
    
    # Target sizes
    total_size = len(usable_df)
    target_sizes = {
        'Train': int(0.70 * total_size),
        'Validation': int(0.15 * total_size),
        'Test': total_size - int(0.70 * total_size) - int(0.15 * total_size)
    }
    
    num_classes = usable_df['label'].nunique()
    class_dist_target = usable_df['label'].value_counts(normalize=True).to_dict()
    
    current_counts = {
        'Train': {'total': 0, 'classes': {c: 0 for c in range(num_classes)}},
        'Validation': {'total': 0, 'classes': {c: 0 for c in range(num_classes)}},
        'Test': {'total': 0, 'classes': {c: 0 for c in range(num_classes)}}
    }
    
    split_assignment = {}
    np.random.seed(42) # For tie breaking
    
    for _, row in group_stats.iterrows():
        vg = row['visual_group_id']
        sz = row['size']
        lbl = row['label']
        
        best_split = None
        best_score = float('inf')
        
        candidates = ['Train', 'Validation', 'Test']
        np.random.shuffle(candidates)
        
        for split in candidates:
            sim_total = current_counts[split]['total'] + sz
            sim_classes = current_counts[split]['classes'].copy()
            sim_classes[lbl] += sz
            
            if sim_total > target_sizes[split]:
                size_penalty = 10.0 + (sim_total - target_sizes[split]) / total_size
            else:
                size_penalty = sim_total / target_sizes[split]
            
            class_penalty = 0
            for c in range(num_classes):
                c_ratio = sim_classes[c] / (sim_total + 1e-9)
                class_penalty += (c_ratio - class_dist_target[c]) ** 2
                
            score = size_penalty + class_penalty * 0.5 
            if score < best_score:
                best_score = score
                best_split = split
                
        split_assignment[vg] = best_split
        current_counts[best_split]['total'] += sz
        current_counts[best_split]['classes'][lbl] += sz
        
    df_v2['split'] = df_v2['visual_group_id'].map(split_assignment)
    
    # Calculate statistics
    train_c = len(df_v2[(df_v2['split'] == 'Train') & (df_v2['use_for_model'] == True)])
    val_c = len(df_v2[(df_v2['split'] == 'Validation') & (df_v2['use_for_model'] == True)])
    test_c = len(df_v2[(df_v2['split'] == 'Test') & (df_v2['use_for_model'] == True)])
    
    print(f"Train: {train_c}, Validation: {val_c}, Test: {test_c}")
    
    out_dir = os.path.join(base_dir, "data/manifests/v2_visual_group_stratified_s42")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "split_manifest.csv")
    df_v2.to_csv(out_path, index=False)
    
    max_dev = 0
    # Print per-class distribution
    for split in ['Train', 'Validation', 'Test']:
        split_df = df_v2[(df_v2['split'] == split) & (df_v2['use_for_model'] == True)]
        print(f"\n{split} class distribution:")
        dist = split_df['label'].value_counts(normalize=True)
        print(dist)
        for c, v in dist.items():
            dev = abs(v - class_dist_target[c])
            if dev > max_dev:
                max_dev = dev
                
    print(f"\nMax class deviation: {max_dev:.4f}")
    print(f"Visual groups: {len(components)}")
    print(f"Largest visual group size: {largest_group_size}")
    print(f"Confirmed edges: {len(confirmed_edges)}")
    print(f"Cross-label conflicts: {len(cross_label_conflicts)}")
    
if __name__ == "__main__":
    main()
