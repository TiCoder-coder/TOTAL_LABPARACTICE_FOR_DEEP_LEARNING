import os
import argparse
import pandas as pd
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2
import hashlib

def _hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_v1_and_v2_hashes():
    p1 = "total_practice/practice_2_2/data/manifests/canonical_split/split_manifest.csv"
    p2 = "total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv"
    return _hash_file(p1), _hash_file(p2)

def audit_visual_leakage():
    print("Gathering initial mutation guarantee hashes...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default="total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv", help="Path to manifest")
    args = parser.parse_args()
    
    v2_manifest_path = args.manifest
    v1_hash_start, v2_hash_start = get_v1_and_v2_hashes()
    
    # 1. LOAD MANIFEST
    df = pd.read_csv(v2_manifest_path)
    
    # 2. VERIFY COUNTS
    train_df = df[df['split'] == 'Train']
    val_df = df[df['split'] == 'Validation']
    test_df = df[df['split'] == 'Test']
    
    assert df['is_generated'].sum() == 0, "Generated images found!"
    assert len(df[df['split'] == 'Quarantine']) == 0, "Quarantine images in working split?"
    
    # 3. METADATA LEAKAGE AUDIT
    def check_intersection(series_t, series_v, series_test, name):
        s_t = set(series_t.dropna())
        s_v = set(series_v.dropna())
        s_test = set(series_test.dropna())
        if len(s_t.intersection(s_v)) > 0: raise AssertionError(f"{name} leaked Train-Val")
        if len(s_t.intersection(s_test)) > 0: raise AssertionError(f"{name} leaked Train-Test")
        if len(s_v.intersection(s_test)) > 0: raise AssertionError(f"{name} leaked Val-Test")
        
    check_intersection(train_df['file_hash'], val_df['file_hash'], test_df['file_hash'], "Exact SHA256")
    check_intersection(train_df['duplicate_cluster_id'], val_df['duplicate_cluster_id'], test_df['duplicate_cluster_id'], "duplicate_cluster_id")
    check_intersection(train_df['source_group'], val_df['source_group'], test_df['source_group'], "source_group")
    check_intersection(train_df['effective_group_id'], val_df['effective_group_id'], test_df['effective_group_id'], "effective_group_id")
    
    print("Metadata Isolation Verified.")
    
    # 4. COMPUTE pHash & dHash
    records = df.to_dict('records')
    print("Computing perceptual hashes...")
    hash_data = []
    
    for r in tqdm(records):
        path = os.path.join("total_practice/practice_2_2/data/final/data_clean_balanced", r['image_path'])
        try:
            img = Image.open(path)
            ph = imagehash.phash(img, hash_size=8)
            dh = imagehash.dhash(img, hash_size=8)
            hash_data.append({
                'image_path': path,
                'v2_split': r['split'],
                'label': r['label'],
                'file_hash': r['file_hash'],
                'effective_group_id': r.get('effective_group_id', ''),
                'source_group': r.get('source_group', ''),
                'phash': ph,
                'dhash': dh,
                'phash_bin': ph.hash.flatten(),
                'dhash_bin': dh.hash.flatten()
            })
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    # Convert to fast numpy array for pairwise distance
    phash_mat = np.array([x['phash_bin'] for x in hash_data], dtype=bool) # (N, 64)
    dhash_mat = np.array([x['dhash_bin'] for x in hash_data], dtype=bool) # (N, 64)
    splits = np.array([x['v2_split'] for x in hash_data])
    labels = np.array([x['label'] for x in hash_data])
    
    N = len(hash_data)
    candidates = []
    
    # Define combinations
    cross_split_mask = (splits[:, None] != splits[None, :])
    # Compute distances
    print("Computing pairwise hamming distances...")
    # Because 3000 x 3000 is 9 million, we can do it in memory easily.
    # Distances are sum of XOR
    # phash_dist: (N, N)
    phash_dist = np.sum(phash_mat[:, None, :] ^ phash_mat[None, :, :], axis=2)
    dhash_dist = np.sum(dhash_mat[:, None, :] ^ dhash_mat[None, :, :], axis=2)
    
    # Mask out same split and upper triangle to avoid dupes
    mask = cross_split_mask & np.tri(N, k=-1, dtype=bool).T 
    
    # Find candidates with pHash distance <= 4 or dHash distance <= 4
    candidate_indices = np.where((phash_dist <= 4) & mask)
    
    print(f"Found {len(candidate_indices[0])} raw candidates. Computing SSIM...")
    
    def get_ssim(p1, p2):
        im1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None: return 0.0
        im1 = cv2.resize(im1, (224, 224))
        im2 = cv2.resize(im2, (224, 224))
        score, _ = ssim(im1, im2, full=True)
        return score
        
    for i, j in zip(candidate_indices[0], candidate_indices[1]):
        pd_dist = phash_dist[i, j]
        dd_dist = dhash_dist[i, j]
        
        # We classify risk
        if pd_dist == 0:
            risk = "CRITICAL"
        elif pd_dist <= 2:
            risk = "HIGH"
        elif pd_dist <= 4:
            risk = "MEDIUM"
        else:
            continue # should not happen based on where
            
        r1, r2 = hash_data[i], hash_data[j]
        
        # Secondary similarity (SSIM)
        ssim_score = get_ssim(r1['image_path'], r2['image_path'])
        
        reason = []
        if pd_dist == 0: reason.append("EXTREMELY_SIMILAR_CANDIDATE")
        if r1['label'] != r2['label']: reason.append("POSSIBLE_LABEL_CONFLICT")
        if ssim_score > 0.95: reason.append("HIGH_SSIM_CONFIRMATION")
        if "_aug" in r1['image_path'] or "_aug" in r2['image_path']: reason.append("AUGMENTATION_DERIVED_SUSPICION")
        
        candidates.append({
            'image_a': r1['image_path'],
            'split_a': r1['v2_split'],
            'label_a': r1['label'],
            'image_b': r2['image_path'],
            'split_b': r2['v2_split'],
            'label_b': r2['label'],
            'sha256_equal': r1['file_hash'] == r2['file_hash'],
            'phash_a': str(r1['phash']),
            'phash_b': str(r2['phash']),
            'phash_distance': pd_dist,
            'dhash_distance': dd_dist,
            'secondary_similarity': ssim_score,
            'same_effective_group': r1['effective_group_id'] == r2['effective_group_id'] and r1['effective_group_id'] != '',
            'same_source_group': r1['source_group'] == r2['source_group'] and r1['source_group'] != '',
            'risk_level': risk,
            'review_reason': "|".join(reason) if reason else "Visual Similarity"
        })
        
    out_path = "total_practice/practice_2_2/artifacts/NEAR_DUPLICATE_REVIEW_REQUIRED.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(candidates).to_csv(out_path, index=False)
    
    # 5. MUTATION GUARANTEE
    v1_hash_end, v2_hash_end = get_v1_and_v2_hashes()
    assert v1_hash_start == v1_hash_end, "V1 MANIFEST MUTATED!"
    assert v2_hash_start == v2_hash_end, "V2 MANIFEST MUTATED!"
    
    print("Done. Mutation check passed.")
    
if __name__ == "__main__":
    audit_visual_leakage()
