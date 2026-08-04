import os
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2

def main():
    base_dir = "total_practice/practice_2_2"
    v2_manifest_path = os.path.join(base_dir, "data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
    df = pd.read_csv(v2_manifest_path)
    df = df[df['use_for_model'] == True]
    
    records = df.to_dict('records')
    hash_data = []
    
    print("Computing perceptual hashes...")
    for r in tqdm(records):
        path = os.path.join("total_practice/practice_2_2/data/final/data_clean_balanced", r['image_path'])
        try:
            img = Image.open(path)
            ph = imagehash.phash(img, hash_size=8)
            dh = imagehash.dhash(img, hash_size=8)
            hash_data.append({
                'image_path': path,
                'phash': ph,
                'dhash': dh,
                'label': r['label'],
                'split': r['split'],
                'effective_group_id': r['effective_group_id'],
                'source_group': r['source_group']
            })
        except Exception as e:
            pass

    print("Computing pairwise hamming distances...")
    # Convert to numpy arrays for fast vectorized distance computation
    phash_arrays = np.array([h['phash'].hash.flatten() for h in hash_data])
    
    # Calculate pairwise hamming distance
    phash_mat = phash_arrays.astype(int)
    # Using broadcasting to find differences
    # Shape: (N, 1, 64) ^ (1, N, 64) -> (N, N, 64)
    phash_dist = np.sum(phash_mat[:, None, :] ^ phash_mat[None, :, :], axis=2)
    
    # We only care about upper triangle
    N = len(hash_data)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dists = phash_dist[i_idx, j_idx]
    
    # Filter candidates with distance <= 4
    candidate_mask = dists <= 4
    c_i = i_idx[candidate_mask]
    c_j = j_idx[candidate_mask]
    c_dists = dists[candidate_mask]
    
    print(f"Found {len(c_i)} raw candidates. Computing SSIM...")
    
    def compute_ssim(p1, p2):
        img1 = cv2.imread(p1)
        img2 = cv2.imread(p2)
        if img1 is None or img2 is None: return 0.0
        
        # Resize to same dimensions for SSIM just in case, though they should be same
        img1 = cv2.resize(img1, (224, 224))
        img2 = cv2.resize(img2, (224, 224))
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        score, _ = ssim(img1_gray, img2_gray, full=True)
        return score

    results = []
    for idx in tqdm(range(len(c_i))):
        i = c_i[idx]
        j = c_j[idx]
        p_dist = c_dists[idx]
        
        r1 = hash_data[i]
        r2 = hash_data[j]
        
        d_dist = r1['dhash'] - r2['dhash']
        s_score = compute_ssim(r1['image_path'], r2['image_path'])
        
        results.append({
            'image_a': r1['image_path'],
            'split_a': r1['split'],
            'label_a': r1['label'],
            'image_b': r2['image_path'],
            'split_b': r2['split'],
            'label_b': r2['label'],
            'phash_a': str(r1['phash']),
            'phash_b': str(r2['phash']),
            'phash_distance': int(p_dist),
            'dhash_distance': int(d_dist),
            'secondary_similarity': s_score,
            'same_effective_group': r1['effective_group_id'] == r2['effective_group_id'],
            'same_source_group': r1['source_group'] == r2['source_group']
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(base_dir, "artifacts/ALL_NEAR_DUPLICATES.csv"), index=False)
    print(f"Saved {len(out_df)} total near duplicates to ALL_NEAR_DUPLICATES.csv")

if __name__ == "__main__":
    main()
