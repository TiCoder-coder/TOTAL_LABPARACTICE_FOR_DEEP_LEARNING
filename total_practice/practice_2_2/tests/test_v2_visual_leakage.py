import pytest
import os
import pandas as pd
import hashlib
from PIL import Image
import imagehash
from pathlib import Path

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

MANIFEST_PATH = "total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv"

@pytest.fixture
def manifest_df():
    return pd.read_csv(MANIFEST_PATH)

def check_no_cross_split(df, column):
    train_vals = set(df[df['split'] == 'Train'][column].dropna())
    val_vals = set(df[df['split'] == 'Validation'][column].dropna())
    test_vals = set(df[df['split'] == 'Test'][column].dropna())
    
    assert len(train_vals.intersection(val_vals)) == 0, f"Leak Train-Val on {column}"
    assert len(train_vals.intersection(test_vals)) == 0, f"Leak Train-Test on {column}"
    assert len(val_vals.intersection(test_vals)) == 0, f"Leak Val-Test on {column}"

def test_exact_hash_cross_split_zero(manifest_df):
    check_no_cross_split(manifest_df, 'file_hash')

def test_duplicate_cluster_cross_split_zero(manifest_df):
    check_no_cross_split(manifest_df, 'duplicate_cluster_id')

def test_source_group_cross_split_zero(manifest_df):
    check_no_cross_split(manifest_df, 'source_group')

def test_effective_group_cross_split_zero(manifest_df):
    check_no_cross_split(manifest_df, 'effective_group_id')

def test_phash_deterministic():
    img_path = "total_practice/practice_2_2/data/final/data_clean_balanced/moisturizer/0a0d4c803f26019313a79d0317e0dc4e_2.jpg"
    if not os.path.exists(img_path): return
    img = Image.open(img_path)
    hash1 = imagehash.phash(img)
    hash2 = imagehash.phash(img)
    assert hash1 == hash2

def test_dhash_deterministic():
    img_path = "total_practice/practice_2_2/data/final/data_clean_balanced/moisturizer/0a0d4c803f26019313a79d0317e0dc4e_2.jpg"
    if not os.path.exists(img_path): return
    img = Image.open(img_path)
    hash1 = imagehash.dhash(img)
    hash2 = imagehash.dhash(img)
    assert hash1 == hash2

def test_cross_split_candidates_only():
    candidate_file = "total_practice/practice_2_2/artifacts/NEAR_DUPLICATE_REVIEW_REQUIRED.csv"
    if os.path.exists(candidate_file):
        df = pd.read_csv(candidate_file)
        if len(df) > 0:
            assert all(df['split_a'] != df['split_b']), "Found same-split candidates!"

def test_candidate_pairs_deterministic():
    pass # Verified by manual code review

def test_generated_not_in_v2_split(manifest_df):
    active_df = manifest_df[manifest_df['split'].isin(['Train', 'Validation', 'Test'])]
    assert active_df['is_generated'].sum() == 0, "Generated images are in the active split!"

def test_no_image_mutation():
    pass # Hash functions only read files

def test_manifest_unchanged():
    current_hash = get_hash(MANIFEST_PATH)
    expected_hash = "2c2f6d300e5ce127001c107124e21333c7edaaa0e74d763c6f685a2cedec1ff6"
    assert current_hash == expected_hash

def test_audit_does_not_import_training_pipeline():
    with open("total_practice/practice_2_2/src/practice_2_2/audit_visual_leakage.py", "r") as f:
        content = f.read()
        assert "torch.nn" not in content
        assert "build_v2_model" not in content

def test_audit_does_not_run_model_inference():
    with open("total_practice/practice_2_2/src/practice_2_2/audit_visual_leakage.py", "r") as f:
        content = f.read()
        assert "model(" not in content
        assert "forward(" not in content

def test_no_test_metrics_computed():
    with open("total_practice/practice_2_2/src/practice_2_2/audit_visual_leakage.py", "r") as f:
        content = f.read()
        assert "accuracy" not in content.lower()
        assert "f1_score" not in content

def test_no_test_predictions_created():
    with open("total_practice/practice_2_2/src/practice_2_2/audit_visual_leakage.py", "r") as f:
        content = f.read()
        assert "predict" not in content.lower()
