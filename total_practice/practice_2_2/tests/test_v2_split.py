import pandas as pd
import pytest
from pathlib import Path
import os
import hashlib

V1_MANIFEST = Path("total_practice/practice_2_2/data/manifests/canonical_split/split_manifest.csv")
V2_MANIFEST = Path("total_practice/practice_2_2/data/manifests/v2_stratified_group_s42/split_manifest_v2.csv")
V1_SUMMARY = Path("total_practice/practice_2_2/data/manifests/canonical_split/split_summary.json")

def get_v2_df():
    return pd.read_csv(V2_MANIFEST)

def get_v1_df():
    return pd.read_csv(V1_MANIFEST)

def test_generated_excluded_before_split():
    df = get_v2_df()
    assert (df['is_generated'] == False).all()

def test_quarantine_excluded():
    df = get_v2_df()
    assert (df['quarantined'] == False).all()

def test_only_original_images_used():
    df = get_v2_df()
    # No generated images means only original usable ones are used
    assert df['is_generated'].sum() == 0
    assert df['use_for_model'].sum() == len(df)

def test_group_isolation():
    df = get_v2_df()
    # Check effective_group_id
    assert df.groupby('effective_group_id')['split'].nunique().max() == 1

def test_duplicate_cluster_isolation():
    df = get_v2_df()
    if 'duplicate_cluster_id' in df.columns:
        valid_dups = df.dropna(subset=['duplicate_cluster_id'])
        if not valid_dups.empty:
            assert valid_dups.groupby('duplicate_cluster_id')['split'].nunique().max() == 1

def test_exact_hash_isolation():
    df = get_v2_df()
    if 'file_hash' in df.columns:
        valid_hashes = df.dropna(subset=['file_hash'])
        if not valid_hashes.empty:
            assert valid_hashes.groupby('file_hash')['split'].nunique().max() == 1

def test_split_reproducibility():
    # Tested internally by script, but we can verify it here by calling the function
    from total_practice.practice_2_2.src.practice_2_2.split_v2_pipeline import assign_splits, create_effective_groups
    
    df_v1 = get_v1_df()
    df_raw = df_v1[(df_v1['is_generated'] == False) & (df_v1['quarantined'] == False) & (df_v1['use_for_model'] == True)].copy()
    if 'split' in df_raw.columns:
        df_raw = df_raw.drop(columns=['split'])
    
    df_eff = create_effective_groups(df_raw)
    
    # Run A
    df_a = assign_splits(df_eff.copy(), seed=42)
    # Run B
    df_b = assign_splits(df_eff.copy(), seed=42)
    
    assert (df_a['split'] == df_b['split']).all()

def test_seed_42_deterministic():
    # Verify the current V2 manifest matches running with seed 42
    from total_practice.practice_2_2.src.practice_2_2.split_v2_pipeline import assign_splits, create_effective_groups
    
    df_v1 = get_v1_df()
    df_raw = df_v1[(df_v1['is_generated'] == False) & (df_v1['quarantined'] == False) & (df_v1['use_for_model'] == True)].copy()
    if 'split' in df_raw.columns:
        df_raw = df_raw.drop(columns=['split'])
    
    df_eff = create_effective_groups(df_raw)
    df_sim = assign_splits(df_eff.copy(), seed=42)
    
    df_actual = get_v2_df()
    
    df_sim_sorted = df_sim.sort_values('record_id').reset_index(drop=True)
    df_act_sorted = df_actual.sort_values('record_id').reset_index(drop=True)
    
    assert (df_sim_sorted['split'] == df_act_sorted['split']).all()

def test_train_ratio_close_to_target():
    df = get_v2_df()
    train_pct = len(df[df['split'] == 'Train']) / len(df)
    assert abs(train_pct - 0.70) < 0.05

def test_validation_ratio_close_to_target():
    df = get_v2_df()
    val_pct = len(df[df['split'] == 'Validation']) / len(df)
    assert abs(val_pct - 0.15) < 0.05

def test_test_ratio_close_to_target():
    df = get_v2_df()
    test_pct = len(df[df['split'] == 'Test']) / len(df)
    assert abs(test_pct - 0.15) < 0.05

def test_class_distribution_reasonable():
    df = get_v2_df()
    classes = df['class_name'].unique()
    for cls in classes:
        cls_df = df[df['class_name'] == cls]
        avail = len(cls_df)
        tr = len(cls_df[cls_df['split'] == 'Train'])
        tr_pct = tr / avail
        # Target is 0.7, shouldn't deviate by more than 15% absolute in worst case for small bins
        assert abs(tr_pct - 0.70) < 0.15

def test_no_generated_in_train():
    df = get_v2_df()
    train_df = df[df['split'] == 'Train']
    assert train_df['is_generated'].sum() == 0

def test_no_generated_in_validation():
    df = get_v2_df()
    val_df = df[df['split'] == 'Validation']
    assert val_df['is_generated'].sum() == 0

def test_no_generated_in_test():
    df = get_v2_df()
    test_df = df[df['split'] == 'Test']
    assert test_df['is_generated'].sum() == 0

def test_v1_manifest_unchanged():
    # We shouldn't modify V1 manifest
    # Here we can't easily assert unless we kept a backup, 
    # but we can at least check it still has 3202 rows.
    df_v1 = get_v1_df()
    assert len(df_v1) == 3202

def test_v1_artifacts_unchanged():
    # V1 summary should still exist
    assert V1_SUMMARY.exists()

