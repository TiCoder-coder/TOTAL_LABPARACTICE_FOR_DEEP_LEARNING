import os
import pandas as pd
from datasets import load_dataset

def test_data_prep(demo_sample_size=1600):
    print("Loading original TRAIN split...")
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    assert len(dataset) == 8530, f"Expected 8530, got {len(dataset)}"
    print("Dataset loads: PASS")
    
    df = dataset.to_pandas()
    pos_df = df[df["label"] == 1].sample(n=demo_sample_size // 2, random_state=42)
    neg_df = df[df["label"] == 0].sample(n=demo_sample_size // 2, random_state=42)
    print(f"pos_df rows: {len(pos_df)}")
    print(f"neg_df rows: {len(neg_df)}")
    assert len(pos_df) == 800, f"Expected 800 pos_df, got {len(pos_df)}"
    assert len(neg_df) == 800, f"Expected 800 neg_df, got {len(neg_df)}"
    
    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"balanced_df rows: {len(balanced_df)}")
    assert len(balanced_df) == 1600, f"Expected 1600 balanced_df, got {len(balanced_df)}"
    
    c0 = (balanced_df['label'] == 0).sum()
    c1 = (balanced_df['label'] == 1).sum()
    print(f"Class 0: {c0}")
    print(f"Class 1: {c1}")
    assert c0 == 800, f"Expected 800 class 0, got {c0}"
    assert c1 == 800, f"Expected 800 class 1, got {c1}"
    print("ALL BALANCED_DF CHECKS PASSED.")

if __name__ == "__main__":
    test_data_prep()
