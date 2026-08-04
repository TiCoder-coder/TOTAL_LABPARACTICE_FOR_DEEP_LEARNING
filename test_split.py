import pandas as pd
df = pd.read_csv("total_practice/practice_2_2/data/manifests/canonical_split/split_manifest.csv")
df = df[(df['is_generated'] == False) & (df['quarantined'] == False) & (df['use_for_model'] == True)].copy()
from total_practice.practice_2_2.src.practice_2_2.split_v2_pipeline import create_effective_groups, assign_splits
df = create_effective_groups(df)
df = assign_splits(df)
print(df['split'].value_counts())
