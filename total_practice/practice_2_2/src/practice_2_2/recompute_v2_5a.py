import pandas as pd
import json

df = pd.read_csv("total_practice/practice_2_2/artifacts/NEAR_DUPLICATE_REVIEW_REQUIRED.csv")

total = len(df)
critical = len(df[df['risk_level'] == 'CRITICAL'])
high = len(df[df['risk_level'] == 'HIGH'])
medium = len(df[df['risk_level'] == 'MEDIUM'])
ssim_95 = len(df[df['secondary_similarity'] >= 0.95])
cross_label = len(df[df['label_a'] != df['label_b']])
same_label = len(df[df['label_a'] == df['label_b']])

def get_split_pair(a, b):
    pair = sorted([a, b])
    return f"{pair[0]} ↔ {pair[1]}"

split_pairs = df.apply(lambda row: get_split_pair(row['split_a'], row['split_b']), axis=1).value_counts().to_dict()

out = {
    "total_candidate_pairs": total,
    "CRITICAL": critical,
    "HIGH": high,
    "MEDIUM": medium,
    "SSIM_gte_0.95": ssim_95,
    "same_label_candidate_pairs": same_label,
    "cross_label_POSSIBLE_LABEL_CONFLICT_pairs": cross_label,
    "split_pairs": split_pairs
}

with open("total_practice/practice_2_2/artifacts/V2_5A_CORRECTED_AUDIT_SUMMARY.json", "w") as f:
    json.dump(out, f, indent=4)
    
print(json.dumps(out, indent=4))
