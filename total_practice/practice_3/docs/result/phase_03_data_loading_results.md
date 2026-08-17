# Phase 03 – Data Loading

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 08` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled DataFrame table titled `Dataset split summary` showing sample counts and usage flags for Train, Validation, and Holdout splits.

## Processing Source
- [`dataset_protocol_v2.py`](../../processing_own_phase/dataset_protocol_v2.py)
- Functions: `load_development_pool()`, `create_split_manifest()`, `materialize_train_validation_only()`

## Artifact Sources
- [`dataset_split_reference.json`](./practice_3_v2_3/dataset_split_reference.json)
- [`protocol_manifest.json`](./practice_3_v2_3/protocol_manifest.json)

## Results
| Split | Samples | Used for Training | Used for Selection (Val) | Used for Final Test (Holdout) |
|---|---:|:---:|:---:|:---:|
| **Train** | 7,676 | True | False | False |
| **Validation** | 960 | False | True | False |
| **Holdout** | 960 | False | False | True |
| **Total Development Pool** | 9,596 | — | — | — |
| **Official HF Test (Excluded)** | 1,066 | False | False | False |

## Evidence
- Dataset split summary table in Notebook [`Cell 08`](../../notebook_practice_3/practice_3.ipynb).
- Split metadata verified in [`dataset_split_reference.json`](./practice_3_v2_3/dataset_split_reference.json).
- Strict isolation enforced by [`dataset_protocol_v2.py`](../../processing_own_phase/dataset_protocol_v2.py) with zero cross-split overlap.

## Summary
The Rotten Tomatoes development pool (9,596 samples) is deterministically partitioned into Train (7,676, 80%), Validation (960, 10%), and Holdout (960, 10%) splits. The official Hugging Face test set is strictly excluded to maintain unbiased evaluation.
