# Phase 04 – Exploratory Data Analysis

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 07` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Styled table `Rotten Tomatoes Dataset Summary` followed by the self-contained Clean Academic HTML EDA Dashboard displaying split overview, class distribution progress bars, text length statistics table, and sample review cards.

## Processing Source
- [`eda_visualization.py`](../../processing_own_phase/eda_visualization.py)
- Function: `render_eda_dashboard()`
- [`dataset_protocol_v2.py`](../../processing_own_phase/dataset_protocol_v2.py)
- Function: `load_development_pool()`

## Artifact Sources
- [`eda_visualization.py`](../../processing_own_phase/eda_visualization.py) (direct HTML generator)
- [`dataset_split_reference.json`](./practice_3_v2_3/dataset_split_reference.json)

## Results
### Dataset Overview & Class Distribution
| Split | Samples | Percentage | Negative (Class 0) | Positive (Class 1) |
|---|---:|---:|---:|---:|
| **Train** | 7,676 | 80.0% | 3,838 (50.0%) | 3,838 (50.0%) |
| **Validation** | 960 | 10.0% | 480 (50.0%) | 480 (50.0%) |
| **Holdout** | 960 | 10.0% | 480 (50.0%) | 480 (50.0%) |
| **Total Dataset** | **9,596** | **100.0%** | **4,798 (50.0%)** | **4,798 (50.0%)** |

### Text Length Statistics
| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| **Characters** | 114.0 | 111.0 | 4 | 267 |
| **Words** | 21.0 | 20.0 | 1 | 59 |

- **Maximum token length bound**: `80 tokens` (covers 100% of samples without truncating sentiment content).

### Sample Reviews
- **Negative (Class 0)**: *"simplistic , silly and tedious ."* (32 characters · 6 words)
- **Positive (Class 1)**: *"the rock is destined to be the 21st century's new \" conan \" and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal ."* (177 characters · 34 words)

## Evidence
- `Rotten Tomatoes Dataset Summary` table rendered in Notebook [`Cell 07`](../../notebook_practice_3/practice_3.ipynb).
- Clean Academic HTML EDA Dashboard rendered directly via `render_eda_dashboard()` in Notebook [`Cell 07`](../../notebook_practice_3/practice_3.ipynb).

## Summary
Exploratory Data Analysis verifies a perfectly balanced binary dataset (50% Negative / 50% Positive across all splits). Text lengths average 114 characters (21 words), confirming that a max sequence length of 80 tokens comfortably encompasses all reviews.
