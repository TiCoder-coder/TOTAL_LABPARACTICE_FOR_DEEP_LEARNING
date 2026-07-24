# Phase 2 — Data Loading Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/data.py` — `load_datasets`, `split_train_val`, `make_dataloaders`, `get_class_distribution`, `compute_mean_std`, `sanity_check_sample`

## Transforms Applied

```python
transforms.Compose([
    transforms.ToTensor(),  # PIL -> float32 tensor in [0, 1]
])
```

## Dataset Sizes (after random_split with seed=42)

| Split      | Samples |
|------------|---------|
| Train      | 54,001  |
| Validation |  5,999  |
| Test       | 10,000  |

> Note: 60,000 split 90/10 = 54,000 / 6,000, but `random_split` rounded to 54,001 / 5,999. Both splits are valid.

## Sample Sanity Check

```text
Shape : (1, 28, 28)
Dtype : torch.float32
Min   : 0.000
Max   : 1.000
Label : Ankle boot (class 9)
```

## DataLoader Config

| Loader     | batch_size | shuffle | num_workers |
|------------|------------|---------|-------------|
| Train      | 64         | True    | 0           |
| Validation | 64         | False   | 0           |
| Test       | 64         | False   | 0           |

## Class Distribution

Saved bar chart to `outputs/class_distribution.png`. All 10 classes are present in train/val/test splits.

## Visualizations Saved

- `outputs/data_samples.png` — 20 sample images
- `outputs/class_distribution.png` — bar chart of class counts per split

## Outcome

Data is loaded correctly. Transforms work. Sanity checks pass.
