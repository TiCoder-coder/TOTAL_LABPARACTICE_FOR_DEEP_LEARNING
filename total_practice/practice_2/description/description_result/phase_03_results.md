# Phase 3 — Data Loading Results

[Phase 2 results](phase_02_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 4 results](phase_04_results.md)

| Cell output | Stored result |
|---|---|
| Cell 6 | Train samples: 45,000 |
| Cell 6 | Validation samples: 5,000 |
| Cell 6 | Official Test samples: 10,000 |
| Cell 6 | Train/Validation index intersection is empty |
| Cell 6 | Train has random augmentation; Validation/Test do not |
| Cell 6 | Validation is used for selection; Test is reserved for final evaluation |

The cell also explains that split indices are fixed before transformed dataset views are created and that TorchVision transforms execute lazily.
