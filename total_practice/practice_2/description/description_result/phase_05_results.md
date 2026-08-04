# Phase 5 — Data Preprocessing Results

[Phase 4 results](phase_04_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 6 results](phase_06_results.md)

| Cell output | Stored result |
|---|---|
| Cell 11 | Train transform contains Resize, RandomCrop, RandomHorizontalFlip, ColorJitter, ToTensor, and ImageNet Normalize |
| Cell 11 | Validation/Test transform contains Resize, CenterCrop, ToTensor, and ImageNet Normalize |
| Cell 11 | Validation and Test use equivalent deterministic transform policies |
| Cell 11 | Train and Validation use separate backing dataset objects |
| Cell 11 | Random augmentation is absent from Validation and Test |

The output is a colored audit table with assertions. It is stored directly in the notebook rather than as a separate PNG.
