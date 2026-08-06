# Notebooks

`04_canonical_report.ipynb` remains the frozen artifact-only presentation. It cannot train, construct a Test DataLoader, or repeat Final Test. The E2 comparison row uses the persisted epoch-14 Validation loss `1.0218` and Macro F1 `0.7821`.

`05_accuracy_refactor.ipynb` is the non-canonical Train/Validation development workflow. It preflights manifest authorization, compares controlled staged-transfer configurations across repeated seeds, and selects soft-voting/TTA inference only from Validation evidence. Training is disabled by default, and the notebook has no Final Test data path.
