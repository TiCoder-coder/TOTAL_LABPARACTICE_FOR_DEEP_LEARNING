# Phase 3 — Source and Test Copy Verification

## Source verification

- 8 Practice 2.2-specific modules copied.
- All 8 copies are SHA-256 byte-identical to their old source.
- `final_test_practice_2_2.py` guard implementation is byte-identical and its entry point was not executed.
- Shared `model`, `train`, `evaluate`, `save_load` and `utils` modules are re-export wrappers; `practice_2/processing_own_phase` remains the source of truth.
- Legacy `train_practice_2_2` is exposed through a compatibility wrapper only.
- `train_practice_2_2_improved.py` and `promote_practice_2_2.py` were not copied.

## Test verification

- 10 Practice 2.2 test files copied.
- Assertions and expected canonical hashes were preserved.
- Mechanical changes: `processing_own_phase` imports became `practice_2_2` imports.
- Submission test notebook path changed only to `notebooks/04_canonical_report.ipynb`.

## Results

| Verification | Result |
|---|---|
| New import smoke test | PASS |
| Old import smoke test | PASS |
| New-layout Practice 2.2 tests | 39 passed |
| Full old Practice 2 regression suite | 87 passed, 3 deprecation warnings |
| Canonical notebook report-only safety | PASS through submission tests |
| Checkpoint reload Validation | PASS |
| Reloaded Validation Accuracy | 78.31050228310502% |
| Accuracy delta | 0.0 |
| Reloaded Validation Loss | 1.021770380947688 |
| Loss delta | 0.0 |
| Reloaded Validation Macro F1 | 0.7820960879325867 |
| Test DataLoader constructed | false |
| Final Test evaluated | false |

## Compatibility contract

- Old imports remain functional: `processing_own_phase.*`.
- New imports are functional when `practice_2_2/src` is on `PYTHONPATH`: `practice_2_2.*`.
- Runtime artifact paths and persisted canonical JSON were not changed.
- No canonical entry point has been switched.
