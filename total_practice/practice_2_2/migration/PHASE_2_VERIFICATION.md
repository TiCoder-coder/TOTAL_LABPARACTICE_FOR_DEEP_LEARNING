# Phase 2 — Presentation Copy Verification

## Copy verification

| Item | SHA-256 | Result |
|---|---|---|
| Source notebook `../practice_2/notebooks/practice_2_2_canonical_report.ipynb` | `6aa73abd6933df964091ff44ebc620abba5ad31a9626393480fe295a24dc7370` | PASS |
| Copied notebook `notebooks/04_canonical_report.ipynb` | `6aa73abd6933df964091ff44ebc620abba5ad31a9626393480fe295a24dc7370` | BYTE-IDENTICAL |
| Source HTML | `0a5146ebc9c8fb4ace3fa0743de50f1ca317af467bdf4fa466fb2ac29dc8d629` | PASS |
| Copied HTML | `0a5146ebc9c8fb4ace3fa0743de50f1ca317af467bdf4fa466fb2ac29dc8d629` | BYTE-IDENTICAL |

## Notebook safety

- Metadata mode: `canonical_report_only`
- Training allowed: `false`
- Final Test evaluation allowed: `false`
- `train_model`: absent
- `optimizer.step`: absent
- `scheduler.step`: absent
- `allow_test=True`: absent
- `create_test_dataset`: absent
- `run_final_test`: absent

## Temporary Run All

The notebook was copied to `/private/tmp/practice_2_2_phase2_runall.ipynb` and executed there with the repository supplied only as the read-only artifact working directory.

- Result: PASS
- Cell errors: 0
- Canonical artifact overwrite: none
- Training: not performed
- Test DataLoader: not created
- Final Test inference: not performed

## Integrity after Run All

| Contract | Result |
|---|---|
| Canonical E2 checkpoint SHA-256 unchanged | PASS |
| Canonical split manifest SHA-256 unchanged | PASS |
| `final_selection.json` unchanged | PASS |
| `final_test_summary.json` unchanged | PASS |
| `FINAL_TEST_COMPLETED.json` unchanged | PASS |
| Dataset fingerprint unchanged | PASS |
| Split fingerprint unchanged | PASS |

## Automated tests

```text
tests/test_submission_practice_2_2.py
tests/test_final_test_practice_2_2.py
tests/test_canonical_train_practice_2_2.py

10 passed in 3.33s
```

These tests did not train a model or evaluate Final Test.
