# Phase 0 — Frozen Repository Inventory

This record predates the Phase 1 layout and Phase 2 presentation copies. It is independent of the immutable canonical lineage.

## Migration policy

`COPY → VERIFY → SWITCH LATER → ARCHIVE LATER`

No canonical data, manifest, checkpoint, lineage file or Final Test artifact may be moved, renamed, overwritten or deleted during Phases 0–2.

## Git status before migration

The worktree was already dirty before this migration. The pre-existing status was:

```text
 M total_practice/practice_2/configs/experiment_config.py
 M total_practice/practice_2/notebooks/practice_2_2.ipynb
 D total_practice/practice_2/notebooks/practice_2_2_preprocessing.ipynb
 M total_practice/practice_2/processing_own_phase/data_practice_2_2.py
 M total_practice/practice_2/processing_own_phase/model.py
 M total_practice/practice_2/tests/test_data_practice_2_2.py
 M total_practice/practice_2_2/HOW_TO_RUN.md
 M total_practice/practice_2_2/QUICKSTART.md
 M total_practice/practice_2_2/README.md
?? total_practice/practice_2/notebooks/practice_2_2_canonical_report.html
?? total_practice/practice_2/notebooks/practice_2_2_canonical_report.ipynb
?? total_practice/practice_2/processing_own_phase/canonical_e3_practice_2_2.py
?? total_practice/practice_2/processing_own_phase/canonical_e4_practice_2_2.py
?? total_practice/practice_2/processing_own_phase/canonical_train_practice_2_2.py
?? total_practice/practice_2/processing_own_phase/final_test_practice_2_2.py
?? total_practice/practice_2/tests/test_canonical_e3_practice_2_2.py
?? total_practice/practice_2/tests/test_canonical_e4_practice_2_2.py
?? total_practice/practice_2/tests/test_canonical_train_practice_2_2.py
?? total_practice/practice_2/tests/test_final_test_practice_2_2.py
?? total_practice/practice_2/tests/test_submission_practice_2_2.py
?? total_practice/practice_2/tools/
?? total_practice/practice_2_2/LEGACY_ARTIFACTS.md
?? total_practice/practice_2_2/SUBMISSION_AUDIT.md
?? total_practice/practice_2_2/reports/
```

## Inventory groups

| Group | Current authoritative location | Inventory summary | Migration status |
|---|---|---:|---|
| Notebooks | `../practice_2/notebooks/` | 3 IPYNB, 1 HTML | Phase 2 copies presentation only |
| Documentation | project root, `craw/`, `data_processing/` | 7 Markdown files at Phase 0 | Phase 2 copies selected docs |
| Reports | `../practice_2/reports/`, `reports/practice_2_2/` | 25 files across both projects | HTML only copied in Phase 2 |
| Canonical artifacts | `outputs/practice_2_2/canonical_26dc4625_52aaf974_s42_v1/` | 16 files | Frozen; not copied in Phase 0–2 |
| Checkpoints | `runs/` trees across both projects | 21 `.pt` files | Frozen; not copied in Phase 0–2 |
| Cosmetic dataset | `data_clean_balanced/` | 3,202 JPEG files | Frozen canonical input |
| Duplicate cosmetic dataset | `../practice_2/data/data_clean_balanced/` | 3,202 byte-identical JPEG files | Retained; no deletion |
| Canonical manifests | `outputs/practice_2_2/canonical_split/` | 3 files | Frozen; not copied in Phase 0–2 |
| Practice 2.2 source | `../practice_2/processing_own_phase/`, `craw/`, `data_processing/` | 37 Python files in audited scope | Phase 3 or later |
| Tests | `../practice_2/tests/` | 19 Python files | Phase 3 or later |

## Canonical identity

- Dataset fingerprint: `26dc4625f96c7cb86bc6a4df0fbed34bc8b22fb6472ef0d008df341f99f71fd6`
- Split fingerprint: `52aaf97499ede5dd4689c2bb36dc0679fe04b8ec8139aae7e8fdcde88042043d`
- Canonical E2 checkpoint: `4f65bec200d023c51f95493833d057029b83a92729c3b88dd9159158dd949ae3`

Exact Phase 0 hashes are recorded in `phase_0_hashes.json`.
