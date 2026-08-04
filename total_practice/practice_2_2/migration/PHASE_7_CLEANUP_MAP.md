# Phase 7 — Final Ownership Classification and Cleanup Map

Classification was completed before any removal or archive action.

| Current Practice 2 scope | Class | Planned treatment | Evidence / reason |
|---|---|---|---|
| `notebooks/practice_2_2.ipynb` | ARCHIVE_FIRST | Archive | Same SHA as the other old report; new executed canonical notebook exists |
| `notebooks/practice_2_2_canonical_report.ipynb` | ARCHIVE_FIRST | Archive | Old-path report, SHA recorded; superseded by active `04_canonical_report.ipynb` |
| `notebooks/practice_2_2_canonical_report.html` | ARCHIVE_FIRST | Archive | Superseded by canonical HTML under Practice 2.2 |
| `notebooks/practice_2_presentation.ipynb` | CIFAR_ONLY_KEEP | Keep | CIFAR-10 presentation; explicitly protected |
| `processing_own_phase/*practice_2_2*.py`, canonical E3/E4/Final Test modules | KEEP_COMPATIBILITY_WRAPPER | Archive implementation, replace with thin forwarding wrapper where a canonical module exists | New package is active; old import remains regression-compatible |
| `processing_own_phase/promote_practice_2_2.py`, `train_practice_2_2_improved.py` | ARCHIVE_FIRST | Archive, no active wrapper | Historical/non-canonical utility, no new-runtime dependency |
| `processing_own_phase/data.py`, `model.py`, `train.py`, `evaluate.py`, `save_load.py`, `utils.py`, `experiment.py`, `final_evaluate.py`, `logger.py`, `visualize.py`, `main.py` | SHARED_GENERIC_KEEP | Keep | Used by CIFAR-10 and compatibility adapters |
| `tests/test_*practice_2_2.py`, canonical E3/E4/Final Test tests | ARCHIVE_FIRST | Archive | Equivalent canonical tests exist and pass under Practice 2.2 |
| Other `tests/test_*.py` | CIFAR_ONLY_KEEP | Keep | CIFAR-10/shared regression coverage |
| `data/data_clean_balanced/` (3,202 files) | SAFE_TO_REMOVE | Remove after digest/fingerprint/dependency gates | Byte-identical duplicate of active dataset; directory digest `591a732e...24e3` |
| `data/cifar-10-*` | CIFAR_ONLY_KEEP | Keep | Practice 2 dataset |
| `outputs/practice_2_2/` | ARCHIVE_FIRST | Archive | Legacy Practice 2.2 output collection |
| Root `outputs/*` | CIFAR_ONLY_KEEP | Keep | Referenced by Practice 2 CIFAR documentation/runtime |
| `reports/practice_2_2/` | ARCHIVE_FIRST | Archive | Legacy Practice 2.2 visual reports |
| `runs/practice_2_2/` | ARCHIVE_FIRST | Archive | Legacy non-canonical E1/E2 checkpoints; canonical checkpoints already verified under Practice 2.2 |
| Root `reports/*` | CIFAR_ONLY_KEEP | Keep | CIFAR report set; ambiguous names are referenced by CIFAR notebook/docs |
| `tools/build_practice_2_2_canonical_notebook.py` | ARCHIVE_FIRST | Archive | Superseded old-path notebook builder |
| `configs/experiment_config.py` | SHARED_GENERIC_KEEP | Keep, clean misleading comment only | Active Practice 2 experiment configuration |
| Cache files/directories | SAFE_TO_REMOVE | Remove | Generated, non-authoritative runtime debris |

## Notebook verification before archive

- Both old IPYNB files: `6aa73abd6933df964091ff44ebc620abba5ad31a9626393480fe295a24dc7370`.
- Active new IPYNB: `2784779b136977df803cd479ec088ab88b14aa1b3cb1ba69e126da887ffab972`.
- Old HTML: `0a5146ebc9c8fb4ace3fa0743de50f1ca317af467bdf4fa466fb2ac29dc8d629`.
- Active new HTML: `da644d4872d56ea01d7b77afba84df539b1c3d84b2b4726113bbae06bf7231a2`.

The new notebook differs because Phase 6 switched it to the active registry and
successfully executed all 19 code cells. Documentation no longer uses the old
copies as canonical authority.

## Dataset removal gates

- Old duplicate: 3,202 files, 48,502,262 bytes.
- Active dataset: 3,202 files, 48,502,262 bytes.
- Deterministic directory digests match:
  `591a732e84adad01354799a55d40d7410725c00f322f909eba499315d5dd24e3`.
- Canonical dataset fingerprint matches expected.
- Active registry resolves only `data/final/data_clean_balanced/`.
- Any remaining legacy dataset fallback will be retired before removal.

No item was classified `UNKNOWN_REVIEW`; ambiguous generic root outputs/reports
were conservatively classified CIFAR_ONLY_KEEP rather than deleted.

## Practice 2.2 pre-migration layout classification

| Current scope | Class | Planned treatment | Reason |
|---|---|---|---|
| `data_clean_balanced/` | ARCHIVE_FIRST | Move to `archive/phase_7/pre_migration_practice_2_2/` | Byte-identical predecessor of active `data/final/` |
| `outputs/practice_2_2/` | ARCHIVE_FIRST | Move to Phase 7 archive | Canonical evidence is verified under `artifacts/`; remaining files are historical |
| `runs/practice_2_2/` | ARCHIVE_FIRST | Move to Phase 7 archive | Checkpoints are verified under canonical/ablation artifacts |
| `reports/practice_2_2/` | ARCHIVE_FIRST | Move to Phase 7 archive | Superseded by canonical HTML and persisted artifact figures |
| `craw/`, `data_processing/`, `data_clean/` | SHARED_GENERIC_KEEP | Keep | Practice 2.2 provenance/pipeline ownership, not duplicate authority |

The explicit legacy registry will be updated to the archive locations. Active
registry paths remain unchanged. This removes duplicate top-level authority
shapes without deleting historical evidence.
