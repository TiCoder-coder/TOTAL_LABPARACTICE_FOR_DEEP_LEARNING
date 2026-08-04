# Phase 3 — Source Ownership and Migration Map

Classification was completed before any Phase 3 source copy.

## Ownership classification

| Existing module | Classification | Phase 3 treatment | Reason |
|---|---|---|---|
| `data_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Canonical cosmetic dataset, split and leakage logic |
| `canonical_train_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Canonical E1/E2 contract and lineage generation |
| `canonical_e3_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Controlled E3 ablation |
| `canonical_e4_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Controlled E4 ablation |
| `final_test_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC, HIGH RISK | COPY unchanged; never execute entry point | Required guard/test API; guard logic must remain byte-identical |
| `analyze_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Practice 2.2 overfitting analysis |
| `audit_practice_2_2_data.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Cosmetic dataset audit and duplicate signatures |
| `regularization_practice_2_2.py` | A — PRACTICE_2_2_SPECIFIC | COPY | Practice 2.2 augmentation/EMA helpers |
| `model.py` | B — SHARED_GENERIC | WRAPPER/RE-EXPORT | Used by CIFAR-10 and Practice 2.2; old module remains source of truth |
| `train.py` | B — SHARED_GENERIC | WRAPPER/RE-EXPORT | Shared optimizer, scheduler, train and Validation utilities |
| `evaluate.py` | B — SHARED_GENERIC | WRAPPER/RE-EXPORT | Generic classification metrics |
| `save_load.py` | B — SHARED_GENERIC | WRAPPER/RE-EXPORT | Shared checkpoint serialization/loading |
| `utils.py` | B — SHARED_GENERIC | WRAPPER/RE-EXPORT | Shared device and reproducibility helpers |
| `logger.py` | B — SHARED_GENERIC | NOT COPIED | Not required by migrated Practice 2.2 modules/tests |
| `visualize.py` | C — PRACTICE_2_CIFAR_ONLY with reusable functions | NOT COPIED | Imports CIFAR configuration and contains CIFAR-specific presentation |
| `data.py` | C — PRACTICE_2_CIFAR_ONLY | NOT COPIED | Explicit CIFAR-10 dataset implementation |
| `experiment.py` | C — PRACTICE_2_CIFAR_ONLY | NOT COPIED | Uses Practice 2 CIFAR configs and artifacts |
| `final_evaluate.py` | C — PRACTICE_2_CIFAR_ONLY | NOT COPIED | Loads official CIFAR-10 Test dataset |
| `main.py` | C — PRACTICE_2_CIFAR_ONLY | NOT COPIED | Legacy CIFAR full-pipeline entry point |
| `train_practice_2_2.py` | D — LEGACY_OR_REVIEW | COMPATIBILITY WRAPPER only | Required by a copied legacy-contract unit test; implementation remains old source of truth |
| `train_practice_2_2_improved.py` | D — LEGACY_OR_REVIEW, HIGH RISK | NOT COPIED | Non-canonical improved-training path; creates Test dataset internally |
| `promote_practice_2_2.py` | D — LEGACY_OR_REVIEW, HIGH RISK | NOT COPIED | Mutates/promotes run, output and report artifacts |
| `__init__.py` | B — PACKAGE INFRASTRUCTURE | REPLACED by compatibility package init | Enables old and new import paths without moving old source |

## Phase 3 strategy

- Practice 2.2-specific modules are copied byte-for-byte.
- Shared modules are not duplicated; thin re-export wrappers delegate to `practice_2/processing_own_phase/`.
- The package initializer discovers the sibling `practice_2` directory and adds it to the import search path. It does not change runtime artifact paths.
- Old source remains authoritative during Phase 3; ownership is not switched.
- High-risk entry points are never executed.

## Source mapping

The old → new path and SHA-256 verification table is recorded after copying in `phase_3_hashes.json` and summarized in `PHASE_3_VERIFICATION.md`.
