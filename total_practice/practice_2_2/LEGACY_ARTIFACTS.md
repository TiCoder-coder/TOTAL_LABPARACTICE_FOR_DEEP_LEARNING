# Legacy and Non-Canonical Artifacts

## Reporting rule

Only this lineage may be reported as the canonical result:

```text
artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/
```

The E3 and E4 directories are controlled ablation lineages, not improved or selected models:

```text
canonical_26dc4625_52aaf974_s42_phase25_e3_layer4_1_v1/
canonical_26dc4625_52aaf974_s42_phase26_e4_augmentation_v1/
```

The pre-migration dataset, outputs and runs inside this project are archived
under `archive/phase_7/pre_migration_practice_2_2/` as explicit read-only
compatibility fallbacks. Old Practice 2-owned files were consolidated under
`archive/phase_7/practice_2/`; only documented thin import wrappers remain in
`practice_2/processing_own_phase/`. All root-level files directly
under old `outputs/practice_2_2/`, including `summary.json`, `predictions.csv`,
`classification_report.csv`, `controlled_experiment_*`, `preprocessing_*` and
`validation_verification.json`, are:

> **LEGACY / NON-CANONICAL / DO NOT REPORT**

They are retained only for historical traceability. Metrics such as 78.57% or 81.80% from older runs must not be mixed with the canonical result.

The notebooks `practice_2_2_preprocessing.ipynb` and `practice_2_presentation.ipynb` are historical development notebooks. They are not the official submission presentation and must not be used to regenerate or select the canonical result. The canonical report-only notebook is [`notebooks/04_canonical_report.ipynb`](notebooks/04_canonical_report.ipynb), with a static export at [`reports/html/practice_2_2_canonical_report.html`](reports/html/practice_2_2_canonical_report.html).

No canonical artifact was deleted. The verified redundant dataset under
`practice_2/data/data_clean_balanced/` was removed in Phase 7.
