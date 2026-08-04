# Practice 2.2 — Safe Inspection Guide

The canonical experiment and one-time Final Test are complete. This guide is for inspecting the submission, not regenerating results.

## Official presentation

```bash
cd total_practice/practice_2_2
jupyter notebook notebooks/04_canonical_report.ipynb
```

The static report is available at [`reports/html/practice_2_2_canonical_report.html`](reports/html/practice_2_2_canonical_report.html).

The official notebook is report-only. `Run All` reads existing JSON/CSV/PNG artifacts and does not:

- recreate the split,
- train E1/E2/E3/E4,
- create a Test DataLoader,
- repeat Final Test,
- overwrite canonical artifacts.

If an artifact is missing, the notebook raises `FileNotFoundError` with the missing paths. Do not regenerate Final Test.

## Canonical paths

```text
data/final/data_clean_balanced/
data/manifests/canonical_split/
artifacts/canonical/canonical_26dc4625_52aaf974_s42_v1/
artifacts/ablations/
```

E3/E4 directories are ablations only. Root-level output files are legacy/non-canonical; see [`LEGACY_ARTIFACTS.md`](LEGACY_ARTIFACTS.md).

## Automated tests

```bash
cd total_practice/practice_2_2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
```

Tests are safe: they validate code and persisted contracts without training or Final Test evaluation.

Default code imports `practice_2_2.*` from `src/`. The old dataset, manifest,
checkpoint, output, notebook and `processing_own_phase.*` paths are
**COMPATIBILITY / PRE-MIGRATION LOCATION** and are resolved only through the
explicit legacy registry; they are not runtime defaults.

## Historical data pipeline

The Tiki crawl and cleaning commands below are historical references only and do not reconstruct the canonical dataset exactly because per-image source URLs and historical balancing scripts were not retained:

```bash
cd total_practice/practice_2_2
python3 -m craw.crawl_tiki
python3 -m data_processing.pipeline
```

Running them creates a new dataset lineage; it must not be represented as the frozen canonical result.

## Locked Final Test

```text
FINAL_TEST_COMPLETED = true
final_test_evaluation_count = 1
repeat_evaluation_allowed = false
```

Do not invoke `final_test_practice_2_2.py` again. The script is fail-closed and refuses an existing `final_test/` directory.
