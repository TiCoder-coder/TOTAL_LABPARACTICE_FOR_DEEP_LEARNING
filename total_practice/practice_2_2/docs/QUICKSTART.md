# Practice 2.2 — Submission Quickstart

## View the canonical report

```bash
cd total_practice/practice_2_2
jupyter notebook notebooks/04_canonical_report.ipynb
```

Use `Run All`. It is artifact-only and safe: no training and no Final Test evaluation.

## Verify the project

```bash
cd total_practice/practice_2_2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/matplotlib-cache \
  PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests
```

Canonical result: E2, Final Test Accuracy 76.36%, Macro F1 0.7617. See [`../README.md`](../README.md) for provenance, limitations and hashes. A static export is available at [`../reports/html/practice_2_2_canonical_report.html`](../reports/html/practice_2_2_canonical_report.html).
