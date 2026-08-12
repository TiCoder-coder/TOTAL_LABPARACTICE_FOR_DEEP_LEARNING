# Practice 2 Results by Phase

This index follows the current 30-cell
[`practice_2_presentation.ipynb`](../../notebooks/practice_2_presentation.ipynb).
Cell positions are 1-based; execution counts shown by Jupyter may differ.

| Phase | Notebook cells | Current primary result | Detail |
|---|---:|---|---|
| 1 — Problem Definition | 2 | Leakage-safe multiclass contract | [Phase 1](phase_01_results.md) |
| 2 — Environment | 3–4 | Runtime and MPS evidence | [Phase 2](phase_02_results.md) |
| 3 — Data Loading | 5–6 | 45k/5k/10k split | [Phase 3](phase_03_results.md) |
| 4 — EDA | 7–9 | RGB statistics, quality audit, PCA/t-SNE | [Phase 4](phase_04_results.md) |
| 5 — Preprocessing | 10–11 | Train-only stochastic transforms | [Phase 5](phase_05_results.md) |
| 6 — Model | 12–13 | ResNet18 parameter sanity checks | [Phase 6](phase_06_results.md) |
| 7 — Training | 14–15 | 25-epoch winner log and four-panel dashboard | [Phase 7](phase_07_results.md) |
| 8 — Hyperparameters | 16–17 | Three LR pairs and three heads | [Phase 8](phase_08_results.md) |
| 9 — Selection | 18–21 | Loss-selected SHA256 lock and PASS | [Phase 9](phase_09_results.md) |
| 10 — Final Test | 22–23 | 94.06% accuracy; per-class and ROC/PR | [Phase 10](phase_10_results.md) |
| 11 — Error Analysis | 24–28 | Confidence, confusion and galleries | [Phase 11](phase_11_results.md) |
| 12 — Conclusion | 29–30 | Reproducible locked result | [Phase 12](phase_12_results.md) |

## Current canonical artifacts

- [ranking](../../outputs/hyperparameter_ranking.csv)
- [locked selection](../../outputs/hyperparameter_selection_locked.json)
- [summary](../../outputs/summary.json)
- [Final Test receipt](../../outputs/final_test_receipt_a906600b717f.json)
- [training dashboard](../../reports/winner_training_log_dashboard.png)
- [ROC/PR dashboard](../../reports/roc_pr_curves_notebook.png)

Older E1/E2 files are baseline history, not the current final selection.
