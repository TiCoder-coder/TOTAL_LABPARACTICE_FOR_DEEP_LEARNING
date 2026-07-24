# Phase 7 — Visualization Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/visualize.py` — `plot_data_samples`, `plot_class_distribution`, `plot_loss_curves`, `plot_accuracy_curves`, `plot_experiment_comparison`, `plot_predictions_grid`, `plot_confusion_matrix`

## Plots Generated

| File                              | Description |
|-----------------------------------|-------------|
| `outputs/data_samples.png`        | 20 sample images |
| `outputs/class_distribution.png`  | Bar chart of 10 classes per split |
| `outputs/loss_curve.png`          | Train vs Val loss for E5_adam |
| `outputs/accuracy_curve.png`      | Train vs Val accuracy for E5_adam |
| `outputs/experiment_comparison.png` | Bar chart of best val acc per experiment |
| `outputs/predictions_grid.png`    | 16 test images with true vs predicted (green=correct, red=wrong) |
| `outputs/confusion_matrix.png`    | Normalized 10x10 confusion matrix |

## Notebook

Created `practice_1.ipynb` with the following sections:
1. Title and objectives
2. Environment setup
3. Run full pipeline
4. Visualizations
5. Experiment results table
6. Summary report

## Outcome

All visualizations generated. Notebook summarizes the full pipeline.
