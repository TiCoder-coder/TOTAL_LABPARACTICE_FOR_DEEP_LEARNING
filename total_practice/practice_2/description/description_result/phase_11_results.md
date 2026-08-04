# Phase 11 — Error Analysis Results

[Phase 10 results](phase_10_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 12 results](phase_12_results.md)

| Output | Description | Artifact |
|---|---|---|
| Cell 25 | Confidence distribution for prediction analysis | [confidence_distribution.png](../../reports/confidence_distribution.png) |
| Cell 25 | Highest-confidence correct predictions | [prediction_gallery_correct.png](../../reports/prediction_gallery_correct.png) |
| Cell 25 | Highest-confidence incorrect predictions | [prediction_gallery_incorrect.png](../../reports/prediction_gallery_incorrect.png) |
| Cell 25 | Six challenging correct cases and six confident mistakes | [prediction_grid_mixed.png](../../reports/prediction_grid_mixed.png) |
| Cell 26 | Artifact-consistency status | [summary.json](../../outputs/summary.json) |
| Cell 27 | Bias–variance and generalization diagnostics | [Validation–Test comparison](../../reports/validation_test_comparison_current.png) |
| Cell 28 | Interpretability limitation and Grad-CAM future-work note | [Notebook](../../notebooks/practice_2_presentation.ipynb) |

The largest error directions include cat→dog, dog→cat, bird→cat, airplane→ship, and horse→dog.
