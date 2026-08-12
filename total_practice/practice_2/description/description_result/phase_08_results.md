# Phase 8 — Controlled Experiment Results

[Phase 7 results](phase_07_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 9 results](phase_09_results.md)

| Cell output | Stored result |
|---|---|
| Cell 17 | E1 strategy is `head_only` |
| Cell 17 | E2 strategy is `partial_finetune` |
| Cell 17 | Backbone, split, seed, optimizer, LR, scheduler, batch size, augmentation, loss, epoch budget, and patience match |
| Cell 17 | Assertions confirm that fine-tuning strategy is the only controlled difference |

Configuration source: [experiment_config.py](../../configs/experiment_config.py).
## Hyperparameter search result

Three head/backbone learning-rate pairs and three classifier heads were compared
using Validation only. The winning configuration is ResNet18 partial fine-tune,
head LR `1e-3`, backbone LR `1e-4`, and a linear classifier with no hidden
layer. Its Validation loss `0.43036` is lower than the medium-LR result
`0.44074`, low-LR result `0.46263`, two-hidden-layer result `0.46720`, and
one-hidden-layer result `0.48979`.

The corrected ranking is stored in `outputs/hyperparameter_ranking.csv`.
