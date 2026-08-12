# Phase 12 Results — Reproducibility and Conclusion

The final selected configuration is ResNet18 `partial_finetune`, head LR
`1e-3`, backbone LR `1e-4`, and a linear classifier. Epoch 24 was selected by
minimum Validation loss (`0.430360`) and verified at `94.42%` Validation
accuracy before Test access.

The SHA256-locked checkpoint achieved `94.06%` Test Accuracy and `0.940458`
Macro F1 on 10,000 images. The Final Test receipt records one evaluation.

Canonical entry points:

- [official notebook](../../notebooks/practice_2_presentation.ipynb)
- [ranking](../../outputs/hyperparameter_ranking.csv)
- [locked selection](../../outputs/hyperparameter_selection_locked.json)
- [final summary](../../outputs/summary.json)
- [training dashboard](../../reports/winner_training_log_dashboard.png)
- [ROC/PR dashboard](../../reports/roc_pr_curves_notebook.png)
