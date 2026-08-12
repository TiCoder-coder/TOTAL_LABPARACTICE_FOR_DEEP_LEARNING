# Phase 8 Results — Hyperparameter Search

Notebook cells 16–17 present two Validation-only tables: three learning-rate
pairs and three classifier heads. The complete ranking is stored in
[`hyperparameter_ranking.csv`](../../outputs/hyperparameter_ranking.csv).

The winner uses head LR `1e-3`, backbone LR `1e-4`, a linear head, and no
additional hidden layer. Its Validation loss is `0.430360`. The next-best loss
is `0.440741` for the medium learning rate. MLP-2 reaches `0.467199`; MLP-1
stops early at epoch 9 with `0.489788`.

Test is absent from all selection rows.
