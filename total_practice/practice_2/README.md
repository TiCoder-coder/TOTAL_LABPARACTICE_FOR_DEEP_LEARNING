# Practice 2 — Transfer Learning on CIFAR-10

Practice 2 implements a leakage-safe multiclass image-classification workflow
with PyTorch, TorchVision and an ImageNet-pretrained ResNet18. The canonical
presentation is
[`practice_2_presentation.ipynb`](notebooks/practice_2_presentation.ipynb).
`Run All` reads stored artifacts and repeats Validation verification; it does
not retrain or rerun Final Test.

## Official result

| Item | Value |
|---|---:|
| Train / Validation / Test | 45,000 / 5,000 / 10,000 |
| Model | ResNet18 `partial_finetune`, linear head |
| Head / backbone LR | `1e-3 / 1e-4` |
| Hidden layers | `[]` |
| Best epoch | 24/25 |
| Validation loss | 0.430360 |
| Validation accuracy at selected checkpoint | 94.42% |
| Test accuracy | 94.06% |
| Test loss | 0.226486 |
| Macro F1 | 0.940458 |
| Micro ROC-AUC | 0.996542 |
| Micro Average Precision | 0.982046 |
| Final Test evaluations | 1 |

Canonical values come from
[`hyperparameter_selection_locked.json`](outputs/hyperparameter_selection_locked.json),
[`summary.json`](outputs/summary.json), and
[`roc_pr_summary.json`](outputs/roc_pr_summary.json).

## Current workflow

```text
CIFAR-10 seeded Train/Validation split
        ↓
Train-only stochastic transforms; deterministic Validation/Test transforms
        ↓
Stage 1: compare three head/backbone learning-rate pairs
        ↓
Stage 2: compare linear, one-hidden-layer and two-hidden-layer heads
        ↓
Select minimum Validation loss; break ties by Validation accuracy
        ↓
Lock best_val_loss.pt path, metrics and SHA256
        ↓
Reload with original criterion; verify Validation loss and accuracy
        ↓
After PASS: evaluate official Test exactly once and write SHA256 receipt
        ↓
Aggregate saved records into `outputs/visualization_data.json`
        ↓
Render the Phase 11 HTML dashboard from that single JSON file in the notebook
```

Test does not participate in preprocessing, training, Early Stopping,
hyperparameter search, checkpoint selection, or tie-breaking.

## Hyperparameter search

The six-row [ranking](outputs/hyperparameter_ranking.csv) covers:

- learning rates `3e-4/3e-5`, `6e-4/6e-5`, `1e-3/1e-4`;
- classifier heads `[]`, `[256]`, `[256, 128]`.

Each run has at most 25 epochs and Early Stopping patience 4. The MLP-1 run
stopped after nine epochs; the other five completed 25 epochs.

Only when a new search is intentionally required, run from this directory:

```bash
python -m processing_own_phase.hyperparameter_search --output-dir outputs
```

Monitor a live run in another terminal:

```bash
python -m processing_own_phase.live_training_monitor \
  --runs-dir runs --refresh-seconds 2
```

Do not run these commands merely to view the notebook.

## Checkpoint verification

The selected checkpoint is
[`best_val_loss.pt`](runs/E2_resnet18_partial_2b5b94de/best_val_loss.pt).
Its SHA256 is
`a906600b717f94aa4cf40ca8504f1b82f6618b80cc6bf9b1a7a63813cfd3223b`.

Reloaded Validation accuracy equals the recorded `94.42%` exactly. The loss
delta is `1.34e-9`. SHA256 provides byte-change detection; it is not an
independently signed guarantee of authenticity.

The [Final Test receipt](outputs/final_test_receipt_a906600b717f.json) records a
single 10,000-image Test pass for this checkpoint.

## Notebook presentation

Phase 7 directly embeds the
[four-panel training dashboard](reports/winner_training_log_dashboard.png).
Phase 8 presents both search stages. Phase 9 presents the selection lock,
SHA256 and reload deltas. Phase 10 presents:

- final metrics;
- TP, TN, FP, FN, TPR, FPR and class accuracy for ten classes;
- per-class ROC-AUC and Average Precision;
- directly embedded [ROC/PR curves](reports/roc_pr_curves_notebook.png);
- confusion matrices and prediction evidence.

Phase 11 also builds
[`visualization_data.json`](outputs/visualization_data.json) with
[`visualization_data.py`](processing_own_phase/visualization_data.py), then
[`training_dashboard.py`](processing_own_phase/training_dashboard.py) renders
the self-contained [HTML dashboard](reports/practice_2_training_dashboard.html)
from that one file. Aggregation reads but never moves, deletes, or rewrites the
source records under `runs/`.

Most recent notebook verification: 13/13 code cells executed, zero errors.
Most recent test-suite verification: **51 passed**; the three warnings are
PyTorch JIT deprecation warnings.

## Canonical artifacts

### Selection and training

- [hyperparameter ranking](outputs/hyperparameter_ranking.csv)
- [locked selection](outputs/hyperparameter_selection_locked.json)
- [winner epoch log](runs/E2_resnet18_partial_2b5b94de/metrics.jsonl)
- [checkpoint manifest](runs/E2_resnet18_partial_2b5b94de/checkpoint_manifest.json)
- [training dashboard](reports/winner_training_log_dashboard.png)
- [aggregated visualization data](outputs/visualization_data.json)
- [artifact-only HTML dashboard](reports/practice_2_training_dashboard.html)

### Final evaluation

- [summary](outputs/summary.json)
- [Final Test receipt](outputs/final_test_receipt_a906600b717f.json)
- [classification report](outputs/classification_report.csv)
- [predictions](outputs/predictions.csv)
- [confusion matrix](outputs/confusion_matrix.csv)
- [ROC/PR summary](outputs/roc_pr_summary.json)
- [ROC points](outputs/roc_curve_per_class.csv)
- [PR points](outputs/pr_curve_per_class.csv)
- [ROC/PR image](reports/roc_pr_curves_notebook.png)

## Documentation

- [Documentation home](description/README.md)
- [Phase descriptions](description/description_own_phase/README.md)
- [Phase result index](description/description_result/README.md)
- [Project requirements](description/project_requirements.md)
- [Historical codebase audit](description/code_base_audit/code_base_audit.md)

The E1/E2 controlled artifacts remain baseline history. Where their values
differ from the locked hyperparameter artifacts, the latter are current.

## Limitations

- One fixed seed was used; repeated-seed uncertainty is not reported.
- The search space is intentionally small and cannot prove a global optimum.
- Test must not be reused for further tuning.
- Grad-CAM remains future interpretability work.
