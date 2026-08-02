# Practice 2 — Transfer Learning on CIFAR-10

Practice 2 implements a multiclass image-classification pipeline with PyTorch, TorchVision, and transfer learning. The official presentation notebook is [`notebooks/demo_practice_2.ipynb`](notebooks/demo_practice_2.ipynb). It reads verified artifacts and does not retrain the models or repeat the Final Test evaluation during `Run All`.

## 1. Official results

| Item | Value |
|---|---:|
| Dataset | CIFAR-10 from `torchvision.datasets.CIFAR10` |
| Train / Validation / Test | 45,000 / 5,000 / 10,000 |
| Backbone | ImageNet-pretrained ResNet18 |
| Selected experiment | E2 — `partial_finetune` |
| Best epoch | 4/5 |
| Validation Accuracy | 89.72% |
| Test Accuracy | 88.95% |
| Test Loss | 0.341539 |
| Macro Precision | 89.2836% |
| Macro Recall | 88.9500% |
| Macro F1 | 88.9933% |
| Final Test evaluations | 1 |

These values come from [`outputs/controlled_experiment_selection.json`](outputs/controlled_experiment_selection.json) and [`outputs/summary.json`](outputs/summary.json). They are not manually entered metrics.

## 2. Official workflow

```text
CIFAR-10
   ↓
Determine Train/Validation indices before preprocessing
   ↓
Train transform with random augmentation
Deterministic Validation/Test transform
   ↓
Controlled E1/E2 training
   ↓
Select the experiment using Validation Accuracy
   ↓
Reload best.pt and verify Validation performance
   ↓
If verification passes: evaluate the Final Test exactly once
   ↓
Generate summary, predictions, classification report, and visualizations
```

Data-use rules:

- Train is used to optimize model parameters.
- Validation is used for epoch monitoring, Early Stopping, experiment selection, and checkpoint selection.
- Test does not participate in preprocessing decisions, training, or model selection.
- The official Test set is evaluated only after the selected checkpoint reproduces its recorded Validation Accuracy.

## 3. Which notebook to use

- **Official demo and presentation notebook:** [`notebooks/demo_practice_2.ipynb`](notebooks/demo_practice_2.ipynb)

The previous baseline notebook is no longer present in the current workspace. All documentation therefore points to `demo_practice_2.ipynb` as the single canonical notebook.

The demo notebook contains 12 phases, GPU-training evidence, controlled E1/E2 comparison, multi-epoch learning curves, a Validation–Test comparison, raw and normalized confusion matrices, confidence analysis, correct/incorrect galleries, and a mixed prediction grid.

## 4. Documentation map

- [Documentation index](description/README.md): single entry point for all Practice 2 documentation.
- [Project requirements](description/project_requirements.md): assignment scope and expected deliverables.
- [Reference workflow](description/reference/working_flow.md): original workflow used to guide the notebook structure.
- [Codebase audit](description/code_base_audit/code_base_audit.md): architecture, correctness, leakage controls, Test isolation, artifacts, tests, and technical debt.
- [Phase description index](description/description_own_phase/README.md): detailed explanation of all 12 notebook phases.
- [Phase result index](description/description_result/README.md): Phase → Cell → source code → canonical output/artifact links.

## 5. Repository structure

```text
practice_2/
├── configs/                    # Core paths and experiment/training configuration
├── data/                       # CIFAR-10 downloaded by TorchVision
├── description/
│   ├── code_base_audit/        # Technical and correctness audit
│   ├── description_own_phase/  # Detailed descriptions for all 12 phases
│   ├── description_result/     # Phase/cell/source/output mapping and results
│   ├── reference/              # Reference workflow
│   ├── project_requirements.md # Assignment description and requirements
│   └── README.md               # Documentation index
├── notebooks/
│   └── demo_practice_2.ipynb   # Official presentation notebook
├── outputs/                    # Machine-readable JSON and CSV artifacts
├── processing_own_phase/       # Reusable implementation package
├── reports/                    # Visual artifacts used by the notebook
├── runs/                       # Per-run checkpoints, logs, and TensorBoard files
├── tests/                      # Unit and integration tests
└── README.md
```

## 6. Source modules

| Module | Responsibility |
|---|---|
| `data.py` | Split construction, dataset views, transforms, DataLoaders, and class distribution |
| `model.py` | Pretrained architectures and parameter-freezing strategies |
| `train.py` | Train/Validation loops, scheduler, checkpointing, and Early Stopping |
| `experiment.py` | Controlled experiments and Validation-only model selection |
| `evaluate.py` | Loss, Accuracy, Precision, Recall, F1, and prediction collection |
| `final_evaluate.py` | Checkpoint verification, single Test pass, and final artifacts |
| `visualize.py` | EDA, learning curves, confusion matrices, and prediction galleries |
| `save_load.py` | Checkpoint persistence and output-equivalence verification |

## 7. Environment setup

```bash
cd total_practice/practice_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Select the `venv-practice-2` kernel when opening the notebook.

The device-selection policy is CUDA → Apple MPS → CPU. The official E1/E2 logs record `Using device: mps`, so the controlled experiments were trained on the Apple GPU through Metal Performance Shaders.

## 8. Safe notebook execution

```bash
jupyter notebook notebooks/demo_practice_2.ipynb
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

The demo notebook:

- reads controlled history and final artifacts;
- reads the selected training log to prove the accelerator used during training;
- verifies the selected checkpoint on Validation;
- displays previously generated Test artifacts;
- accesses Test images only to assemble the artifact-backed error-analysis grid;
- does not call `run_controlled_experiments()`;
- does not call `regenerate_final_artifacts()`;
- does not run model inference on Test again.

## 9. Training and Final Test

Do not repeat training or Final Test merely to view the notebook. The official artifacts already exist.

If a new experiment is intentionally required, run `run_controlled_experiments()` as a new controlled run with separate checkpoints and artifacts. If Final Test regeneration is explicitly required, ensure the new selection artifact is valid and understand that `regenerate_final_artifacts()` evaluates the official Test set.

> **Warning:** `python -m processing_own_phase.main` is a legacy artifact-writing path. Although only E1/E2 remain, it can overwrite official artifacts. Review the [codebase audit](description/code_base_audit/code_base_audit.md) before using it.

## 10. Tests

```bash
pytest -q
git diff --check
```

Most recent verification: **45 tests passed**. The three remaining warnings are `torch.jit.trace` deprecation warnings and do not affect pipeline correctness.

## 11. Canonical artifacts

### Controlled selection and history

- [`outputs/controlled_experiment_comparison.csv`](outputs/controlled_experiment_comparison.csv)
- [`outputs/controlled_experiment_selection.json`](outputs/controlled_experiment_selection.json)
- [`outputs/controlled_training_history.json`](outputs/controlled_training_history.json)

### Final evaluation

- [`outputs/summary.json`](outputs/summary.json)
- [`outputs/classification_report.csv`](outputs/classification_report.csv)
- [`outputs/predictions.csv`](outputs/predictions.csv)
- [`outputs/confusion_matrix.csv`](outputs/confusion_matrix.csv)

### Visualizations used by the demo

- [`reports/controlled_training_curves.png`](reports/controlled_training_curves.png)
- [`reports/controlled_learning_rate.png`](reports/controlled_learning_rate.png)
- [`reports/controlled_experiment_comparison.png`](reports/controlled_experiment_comparison.png)
- [`reports/validation_test_comparison_current.png`](reports/validation_test_comparison_current.png)
- [`reports/confusion_matrix_raw.png`](reports/confusion_matrix_raw.png)
- [`reports/confusion_matrix_normalized.png`](reports/confusion_matrix_normalized.png)
- [`reports/prediction_grid_mixed.png`](reports/prediction_grid_mixed.png)

## 12. Known limitations

- The controlled comparison uses one seed; mean and standard deviation across multiple seeds are not available.
- The `head_only` strategy still requires care with BatchNorm running statistics while the full model is in `train()` mode.
- `main.py` is a legacy orchestration path and is not fully aligned with the guarded final-evaluation workflow.
- Only the two official controlled runs and the artifacts linked by the demo notebook are retained as canonical presentation evidence.
- Grad-CAM is not implemented. The notebook presents it as future interpretability work and does not claim attribution evidence.
- The five-epoch budget is sufficient for a meaningful multi-point comparison but does not prove full convergence.

## 13. Presentation checklist

1. Open `demo_practice_2.ipynb`.
2. Use Cell 4 to prove E2 training used the Apple GPU through MPS.
3. Use Cell 6 to explain the 45,000/5,000/10,000 split.
4. Use Cell 11 to explain transform separation and leakage prevention.
5. Use Cell 19 to compare controlled E1 and E2.
6. Confirm that Cell 21 reports Validation Verification `PASS`.
7. Use Cell 23 for Final Test metrics and confusion matrices.
8. Use Cell 25 for the mixed error-analysis grid.
9. Present only the canonical plots linked by the current notebook and phase-result documentation.
10. Do not retrain or repeat Final Test during the presentation.
