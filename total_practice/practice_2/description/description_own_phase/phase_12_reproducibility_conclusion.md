# Phase 12 — Reproducibility, Limitations, and Conclusion

## Notebook location

- Notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb)
- Main cells: **Cells 29–30**
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

The final phase does not create another model decision. It packages the evidence required for another reviewer to understand, audit, and rerun the workflow without relying on hidden notebook state.

## 2. Reproducibility mechanisms

### Centralized configuration

Split ratio, seed, batch size, optimizer, learning rate, scheduler, epoch budget, patience, image size, and strategy are stored in configuration and artifact files rather than scattered through Markdown text.

Related files:

- [experiment_config.py](../../configs/experiment_config.py)
- [controlled_experiment_selection.json](../../outputs/controlled_experiment_selection.json)

### Seed

Seed 42 is applied to the split and primary random-number generators. It reproduces Train/Validation membership and reduces training variance. Hardware-specific operations can still prevent exact bitwise equivalence across CPU, MPS, and CUDA.

### Compute provenance

Persistent E1/E2 logs record `Using device: mps`, proving that official controlled training ran on the Apple GPU. Cell 4 displays this evidence alongside the final-evaluation device and current runtime backend.

The regenerated `summary.json` records the Final Evaluation backend as `cpu`. This does not change training provenance: E2 training used MPS, whereas deterministic checkpoint verification and artifact regeneration were performed in a separate CPU evaluation process.

### Isolated run directories

Each controlled experiment has its own directory:

- [E1_resnet18_head_3069508a](../../runs/E1_resnet18_head_3069508a)
- [E2_resnet18_partial_6c5d4ec5](../../runs/E2_resnet18_partial_6c5d4ec5)

This prevents checkpoints and logs from being mixed across experiments.

### Checkpoint metadata

The best checkpoint stores reconstruction metadata and Validation information. Final evaluation reloads the file into a new model instead of relying on the in-memory state left after training.

### Artifact lineage

```text
Controlled configuration
    ↓
E1/E2 histories, logs, and best checkpoints
    ↓
controlled_experiment_comparison.csv
    ↓
controlled_experiment_selection.json
    ↓
Validation Verification
    ↓
summary.json, predictions, classification report, and figures
```

Use the [Cell–Output Map](../description_result/README.md) to trace every notebook phase to its implementation and output.

## 3. Safe rerun procedures

### Presentation-only execution

Open [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb) and run its artifact-reading cells. The notebook reads the selected training log, verifies Validation, and displays saved Test artifacts. It does not retrain or repeat Test inference.

### Reproducing the complete experiment

1. Create the environment from [requirements.txt](../../requirements.txt).
2. Run data, model, and training tests.
3. Run the controlled E1/E2 experiments.
4. inspect histories and the selection artifact.
5. reload the selected checkpoint and run Validation Verification.
6. proceed to one Final Test pass only after PASS.
7. run the notebook with Restart Kernel → Run All Cells.

Do not use legacy `processing_own_phase/main.py` as the official entry point. Although only E1/E2 remain, it can still overwrite `summary.json` with an older schema. See the [Codebase Audit](../code_base_audit/code_base_audit.md).

## 4. Current limitations

### Single seed

The result represents seed 42 only. Stability across random initialization and stochastic augmentation has not been quantified. A stronger study should report mean and standard deviation across multiple seeds.

### Limited epoch budget

Five epochs create valid multi-point curves and a meaningful controlled comparison, but they do not prove that partial fine-tuning has converged fully.

### BatchNorm behavior

Freezing weights with `requires_grad=False` does not automatically freeze running statistics. A strict head-only protocol should define BatchNorm behavior explicitly.

### Historical artifacts

Historical, quick-run, duplicate, and test-only files were removed during the approved cleanup. Presentation should use only artifacts linked by the current notebook and Phase Result Index.

### Interpretability

Error analysis is implemented, but Grad-CAM or feature attribution is not. This is a future extension, not a claimed result.

### Future Test reuse

The official Test set has already been opened for final reporting. If future model changes are designed from current Test errors and repeatedly evaluated on the same set, Test will gradually become implicit Validation. Future development should use Validation or a newly defined holdout protocol.

## 5. Technical conclusion

The current pipeline satisfies the core requirements:

- 45,000/5,000/10,000 Train/Validation/Test split;
- split indices determined before preprocessing;
- Train augmentation separated from deterministic Validation/Test preprocessing;
- controlled E1/E2 comparison differing only by fine-tuning strategy;
- official E1/E2 training on Apple GPU through MPS;
- E2 selected through Validation;
- selected checkpoint reloaded and Validation Verification passed;
- Final Test executed after verification;
- Accuracy, Loss, Macro Precision/Recall/F1, per-class report, confusion matrices, confidence analysis, and prediction galleries;
- One-vs-Rest ROC and Precision–Recall analysis derived from exported per-class probabilities;
- metrics and outputs read from artifacts instead of hardcoded notebook values.

Current official result:

| Item | Value |
|---|---:|
| Selected strategy | `partial_finetune` |
| Best epoch | 4 |
| Validation Accuracy | 89.72% |
| Test Accuracy | 88.95% |
| Test Macro F1 | 88.99% |
| Training backend | Apple MPS GPU |
| Final Evaluation backend | CPU |

## 6. Submission and presentation checklist

- Open [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb).
- Confirm Cell 4 shows E2 training device `mps` and GPU accelerated `True`.
- Confirm Cell 6 displays the correct split sizes.
- Confirm Cell 11 proves transform separation.
- Confirm learning curves contain five epoch points.
- Confirm the E1/E2 table contains no Test metric.
- Confirm Cell 21 reports Validation Verification PASS.
- Confirm Cell 23 reads `summary.json`.
- Confirm the raw confusion matrix sums to 10,000.
- Confirm the mixed prediction grid is visible.
- Do not run legacy `main.py` before presentation.
- Do not retrain or repeat Final Test during presentation.
- Do not commit or push unintended artifacts.

## 7. Suggested closing statement

> Every model decision was made on Validation. The E2 epoch-4 checkpoint was reloaded and reproduced 89.72% Validation Accuracy before Test access was allowed. It achieved 88.95% Test Accuracy and 88.99% Macro F1 on 10,000 images. Future work should evaluate multiple seeds, define BatchNorm freezing explicitly, and add interpretability without repeatedly optimizing against the current Test set.

## 8. Related documentation

- [Project README](../../README.md)
- [Codebase Audit](../code_base_audit/code_base_audit.md)
- [Cell–Output Map](../description_result/README.md)
- [Phase Documentation Index](README.md)
