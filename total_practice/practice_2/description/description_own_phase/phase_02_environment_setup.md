# Phase 2 — Environment Setup and GPU Evidence

## Notebook location

- Notebook: [demo_practice_2.ipynb](../../notebooks/demo_practice_2.ipynb)
- Main cells: **Cells 3–4**
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Objective

This phase standardizes the runtime before any data loading or model construction. It ensures that the notebook resolves the correct project root, imports the intended internal modules, establishes reproducibility, and reports both the live device and the device recorded during official training.

## 2. Environment initialization

### Project root and module path

The notebook detects the Practice 2 root whether it is opened from the project root or the `notebooks` directory. It adds the resolved path to `sys.path`, allowing imports from `processing_own_phase` and `configs` without depending on hidden notebook state.

### Primary libraries

- PyTorch and TorchVision for models, tensors, transforms, and CIFAR-10.
- NumPy and Pandas for numerical operations and tables.
- Matplotlib for visualization.
- scikit-learn for classification metrics and confusion analysis.
- IPython display utilities for presentation output.

### Random seed

The official seed is **42**. It is applied to Python, NumPy, PyTorch, the Train/Validation permutation, and supported device-specific generators. A fixed seed reduces variance between repeated runs, although exact bitwise reproducibility can still depend on hardware and backend kernels.

### Device policy

[`utils.py`](../../processing_own_phase/utils.py) selects the best available backend in this order:

```text
CUDA → Apple MPS → CPU
```

The selected controlled E1/E2 runs used `mps`, meaning computation was accelerated by the Apple GPU through Metal Performance Shaders.

## 3. Persistent GPU-training evidence

Cell 4 does more than check whether a GPU is currently available. It reads the selection artifact, resolves the selected E2 checkpoint, finds the persistent log stored beside that checkpoint, and extracts the exact `Using device: mps` line.

This is stronger evidence than a live `torch.backends.mps.is_available()` check. Live availability proves only what the current kernel can use; the run log proves what the training process that produced `best.pt` actually used.

The `GPU accelerator evidence` table compares three sources:

| Stage | Source | Expected device |
|---|---|---|
| Selected E2 controlled training | Persistent run log | `mps` |
| Final Evaluation | `outputs/summary.json` | `mps` |
| Current notebook runtime | Live `get_device()` call | `mps` on the current machine |

The cell asserts that the recorded training device belongs to `{cuda, mps}`. It fails clearly if the selected run was trained on CPU. The cell does not retrain the model.

## 4. Expected cell output

The setup cells display:

- Python version.
- PyTorch version.
- TorchVision version.
- Resolved project root.
- GPU accelerator evidence table.
- A green `True` indicator for GPU-accelerated rows.

These outputs are environment and provenance evidence, not model-performance metrics.

## 5. Colored-table helper

The notebook defines a reusable Pandas Styler helper that applies headers, row highlighting, gradients, and green/gray boolean cells. Styling affects only HTML presentation. It does not change the underlying values or artifacts.

If the notebook is opened in a renderer that does not support HTML styling, tables may revert to the default DataFrame appearance while retaining correct data.

## 6. Correctness checks

Before proceeding, verify that:

- internal imports complete without `ModuleNotFoundError`;
- the project root resolves to Practice 2;
- the selected checkpoint and its log exist;
- the log contains `Using device:`;
- the recorded training device is `mps` or `cuda`;
- the selection artifact records `test_data_used = false`;
- the device used for tensors is consistent with the model device.

## 7. Common risks

- Running later cells without first executing setup leaves required variables undefined.
- Selecting a different Jupyter kernel can produce dependency or version mismatches.
- Relative paths can fail when the notebook is launched from an unexpected directory.
- CPU, MPS, and CUDA can produce slightly different floating-point results despite the same seed.
- MPS is GPU acceleration, but it is not CUDA. CUDA-specific automatic mixed precision is not active on MPS in the current training implementation.

The notebook should therefore be tested with **Restart Kernel → Run All Cells** rather than relying on old kernel state.

## 8. Related implementation

- [core_config.py](../../configs/core_config.py)
- [experiment_config.py](../../configs/experiment_config.py)
- [utils.py](../../processing_own_phase/utils.py)
- [Selected E2 training log](../../runs/E2_resnet18_partial_6c5d4ec5/E2_resnet18_partial_6c5d4ec5.log)
- [Final summary](../../outputs/summary.json)

## 9. Suggested presentation script

> I fixed seed 42 and verified the software environment before data processing. To prove the training device, the notebook reads the persistent log belonging to the selected E2 checkpoint. The log records `Using device: mps`, so the model was trained on the Apple GPU through Metal rather than merely detecting a GPU in the current session.

## 10. Transition to the next phase

[Phase 3](phase_03_data_loading.md) loads CIFAR-10 and constructs Train, Validation, and Test according to the leakage-prevention requirements.
