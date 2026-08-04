# Phase 2 — Environment Setup Results

[Phase 1 results](phase_01_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 3 results](phase_03_results.md)

| Cell output | Description | Evidence |
|---|---|---|
| Cell 4 | Python, PyTorch, TorchVision, and project-root information | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 4 | `GPU accelerator evidence` table | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 4 | Selected E2 training device is `mps`; assertion confirms GPU acceleration | [Selected E2 log](../../runs/E2_resnet18_partial_6c5d4ec5/E2_resnet18_partial_6c5d4ec5.log) |
| Cell 4 | Final-evaluation device is `mps` | [Final summary](../../outputs/summary.json) |

The training-device claim is derived from the persistent run log rather than inferred only from current hardware availability.
