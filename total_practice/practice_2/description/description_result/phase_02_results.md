# Phase 2 — Environment Setup Results

[Phase 1 results](phase_01_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb) | [Phase 3 results](phase_03_results.md)

| Cell output | Description | Evidence |
|---|---|---|
| Cell 4 | Python, PyTorch, TorchVision, and project-root information | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 4 | `GPU accelerator evidence` table | [Notebook](../../notebooks/practice_2_presentation.ipynb) |
| Cell 4 | Selected winner training device is `mps`; assertion confirms GPU acceleration | [Winner log](../../runs/E2_resnet18_partial_2b5b94de/E2_resnet18_partial_2b5b94de.log) |
| Cell 4 | Final-evaluation device is `cpu`; this is distinct from the MPS training provenance | [Final summary](../../outputs/summary.json) |

The training-device claim is derived from the persistent run log rather than inferred from either the current runtime or the backend used to regenerate the final artifacts.
