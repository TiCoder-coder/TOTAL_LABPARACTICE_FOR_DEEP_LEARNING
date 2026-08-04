# Phase 12 — Reproducibility and Conclusion Results

[Phase 11 results](phase_11_results.md) | [Result index](README.md) | [Open notebook](../../notebooks/practice_2_presentation.ipynb)

Phase 12 is primarily a Markdown conclusion and reproducibility checklist in Cells 29–30.

Stored evidence referenced by the phase:

- selected checkpoint path and SHA-256: [summary.json](../../outputs/summary.json);
- controlled selection source: [controlled_experiment_selection.json](../../outputs/controlled_experiment_selection.json);
- E1 training provenance: [E1 run](../../runs/E1_resnet18_head_3069508a);
- E2 training provenance: [E2 run](../../runs/E2_resnet18_partial_6c5d4ec5);
- final notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb).

Final conclusion: E2 `partial_finetune`, trained on Apple MPS, was selected using Validation only, passed checkpoint verification, and achieved 88.95% Test Accuracy with 88.99% Macro F1.
