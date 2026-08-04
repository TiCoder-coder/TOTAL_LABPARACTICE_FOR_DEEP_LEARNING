# Practice 2 Documentation

This directory follows the documentation layout used by Practice 1 while preserving the twelve-phase structure of the current Practice 2 notebook.

## Documentation groups

| Group | Purpose | Entry point |
|---|---|---|
| Project requirements | Original assignment objective and deliverables | [Project requirements](project_requirements.md) |
| Workflow reference | General ML/DL workflow used as design guidance | [Working flow](reference/working_flow.md) |
| Codebase audit | Technical review, risks, evidence, and remediation priorities | [Codebase audit](code_base_audit/code_base_audit.md) |
| Phase descriptions | Detailed explanation of implementation and reasoning for all 12 phases | [Phase description index](description_own_phase/README.md) |
| Phase results | Direct mapping from phases and notebook cells to stored outputs and artifacts | [Phase result index](description_result/README.md) |

## Canonical notebook

- [practice_2_presentation.ipynb](../notebooks/practice_2_presentation.ipynb)

The notebook is the presentation layer. Reusable training and evaluation logic remains under [`processing_own_phase/`](../processing_own_phase/).

## Official pipeline

```text
Problem definition
    → environment and GPU evidence
    → data loading and split
    → exploratory data analysis
    → preprocessing and leakage control
    → ResNet18 construction
    → multi-epoch training
    → controlled E1/E2 comparison
    → Validation-only selection
    → checkpoint verification
    → one Final Test evaluation
    → error analysis and conclusion
```

## Official results

| Item | Value |
|---|---:|
| Selected experiment | E2 `partial_finetune` |
| Training backend | Apple GPU through MPS |
| Best epoch | 4 |
| Validation Accuracy | 89.72% |
| Test Accuracy | 88.95% |
| Macro F1 | 88.99% |
| Test samples | 10,000 |
| Test evaluation count | 1 |

Values are read from [`outputs/controlled_experiment_selection.json`](../outputs/controlled_experiment_selection.json) and [`outputs/summary.json`](../outputs/summary.json).
