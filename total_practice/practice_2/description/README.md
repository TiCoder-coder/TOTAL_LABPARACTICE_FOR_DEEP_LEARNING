# Practice 2 Documentation

The canonical presentation is
[`practice_2_presentation.ipynb`](../notebooks/practice_2_presentation.ipynb).

## Current result

| Item | Value |
|---|---:|
| Winner | ResNet18 `partial_finetune`, linear head |
| Head / backbone LR | `1e-3 / 1e-4` |
| Best epoch | 24 |
| Validation loss / accuracy | 0.430360 / 94.42% |
| Test accuracy / loss | 94.06% / 0.226486 |
| Macro F1 | 0.940458 |
| Micro ROC-AUC / AP | 0.996542 / 0.982046 |
| Final Test count | 1 |

Values come from [locked selection](../outputs/hyperparameter_selection_locked.json),
[summary](../outputs/summary.json), and
[ROC/PR summary](../outputs/roc_pr_summary.json).

For notebook reporting, the visualization layer aggregates the relevant saved
fields into one [visualization data file](../outputs/visualization_data.json).
The [HTML dashboard](../reports/practice_2_training_dashboard.html) renders
from that file only; original records under `runs/` are preserved unchanged.

## Documentation map

- [Project requirements](project_requirements.md)
- [Reference workflow](reference/working_flow.md)
- [Phase descriptions](description_own_phase/README.md)
- [Phase result index](description_result/README.md)
- [Historical codebase audit](code_base_audit/code_base_audit.md) — baseline
  snapshot; superseded by the current artifacts above where values differ.
