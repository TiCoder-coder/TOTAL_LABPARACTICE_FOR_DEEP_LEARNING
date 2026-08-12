# Phase 8 — Controlled Experiments E1 versus E2

## Notebook location

- Notebook: [practice_2_presentation.ipynb](../../notebooks/practice_2_presentation.ipynb)
- Main cells: **Cells 16–17**
- Source: [experiment.py](../../processing_own_phase/experiment.py)
- Configuration: [experiment_config.py](../../configs/experiment_config.py)
- Full mapping: [Cell–Output Map](../description_result/README.md)

## 1. Experimental objective

The controlled comparison answers one question:

> When all background conditions remain fixed, how does training only the classification head compare with fine-tuning the final ResNet block and head?

A meaningful conclusion requires that learning rate, optimizer, augmentation, epoch budget, and every other background factor remain identical.

## 2. Independent variable

| Experiment | Strategy | Trainable region |
|---|---|---|
| E1 | `head_only` | Classification head |
| E2 | `partial_finetune` | ResNet18 `layer4` and classification head |

Fine-tuning strategy is the only intentionally changed experimental variable.

## 3. Controlled factors

E1 and E2 share:

- ResNet18 architecture;
- ImageNet-pretrained initialization policy;
- dataset split and split seed;
- global seed 42;
- batch size 64;
- Adam optimizer;
- learning rate `0.001`;
- weight decay `1e-4`;
- ReduceLROnPlateau scheduler;
- five-epoch budget;
- Train augmentation;
- deterministic Validation transform;
- Cross Entropy loss;
- Early Stopping patience 3;
- Validation Accuracy selection metric;
- Apple MPS compute backend for the official runs.

Cell 17 displays these fields side by side and asserts that only `training_mode` differs.

## 4. Per-experiment workflow

```text
Initialize ResNet18 with the same pretrained policy
        ↓
Apply E1 or E2 training mode
        ↓
Train with the same Train loader and base configuration
        ↓
Evaluate each epoch with the same Validation loader
        ↓
Save best.pt by Validation Accuracy

## Two-stage hyperparameter search

The new Validation-only search first compares head/backbone learning-rate pairs
`3e-4/3e-5`, `6e-4/6e-5`, and `1e-3/1e-4`. It then holds the winning learning
rate fixed and compares a linear classifier, one hidden layer `[256]`, and two
hidden layers `[256, 128]`. Runs use at most 25 epochs and early stopping with a
patience of four epochs. The Test set is not loaded for either stage.

The command is:

```bash
python -m processing_own_phase.hyperparameter_search --output-dir outputs
```

The final ranking is written to `outputs/hyperparameter_ranking.csv`; primary
ordering is minimum Validation loss and the tie-breaker is maximum Validation
accuracy.
        ↓
Store run history, metadata, and Validation metrics
```

The Test DataLoader is not part of this workflow.

## 5. Fairness and limitations

### Controlled correctly

- Same data membership and seed.
- Same optimization budget and objective.
- Same preprocessing policies.
- Same selection rule.
- No Test metric in the E1/E2 comparison table.
- Persistent logs confirm both official runs used MPS.

### Limitations

- Only one seed is available, so variance across random initializations is unknown.
- Five epochs provide a valid multi-point comparison but may not reach full convergence.
- Trainable parameter count differs by design because it is a direct consequence of strategy.
- BatchNorm running statistics remain a caveat when freezing parameters through `requires_grad` alone.

## 6. Canonical artifacts

- [experiment_config.py](../../configs/experiment_config.py)
- [controlled_experiment_comparison.csv](../../outputs/controlled_experiment_comparison.csv)
- [controlled_experiment_comparison.png](../../reports/controlled_experiment_comparison.png)
- [E1 run summary](../../runs/E1_resnet18_head_3069508a/E1_resnet18_head_3069508a_summary.json)
- [E2 run summary](../../runs/E2_resnet18_partial_6c5d4ec5/E2_resnet18_partial_6c5d4ec5_summary.json)
- [Combined controlled history](../../outputs/controlled_training_history.json)

The `controlled_` prefix is important because similarly named report files can belong to historical quick runs.

## 7. Selection criterion

The winning experiment is selected by **best Validation Accuracy**. If an exact tie occurs, any tie-break rule must be declared before inspecting Test, for example lower Validation Loss or the simpler model. Test Accuracy must never be used to break a tie.

## 8. Experiment scope

Only E1 and E2 are defined and retained. No additional exploratory experiment is part of the codebase, artifact folders, or presentation selection table.

## 9. Suggested presentation script

> E1 and E2 use the same ResNet18, split, seed, optimizer, learning rate, scheduler, augmentation, loss, batch size, five-epoch budget, and MPS backend. The only intended change is that E1 trains the head while E2 trains layer4 plus the head. Therefore, the Validation difference can be attributed primarily to the fine-tuning strategy.

## 10. Transition to the next phase

[Phase 9](phase_09_selection_verification.md) reads the controlled artifacts, selects E2 using Validation only, and verifies the selected checkpoint before Test access is permitted.
