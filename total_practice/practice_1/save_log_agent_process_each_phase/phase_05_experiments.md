# Phase 5 — Experiments Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Files Created

- `processing_own_phase/experiment.py` — `ExperimentRunner`, `build_optimizer`

## Experiment Results

| ID          | Architecture | Optimizer | LR   | Dropout | Best Val Acc | Best Epoch | Params   | Time     |
|-------------|--------------|-----------|------|---------|--------------|------------|----------|----------|
| E0_baseline | [128]        | SGD       | 0.01 | 0.0     | 0.8813       | 10         | 101,770  | 47.5s    |
| E1_lr_low   | [128]        | SGD       | 0.001| 0.0     | 0.8431       | 10         | 101,770  | 45.8s    |
| E2_lr_high  | [128]        | SGD       | 0.1  | 0.0     | 0.8585       | 7          | 101,770  | 47.8s    |
| E3_deeper   | [256, 128]   | SGD       | 0.01 | 0.0     | 0.8845       | 10         | 235,146  | 51.6s    |
| E4_dropout  | [256, 128]   | SGD       | 0.01 | 0.2     | 0.8801       | 8          | 235,146  | 55.8s    |
| E5_adam     | [256, 128]   | Adam      | 0.001| 0.2     | **0.8858**   | 10         | 235,146  | 67.9s    |

## Findings

### Phase B — Learning Rate
- **LR=0.001** (E1): Too slow. After 10 epochs only 0.8431. Did not fully converge.
- **LR=0.01** (E0): Best of the three. Converges smoothly to 0.8813.
- **LR=0.1** (E2): Reaches 0.8585 at epoch 7 then plateaus/diverges. Validation loss spikes at epoch 5 (0.5233) — likely oscillating.

### Phase C — Architecture
- **E0 [128] -> 0.8813**
- **E3 [256, 128] -> 0.8845** (+0.32 pp). Added ~133k params for a small gain.

### Phase D — Regularization
- **E3 (no dropout) -> 0.8845**
- **E4 (dropout=0.2) -> 0.8801** (-0.44 pp). At 10 epochs, dropout did not help; may need more epochs.

### Phase E — Optimizer
- **E4 SGD -> 0.8801**
- **E5 Adam -> 0.8858** (+0.57 pp). Adam trains faster and reaches the best score.

## Best Experiment

**E5_adam** — `[256, 128]`, Adam, lr=0.001, dropout=0.2 — **Val Acc 0.8858**.

## Outcome

Experiment runner works. CSV saved to `save_log_agent_process_each_phase/experiment_results.csv`. Best config selected.
