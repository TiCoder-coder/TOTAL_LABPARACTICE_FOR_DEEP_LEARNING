# Phase 7 — Model Training

## Current notebook contract

The presentation notebook does not retrain. It reads the completed winner log
from [`metrics.jsonl`](../../runs/E2_resnet18_partial_2b5b94de/metrics.jsonl)
and displays the stored training evidence.

The approved training configuration uses an ImageNet-pretrained ResNet18,
`partial_finetune`, AdamW, label-smoothed Cross Entropy, differential learning
rates, Warmup-Cosine scheduling, gradient clipping, strong Train-only
augmentation, and deterministic Validation preprocessing.

Each search run has at most 25 epochs. Early Stopping monitors Validation loss
with patience 4. Test is neither constructed nor evaluated during training.

## Durable logging and dashboard

After every completed epoch, [`train.py`](../../processing_own_phase/train.py)
appends a JSON object to `metrics.jsonl` containing:

- Train and Validation loss;
- Train and Validation accuracy;
- Validation macro F1;
- generalization gap;
- head and backbone learning rates;
- epoch duration and current best epochs.

[`live_training_monitor.py`](../../processing_own_phase/live_training_monitor.py)
can read this file during training without modifying model state. The notebook
shows the same four-panel evidence as a directly embedded image:

- [winner training-log dashboard](../../reports/winner_training_log_dashboard.png)

The panels show loss, accuracy, macro F1/generalization gap, and differential
learning rates. The minimum-loss epoch, maximum-accuracy epoch, and final epoch
are marked explicitly.

## Checkpoint policy

Each run writes:

- `latest.pt` — latest completed epoch;
- `best_val_loss.pt` — minimum Validation loss;
- `best_val_accuracy.pt` — maximum Validation accuracy;
- `best.pt` — compatibility alias for the configured primary metric;
- `checkpoint_manifest.json` — file size, SHA256, epoch, metrics, Git commit,
  and payload-load status.

The current selection metric is minimum Validation loss. The winner is epoch 24
with Validation loss `0.4303597612` and Validation accuracy `94.42%` at that
same checkpoint. The fact that another epoch may have a higher accuracy is not
substituted into the selected-checkpoint row.

## Reproduction commands

Run from `total_practice/practice_2`:

```bash
python -m processing_own_phase.hyperparameter_search --output-dir outputs
```

In another terminal, monitor without influencing training:

```bash
python -m processing_own_phase.live_training_monitor \
  --runs-dir runs --refresh-seconds 2
```

Do not rerun the search merely to view the notebook. Continue with
[Phase 8](phase_08_controlled_experiments.md) for the two-stage comparison.
