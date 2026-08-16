# Practice 3 — Fine-Tuning DistilBERT for Sentiment Classification

Practice 3 implements a binary movie-review sentiment-classification workflow using Hugging Face Transformers and a pretrained DistilBERT model. It demonstrates end-to-end transfer learning from generic language representation to downstream Negative/Positive classification. The project features a controlled hyperparameter search, Validation-only model selection, a rigorous one-time final Holdout evaluation, and an artifact-driven final notebook.

The canonical presentation is [`practice_3.ipynb`](notebook_practice_3/practice_3.ipynb). `Run All` reads stored final artifacts to render tables and charts; it does not retrain experiments, rerun Holdout evaluation, or load the Official Test split.

## Official Result (Practice 3 v2.3)

| Item | Value |
|---|---:|
| Train / Validation / Holdout | 7,676 / 960 / 960 |
| Model | DistilBERT binary classifier |
| Final winner | `E4_weight_decay_0.05` |
| Learning rate | `1e-5` |
| Weight decay | `0.05` |
| Classifier dropout | `0.20` |
| Best epoch | `2` |
| Stopped epoch | `6` |
| Validation loss | `0.415336` |
| Validation accuracy | `86.77%` |
| Validation F1 | `86.78%` |
| Holdout loss | `0.353578` |
| Holdout accuracy | `85.31%` |
| Holdout precision | `86.30%` |
| Holdout recall | `83.96%` |
| Holdout F1 | `85.11%` |
| Final Holdout evaluations | `1` |
| Official Hugging Face Test | `NOT USED` |

Canonical values come from [`final_validation_winner_lock.json`](docs/result/practice_3_v2_3/final_validation_winner_lock.json), [`final_holdout_metrics.json`](docs/result/practice_3_v2_3/final_holdout_metrics.json), and [`final_validation_ranking.csv`](docs/result/practice_3_v2_3/final_validation_ranking.csv).

## Current Workflow

```text
Movie-review dataset
        ↓
EDA and split-integrity checks
        ↓
Frozen Train / Validation / Holdout protocol
        ↓
DistilBERT tokenization
        ↓
Stage A: compare learning rates 1e-5 / 2e-5 / 3e-5
        ↓
Stage B: test weight decay, classifier dropout and staged fine-tuning
        ↓
Rank experiments by minimum Validation loss
        ↓
Lock E4_weight_decay_0.05 checkpoint and SHA256
        ↓
Evaluate frozen Holdout exactly once
        ↓
Read saved confusion-matrix, error-analysis, inference and reload artifacts
        ↓
Present everything in practice_3.ipynb without retraining
```

Holdout does not participate in preprocessing, training, Early Stopping, hyperparameter search, checkpoint selection, or tie-breaking.

## Hyperparameter Search

The search was divided into two controlled stages. Stage A explored learning rates (1e-5, 2e-5, 3e-5), establishing 1e-5 as the most stable baseline. Stage B introduced regularization and architectural tweaks, testing increased weight decay, increased classifier dropout, and a staged fine-tuning strategy (freezing the backbone before unfreezing). The winner was selected strictly based on the lowest Validation loss across all valid candidates.

## Winner Checkpoint Verification

The selected winner (`E4_weight_decay_0.05`) achieved its minimum Validation loss at epoch 2. Its state dictionary and exact weights are preserved and verified via SHA-256 fingerprint:
`d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191` (checkpoint-960).

## Final Holdout Evaluation

Following the checkpoint lock, the isolated Holdout split (960 samples) was evaluated exactly once (`evaluation_count = 1`). The resulting Holdout loss was 0.3536 with an F1 score of 85.11%. The official Hugging Face Test split remained unloaded (`NOT USED`) to guarantee strict data isolation.

## Error Analysis

The final Holdout confusion matrix demonstrates a slight tendency toward missing positive reviews rather than over-predicting them:
- **True Positives**: 403
- **True Negatives**: 416
- **False Positives**: 64
- **False Negatives**: 77

Qualitative error analysis of the 141 wrong predictions indicates that the model struggles primarily with subtle/mixed sentiment, ambiguous wording, and heavy sarcasm or irony.

## Custom Inference

To demonstrate production behavior, the locked model was tested against 9 novel, user-authored sentences representing clear positive, clear negative, and mixed sentiments. Inference results matched expectations, succeeding on clear language while faltering gracefully on nuanced, mixed reviews. These sentences were for demonstration only and did not influence model selection.

## Save / Reload Verification

The final model architecture and tokenization pipeline were successfully exported to disk and reloaded. Verification confirmed 0 parameter mismatches across the 66.9M weights. Reloaded predictions on the custom inference inputs matched original logit outputs perfectly (0.0 probability difference), confirming the model can be used reliably in downstream applications without retraining.

> **Git LFS Note**: The final model binary in `docs/result/practice_3_v2_3/final_saved_model/` is ~255MB and requires Git LFS for version control.

## Notebook Presentation

The final pipeline is logically encapsulated within `notebook_practice_3/practice_3.ipynb`. The notebook acts purely as a presentation layer—loading pre-calculated JSON metrics, CSV logs, and PNG charts—rather than initiating expensive training operations or contaminating Holdout metrics.

## Live Training Demonstration (Terminal + Notebook Monitor)

To provide an interactive classroom experience without breaking scientific isolation, Practice 3 includes a specialized **Live Training Demonstration** decoupled into two processes:

1. **Terminal Training Process**: Executes an isolated, fast-running training loop on a small demo dataset.
   ```bash
   .venv/bin/python -m total_practice.practice_3.processing_own_phase.live_terminal_training
   ```
2. **Notebook Monitoring Client**: A single cell in `practice_3.ipynb` tracks the terminal process in real-time, parsing stream metrics (`metrics.jsonl`) and rendering a live 2x2 dashboard inside Jupyter.

The notebook does NOT execute model training, and the demo is strictly excluded from official ranking and evaluation.

## Canonical Artifacts

- **Final Summary**: [`docs/result/practice_3_v2_3/final_summary.md`](docs/result/practice_3_v2_3/final_summary.md)
- **Winner Lock**: [`docs/result/practice_3_v2_3/final_validation_winner_lock.json`](docs/result/practice_3_v2_3/final_validation_winner_lock.json)
- **Holdout Metrics**: [`docs/result/practice_3_v2_3/final_holdout_metrics.json`](docs/result/practice_3_v2_3/final_holdout_metrics.json)

## Documentation

- **Codebase Audit**: [`docs/code_base_audit.md`](docs/code_base_audit.md)
- **Current Flow**: [`docs/current_flow/practice_3_current_flow.md`](docs/current_flow/practice_3_current_flow.md)
- **Process Logs**: [`docs/save_process_proceduce_own_phase_refactor&fix/`](docs/save_process_proceduce_own_phase_refactor&fix/)

## Limitations

- Generalization suffers on text containing irony, sarcasm, and subtle/mixed sentiment.
- The training dataset is relatively small, limiting the diversity of linguistic constructs the model can learn.
- Some false predictions manifest with very high model confidence, indicating suboptimal calibration on hard examples.
