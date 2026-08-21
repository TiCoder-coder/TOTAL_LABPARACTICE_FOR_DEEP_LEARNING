# Practice 3 Final Canonical Workflow (v2.3)

This document describes the final, authoritative execution flow of Practice 3 v2.3 from end to end.

## 1. Problem Definition
- **WHAT**: Binary Sentiment Classification of movie reviews.
- **INPUT**: English movie review text.
- **OUTPUT**: `NEGATIVE` (0) or `POSITIVE` (1).
- **WHY**: To demonstrate the fine-tuning of a pretrained language model (DistilBERT) for a downstream Natural Language Processing classification task.

## 2. Environment Setup
- **WHAT**: Determine compute device and fix random seeds.
- **PROCESS**: Initialize MPS (Apple Silicon), CUDA, or CPU fallback. Lock seeds to `42` across Python, NumPy, and PyTorch.
- **DECISION**: A deterministic environment is required for controlled hyperparameter searches.

## 3. Pretrained Sentiment Demo
- **WHAT**: Inference using the out-of-the-box Hugging Face `sentiment-analysis` pipeline.
- **WHY**: To demonstrate the baseline capabilities of a generic pretrained classifier before applying custom fine-tuning.

## 4. Frozen Data Protocol (Dataset Loading)
- **WHAT**: Isolate and partition the `rotten_tomatoes` dataset into a strict evaluation protocol.
- **PROCESS**: Load the original Hugging Face Train + Validation splits, re-shuffle, and enforce a custom 3-way split:

| Split | Size | Purpose |
|---|---|---|
| **TRAIN** | 7676 | Optimize model parameters via backpropagation. |
| **VALIDATION** | 960 | Rank experiments, trigger Early Stopping, select the final Winner. |
| **HOLDOUT** | 960 | A one-time final generalization evaluation, accessed only *after* Winner Lock. |

- **DECISION**: The official Hugging Face Test split is **NOT USED** to guarantee strict data isolation.

## 5. Exploratory Data Analysis (EDA)
- **WHAT**: Investigate dataset properties.
- **PROCESS**: Verify class balance (exact 50/50 split across all partitions) and measure token length distribution.
- **DECISION**: Most sequences are under 60 tokens, allowing us to safely define an optimal `max_length`.

## 6. Tokenization / Preprocessing
- **WHAT**: Convert raw text into numerical tensors compatible with DistilBERT.
- **PROCESS**: Use `distilbert-base-uncased` (vocab: 30522).
  - Raw Text → Tokenizer → `input_ids`, `attention_mask`
- **DECISION**: `max_length` is fixed at 80 tokens with aggressive truncation and dynamic padding.

## 7. DistilBERT Model
- **WHAT**: Instantiate the neural network architecture.
- **PROCESS**: Load pretrained DistilBERT weights + a randomly initialized sequence classification head.
- **WHY**: Transfer learning relies on the deep contextual embeddings learned during DistilBERT's pretraining phase, adapting the final layer directly to the two target sentiment classes.

## 8. Training Configuration
- **WHAT**: Define the canonical hyperparameter baseline.
- **PROCESS**:
  - `learning_rate`: [Search space: 1e-5, 2e-5, 3e-5]
  - `weight_decay`: 0.05
  - `batch_size`: 16
  - `max_epochs`: 15
  - `warmup_steps`: 720
  - `label_smoothing_factor`: 0.1
  - `gradient_clipping`: 1.0
  - `seq_classif_dropout`: 0.20
  - `early_stopping_patience`: 4
- **DECISION**: 15 epochs is a *maximum budget*, not a requirement. Early stopping will halt training if `eval_loss` does not improve.

## 9. Hyperparameter Search
- **WHAT**: Execute multiple fine-tuning runs in stages to optimize generalization.
- **PROCESS**:
  - **Stage A**: Baseline sweeps (E1: 1e-5, E2: 2e-5, E3: 3e-5).
  - **Stage B**: Controlled follow-ups (E4: weight_decay=0.05, E5: dropout=0.40, E6: staged fine-tuning).
- **DECISION**: Controlled experiments vary exactly one condition at a time against the baseline. 
- **STATUS**: All six clean experiment records are `PLANNED`; no new ranking exists yet.

## 10. Validation Ranking & Learning Curves
- **WHAT**: Select the best model checkpoint based strictly on Validation set metrics.
- **PROCESS**: After training, rank E1, E2, E3, E4, E5, and E6 by their minimum Validation Loss.
- **WHY**: While Accuracy measures classification correctness, Cross-Entropy Loss measures confidence and probability density, providing a superior signal for early stopping and model selection.
- **OUTPUT**: Pending clean-sequence training.

## 11. Winner Lock
- **WHAT**: Secure the final model checkpoint.
- **PROCESS**: Record the SHA256 cryptographic hash of `checkpoint-960` from the `E4` run.
- **DECISION**: No further hyperparameter tuning is permitted once the winner is locked.

## 12. ONE-TIME Holdout Evaluation
- **WHAT**: Evaluate the generalized performance of the locked model on unseen data.
- **PROCESS**:
  - Locked E4 → Open Holdout → One single inference pass → Final metrics.
- **OUTPUT**:
  - Accuracy: 85.31%
  - F1: 85.11%
  - Loss: 0.3536
  - `evaluation_count = 1`
- **DECISION**: The Holdout result is for reporting *only*. It cannot trigger new hyperparameter searches.

## 13. Confusion Matrix + Error Analysis
- **WHAT**: Diagnose the types of mistakes made by the final model.
- **PROCESS**: Calculate True Positives (403), True Negatives (416), False Positives (64), and False Negatives (77) from the Holdout predictions.
- **OUTPUT**: The model is slightly more prone to missing positive reviews (FN > FP). Qualitative analysis reveals difficulties with subtle/mixed sentiment, sarcasm, and ambiguity.

## 14. Custom Sentence Inference
- **WHAT**: Demonstrate the model interacting with 9 novel, user-authored sentences.
- **PROCESS**: Pass custom strings through the pipeline to observe logits, softmax probabilities, and final labels.
- **WHY**: To provide qualitative evidence that the model behaves as expected in production-like scenarios, including predictable failures on mixed sentiment.

## 15. Save / Reload Verification
- **WHAT**: Guarantee the model can be persisted to disk and reused identically.
- **PROCESS**: Export `E4` via `save_pretrained()`, erase memory, load from disk, and pass the 9 custom sentences.
- **OUTPUT**: 0 parameter mismatches. 9/9 predictions match identically with 0.0 maximum probability difference.

## 16. Final Notebook Presentation
- **WHAT**: Provide a comprehensive, read-only presentation of the entire project.
- **PROCESS**: The notebook `practice_3.ipynb` orchestrates the 16 phases by importing modular functions, rendering JSON metrics, and plotting PNG charts.
- **DECISION**: The notebook intentionally does NOT invoke `Trainer.train()` or evaluate on Holdout, ensuring safety, repeatability, and instantaneous execution.

## 17. Live Training Demonstration (Optional)
- **WHAT**: A real-time training visualization decoupled into two processes.
- **PROCESS**:
  - A separate terminal process (`live_terminal_training.py`) trains an isolated demo subset and streams metrics to `metrics.jsonl`.
  - The notebook acts purely as a passive monitor (`monitor_live_training`), polling the file to update a 2x2 dashboard dynamically.
- **DECISION**: By decoupling, the notebook never hangs on training loops and avoids leaking memory or state into the presentation layer.

---

### Directory Architecture

- `notebook_practice_3/` → Final presentation notebook orchestrating the flow (`practice_3.ipynb`).
- `processing_own_phase/` → Standalone Python implementations and metric aggregators.
- `runs/practice_3_v2_3/` → Raw TensorBoard logs and checkpoint weights.
- `docs/result/practice_3_v2_3/` → Final locked JSON artifacts, charts, and saved binaries.
- `docs/plan-doc/` → Planning documents and historical error analysis.
- `docs/save_process_proceduce_own_phase_refactor&fix/` → Process verification logs.
- `docs/current_flow/` → This canonical workflow document.

---

### Final Metric Summary

| Item | Result |
|---|---|
| Model | DistilBERT |
| Winner | `E4_weight_decay_0.05` |
| Learning Rate | 1e-5 |
| Weight Decay | 0.05 |
| Best Epoch | 2 |
| Validation Accuracy | 86.77% |
| Validation F1 | 86.78% |
| Holdout Accuracy | 85.31% |
| Holdout F1 | 85.11% |
| Save / Reload | PASS |
| Official Test | NOT USED |
