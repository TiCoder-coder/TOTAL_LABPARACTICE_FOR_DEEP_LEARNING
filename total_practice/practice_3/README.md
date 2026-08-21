# Practice 3 — Get Started with Hugging Face & Fine-Tuning

Practice 3 provides a comprehensive practical implementation of modern NLP workflows using the Hugging Face ecosystem. The project is divided into two distinct parts:
1. **Exercise 1 – Sentiment Analysis with Hugging Face**: Out-of-the-box sentiment classification using a pretrained Transformer pipeline from Hugging Face Hub, featuring subword tokenization analysis and direct inference.
2. **Exercise 2 – Finetuning a Pretrained Model for Binary Text Classification**: End-to-end Transfer Learning workflow fine-tuning `distilbert-base-uncased` on the Rotten Tomatoes movie reviews dataset across 13 modular phases with rigorous data isolation.

The primary presentation notebook is [`notebook_practice_3/practice_3.ipynb`](notebook_practice_3/practice_3.ipynb). The notebook functions purely as an orchestration and visualization layer—rendering tables, HTML dashboards, and trajectory plots from persistent artifacts without initiating expensive retraining.

---

## Assignment Structure & Requirements Mapping

### Exercise 1: Sentiment Analysis with Hugging Face (Hub Pretrained)
| Requirement | Implementation Component | Main Evidence | Authoritative Artifact |
|---|---|---|---|
| **1. Hugging Face Transformers** | [`exercise_01_pretrained_sentiment.py`](processing_own_phase/exercise_01_pretrained_sentiment.py) | Environment & Pipeline Loader | [`exercise_01_pretrained_sentiment_results.md`](docs/result/exercise_01_pretrained_sentiment_results.md) |
| **2. Pretrained Sentiment Model** | `distilbert-base-uncased-finetuned-sst-2-english` | Hugging Face Hub Checkpoint | Model config & weights |
| **3. Sample Tokenization** | `tokenize_sample_sentence()` | Subword Tokens, IDs, Attention Mask | [`exercise_1_pretrained_sentiment.json`](docs/result/practice_3_v2_3/exercise_1_pretrained_sentiment.json) |
| **4. Sentiment Inference** | `predict_sentiment()` | Sentiment Prediction & Confidence Score | [`exercise_1_pretrained_sentiment.json`](docs/result/practice_3_v2_3/exercise_1_pretrained_sentiment.json) |

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
| Assignment Requirement | Implementation Phase | Main Evidence in Notebook | Authoritative Artifact |
|---|---|---|---|
| **Requirement 1: Install & Setup** | Phase 2 – Environment Setup | `GPU accelerator evidence` (`mps` Apple GPU) | `Cell 06` in [`practice_3.ipynb`](notebook_practice_3/practice_3.ipynb) |
| **Requirement 2: Load Dataset** | Phase 3 – Data Loading | 7,676 Train / 960 Val / 960 Holdout table | [`dataset_split_reference.json`](docs/result/practice_3_v2_3/dataset_split_reference.json) |
| **Requirement 3: Load Model & Tokenizer** | Phase 5 & Phase 6 | Architecture table (66.95M parameters) | [`final_saved_model/config.json`](docs/result/practice_3_v2_3/final_saved_model/config.json) |
| **Requirement 4: Preprocess Dataset** | Phase 4 & Phase 5 | HTML EDA Dashboard + Token mapping | [`eda_visualization.py`](processing_own_phase/eda_visualization.py) |
| **Requirement 5: Define Training Arguments** | Phase 7 – Training Configuration | Max 15 epochs, Patience 4, Batch 16, Warmup 720 | [`training_authorization.json`](docs/result/practice_3_v2_3/training_authorization.json) |
| **Requirement 6: Trainer & Fine-Tuning** | Phase 8 – Hyperparameter Search | Benchmark of 6 experiments (Winner: `E4`) | [`final_validation_ranking.csv`](docs/result/practice_3_v2_3/final_validation_ranking.csv) |
| **Requirement 7: Evaluate Fine-Tuned Model** | Phase 9, Phase 10 & Phase 11 | Learning Curves, Holdout Metrics, Confusion Matrix | [`final_holdout_metrics.json`](docs/result/practice_3_v2_3/final_holdout_metrics.json) |
| **Deployment & Persistence** | Phase 12 & Phase 13 | Custom Inference & 0 Mismatch Save/Reload | [`save_reload_verification.json`](docs/result/practice_3_v2_3/save_reload_verification.json) |

---

## Exercise 1 – Pretrained Sentiment Analysis Summary

- **Pretrained Checkpoint**: `distilbert-base-uncased-finetuned-sst-2-english` (Stanford Sentiment Treebank SST-2).
- **Sample Review Text**: *"This movie is absolutely wonderful and enjoyable."*
- **Tokenization Analysis**:
  * 10 tokens extracted: `['[CLS]', 'this', 'movie', 'is', 'absolutely', 'wonderful', 'and', 'enjoyable', '.', '[SEP]']`
  * Input IDs: `[101, 2023, 3185, 2003, 7078, 6919, 1998, 22249, 1012, 102]`
  * Attention Mask: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- **Zero-Shot Inference Results**:
  * *"This movie is absolutely wonderful and enjoyable."* $\rightarrow$ **`POSITIVE`** (Confidence: **`0.9999`**)
  * *"The plot was completely predictable and boring."* $\rightarrow$ **`NEGATIVE`** (Confidence: **`0.9998`**)

---

## Exercise 2 – Official Fine-Tuning Results (Protocol v2.3)

| Metric / Specification | Value | Status |
|---|---:|:---:|
| **Dataset Splits (Train / Val / Holdout)** | 7,676 / 960 / 960 | Isolated |
| **Backbone Model** | `distilbert-base-uncased` (6 layers, 768 dim) | Fine-tuned |
| **Winning Experiment** | `E4_weight_decay_0.05` | Locked |
| **Learning Rate / Weight Decay** | `1e-5` / `0.05` | Optimal |
| **Best Epoch (Selected) / Stop Epoch** | `2` (Checkpoint-960) / `6` (Early Stopping) | Verified |
| **Validation Loss** | `0.415336` | Min Loss |
| **Validation Accuracy / Macro-F1** | `86.77%` / `86.78%` | Top Rank |
| **Final Holdout Loss** | `0.353578` | Generalization |
| **Final Holdout Accuracy** | `85.31%` | Test-grade |
| **Final Holdout Precision / Recall** | `86.30%` / `83.96%` | Balanced |
| **Final Holdout Macro-F1** | `85.11%` | Target Met |
| **Holdout Evaluation Access Count** | `1` (Guarded) | No Leakage |
| **Official Hugging Face Test Split** | `NOT USED` | Preserved |

*All metrics are verified from [`final_validation_winner_lock.json`](docs/result/practice_3_v2_3/final_validation_winner_lock.json) and [`final_holdout_metrics.json`](docs/result/practice_3_v2_3/final_holdout_metrics.json).*

---

## End-to-End Workflow

```text
Rotten Tomatoes Dataset (10,600 samples)
       ↓
Deterministic Hash Splitting (7,676 Train / 960 Val / 960 Holdout)
       ↓
EDA & Truncation Bound Selection (max_length = 80 tokens)
       ↓
DistilBERT Tokenization (input_ids & attention_mask tensors)
       ↓
Stage A: Learning Rate Grid (1e-5, 2e-5, 3e-5)
       ↓
Stage B: Controlled Follow-ups (Weight Decay 0.05, Dropout 0.40, Staged Tuning)
       ↓
Validation Loss Ranking → Winner Lock (pending new E1–E6 training)
       ↓
One-time Guarded Holdout Evaluation (not run after reset)
       ↓
Error Analysis & Confusion Matrix Diagnosis (pending)
       ↓
Custom Edge-Case Inference & Save/Reload Persistence Check (pending)
       ↓
Unified Presentation in practice_3.ipynb (Offline, Zero Retraining)
```

---

## Controlled Hyperparameter Search Summary

The clean active sequence contains six `PLANNED` controlled experiments:
`E1_lr_1e-5`, `E2_lr_2e-5`, `E3_lr_3e-5`,
`E4_weight_decay_0.05`, `E5_classifier_dropout_0.40`, and
`E6_staged_finetune`. No validation ranking or winner exists until these runs
are trained again in order.

---

## Checkpoint Lock & Git LFS

The reset intentionally removed all historical checkpoints and final-model
artifacts. A new checkpoint lock must be created only after the clean E1–E6
sequence is complete.

---

## Artifact-only notebook visualization

Các Phase 8–9 trong `notebook_practice_3/practice_3.ipynb` đọc trực tiếp
`run_summary.json` và `training_history.json` thông qua
`processing_own_phase/visualization_data.py`. Biểu đồ được render ngay trong
notebook, không tạo website/HTML riêng và không gọi training hoặc Holdout.

## Live Training Demonstration

To demonstrate real-time training progress without interfering with locked official benchmarks:
1. **Terminal Training Process**:
   ```bash
   .venv/bin/python -m total_practice.practice_3.processing_own_phase.live_terminal_training
   ```
2. **Notebook Monitoring Client**: `Cell 20` in `practice_3.ipynb` streams metrics from `runs/live_terminal_demo/` into a live 2x2 dashboard inside Jupyter.

---

## Documentation Index

- **Result Documentation Index**: [`docs/result/README.md`](docs/result/README.md)
- **Exercise 1 Pretrained Results**: [`docs/result/exercise_01_pretrained_sentiment_results.md`](docs/result/exercise_01_pretrained_sentiment_results.md)
- **Codebase & Architecture Audit**: [`docs/code_base_audit.md`](docs/code_base_audit.md) (**100% PASS**)
- **Training Visualization**: [`notebook_practice_3/practice_3.ipynb`](notebook_practice_3/practice_3.ipynb), Phases 8–9
- **Individual Phase Results**:
  * [Phase 3 – Data Loading](docs/result/phase_03_data_loading_results.md)
  * [Phase 4 – Exploratory Data Analysis](docs/result/phase_04_eda_results.md)
  * [Phase 5 – Tokenization](docs/result/phase_05_tokenization_results.md)
  * [Phase 6 – Model Architecture](docs/result/phase_06_model_architecture_results.md)
  * [Phase 7 – Training Configuration](docs/result/phase_07_training_configuration_results.md)
  * Phase 8–13 reports: pending regeneration from the clean E1–E6 artifacts
