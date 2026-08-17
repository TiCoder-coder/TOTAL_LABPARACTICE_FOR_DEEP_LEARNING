# Practice 3 Comprehensive Codebase & Presentation Audit

**Audit Date**: 2026-08-17  
**Protocol Version**: Practice 3 v2.3 Protocol (`practice_3_v2.3`)  
**Scope**: End-to-End Pipeline Audit across Notebook Presentation (`notebook_practice_3/practice_3.ipynb`), Core Python Modules (`processing_own_phase/`), Authoritative Result Artifacts (`docs/result/practice_3_v2_3/`), and Documentation Results (`docs/result/phase_*.md`).

---

## 1. Overall Audit Summary

```text
Notebook execution flow:               PASS
Notebook → Python module separation:  PASS
Data split protocol:                   PASS
EDA + HTML Visualization:              PASS
Tokenization:                          PASS
Model Architecture:                    PASS
Training Configuration:                PASS
Hyperparameter Search:                 PASS
Best Epoch Selection:                  PASS
Final Holdout Evaluation:              PASS
Error Analysis:                        PASS
Custom Inference:                      PASS
Save / Reload Verification:            PASS
Result Documentation:                  PASS
```

**Overall Status**: **100% PASS** (13/13 Phases verified with authoritative evidence, zero blocking issues).

---

## 2. Source of Truth Hierarchy

When resolving any potential discrepancies between historical text and active code, the following hierarchy strictly governs:
1. **Current Notebook Outputs**: [`notebook_practice_3/practice_3.ipynb`](../notebook_practice_3/practice_3.ipynb) (Rendered tables, metrics, and visualization dashboards).
2. **Locked Experiment Artifacts**: [`docs/result/practice_3_v2_3/`](./result/practice_3_v2_3/) (Cryptographically locked JSON manifests, ranking CSVs, and model binaries).
3. **Python Processing Modules**: [`processing_own_phase/`](../processing_own_phase/) (Deterministic protocol logic, training runners, and evaluation guards).
4. **Result Documentation**: [`docs/result/README.md`](./result/README.md) & [`docs/result/phase_*.md`](./result/) (Formal evidence traceability records).

---

## 3. Central Audit Matrix (13 Phases + Orchestration)

| Phase | Notebook Cell | Processing Source | Main Evidence | Result Doc | Status |
|---|---|---|---|---|:---:|
| **Phase 1 – Problem Definition** | [`Cell 01`](../notebook_practice_3/practice_3.ipynb) (Markdown) | [`config.py`](../processing_own_phase/config.py) | Problem Contract Table (Binary classification, `[N, 80]` tensors, DistilBERT) | Notebook Cell 01 | **PASS** |
| **Phase 2 – Environment Setup** | [`Cell 02`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 03`](../notebook_practice_3/practice_3.ipynb) (Code) | [`config.py`](../processing_own_phase/config.py) | Styled `colored_table` + GPU evidence table (`mps` Apple GPU through Metal) | Notebook Cell 03 | **PASS** |
| **Phase 3 – Data Loading** | [`Cell 04`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 05`](../notebook_practice_3/practice_3.ipynb) (Code) | [`dataset_protocol_v2.py`](../processing_own_phase/dataset_protocol_v2.py) | `Dataset split summary` table (Train: 7,676, Val: 960, Holdout: 960) | [`phase_03_data_loading_results.md`](./result/phase_03_data_loading_results.md) | **PASS** |
| **Phase 4 – Exploratory Data Analysis** | [`Cell 06`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 07`](../notebook_practice_3/practice_3.ipynb) (Code) | [`eda_visualization.py`](../processing_own_phase/eda_visualization.py) | `Rotten Tomatoes Dataset Summary` + Clean Academic HTML EDA Dashboard | [`phase_04_eda_results.md`](./result/phase_04_eda_results.md) | **PASS** |
| **Phase 5 – Tokenization** | [`Cell 08`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 09`](../notebook_practice_3/practice_3.ipynb) (Code) | [`pretraining_integration_v2.py`](../processing_own_phase/pretraining_integration_v2.py) | Sample raw text + `Tokenization Example` table (`distilbert-base-uncased` WordPiece IDs) | [`phase_05_tokenization_results.md`](./result/phase_05_tokenization_results.md) | **PASS** |
| **Phase 6 – Model Architecture** | [`Cell 10`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 11`](../notebook_practice_3/practice_3.ipynb) (Code) | [`pretraining_integration_v2.py`](../processing_own_phase/pretraining_integration_v2.py) | `Model Architecture Configuration` table (6 layers, 768 dim, 12 heads, 66.95M params) | [`phase_06_model_architecture_results.md`](./result/phase_06_model_architecture_results.md) | **PASS** |
| **Phase 7 – Training Configuration** | [`Cell 12`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 13`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_protocol_v2.py`](../processing_own_phase/experiment_protocol_v2.py), [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py) | `Training Configuration` table (Max 15 epochs, Patience 4, Batch 16, Warmup 720) | [`phase_07_training_configuration_results.md`](./result/phase_07_training_configuration_results.md) | **PASS** |
| **Phase 8 – Hyperparameter Search** | [`Cell 14`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 15`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py), [`experiment_registry_v2.py`](../processing_own_phase/experiment_registry_v2.py) | `Final Validation Ranking` table + Validation Loss comparison chart | [`phase_08_hyperparameter_search_results.md`](./result/phase_08_hyperparameter_search_results.md) | **PASS** |
| **Live Training Demonstration** | [`Cell 16`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 17`](../notebook_practice_3/practice_3.ipynb) (Code) | [`live_training_monitor.py`](../processing_own_phase/live_training_monitor.py) | Live terminal training metrics synchronization monitor | Notebook Cell 17 | **PASS** |
| **Phase 9 – Learning Curves & Overfitting** | [`Cell 18`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 19`](../notebook_practice_3/practice_3.ipynb) & [`Cell 21`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py) | 9 graphical plots (`winner_train_val_loss.png`, `experiment_val_loss_curves.png`, etc.) | [`phase_09_learning_curve_results.md`](./result/phase_09_learning_curve_results.md) | **PASS** |
| **Phase 10 – Final Holdout Evaluation** | [`Cell 22`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 23`](../notebook_practice_3/practice_3.ipynb) (Code) | [`final_holdout_evaluation_v2_3.py`](../processing_own_phase/final_holdout_evaluation_v2_3.py), [`holdout_guard_v2.py`](../processing_own_phase/holdout_guard_v2.py) | `Validation vs Holdout Metrics` table (Holdout Acc: 85.31%, F1: 0.8511) | [`phase_10_holdout_evaluation_results.md`](./result/phase_10_holdout_evaluation_results.md) | **PASS** |
| **Phase 11 – Error Analysis** | [`Cell 24`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 25`](../notebook_practice_3/practice_3.ipynb) & [`Cell 26`](../notebook_practice_3/practice_3.ipynb) (Code) | [`final_holdout_evaluation_v2_3.py`](../processing_own_phase/final_holdout_evaluation_v2_3.py) | Confusion matrix plot, Confidence histogram, FP/FN/High Conf error tables | [`phase_11_error_analysis_results.md`](./result/phase_11_error_analysis_results.md) | **PASS** |
| **Phase 12 – Custom Inference** | [`Cell 27`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 28`](../notebook_practice_3/practice_3.ipynb) (Code) | [`custom_inference_v2_3.py`](../processing_own_phase/custom_inference_v2_3.py) | `Custom Inference Results` table (9 test sentences across 6 linguistic categories) | [`phase_12_custom_inference_results.md`](./result/phase_12_custom_inference_results.md) | **PASS** |
| **Phase 13 – Save / Reload Verification** | [`Cell 29`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 30`](../notebook_practice_3/practice_3.ipynb) (Code) | [`save_reload_verification_v2_3.py`](../processing_own_phase/save_reload_verification_v2_3.py) | `Save / Reload Evidence` table (0 parameter mismatches, 9/9 prediction matches) | [`phase_13_save_reload_verification_results.md`](./result/phase_13_save_reload_verification_results.md) | **PASS** |

---

## 4. Architectural Separation Audit

The codebase strictly adheres to the Clean Architecture separation principle:
- **Notebook Layer (`notebook_practice_3/practice_3.ipynb`)**: Operates exclusively as a presentation and orchestration layer. It loads precomputed artifacts, renders styled DataFrames and figures, and contains zero duplicate training loops or heavy logic blocks.
- **Processing Layer (`processing_own_phase/`)**: Contains modular, testable, and reusable Python scripts.
  * `dataset_protocol_v2.py`: Deterministic data splitting with hash verification.
  * `eda_visualization.py`: Self-contained, offline HTML/CSS dashboard generator (`render_eda_dashboard()`).
  * `pretraining_integration_v2.py`: Model instantiation and tokenizer verification.
  * `experiment_runner_v2_3.py`: Automated controlled hyperparameter grid training.
  * `holdout_guard_v2.py` & `final_holdout_evaluation_v2_3.py`: Strict one-time holdout access enforcement.
  * `custom_inference_v2_3.py`: Real-time inference on arbitrary review text.
  * `save_reload_verification_v2_3.py`: Exact parameter and probability numerical consistency verification.
- **Architectural Status**: **PASS**.

---

## 5. Detailed Phase Audit Findings

### Phase 1–2: Problem Definition & Environment
- **Task Contract**: Supervised binary sentiment classification on movie reviews (`0: NEGATIVE`, `1: POSITIVE`).
- **Input Contract**: Sequences padded/truncated to `[N, 80]` tensors via `distilbert-base-uncased`.
- **Hardware Acceleration**: Confirmed active on `mps` (Apple Metal GPU).
- **Status**: **PASS**.

### Phase 3: Data Loading & Split Isolation
- **Dataset**: `cornell-movie-review-data/rotten_tomatoes`.
- **Split Distribution**: Total Development Pool = 9,596 samples.
  * **Train**: 7,676 samples (80.0%) — 3,838 Negative / 3,838 Positive.
  * **Validation**: 960 samples (10.0%) — 480 Negative / 480 Positive.
  * **Holdout**: 960 samples (10.0%) — 480 Negative / 480 Positive.
- **Official Test Exclusion**: Official Hugging Face test split (1,066 samples) is strictly ignored.
- **Data Leakage Control**: **PASS** (Zero cross-split ID or text overlap verified).

### Phase 4: Exploratory Data Analysis & HTML Dashboard
- **HTML Visualization**: Integrated via `eda_visualization.py` (`render_eda_dashboard()`).
- **Dashboard Features**: Responsive split overview cards, clean progress bars for class balance, statistical table for character/word lengths, and sample review cards.
- **Context Correction**: All references to CIFAR-10 have been completely eliminated (`Rotten Tomatoes Dataset Summary`).
- **Text Statistics**: Mean char length = 114.0, Median = 111.0, Range = `[4, 267]`. Mean word count = 21.0, Median = 20.0, Range = `[1, 59]`. Safe token bound = 80 tokens (100% sample coverage).
- **Status**: **PASS**.

### Phase 5–6: Tokenization & Model Architecture
- **Tokenizer**: WordPiece algorithm with 30,522 vocabulary size, dynamic padding.
- **Model**: `DistilBertForSequenceClassification` loaded dynamically from [`final_saved_model/config.json`](./result/practice_3_v2_3/final_saved_model/config.json).
- **Parameters**: 6 Transformer encoder layers, 768 hidden dimension, 12 attention heads, 3,072 intermediate feed-forward dimension, 66,955,010 total parameters (100% fine-tuned).
- **Status**: **PASS**.

### Phase 7–8: Training Configuration & Hyperparameter Search
- **Training Budget**: Maximum 15 epochs, Early Stopping patience = 4 on `eval_loss`, Batch size = 16, Warmup steps = 720, Label smoothing = 0.1, Optimizer = AdamW.
- **Primary Selection Metric**: `Validation Loss (lower is better)`.
- **Winning Configuration**: **`E4_weight_decay_0.05`**
  * Learning Rate: `1e-5`
  * Weight Decay: `0.05`
  * Classifier Dropout: `0.2`
  * Strategy: `full`
  * Best Validation Loss: **`0.4153`** (Lowest across all 6 runs)
  * Best Validation Accuracy: **`0.8677`** (86.77%)
  * Best Validation Macro-F1: **`0.8678`** (86.78%)
- **Status**: **PASS**.

### Phase 9: Learning Curves & Best Epoch Distinction
- **Maximum Epoch Budget**: 15 epochs.
- **Selected Best Epoch**: **Epoch 2.0** (Checkpoint-960 with minimum validation loss `0.4153`).
- **Actual Stop Epoch**: **Epoch 6.0** (Early stopping safely halted training after 4 consecutive non-improving epochs).
- **Overfitting Control**: **PASS** (Best Epoch $\ne$ Stop Epoch distinction verified; checkpoint locked at true loss minimum before overfitting divergence).

### Phase 10: Final Holdout Evaluation
- **Evaluation Discipline**: Single, locked evaluation executed strictly once on the sealed Holdout split (960 samples).
- **Holdout Performance**:
  * Holdout Loss: **`0.3536`**
  * Holdout Accuracy: **`0.8531`** (85.31%)
  * Holdout Macro-F1: **`0.8511`** (85.11%)
  * Holdout Precision: **`0.8630`**
  * Holdout Recall: **`0.8396`**
- **Generalization Gap**: Validation Acc 86.77% vs Holdout Acc 85.31% ($\Delta = -1.46\%$). Minimal gap confirms out-of-sample robustness without over-optimistic tuning bias.
- **Status**: **PASS**.

### Phase 11: Error Analysis
- **Total Errors**: 141 out of 960 (Error Rate = 14.69%).
- **Confusion Matrix**: True Negatives = 416, True Positives = 403, False Positives = 64, False Negatives = 77 (symmetric error distribution).
- **Calibration**: Brier Score = 0.1092, ECE = 0.0422. Overall confidence = 0.8489; error confidence = 0.7483 (errors are concentrated near the 0.5 decision boundary).
- **Status**: **PASS**.

### Phase 12: Custom Inference
- **Test Suite**: 9 handcrafted sentences across categories (STANDARD_POSITIVE, STANDARD_NEGATIVE, SUBTLE_POSITIVE, SUBTLE_NEGATIVE, AMBIGUOUS, CONTRAST, NEGATION).
- **Behavior**: High confidence on standard and negation patterns (90–98%), appropriate uncertainty on subtle mixed reviews (~54–58%).
- **Status**: **PASS**.

### Phase 13: Save / Reload Verification
- **Model Bundle**: [`final_saved_model/`](./result/practice_3_v2_3/final_saved_model/) (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`).
- **Verification Metrics**:
  * Parameter mismatch count: **`0`** (66,955,010 parameters checked, max diff = `0.000000`).
  * Prediction matches: **`9 / 9`** (100% agreement on test suite).
  * Max probability difference: **`0.000000`** (Exact floating-point equivalence).
- **Status**: **PASS**.

---

## 6. Result Documentation Coverage

Authoritative Markdown result documentation is maintained in [`docs/result/`](./result/README.md) with 100% verifiable relative hyperlinks:

| Result File | Phase Reference | Evidence Verified | Link Status |
|---|---|---|:---:|
| [`phase_03_data_loading_results.md`](./result/phase_03_data_loading_results.md) | Phase 3 (Cell 05) | Split summary table & dataset split reference | **PASS** |
| [`phase_04_eda_results.md`](./result/phase_04_eda_results.md) | Phase 4 (Cell 07) | Clean Academic HTML Dashboard & dataset stats | **PASS** |
| [`phase_05_tokenization_results.md`](./result/phase_05_tokenization_results.md) | Phase 5 (Cell 09) | Tokenization mapping demo table | **PASS** |
| [`phase_06_model_architecture_results.md`](./result/phase_06_model_architecture_results.md) | Phase 6 (Cell 11) | DistilBERT model architecture config table | **PASS** |
| [`phase_07_training_configuration_results.md`](./result/phase_07_training_configuration_results.md) | Phase 7 (Cell 13) | Training hyperparameters table | **PASS** |
| [`phase_08_hyperparameter_search_results.md`](./result/phase_08_hyperparameter_search_results.md) | Phase 8 (Cell 15) | Final validation ranking & Winner E4 lock | **PASS** |
| [`phase_09_learning_curve_results.md`](./result/phase_09_learning_curve_results.md) | Phase 9 (Cell 19, 21) | 9 graphical trajectory figures | **PASS** |
| [`phase_10_holdout_evaluation_results.md`](./result/phase_10_holdout_evaluation_results.md) | Phase 10 (Cell 23) | Holdout evaluation metrics table | **PASS** |
| [`phase_11_error_analysis_results.md`](./result/phase_11_error_analysis_results.md) | Phase 11 (Cell 25, 26) | Confusion matrix & 3 error DataFrames | **PASS** |
| [`phase_12_custom_inference_results.md`](./result/phase_12_custom_inference_results.md) | Phase 12 (Cell 28) | Custom test predictions table | **PASS** |
| [`phase_13_save_reload_verification_results.md`](./result/phase_13_save_reload_verification_results.md) | Phase 13 (Cell 30) | Save/Reload numerical verification table | **PASS** |
| [`README.md`](./result/README.md) | Index Overview | Comprehensive traceability index | **PASS** |

---

## 7. Remaining Issues / Risks

- **Blocking Issues**: **None**. (All 13 phases execute and render cleanly without errors or data leakage).
- **Non-blocking Historical Notes**:
  * Directory [`save_process_proceduce_own_phase_refactor&fix/`](./save_process_proceduce_own_phase_refactor&fix/) contains historical development logs from earlier iterations (v1 and v2). The canonical production protocol is strictly isolated in [`docs/result/practice_3_v2_3/`](./result/practice_3_v2_3/).
