# Practice 3 Comprehensive Codebase & Presentation Audit

**Audit Date**: 2026-08-17  
**Protocol Version**: Practice 3 v2.3 Protocol (`practice_3_v2.3`)  
**Scope**: End-to-End Assignment Audit across:
- **Exercise 1**: Sentiment Analysis with Hugging Face (Pretrained Hub Zero-Shot)
- **Exercise 2**: Finetuning a Pretrained Model for Binary Text Classification (Phases 1–13)
- **Presentation Notebook**: [`notebook_practice_3/practice_3.ipynb`](../notebook_practice_3/practice_3.ipynb) (34 cells)
- **Python Processing Layer**: [`processing_own_phase/`](../processing_own_phase/)
- **Authoritative Result Artifacts**: [`docs/result/practice_3_v2_3/`](./result/practice_3_v2_3/)
- **Result Documentation**: [`docs/result/README.md`](./result/README.md) & [`docs/result/`](./result/)

---

## 1. Overall Audit Summary

### Exercise 1: Sentiment Analysis with Hugging Face
```text
Hugging Face Transformers:             PASS
Pretrained Hub Model Loading:          PASS (distilbert-base-uncased-finetuned-sst-2-english)
Sample Sentence Tokenization:          PASS (Subwords, Input IDs, Attention Mask)
Direct Sentiment Pipeline Inference:   PASS (Positive: 0.9999, Negative: 0.9998)
```

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
```text
Requirement 1 (Environment Setup):     PASS (MPS/GPU Hardware verification)
Requirement 2 (Data Loading):          PASS (Rotten Tomatoes 7676 / 960 / 960 isolated splits)
Requirement 3 (Model & Tokenizer):     PASS (distilbert-base-uncased, 6 layers, 66.95M params)
Requirement 4 (Preprocessing & EDA):   PASS (Clean Academic HTML Dashboard, max_len=80)
Requirement 5 (Training Arguments):    PASS (Max 15 epochs, Early Stopping p=4, AdamW)
Requirement 6 (Trainer & Tuning):      PASS (6 Controlled experiments, Winner: E4_weight_decay_0.05)
Requirement 7 (Model Evaluation):      PASS (Holdout Accuracy: 85.31%, F1: 0.8511, ECE: 0.0422)
Verification & Persistence:            PASS (Custom Inference + 0 mismatch Save/Reload)
Result Documentation:                  PASS (12 dedicated .md files, 100% verified links)
```

**Overall Status**: **100% PASS** (All requirements satisfied, zero blocking issues).

---

## 2. Source of Truth Hierarchy

When resolving any potential discrepancies between historical text and active code, the following hierarchy strictly governs:
1. **Current Notebook Outputs**: [`notebook_practice_3/practice_3.ipynb`](../notebook_practice_3/practice_3.ipynb) (Rendered tables, metrics, and visualization dashboards).
2. **Locked Experiment Artifacts**: [`docs/result/practice_3_v2_3/`](./result/practice_3_v2_3/) (Cryptographically locked JSON manifests, ranking CSVs, and model binaries).
3. **Python Processing Modules**: [`processing_own_phase/`](../processing_own_phase/) (Deterministic protocol logic, training runners, and evaluation guards).
4. **Result Documentation**: [`docs/result/README.md`](./result/README.md) & [`docs/result/phase_*.md`](./result/) (Formal evidence traceability records).

---

## 3. Central Assignment Audit Matrix

### Exercise 1: Sentiment Analysis with Hugging Face Hub (Zero-Shot)
| Part | Notebook Cell | Processing Source | Main Evidence | Result Doc | Status |
|---|---|---|---|---|:---:|
| **Exercise 1 – Overview** | [`Cell 01`](../notebook_practice_3/practice_3.ipynb) (Markdown) | [`exercise_1_pretrained_sentiment.py`](../processing_own_phase/exercise_1_pretrained_sentiment.py) | Section header & objective definition | Notebook Cell 01 | **PASS** |
| **Exercise 1 – Tokenization & Inference** | [`Cell 02`](../notebook_practice_3/practice_3.ipynb) (Code) | [`exercise_1_pretrained_sentiment.py`](../processing_own_phase/exercise_1_pretrained_sentiment.py) | Sample tokenization table + Pretrained sentiment predictions table | [`exercise_01_pretrained_sentiment_results.md`](./result/exercise_01_pretrained_sentiment_results.md) | **PASS** |

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
| Phase | Assignment Requirement | Notebook Cell | Processing Source | Main Evidence | Result Doc | Status |
|---|---|---|---|---|---|:---:|
| **Exercise 2 – Overview** | Assignment Mapping | [`Cell 03`](../notebook_practice_3/practice_3.ipynb) (Markdown) | Architecture contract | Requirement-to-Phase mapping table | Notebook Cell 03 | **PASS** |
| **Phase 1 – Problem Definition** | Problem Contract | [`Cell 04`](../notebook_practice_3/practice_3.ipynb) (Markdown) | [`config.py`](../processing_own_phase/config.py) | Problem Contract Table (`[N, 80]` tensors, DistilBERT) | Notebook Cell 04 | **PASS** |
| **Phase 2 – Environment Setup** | **Req 1: Install & Setup** | [`Cell 05`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 06`](../notebook_practice_3/practice_3.ipynb) (Code) | [`config.py`](../processing_own_phase/config.py) | Styled GPU/MPS evidence table (`mps` Apple GPU) | Notebook Cell 06 | **PASS** |
| **Phase 3 – Data Loading** | **Req 2: Load Binary Dataset** | [`Cell 07`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 08`](../notebook_practice_3/practice_3.ipynb) (Code) | [`dataset_protocol_v2.py`](../processing_own_phase/dataset_protocol_v2.py) | `Dataset split summary` table (Train: 7676, Val: 960, Holdout: 960) | [`phase_03_data_loading_results.md`](./result/phase_03_data_loading_results.md) | **PASS** |
| **Phase 4 – Exploratory Data Analysis** | **Req 4: Preprocess (EDA)** | [`Cell 09`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 10`](../notebook_practice_3/practice_3.ipynb) (Code) | [`eda_visualization.py`](../processing_own_phase/eda_visualization.py) | `Rotten Tomatoes Dataset Summary` + Clean Academic HTML EDA Dashboard | [`phase_04_eda_results.md`](./result/phase_04_eda_results.md) | **PASS** |
| **Phase 5 – Tokenization** | **Req 3 & 4: Tokenizer & Preprocess** | [`Cell 11`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 12`](../notebook_practice_3/practice_3.ipynb) (Code) | [`pretraining_integration_v2.py`](../processing_own_phase/pretraining_integration_v2.py) | Sample raw text + `Tokenization Example` table (`distilbert-base-uncased`) | [`phase_05_tokenization_results.md`](./result/phase_05_tokenization_results.md) | **PASS** |
| **Phase 6 – Model Architecture** | **Req 3: Pretrained Architecture** | [`Cell 13`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 14`](../notebook_practice_3/practice_3.ipynb) (Code) | [`pretraining_integration_v2.py`](../processing_own_phase/pretraining_integration_v2.py) | `Model Architecture Configuration` table (6 layers, 768 dim, 66.95M params) | [`phase_06_model_architecture_results.md`](./result/phase_06_model_architecture_results.md) | **PASS** |
| **Phase 7 – Training Configuration** | **Req 5: Training Arguments** | [`Cell 15`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 16`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_protocol_v2.py`](../processing_own_phase/experiment_protocol_v2.py), [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py) | `Training Configuration` table (Max 15 epochs, Patience 4, Batch 16, Warmup 720) | [`phase_07_training_configuration_results.md`](./result/phase_07_training_configuration_results.md) | **PASS** |
| **Phase 8 – Hyperparameter Search** | **Req 6: Trainer & Fine-Tuning** | [`Cell 17`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 18`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py), [`experiment_registry_v2.py`](../processing_own_phase/experiment_registry_v2.py) | `Final Validation Ranking` table + Validation Loss comparison chart | [`phase_08_hyperparameter_search_results.md`](./result/phase_08_hyperparameter_search_results.md) | **PASS** |
| **Live Training Demo** | Monitoring | [`Cell 19`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 20`](../notebook_practice_3/practice_3.ipynb) (Code) | [`live_training_monitor.py`](../processing_own_phase/live_training_monitor.py) | Live terminal training metrics synchronization monitor | Notebook Cell 20 | **PASS** |
| **Phase 9 – Learning Curves & Overfitting** | **Req 7: Evaluate (Curves)** | [`Cell 21`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 22`](../notebook_practice_3/practice_3.ipynb) & [`Cell 24`](../notebook_practice_3/practice_3.ipynb) (Code) | [`experiment_runner_v2_3.py`](../processing_own_phase/experiment_runner_v2_3.py) | 9 graphical trajectory plots (`winner_train_val_loss.png`, etc.) | [`phase_09_learning_curve_results.md`](./result/phase_09_learning_curve_results.md) | **PASS** |
| **Phase 10 – Final Holdout Evaluation** | **Req 7: Holdout Metrics** | [`Cell 25`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 26`](../notebook_practice_3/practice_3.ipynb) (Code) | [`final_holdout_evaluation_v2_3.py`](../processing_own_phase/final_holdout_evaluation_v2_3.py), [`holdout_guard_v2.py`](../processing_own_phase/holdout_guard_v2.py) | `Validation vs Holdout Metrics` table (Holdout Acc: 85.31%, F1: 0.8511) | [`phase_10_holdout_evaluation_results.md`](./result/phase_10_holdout_evaluation_results.md) | **PASS** |
| **Phase 11 – Error Analysis** | **Req 7: Diagnostics** | [`Cell 27`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 28`](../notebook_practice_3/practice_3.ipynb) & [`Cell 29`](../notebook_practice_3/practice_3.ipynb) (Code) | [`final_holdout_evaluation_v2_3.py`](../processing_own_phase/final_holdout_evaluation_v2_3.py) | Confusion matrix plot, Confidence histogram, FP/FN/High Conf error tables | [`phase_11_error_analysis_results.md`](./result/phase_11_error_analysis_results.md) | **PASS** |
| **Phase 12 – Custom Inference** | Deployment Validation | [`Cell 30`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 31`](../notebook_practice_3/practice_3.ipynb) (Code) | [`custom_inference_v2_3.py`](../processing_own_phase/custom_inference_v2_3.py) | `Custom Inference Results` table (9 test sentences across 6 linguistic categories) | [`phase_12_custom_inference_results.md`](./result/phase_12_custom_inference_results.md) | **PASS** |
| **Phase 13 – Save / Reload Verification** | Persistence Integrity | [`Cell 32`](../notebook_practice_3/practice_3.ipynb) (MD), [`Cell 33`](../notebook_practice_3/practice_3.ipynb) (Code) | [`save_reload_verification_v2_3.py`](../processing_own_phase/save_reload_verification_v2_3.py) | `Save / Reload Evidence` table (0 parameter mismatches, 9/9 prediction matches) | [`phase_13_save_reload_verification_results.md`](./result/phase_13_save_reload_verification_results.md) | **PASS** |

---

## 4. Architectural Separation Audit

The codebase strictly adheres to the Clean Architecture separation principle:
- **Notebook Layer (`notebook_practice_3/practice_3.ipynb`)**: Operates exclusively as a presentation and orchestration layer.
- **Processing Layer (`processing_own_phase/`)**: Contains modular, testable, and reusable Python scripts.
  * `exercise_1_pretrained_sentiment.py`: Out-of-the-box Hugging Face Hub pretrained pipeline (`run_exercise_1_pipeline()`).
  * `dataset_protocol_v2.py`: Deterministic data splitting with hash verification.
  * `eda_visualization.py`: Self-contained, offline HTML/CSS dashboard generator (`render_eda_dashboard()`).
  * `pretraining_integration_v2.py`: Model instantiation and tokenizer verification.
  * `experiment_runner_v2_3.py`: Automated controlled hyperparameter grid training.
  * `holdout_guard_v2.py` & `final_holdout_evaluation_v2_3.py`: Strict one-time holdout access enforcement.
  * `custom_inference_v2_3.py`: Real-time inference on arbitrary review text.
  * `save_reload_verification_v2_3.py`: Exact parameter and probability numerical consistency verification.
- **Architectural Status**: **PASS**.

---

## 5. Detailed Findings & Disambiguation

### Disambiguation: Exercise vs Experiment
- **Exercise 1 / Exercise 2**: High-level assignment divisions (Exercise 1 = Hub Pretrained Zero-Shot; Exercise 2 = Transfer Learning & Fine-Tuning).
- **Experiment E1 / E2 / E3 / E4 / E5b / E6c**: Controlled hyperparameter tuning configurations evaluated inside Phase 8 of Exercise 2.
- **Winner Experiment**: **`E4_weight_decay_0.05`** (Best Val Loss: `0.4153`, Val Acc: `86.77%`, Val Macro-F1: `0.8678`).

### Exercise 1 Verification
- **Model Checkpoint**: `distilbert-base-uncased-finetuned-sst-2-english` (loaded directly from Hugging Face Hub).
- **Sample Sentence**: *"This movie is absolutely wonderful and enjoyable."*
- **Tokenization**: 10 tokens `[[101, 2023, 3185, 2003, 7078, 6919, 1998, 22249, 1012, 102]]` with attention mask `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`.
- **Predicted Sentiment**: `POSITIVE` (Confidence: `0.9999`).

### Exercise 2 Holdout & Overfitting Verification
- **Best Epoch vs Stop Epoch**: Selected Best Epoch = `2.0` (Checkpoint-960), Actual Stop Epoch = `6.0` (Early Stopping with patience = 4).
- **Holdout Evaluation**: Single guarded evaluation on 960 unseen samples $\rightarrow$ Loss: `0.3536`, Accuracy: `85.31%`, Macro-F1: `0.8511`. Generalization gap = `-1.46%`.
- **Save / Reload**: 66,955,010 parameters checked with 0 mismatch (`0.000000` difference).

---

## 6. Remaining Issues / Risks

- **Blocking Issues**: **None**. (All 34 cells in `practice_3.ipynb` execute and render cleanly without errors or data leakage).
- **Non-blocking Historical Notes**: Directory [`save_process_proceduce_own_phase_refactor&fix/`](./save_process_proceduce_own_phase_refactor&fix/) contains historical development logs from earlier iterations. Canonical production data is strictly in [`docs/result/practice_3_v2_3/`](./result/practice_3_v2_3/).
