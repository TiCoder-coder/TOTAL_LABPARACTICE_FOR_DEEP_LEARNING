# Practice 3 Result Documentation Index

This directory contains authoritative result documentation files for Practice 3 (Get Started with Hugging Face & Fine-Tuning). Each documentation file maps directly to an assignment exercise / phase, its execution cell, output evidence, processing scripts, and artifact sources.

## Assignment Structure & Results Provenance Table

### Exercise 1: Sentiment Analysis with Hugging Face (Pretrained Zero-Shot)
| Part | Title | Documentation File | Main Evidence in Notebook | Authoritative Artifact |
|:---:|---|---|---|---|
| **Exercise 1** | Pretrained Hub Sentiment Analysis | [`exercise_01_pretrained_sentiment_results.md`](./exercise_01_pretrained_sentiment_results.md) | [`Cell 02`](../../notebook_practice_3/practice_3.ipynb): Tokenization & Inference tables | [`exercise_1_pretrained_sentiment.json`](./practice_3_v2_3/exercise_1_pretrained_sentiment.json) |

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
| Phase | Assignment Requirement | Documentation File | Main Evidence in Notebook | Authoritative Artifact |
|:---:|---|---|---|---|
| **Phase 3** | Req 2: Load Binary Dataset | [`phase_03_data_loading_results.md`](./phase_03_data_loading_results.md) | [`Cell 08`](../../notebook_practice_3/practice_3.ipynb): Dataset split summary table | [`dataset_split_reference.json`](./practice_3_v2_3/dataset_split_reference.json) |
| **Phase 4** | Req 4: Preprocess Dataset (EDA) | [`phase_04_eda_results.md`](./phase_04_eda_results.md) | [`Cell 10`](../../notebook_practice_3/practice_3.ipynb): HTML EDA Dashboard | [`eda_visualization.py`](../../processing_own_phase/eda_visualization.py) |
| **Phase 5** | Req 3 & 4: Tokenization & Preprocess | [`phase_05_tokenization_results.md`](./phase_05_tokenization_results.md) | [`Cell 12`](../../notebook_practice_3/practice_3.ipynb): Tokenization mapping table | [`tokenization_demo.json`](./practice_3_v2_3/tokenization_demo.json) |
| **Phase 6** | Req 3: Pretrained Model Architecture | [`phase_06_model_architecture_results.md`](./phase_06_model_architecture_results.md) | [`Cell 14`](../../notebook_practice_3/practice_3.ipynb): Architecture config table | [`final_saved_model/config.json`](./practice_3_v2_3/final_saved_model/config.json) |
| **Phase 7** | Req 5: Define Training Arguments | [`phase_07_training_configuration_results.md`](./phase_07_training_configuration_results.md) | [`Cell 16`](../../notebook_practice_3/practice_3.ipynb): Training hyperparameter table | [`training_authorization.json`](./practice_3_v2_3/training_authorization.json) |
| **Phase 8** | Req 6: Trainer & Fine-Tuning | [`phase_08_hyperparameter_search_results.md`](./phase_08_hyperparameter_search_results.md) | [`Cell 18`](../../notebook_practice_3/practice_3.ipynb): Final ranking table & Loss chart | [`final_validation_ranking.csv`](./practice_3_v2_3/final_validation_ranking.csv) |
| **Phase 9** | Req 7: Evaluate Model (Curves) | [`phase_09_learning_curve_results.md`](./phase_09_learning_curve_results.md) | [`Cell 22`](../../notebook_practice_3/practice_3.ipynb) & [`Cell 24`](../../notebook_practice_3/practice_3.ipynb): Training curves & Stop epoch plots | [`figures/`](./practice_3_v2_3/figures/) |
| **Phase 10** | Req 7: Final Holdout Evaluation | [`phase_10_holdout_evaluation_results.md`](./phase_10_holdout_evaluation_results.md) | [`Cell 26`](../../notebook_practice_3/practice_3.ipynb): Validation vs Holdout table | [`final_holdout_metrics.json`](./practice_3_v2_3/final_holdout_metrics.json) |
| **Phase 11** | Req 7: Error Analysis | [`phase_11_error_analysis_results.md`](./phase_11_error_analysis_results.md) | [`Cell 28`](../../notebook_practice_3/practice_3.ipynb) & [`Cell 29`](../../notebook_practice_3/practice_3.ipynb): Confusion matrix & Error tables | [`holdout_errors.csv`](./practice_3_v2_3/holdout_errors.csv) |
| **Phase 12** | Deployment: Custom Inference | [`phase_12_custom_inference_results.md`](./phase_12_custom_inference_results.md) | [`Cell 31`](../../notebook_practice_3/practice_3.ipynb): Custom test predictions table | [`custom_inference_results.csv`](./practice_3_v2_3/custom_inference_results.csv) |
| **Phase 13** | Persistence: Save & Reload | [`phase_13_save_reload_verification_results.md`](./phase_13_save_reload_verification_results.md) | [`Cell 33`](../../notebook_practice_3/practice_3.ipynb): Save/Reload verification table | [`save_reload_verification.json`](./practice_3_v2_3/save_reload_verification.json) |

## Traceability Chain
```text
Markdown Result Documentation (.md)
       ↓
Notebook Exercise & Phase Reference (practice_3.ipynb)
       ↓
Execution Output (HTML Table / Dashboard / Plot)
       ↓
Processing Script & Artifact Data (processing_own_phase/ & docs/result/practice_3_v2_3/)
```
