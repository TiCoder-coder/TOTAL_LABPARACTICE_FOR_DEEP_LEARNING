# Phase v2.3 Current Flow Plan (2026-08-16)

## 1. Purpose of current_flow
`practice_3_current_flow.md` serves as the final, canonical workflow explanation for Practice 3 v2.3. It answers WHAT happens, WHY it happens, WHAT data is used, WHAT artifacts are produced, and WHAT decisions are made at each major stage. It is not a debugging history.

## 2. Authoritative Sources
The flow will be based strictly on the final locked artifacts:
- `final_summary.md`
- `final_validation_ranking.json`
- `final_validation_winner_lock.json`
- `final_holdout_metrics.json`
- `final_holdout_error_analysis.md`
- `custom_inference_analysis.md`
- `save_reload_verification.json`
- `code_base_audit.md`

## 3. Target Structure
The flow will trace 16 precise phases:
1. Problem Definition
2. Environment Setup
3. Pretrained Sentiment Demo
4. Dataset Loading (Train/Val/Holdout roles)
5. EDA
6. Preprocessing & Tokenization
7. DistilBERT Model init
8. Training Configuration
9. Hyperparameter Search (Stages A & B)
10. Validation Ranking & Learning Curves
11. Winner Lock
12. ONE-TIME Holdout Evaluation
13. Confusion Matrix + Error Analysis
14. Custom Sentence Inference
15. Save / Reload Verification
16. Final Notebook Presentation

## 4. Level of Detail
- Clear, step-by-step logic (WHAT, WHY, INPUT, PROCESS, OUTPUT, DECISION).
- No massive JSON blocks or traceback dumps.
- Focus on the *why* of the architecture decisions (e.g., why early stopping, why cross-entropy loss).

## 5. Historical Information Policy
- E5, E6, and E6b will be mentioned briefly as excluded historical experiments, keeping focus on the 6 valid experiments.
- Previous debug attempts are omitted in favor of the finalized, canonical process.

## 6. Safety Constraints
- Read-only execution. No `Trainer.train()` or `Trainer.evaluate()` calls.
- Holdout evaluation count remains at 1. Official Hugging Face Test remains NOT USED.

## 7. Completion Criteria
- Generation of `docs/current_flow/practice_3_current_flow.md`.
- Process logged in `practice_3_v2_3_current_flow_process_2026-08-16.md`.
- Absolute consistency with final metrics.
