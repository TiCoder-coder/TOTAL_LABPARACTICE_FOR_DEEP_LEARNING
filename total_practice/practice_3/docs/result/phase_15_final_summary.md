# Practice 3 — Final Summary

**Status:** PASS — PRACTICE 3 COMPLETE

## Pipeline

| Phase | Block | Processing | Verified output | Status |
|---|---|---|---|---|
| 0–1 | Overview & Environment | Define the two-exercise workflow and reproducible runtime | Environment/package contract | PASS |
| 2 | Pretrained Inference | Run the already fine-tuned SST-2 sentiment model | Sentiment labels and confidence examples | PASS |
| 3 | Tokenization Investigation | Inspect tokens, IDs, masks and decode behavior | Verified tokenizer understanding | PASS |
| 4 | Dataset Loading | Load and verify Rotten Tomatoes fixed splits | Train/Validation/Test contract | PASS |
| 5 | EDA & Sanity Checks | Inspect text quality, labels, duplicates, overlaps and lengths | Evidence for maximum token length | PASS |
| 6 | Preprocessing | Tokenize all splits with truncation and dynamic padding | Verified model-ready inputs | PASS |
| 7 | Model Construction | Attach a two-label classification head to generic DistilBERT | Forward-pass-verified baseline model | PASS |
| 8 | Metrics & Configuration | Define metrics, training arguments and checkpoint policy | Locked baseline configuration | PASS |
| 9 | Fine-Tuning | Train for three epochs using Train and Validation only | History and three checkpoints | PASS |
| 10 | Learning Curves | Analyze artifact-only training/Validation trajectories | Selected epoch and cautious generalization evidence | PASS |
| 11 | Validation & Final Test | Verify checkpoint reload on Validation, then evaluate Test once | Frozen final metrics and predictions | PASS |
| 12 | Error Analysis | Derive confusion matrix and deterministic errors from frozen predictions | Class/error evidence without reevaluation | PASS |
| 13 | New-Sentence Inference | Run guarded inference on four custom sentences | Verified custom prediction artifact | PASS |
| 14 | Save & Reload | Save a reusable local package and prove exact equivalence | Integrity-verified local model/tokenizer package | PASS |

## Final Model

- Architecture: DistilBERT for sequence classification
- Selected checkpoint: `checkpoint-1068`
- Epoch / step: 2 / 1068
- Selection reason: minimum eval_loss
- Selected Validation loss: 0.3904653192
- Selected Validation F1: 0.8579387187
- Test used for selection: False
- Authoritative SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`

## Validation and Final Test Metrics

| Metric | Validation | Final Test |
|---|---:|---:|
| Loss | 0.3904653192 | 0.4485123158 |
| Accuracy | 0.8564727955 | 0.8452157598 |
| Precision | 0.8492647059 | 0.8458646617 |
| Recall | 0.8667917448 | 0.8442776735 |
| F1 | 0.8579387187 | 0.8450704225 |

The official Test split was evaluated exactly once after the Validation reload gate passed; it was not used for checkpoint selection.

## Training and Generalization

Epoch 1 to 2 improves both fit and Validation behavior. From epoch 2 to 3, Train loss continues to decrease while Validation loss increases and Validation Accuracy/F1 do not improve. This indicates generalization begins to worsen after epoch 2 and is consistent with the onset of overfitting. With only three epochs and one run, this is limited evidence rather than a definitive broad conclusion.

Evidence limit: Three epochs and one seeded baseline run are limited evidence; no broad conclusion is claimed.

- [Loss curves](phase_10_loss_curves.png)
- [Validation metrics](phase_10_validation_metrics.png)

## Confusion Matrix and Error Analysis

- TN=451, FP=82, FN=83, TP=450
- Total=1066, correct=901, incorrect=165
- NEGATIVE→POSITIVE errors: 82
- POSITIVE→NEGATIVE errors: 83
- [Confusion matrix figure](phase_12_confusion_matrix.png)

The frozen artifact contains 165 errors: 82 NEGATIVE→POSITIVE and 83 POSITIVE→NEGATIVE. The highest-confidence error is 0.9915. Length, confidence and lexical flags below are descriptive co-occurrences only; they do not establish that a text feature caused an error.

## Custom Inference Evidence

| Text | Prediction | Confidence |
|---|---|---:|
| The performances were warm, convincing, and deeply moving. | POSITIVE | 0.993701 |
| The story was tedious, predictable, and painfully slow. | NEGATIVE | 0.987616 |
| It is not a bad movie at all. | POSITIVE | 0.925192 |
| The acting is excellent, but the plot is disappointingly shallow. | NEGATIVE | 0.977097 |

## Save/Reload Evidence

- Local package: `/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/total_practice/practice_3/docs/result/phase_14_saved_model`
- Exact-equal state tensors: 104
- Maximum logits difference: 0.0
- Maximum probability difference: 0.0
- Maximum confidence difference: 0.0
- Package SHA-256: `22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660`

## Limitations

- The baseline was trained for only three epochs.
- Fine-tuning evidence comes from one DistilBERT baseline configuration and one full seeded run.
- No broad hyperparameter search or multi-seed robustness study was performed.
- The implemented task is English binary sentiment classification only.
- Evaluation evidence is tied to the Rotten Tomatoes dataset and its fixed splits.
- Mixed sentiment, negation and high-confidence errors remain observable challenges; descriptive flags do not establish causes.
- The project reports aggregate evaluation and artifact-based error analysis, not production monitoring.
- A local package was verified, but no production API, serving infrastructure, load/latency test, drift monitoring, security review or deployment was implemented.

## Final Conclusion

Phase 0–14 evidence is internally consistent and PASS; Practice 3 implementation is complete without new model execution in Phase 15.

Phase 15 was artifact-only: no model/dataset loading, training, inference, Validation/Test evaluation, checkpoint change or package resave occurred.
