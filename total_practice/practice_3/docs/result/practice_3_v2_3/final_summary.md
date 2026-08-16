# Practice 3 Final Summary (v2.3)

## 1. Problem
Binary sentiment classification.

Input:
English movie review text.

Output:
Negative (0) / Positive (1).

Model:
DistilBERT pretrained Transformer (`distilbert-base-uncased`).

## 2. Dataset Protocol
Development source:
Hugging Face `rotten_tomatoes` official Train + Validation.

Final practice_3_v2.3 protocol split:
Train = 7676
Validation = 960
Holdout = 960

Official Hugging Face Test:
NOT USED.

Explain:
Train: 
optimization and model updating.

Validation: 
hyperparameter search and model selection (Early Stopping).

Holdout: 
final generalization evaluation exactly once.

## 3. Preprocessing
Tokenizer:
`distilbert-base-uncased`

Vocabulary:
30522

Describe:
text
→ tokenizer
→ input_ids, attention_mask

Max sequence length (`max_length`): 80 (based on EDA token distribution).

## 4. Training Setup
Summarize active settings:

learning rate: 1e-5, 2e-5, 3e-5
weight decay: 0.05
batch size: 16
eval batch size: 16
max epochs: 15
warmup steps: 720
label smoothing: 0.1
gradient clipping: 1.0
dropout: 0.20
early stopping patience: 4
fine-tuning strategy: baseline

Explain:
15 epochs is the maximum computational budget.
Early Stopping terminating training if `eval_loss` does not improve for 4 epochs, saving compute and preventing overfitting.

## 5. Hyperparameter Search
Explain Stage A:
Explored Learning Rates:
1e-5 (E1)
2e-5 (E2)
3e-5 (E3)

Explain Stage B:
Explored Regularization and Fine-Tuning:
Weight Decay = 0.05 (E4)
Dropout = 0.40 (E5b)
Staged Fine-tuning (E6c)

Valid runs:
E1, E2, E3, E4, E5b, E6c

Excluded historical runs:
E5: invalid controlled experiment (dropout was accidentally baseline)
E6: failed MPS SDPA (dropout unsupported on MPS backend)
E6b: failed optimizer/scheduler mismatch during unfreezing

## 6. Final Ranking

| Run | Changed Parameter | Best Epoch | Stopped Epoch | Validation Loss | Validation Accuracy | Validation F1 | Rank |
|---|---|---|---|---|---|---|---|
| E4_weight_decay_0.05 | Weight Decay (0.05) | 2 | 6 | 0.4153 | 86.77% | 86.78% | 1 |
| E1_lr_1e-5 | Baseline (1e-5) | 2 | 6 | 0.4285 | 84.79% | 84.86% | 2 |
| E6c_staged_finetune | Staged FT | 4 | 8 | 0.4431 | 86.46% | 86.46% | 3 |
| E5b_classifier_dropout_0.40 | Dropout (0.40) | 2 | 6 | 0.4578 | 85.52% | 85.49% | 4 |
| E2_lr_2e-5 | LR (2e-5) | 1 | 5 | 0.4907 | 84.90% | 84.86% | 5 |
| E3_lr_3e-5 | LR (3e-5) | 1 | 5 | 0.5401 | 83.23% | 83.47% | 6 |

Explain:
`E4_weight_decay_0.05` was selected as the final winner by achieving the lowest Validation Loss.

## 7. Winner
Winner:
`E4_weight_decay_0.05`

LR:
1e-5

Weight Decay:
0.05

Best Epoch:
2

Stopped Epoch:
6

Validation Loss:
0.41533583402633667

Validation Accuracy:
86.77%

Validation F1:
86.78%

## 8. Final Holdout Evaluation
Holdout Loss:
0.3536

Accuracy:
85.31%

Precision:
86.30%

Recall:
83.96%

F1:
85.11%

State clearly:
The Winner was locked BEFORE Holdout evaluation.
Holdout was evaluated exactly once (`evaluation_count = 1`).
Holdout was not used for hyperparameter selection.

## 9. Generalization
Compare Validation vs Holdout:

Validation Accuracy:
86.77%

Holdout Accuracy:
85.31%

Accuracy gap:
≈ 1.46 percentage points

Validation F1:
86.78%

Holdout F1:
85.11%

F1 gap:
≈ 1.67 percentage points

Interpretation:
Generalization remains relatively stable between the validation and unseen dataset.

## 10. Confusion Matrix
TN = 416
FP = 64
FN = 77
TP = 403

Explain:
FN > FP
Therefore the model is slightly more likely to miss Positive reviews than to incorrectly predict Positive sentiment for Negative reviews.

## 11. Error Analysis
Summarize:
141 / 960 errors

Main observed difficulties:
- Subtle/Mixed Sentiment
- Sarcasm/Irony
- Ambiguous Wording

High-confidence errors:
14 errors had confidence >= 0.90.

Limitations:
The model successfully classifies clear sentiment but falters on nuanced language, indicating limits of the base DistilBERT representations or dataset size.

## 12. Custom Inference
Representative examples:

Clear positive:
"The cinematography was absolutely stunning and the plot kept me engaged the entire time."
Predicted: Positive (0.9859)

Clear negative:
"A complete waste of time. The script was awful and the acting was wooden."
Predicted: Negative (0.9897)

Mixed/subtle failure:
"The movie looks beautiful, although there is little else to recommend about it."
Predicted: Positive (0.5888) -> FAILURE (True sentiment is negative)

Custom inference is for demonstration only, not model selection.

## 13. Save / Reload
Status:
PASS

Tokenizer:
30522 → 30522

Parameters compared:
66,955,010

Mismatches:
0

Prediction match:
9 / 9

Max probability difference:
0.0

Conclusion:
The model can be perfectly reloaded and reused without retraining.

## 14. Engineering Note
Final model:
≈ 255.43 MB

GitHub standard file size limit is insufficient.
Git LFS is required if committing the binary.

## 15. Strengths
- pretrained Transformer transfer learning
- controlled parameter search
- Validation-only model selection
- one-time final Holdout protocol
- reproducible save/reload
- artifact-driven final notebook presentation

## 16. Limitations
- struggles with subtle/mixed sentiment
- struggles with ambiguous wording
- struggles with sarcasm/irony
- some confident wrong predictions
- relatively small training dataset

## 17. Future Work
- more training data
- calibration improvement
- alternative fine-tuning strategies
- stronger pretrained language models

## Final Metric Table

| Metric | Validation | Holdout |
|---|---|---|
| Loss | 0.4153 | 0.3536 |
| Accuracy | 86.77% | 85.31% |
| Precision | N/A | 86.30% |
| Recall | N/A | 83.96% |
| F1 | 86.78% | 85.11% |

## Final Parameter Table

| Parameter | Final Value | Meaning |
|---|---|---|
| learning_rate | 1e-5 | Step size during gradient descent |
| weight_decay | 0.05 | L2 regularization penalizing large weights |
| max_epochs | 15 | Maximum number of training passes over the dataset |
| batch_size | 16 | Number of samples per training forward/backward pass |
| eval_batch_size | 16 | Number of samples per evaluation batch |
| warmup_steps | 720 | Gradual learning rate increase to prevent early divergence |
| gradient_clipping | 1.0 | Limits gradient norms to prevent exploding gradients |
| label_smoothing_factor| 0.1 | Softens target probabilities to reduce overconfidence |
| early_stopping_patience| 4 | Epochs without improvement before stopping |
| early_stopping_threshold| 0.0 | Minimum required improvement to reset patience |
| seq_classif_dropout | 0.20 | Dropout probability before classification head |
| attn_implementation | eager | MPS-compatible eager attention calculation |
| fine_tuning_strategy | baseline | Standard end-to-end backpropagation |
