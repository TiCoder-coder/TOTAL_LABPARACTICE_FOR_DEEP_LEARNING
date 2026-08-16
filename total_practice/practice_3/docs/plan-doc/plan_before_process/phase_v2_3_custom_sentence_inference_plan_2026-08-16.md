# Phase v2.3 Custom Sentence Inference Plan

**Date:** 2026-08-16
**Protocol:** practice_3_v2.3

## 1. Purpose
The purpose of this phase is to demonstrate the final locked model's practical inference behavior on completely new, user-defined sentences. This simulates real-world usage and provides an intuitive demonstration of tokenization and sentiment prediction.

## 2. Source of Custom Sentences
I will manually craft 8-10 custom sentences spanning various linguistic scenarios:
- Clear Positive / Clear Negative
- Negation
- Mixed Sentiment
- Subtle Positive / Subtle Negative
- Ambiguous
- Complex / Contrast
These will be entirely new and not derived from Train, Validation, Holdout, or the Official Test split.

## 3. Winner Checkpoint
The inference will exclusively use the locked `E4_weight_decay_0.05` checkpoint:
- Path: `total_practice/practice_3/runs/practice_3_v2_3/E4_weight_decay_0.05/checkpoints/checkpoint-960`
- SHA256: `d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191`

## 4. Preprocessing & Tokenization
- The `distilbert-base-uncased` tokenizer will be loaded.
- For one representative sentence, the tokenization process (String → Tokens → Input IDs / Attention Mask) will be explicitly demonstrated and saved for notebook presentation.

## 5. Inference Method & Confidence Calculation
- The model will run in `eval()` mode with `torch.no_grad()`.
- `Trainer` methods and `Dataset` objects will be bypassed to emphasize raw manual inference.
- Logits will be passed through a softmax function to obtain raw probabilities.
- Predicted class = `argmax(probabilities)`
- Confidence = `max(probabilities)`

## 6. Strict Rules & Data Protection
- **No Training**: The model parameters, hyperparameters, and winner metadata will not be changed.
- **Holdout / Test Protection**: Neither the Holdout split nor the Official Test split will be loaded, accessed, or inferred during this phase.

## 7. Expected Artifacts
- `custom_inference_inputs.json`: The crafted input sentences and expected categories.
- `custom_inference_v2_3.py`: The python script performing the inference.
- `custom_inference_results.csv` and `custom_inference_results.json`: Detailed inference outputs.
- `custom_inference_analysis.md`: Qualitative breakdown of model behavior.
- `practice_3_v2_3_custom_sentence_inference_process_2026-08-16.md`: Process log.
- A concise text/JSON demonstration of the inference flow and tokenization for a single sentence.
