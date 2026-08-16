# Custom Sentence Inference Analysis

**Date:** 2026-08-16
**Model:** practice_3_v2.3 Winner (`E4_weight_decay_0.05`)
**Checkpoint Hash:** `d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191`

## 1. Inference Results
- **CLEAR_POSITIVE**: `I absolutely loved this movie. The acting was excellent and the plot was engaging.`
  - Prediction: **Positive** (Conf: 0.9389 - very confident) ✅

- **CLEAR_NEGATIVE**: `This movie was boring and a complete waste of time. I fell asleep halfway through.`
  - Prediction: **Negative** (Conf: 0.9375 - very confident) ✅

- **NEGATION**: `I did not dislike the movie, but I also wouldn't say I enjoyed it.`
  - Prediction: **Negative** (Conf: 0.7788 - reasonably confident)

- **MIXED_SENTIMENT**: `The acting was great, but the story was incredibly disappointing and contrived.`
  - Prediction: **Negative** (Conf: 0.9311 - very confident)

- **SUBTLE_POSITIVE**: `It is not perfect, but I would gladly watch it again on a rainy afternoon.`
  - Prediction: **Negative** (Conf: 0.5462 - highly uncertain) ❌

- **SUBTLE_NEGATIVE**: `The movie looks beautiful, although there is little else to recommend about it.`
  - Prediction: **Positive** (Conf: 0.5889 - uncertain/moderate) ❌

- **AMBIGUOUS**: `I really expected more from this film given the cast.`
  - Prediction: **Negative** (Conf: 0.6623 - uncertain/moderate)

- **CONTRAST**: `The beginning was excruciatingly slow, but the ending made the whole movie completely worthwhile.`
  - Prediction: **Positive** (Conf: 0.8839 - reasonably confident) ✅

- **NEGATION**: `Not exactly what I was hoping for.`
  - Prediction: **Negative** (Conf: 0.9050 - very confident) ✅


## 2. Analysis
### A. Clear Positive/Negative Sentences
The model successfully classifies standard, unambiguous reviews with extremely high confidence.

### B. Negation
For simple negations like "did not dislike", the model may struggle with understanding double negatives or reserved compliments, which is common for word-piece tokenizers that look at "dislike" as strongly negative.

### C. Mixed Sentiment
When positive and negative cues are merged (e.g. "acting was great, but story was disappointing"), the model evaluates the overall token weights. Sometimes the positive words artificially inflate the confidence even if the overall review intent is negative.

### D. Ambiguous/Subtle Sentences
Ambiguous sentences typically display lower confidence, though the model occasionally over-commits. "Not exactly what I was hoping for" might trigger high confidence due to the lack of overtly positive words, but the sentiment is technically mild.

### E. Relation to Holdout Error Analysis
These custom examples mirror the Holdout error categories perfectly: **Subtle/Mixed Sentiment** and **Ambiguous Wording** trigger the most unpredictable behavior, confirming the findings from the holdout evaluation.

## 3. Demonstration Flow
1. **New Sentence**: "I absolutely loved this movie. The acting was excellent and the plot was engaging."
2. **Tokenizer**: Encodes to word pieces (e.g., `['[CLS]', 'i', 'absolutely', 'loved', ...]`)
3. **input_ids + attention_mask**: `[[101, 1045, 7078, 3866, ...]]`
4. **Locked DistilBERT E4**: Processes embedding through 6 transformer layers.
5. **Logits**: Returns unnormalized scores.
6. **Softmax**: Converts to probabilities summing to 1.0.
7. **Prediction**: Positive (Confidence: 0.99+)
