# Phase 2 — Pretrained Sentiment Inference Log

**Original implementation/fix date:** 2026-08-12  
**Separated from combined Priority 1 log:** 2026-08-14  
**Status:** PASS

## Objective

Demonstrate Exercise 1 using an already sentiment-fine-tuned checkpoint:

- load the pretrained sentiment pipeline;
- run inference on custom example sentences;
- inspect the tokenizer output;
- distinguish per-sentence confidence from dataset-level accuracy;
- perform no fine-tuning.

## Implementation

Core module:

`processing_own_phase/phase_02_pretrained_inference.py`

Checkpoint:

`distilbert-base-uncased-finetuned-sst-2-english`

The notebook calls:

- `get_sentiment_pipeline()`;
- `run_inference(pipe, test_sentences)`;
- `inspect_tokenizer(test_sentences[0])`.

The module owns pipeline loading, inference and tokenizer inspection. The notebook supplies three presentation examples and displays the returned results.

## Latest executed evidence

| Example | Predicted label | Confidence |
|---|---|---:|
| “I absolutely loved this movie! The performances were outstanding.” | POSITIVE | 0.9999 |
| “This film was a complete waste of time. Terrible acting.” | NEGATIVE | 0.9998 |
| “It was okay, nothing special.” | NEGATIVE | 0.9821 |

Tokenizer inspection for the first sentence:

- vocabulary size: 30,522;
- first token: `[CLS]`;
- final token: `[SEP]`;
- tokens include punctuation as separate tokenizer units;
- tokenizer output was produced successfully.

These confidence values are individual prediction scores and are not reported as model accuracy.

## Notebook verification

- Phase 2 code-cell execution count in the latest Run All: 3
- Example count: 3
- Prediction count: 3
- Tokenizer inspection: PASS
- Runtime/import error: none

## Isolation and scope

- Uses an already fine-tuned model for Exercise 1 only.
- No training, backward pass, optimizer or scheduler operation occurs.
- No Rotten Tomatoes Train/Validation/Test split is accessed.
- Phase 2 results are not used to select the Exercise 2 checkpoint.

Phase 2 evidence is stored in the executed notebook output rather than a separate result JSON.

## Conclusion

`PHASE 2: PASS — PRETRAINED SENTIMENT INFERENCE VERIFIED`
