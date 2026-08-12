# Phase 2 Pretrained Sentiment Inference Plan

**Date:** 2026-08-10  
**Phase:** 2/15  
**File path:** `docs/plan-doc/plan_before_process/phase_02_pretrained_inference_plan_2026-08-12.md`  
**Dependencies:** Phase 1 (Environment & Reproducibility)

---

## 1. Objective

Implement **Exercise 1** use a pre-fine-tuned model for sentiment analysis inference.

Specifically:

- Load `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face Hub.
- Run inference on 3 sample sentences (positive, negative, neutral).
- Inspect the tokenizer to understand subword tokenization.
- Demonstrate the **Inference** layer of Transfer Learning.

---

## 2. Input

| Source                | Content                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| Model Checkpoint      | `distilbert-base-uncased-finetuned-sst-2-english` (fine-tuned on SST-2) |
| Sample Sentences      | 3 sentences: positive, negative, neutral                                |
| Hugging Face Pipeline | `pipeline("sentiment-analysis")`                                        |
| Environment           | Python 3.11.9, transformers 5.14.1, torch 2.13.0+cpu                    |

---

## 3. Processing

### 3.1 Load Model and Pipeline

- Use `pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")`.
- Set `device=-1` (CPU) – matches the environment.
- First run downloads model (~260MB).

### 3.2 Run Inference

- 3 sample sentences:
  1. Positive: `"I absolutely loved this movie! The performances were outstanding."`
  2. Negative: `"This film was a complete waste of time. Terrible acting."`
  3. Neutral: `"It was okay, nothing special."`
- Print label and confidence score for each.

### 3.3 Inspect Tokenizer

- Load tokenizer from the same checkpoint.
- Tokenize the positive sentence.
- Display: `input_ids`, `attention_mask`, tokens (subword), `vocab_size`.
- Explain subword tokenization.

---

## 4. Module Functions (3 functions)

| Function                                                      | Purpose                                                                               |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `get_sentiment_pipeline() -> pipeline`                        | Load and return sentiment analysis pipeline with SST-2 model.                         |
| `run_inference(pipeline_obj, texts: List[str]) -> List[Dict]` | Run inference on a list of texts, return `[{label, score}]`.                          |
| `inspect_tokenizer(text: str) -> Dict`                        | Tokenize a single text; return `input_ids`, `attention_mask`, `tokens`, `vocab_size`. |

**Note:** No file-saving function to keep the module simple. Notebook can save outputs as needed.

---

## 5. Expected Output

- `phase_02_pretrained_inference.py` with 3 public functions.
- Notebook cells:
  - Import 3 functions.
  - Call `get_sentiment_pipeline()` to load model.
  - Call `run_inference()` with 3 sentences.
  - Call `inspect_tokenizer()` and display tokenization results.
- No hard-coded `device = "cpu"` (use `device=-1` for pipeline).

---

## 6. Files Affected

| File                                                                                 | Action            |
| ------------------------------------------------------------------------------------ | ----------------- |
| `docs/plan-doc/plan_before_process/phase_02_pretrained_inference_plan_2026-08-12.md` | Create            |
| `processing_own_phase/phase_02_pretrained_inference.py`                              | Create            |
| `notebook_practice_3/practice_3.ipynb`                                               | Add Phase 2 cells |

---

## 7. Validation / Sanity Checks

- [ ] `get_sentiment_pipeline()` returns a pipeline object.
- [ ] `run_inference()` returns `[{label, score}]` with correct types.
- [ ] `inspect_tokenizer()` returns `vocab_size = 30522` (DistilBERT base).
- [ ] Sample sentences are classified correctly.
- [ ] No output when importing the module.
- [ ] `python phase_02_pretrained_inference.py` runs without errors.

---

## 8. Completion Criteria

Phase 2 is complete when:

1. `phase_02_pretrained_inference.py` runs correctly with all 3 functions.
2. Notebook displays inference results and tokenizer inspection.
3. Clear distinction is shown between Exercise 1 (inference) and Exercise 2 (fine-tuning).

---

## 9. Decision Log

| Decision           | Details                                      | Reason                                              |
| ------------------ | -------------------------------------------- | --------------------------------------------------- |
| Use pipeline       | `pipeline()` instead of manual model loading | Simpler, matches Exercise 1 intent                  |
| Use `device=-1`    | Force CPU for pipeline                       | Pipeline requires integer device ID                 |
| 3 sample sentences | Positive, negative, neutral                  | Demonstrates model behavior across different inputs |

---

## 10. Next Step

After Phase 2 plan is approved → implement `phase_02_pretrained_inference.py` and test in notebook.

---

**End of Phase 2 Plan**
