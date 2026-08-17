# Exercise 01 – Pretrained Sentiment Analysis with Hugging Face

## Notebook Reference
Notebook:
[`practice_3.ipynb`](../../notebook_practice_3/practice_3.ipynb)

Cell:
[`Cell 02` (Code)](../../notebook_practice_3/practice_3.ipynb)

Cell output:
Two styled tables:
1. `Exercise 1: Sample Sentence Tokenization` displaying subword tokens, token IDs, and attention masks.
2. `Exercise 1: Pretrained Sentiment Analysis Predictions` displaying out-of-the-box sentiment classification and confidence scores.

## Processing Source
- [`exercise_1_pretrained_sentiment.py`](../../processing_own_phase/exercise_1_pretrained_sentiment.py)
- Functions: `run_exercise_1_pipeline()`, `load_pretrained_sentiment_pipeline()`, `tokenize_sample_sentence()`, `predict_sentiment_single()`

## Artifact Sources
- [`exercise_1_pretrained_sentiment.json`](./practice_3_v2_3/exercise_1_pretrained_sentiment.json)

## Pretrained Hub Model Specification
| Item | Specification |
|---|---|
| **Model Hub Checkpoint** | `distilbert-base-uncased-finetuned-sst-2-english` |
| **Backbone Architecture** | 6-layer DistilBERT |
| **Pretrained Dataset** | Stanford Sentiment Treebank (SST-2) |
| **Output Classes** | 2 (`NEGATIVE`, `POSITIVE`) |
| **Training Required** | None (Direct out-of-the-box pipeline inference) |

## Results

### 1. Tokenization Output on Sample Sentence
- **Sample Sentence**: *"This movie is absolutely wonderful and enjoyable."*

| Position | Subword Token | Input ID | Attention Mask |
|:---:|:---:|:---:|:---:|
| 0 | `[CLS]` | 101 | 1 |
| 1 | `this` | 2023 | 1 |
| 2 | `movie` | 3185 | 1 |
| 3 | `is` | 2003 | 1 |
| 4 | `absolutely` | 7078 | 1 |
| 5 | `wonderful` | 6919 | 1 |
| 6 | `and` | 1998 | 1 |
| 7 | `enjoyable` | 22249 | 1 |
| 8 | `.` | 1012 | 1 |
| 9 | `[SEP]` | 102 | 1 |

### 2. Pretrained Sentiment Analysis Predictions
| Sentence | Predicted Sentiment | Confidence Score | Model Checkpoint |
|---|:---:|:---:|---|
| *"This movie is absolutely wonderful and enjoyable."* | **POSITIVE** | **0.9999** | `distilbert-base-uncased-finetuned-sst-2-english` |
| *"The plot was completely predictable and boring."* | **NEGATIVE** | **0.9998** | `distilbert-base-uncased-finetuned-sst-2-english` |

## Evidence
- Tokenization and prediction tables rendered in Notebook [`Cell 02`](../../notebook_practice_3/practice_3.ipynb).
- Exported JSON artifact in [`exercise_1_pretrained_sentiment.json`](./practice_3_v2_3/exercise_1_pretrained_sentiment.json).

## Summary
Exercise 1 satisfies all 4 assignment requirements: loading Hugging Face transformers, utilizing the pretrained `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face Hub, tokenizing sample text with attention masks, and performing zero-shot sentiment inference without local training.
