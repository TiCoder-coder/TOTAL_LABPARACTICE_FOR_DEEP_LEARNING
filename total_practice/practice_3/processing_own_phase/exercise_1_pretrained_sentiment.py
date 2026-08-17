"""Exercise 1 – Pretrained Sentiment Analysis with Hugging Face.

Demonstrates out-of-the-box sentiment classification using the pretrained
distilbert-base-uncased-finetuned-sst-2-english checkpoint from Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
# pyrefly: ignore [missing-import]
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# pyrefly: ignore [missing-import]
from transformers import logging as transformers_logging
# pyrefly: ignore [missing-import]
from huggingface_hub.utils import logging as hf_hub_logging

# Suppress Hugging Face Hub unauthenticated request and telemetry warnings permanently
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_hub_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from .config import RESULT_DIR

EXERCISE_1_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_SAMPLE_SENTENCE = "This movie is absolutely wonderful and enjoyable."


def load_pretrained_sentiment_pipeline(
    model_name: str = EXERCISE_1_MODEL_NAME,
) -> tuple[Any, Any, Any]:
    """Load pretrained tokenizer, model, and sequence classification pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return tokenizer, model, sentiment_pipe


def tokenize_sample_sentence(
    sentence: str = DEFAULT_SAMPLE_SENTENCE,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tokenize a sample sentence and construct visualization DataFrame."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(EXERCISE_1_MODEL_NAME)

    tokens = tokenizer.tokenize(sentence)
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    # Align subword tokens with special tokens ([CLS], tokens..., [SEP])
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]

    token_rows = []
    for idx, (token_str, tid, mask_val) in enumerate(
        zip(decoded_tokens, input_ids, attention_mask)
    ):
        token_rows.append({
            "Position": idx,
            "Subword Token": token_str,
            "Input ID": tid,
            "Attention Mask": mask_val,
        })

    tokens_df = pd.DataFrame(token_rows)
    token_dict = {
        "sentence": sentence,
        "raw_tokens": tokens,
        "decoded_tokens": decoded_tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    return token_dict, tokens_df


def predict_sentiment_single(
    sentence: str = DEFAULT_SAMPLE_SENTENCE,
    classifier: Any | None = None,
) -> dict[str, Any]:
    """Perform sentiment analysis on a single sentence using pretrained pipeline."""
    if classifier is None:
        _, _, classifier = load_pretrained_sentiment_pipeline()

    raw_result = classifier(sentence)[0]
    return {
        "sentence": sentence,
        "predicted_label": raw_result["label"],
        "confidence_score": float(raw_result["score"]),
        "model_name": EXERCISE_1_MODEL_NAME,
    }


def run_exercise_1_pipeline(
    sentences: list[str] | None = None,
    save_artifact: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Exercise 1 workflow and return token and prediction DataFrames."""
    if sentences is None:
        sentences = [
            "This movie is absolutely wonderful and enjoyable.",
            "The plot was completely predictable and boring.",
        ]

    tokenizer, _, classifier = load_pretrained_sentiment_pipeline()

    # 1. Tokenize primary sample sentence
    primary_sentence = sentences[0]
    token_dict, tokens_df = tokenize_sample_sentence(primary_sentence, tokenizer=tokenizer)

    # 2. Sentiment analysis across sample sentences
    prediction_records = []
    for text in sentences:
        pred = predict_sentiment_single(text, classifier=classifier)
        prediction_records.append({
            "Sentence": pred["sentence"],
            "Predicted Sentiment": pred["predicted_label"],
            "Confidence Score": f"{pred['confidence_score']:.4f}",
            "Raw Confidence": pred["confidence_score"],
            "Model Hub Checkpoint": pred["model_name"],
        })

    predictions_df = pd.DataFrame(prediction_records)

    if save_artifact:
        out_dir = RESULT_DIR / "practice_3_v2_3"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_payload = {
            "exercise": "Exercise 1: Sentiment Analysis with Hugging Face",
            "model_checkpoint": EXERCISE_1_MODEL_NAME,
            "primary_sample_sentence": primary_sentence,
            "tokenization": token_dict,
            "predictions": prediction_records,
        }
        out_path = out_dir / "exercise_1_pretrained_sentiment.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact_payload, f, indent=2)

    return tokens_df, predictions_df
