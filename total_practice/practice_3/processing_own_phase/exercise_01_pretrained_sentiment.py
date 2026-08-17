"""Exercise 01 – Pretrained Sentiment Analysis with Hugging Face.

Exports functions matching naming convention:
- load_pretrained_sentiment()
- tokenize_sample_sentence()
- predict_sentiment()
- run_exercise_1_pipeline()
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from .exercise_1_pretrained_sentiment import (
    DEFAULT_SAMPLE_SENTENCE,
    EXERCISE_1_MODEL_NAME,
    load_pretrained_sentiment_pipeline,
    predict_sentiment_single,
    run_exercise_1_pipeline,
    tokenize_sample_sentence,
)


def load_pretrained_sentiment(model_name: str = EXERCISE_1_MODEL_NAME) -> tuple[Any, Any, Any]:
    """Alias for load_pretrained_sentiment_pipeline."""
    return load_pretrained_sentiment_pipeline(model_name=model_name)


def predict_sentiment(
    sentence: str = DEFAULT_SAMPLE_SENTENCE,
    classifier: Any | None = None,
) -> dict[str, Any]:
    """Alias for predict_sentiment_single."""
    return predict_sentiment_single(sentence=sentence, classifier=classifier)


__all__ = [
    "DEFAULT_SAMPLE_SENTENCE",
    "EXERCISE_1_MODEL_NAME",
    "load_pretrained_sentiment",
    "load_pretrained_sentiment_pipeline",
    "predict_sentiment",
    "predict_sentiment_single",
    "run_exercise_1_pipeline",
    "tokenize_sample_sentence",
]
