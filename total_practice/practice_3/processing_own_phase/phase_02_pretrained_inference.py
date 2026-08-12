from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer


def get_sentiment_pipeline() -> pipeline:
    """
    Load the pre-trained sentiment analysis pipeline with DistilBERT fine-tuned on SST-2.
    Uses CPU (device=-1) to match the environment.
    """
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU
    )


def run_inference(pipeline_obj: pipeline, texts: List[str]) -> List[Dict[str, Any]]:
    """
    Run sentiment inference on a list of texts.
    Returns a list of dicts with 'label' and 'score'.
    """
    return pipeline_obj(texts)


def inspect_tokenizer(text: str) -> Dict[str, Any]:
    """
    Load the tokenizer from the fine-tuned model and tokenize the input text.
    Returns a dict with input_ids, attention_mask, tokens, and vocab_size.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    return {
        "input_ids": encoding["input_ids"][0].tolist(),
        "attention_mask": encoding["attention_mask"][0].tolist(),
        "tokens": tokens,
        "vocab_size": tokenizer.vocab_size
    }


if __name__ == "__main__":
    print("Loading sentiment pipeline...")
    pipe = get_sentiment_pipeline()

    sample_texts = [
        "I absolutely loved this movie! The performances were outstanding.",
        "This film was a complete waste of time. Terrible acting.",
        "It was okay, nothing special."
    ]

    print("\n--- Running Inference ---")
    results = run_inference(pipe, sample_texts)
    for text, result in zip(sample_texts, results):
        print(f"Text: {text[:60]}...")
        print(f"  -> {result['label']} (score: {result['score']:.4f})")

    print("\n--- Tokenizer Inspection (first sentence) ---")
    token_info = inspect_tokenizer(sample_texts[0])
    print(f"Vocab size: {token_info['vocab_size']}")
    print(f"Tokens (first 20): {token_info['tokens'][:20]}")
    print(f"Input IDs (first 20): {token_info['input_ids'][:20]}")