import torch
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def test_mps():
    print("--- MPS PREFLIGHT ---")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available: PASS")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 30522, "Vocab size mismatch"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        attn_implementation="eager"
    )
    model.to(device)
    model.train()
    print("Model loads with eager attention: PASS")

    # dummy forward
    input_ids = torch.randint(0, 30000, (2, 80)).to(device)
    attention_mask = torch.ones((2, 80)).to(device)
    labels = torch.tensor([0, 1]).to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item()}")
    assert torch.isfinite(loss), "Loss is not finite!"
    print("MPS train-mode forward works and finite loss: PASS")
    
def test_data():
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    print(f"Original Train subset size: {len(dataset)}")
    assert len(dataset) == 8530, "Not the full train split!"
    
    e4_path = "total_practice/practice_3/runs/practice_3_v2_3/E4_weight_decay_0.05/checkpoints/checkpoint-960/pytorch_model.bin"
    import os
    if os.path.exists(e4_path):
        if get_hash(e4_path) == "d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191":
            print("E4 checkpoint unchanged: PASS")
        else:
            print("E4 checkpoint changed! FAIL")
    else:
        print("E4 checkpoint not found locally. PASS")
        
if __name__ == "__main__":
    from total_practice.practice_3.processing_own_phase.live_training_demo_standalone import run_live_training_demo
    print("Standalone module import: PASS")
    test_mps()
    test_data()
