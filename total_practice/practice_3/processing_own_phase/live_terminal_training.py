import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime, timezone

DEMO_DIR = Path("total_practice/practice_3/runs/live_terminal_demo")
METRICS_FILE = DEMO_DIR / "metrics.jsonl"
STATUS_FILE = DEMO_DIR / "status.json"
FINAL_FILE = DEMO_DIR / "final_summary.json"


def _write_status(status_obj: dict):
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status_obj, f, indent=2)
    tmp.replace(STATUS_FILE)


def _append_metric(record: dict):
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_demo_data(demo_sample_size: int = 1600, seed: int = 42):
    print("Loading Hugging Face original TRAIN split only...")
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    df = dataset.to_pandas()
    half = demo_sample_size // 2
    pos_df = df[df["label"] == 1].sample(n=half, random_state=seed)
    neg_df = df[df["label"] == 0].sample(n=half, random_state=seed)
    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
    balanced_dataset = Dataset.from_pandas(balanced_df)
    split = balanced_dataset.train_test_split(test_size=0.2, seed=seed)
    print(f"Demo Train: {len(split['train'])}  Demo Validation: {len(split['test'])}")
    return split["train"], split["test"]


def _tokenize(demo_train, demo_val, tokenizer, max_length: int = 80):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    tok_train = demo_train.map(tokenize_fn, batched=True)
    tok_val = demo_val.map(tokenize_fn, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    tok_train.set_format(type="torch", columns=cols)
    tok_val.set_format(type="torch", columns=cols)
    return tok_train, tok_val


def _evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item() * len(labels)
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
    model.train()
    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels_list, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels_list, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1


def run():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous run
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()

    pid = os.getpid()
    started_at = datetime.now(timezone.utc).isoformat()
    status = {
        "status": "RUNNING",
        "pid": pid,
        "started_at": started_at,
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "last_update": started_at,
        "error": None,
    }
    _write_status(status)

    try:
        print("==================================================")
        print("LIVE TRAINING DEMONSTRATION ONLY")
        print("Official Validation used: NO  |  Holdout used: NO  |  Official Test used: NO")
        print("==================================================")

        device = _detect_device()
        print(f"Device: {device}")

        demo_train, demo_val = _load_demo_data()

        model_name = "distilbert/distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer vocab: {tokenizer.vocab_size}")

        tok_train, tok_val = _tokenize(demo_train, demo_val, tokenizer)

        train_loader = torch.utils.data.DataLoader(tok_train, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(tok_val, batch_size=32, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, attn_implementation="eager"
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

        demo_epochs = 2
        total_steps = len(train_loader) * demo_epochs
        status["total_steps"] = total_steps
        _write_status(status)

        current_step = 0
        recent_losses = []
        start_time = time.time()

        for epoch in range(demo_epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                out = model(ids, attention_mask=mask, labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()

                current_step += 1
                recent_losses.append(loss.item())
                current_lr = optimizer.param_groups[0]["lr"]
                epoch_float = round(epoch + (batch_idx + 1) / len(train_loader), 4)

                # Log train every 5 steps
                if current_step % 5 == 0 or current_step == total_steps:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    recent_losses = []
                    _append_metric({
                        "type": "train",
                        "step": current_step,
                        "epoch": epoch_float,
                        "train_loss": round(avg_loss, 6),
                        "learning_rate": round(current_lr, 10),
                    })

                # Evaluate every 20 steps
                if current_step % 20 == 0 or current_step == total_steps:
                    v_loss, v_acc, v_prec, v_rec, v_f1 = _evaluate(model, val_loader, device)
                    _append_metric({
                        "type": "eval",
                        "step": current_step,
                        "epoch": epoch_float,
                        "val_loss": round(v_loss, 6),
                        "accuracy": round(v_acc, 6),
                        "precision": round(v_prec, 6),
                        "recall": round(v_rec, 6),
                        "f1": round(v_f1, 6),
                    })
                    print(f"[step {current_step}/{total_steps}] epoch={epoch_float:.2f} "
                          f"val_loss={v_loss:.4f} acc={v_acc:.4f} f1={v_f1:.4f}")

                status.update({
                    "current_epoch": epoch_float,
                    "current_step": current_step,
                    "last_update": datetime.now(timezone.utc).isoformat(),
                })
                _write_status(status)

        runtime = round(time.time() - start_time, 2)
        final_v_loss, final_v_acc, final_v_prec, final_v_rec, final_v_f1 = _evaluate(model, val_loader, device)

        summary = {
            "official_experiment": False,
            "used_for_model_selection": False,
            "used_for_holdout_evaluation": False,
            "demo_train_samples": len(demo_train),
            "demo_val_samples": len(demo_val),
            "final_val_loss": round(final_v_loss, 6),
            "final_accuracy": round(final_v_acc, 6),
            "final_precision": round(final_v_prec, 6),
            "final_recall": round(final_v_rec, 6),
            "final_f1": round(final_v_f1, 6),
            "runtime_seconds": runtime,
        }
        with open(FINAL_FILE, "w") as f:
            json.dump(summary, f, indent=2)

        status.update({"status": "COMPLETED", "last_update": datetime.now(timezone.utc).isoformat()})
        _write_status(status)

        print("\n==================================================")
        print("LIVE TRAINING DEMONSTRATION COMPLETE")
        print(f"Final Demo Val F1: {final_v_f1:.4f}  Accuracy: {final_v_acc:.4f}")
        print("These metrics are DEMO only. NOT part of official E1-E6c ranking.")
        print("==================================================")

    except Exception as e:
        status.update({"status": "FAILED", "error": str(e), "last_update": datetime.now(timezone.utc).isoformat()})
        _write_status(status)
        raise


if __name__ == "__main__":
    run()
