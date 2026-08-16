import json
import csv
import hashlib
import uuid
import datetime
from pathlib import Path
import sys
import numpy as np
from scipy.special import softmax

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

from .dataset_protocol_v2 import materialize_holdout_only

PROJECT_ROOT = Path("/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING")
PRACTICE_ROOT = PROJECT_ROOT / "total_practice" / "practice_3"
V23_RESULT_DIR = PRACTICE_ROOT / "docs" / "result" / "practice_3_v2_3"
EXP_DIR = V23_RESULT_DIR / "experiments"
HOLDOUT_PATH = PRACTICE_ROOT / "datasets" / "Holdout.csv"

def sha256_payload(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def calculate_diagnostics(probs, true_labels, predicted_labels):
    n = len(probs)
    # Brier Score = 1/N sum_i (prob_1 - y_i)^2 
    brier_score = np.mean((probs[:, 1] - true_labels) ** 2)
    
    # Entropy = - sum p log p
    # To avoid log(0), add epsilon
    eps = 1e-15
    entropy = -np.sum(probs * np.log(probs + eps), axis=1).mean()
    
    # Mean confidence
    confidences = np.max(probs, axis=1)
    mean_conf = np.mean(confidences)
    
    # Wrong prediction confidence
    wrong_mask = (predicted_labels != true_labels)
    wrong_conf = np.mean(confidences[wrong_mask]) if np.any(wrong_mask) else 0.0
    
    # ECE (Expected Calibration Error) - 10 bins
    bins = 10
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(bin_mask):
            bin_acc = np.mean(predicted_labels[bin_mask] == true_labels[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (np.sum(bin_mask) / n) * np.abs(bin_acc - bin_conf)
            
    return brier_score, entropy, mean_conf, wrong_conf, ece

def main():
    print("=== FINAL HOLDOUT EVALUATION ===")
    
    # 1. VERIFY WINNER LOCK
    lock_path = V23_RESULT_DIR / "final_validation_winner_lock.json"
    if not lock_path.exists():
        raise RuntimeError("Winner lock missing.")
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    
    expected_winner = "E4_weight_decay_0.05"
    if lock_data["winner_run_id"] != expected_winner:
        raise RuntimeError(f"Lock winner mismatch: {lock_data['winner_run_id']}")
    
    if lock_data["checkpoint_hash"] != "d4a8b8377de3ade5edf5d03326f5421e52065988cb27a75ed77d52d62a073191":
        raise RuntimeError("Checkpoint hash changed!")
    
    if lock_data["split_fingerprint"] != "4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13":
        raise RuntimeError("Split fingerprint mismatch.")
    
    if lock_data["holdout_accessed"] is not False or lock_data["official_test_loaded"] is not False:
        raise RuntimeError("Holdout/Test flag violation.")
    
    # Verify registry global state
    reg_path = V23_RESULT_DIR / "experiment_registry.json"
    registry = json.loads(reg_path.read_text(encoding="utf-8"))
    
    e4_reg = next((e for e in registry["experiments"] if e["experiment_id"] == expected_winner), None)
    if e4_reg["status"] != "COMPLETED":
        raise RuntimeError("E4 status is not COMPLETED.")
    
    if registry.get("holdout_status", "SEALED") != "SEALED" or registry.get("holdout_evaluation_count", 0) != 0:
        raise RuntimeError("Holdout is not SEALED or count != 0.")
    
    print("Pre-evaluation checks: PASS")
    
    # 2. HOLDOUT CLAIM
    attempt_id = str(uuid.uuid4())
    claim_path = V23_RESULT_DIR / "holdout_claim.json"
    if claim_path.exists():
        raise RuntimeError("Holdout has already been claimed! Stop.")
    
    claim = {
        "protocol_version": "practice_3_v2.3",
        "winner_run_id": expected_winner,
        "winner_lock_hash": lock_data["lock_hash"],
        "checkpoint_hash": lock_data["checkpoint_hash"],
        "split_fingerprint": lock_data["split_fingerprint"],
        "evaluation_attempt_id": attempt_id,
        "claimed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "CLAIMED"
    }
    claim_path.write_text(json.dumps(claim, indent=2), encoding="utf-8")
    print(f"One-Time Holdout CLAIMED: {attempt_id}")
    
    # 3. LOAD DATASET FROM PROTOCOL MANIFEST
    manifest_source_path = PRACTICE_ROOT / "docs" / "result" / "practice_3_v2_1" / "dataset_split_manifest.json"
    dataset_manifest = json.loads(manifest_source_path.read_text(encoding="utf-8"))
    
    holdout_dataset, _ = materialize_holdout_only(dataset_manifest)
    
    if len(holdout_dataset) != 960:
        raise ValueError(f"Holdout size is {len(holdout_dataset)}, expected 960.")
    
    labels = holdout_dataset["label"]
    if labels.count(0) != 480 or labels.count(1) != 480:
        raise ValueError("Holdout label distribution is not 480/480.")
    
    print("Holdout loaded via protocol: 960 samples, balanced.")
    
    # 4. LOAD MODEL AND TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tokenizer.vocab_size != 30522:
        raise ValueError("Vocabulary size mismatch.")
    
    texts = [str(t) for t in holdout_dataset["text"]]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=80)
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": holdout_dataset["label"]
    })
    
    checkpoint_path = PROJECT_ROOT / "total_practice" / "practice_3" / lock_data["checkpoint_path"]
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_path),
        num_labels=2,
        local_files_only=True,
        attn_implementation="eager"
    )
    model.eval()
    
    # 5. INFERENCE (NO TRAINING)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir="/tmp/dummy",
            per_device_eval_batch_size=32,
            report_to="none"
        )
    )
    
    print("Executing Trainer.predict()...")
    prediction_output = trainer.predict(dataset, metric_key_prefix="holdout")
    
    logits = prediction_output.predictions
    true_labels = prediction_output.label_ids
    
    # Probabilities & metrics
    probs = softmax(logits, axis=-1)
    pred_labels = np.argmax(probs, axis=-1)
    
    # Ensure standard binary metrics (positive label = 1)
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, pos_label=1)
    rec = recall_score(true_labels, pred_labels, pos_label=1)
    f1 = f1_score(true_labels, pred_labels, pos_label=1)
    loss = log_loss(true_labels, probs)
    
    print(f"Holdout Loss: {loss}")
    print(f"Holdout Accuracy: {acc}")
    print(f"Holdout F1: {f1}")
    
    # Confusion matrix elements
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    
    # Diagnostics
    brier, entropy, mean_conf, wrong_conf, ece = calculate_diagnostics(probs, true_labels, pred_labels)
    
    # 6. SAVE PREDICTIONS CSV
    csv_path = V23_RESULT_DIR / "holdout_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "text", "true_label", "predicted_label", 
                         "logit_class_0", "logit_class_1", "probability_class_0", "probability_class_1", 
                         "predicted_confidence", "correct"])
        for i in range(len(holdout_dataset)):
            writer.writerow([
                i, holdout_dataset[i]["text"], int(true_labels[i]), int(pred_labels[i]),
                float(logits[i][0]), float(logits[i][1]),
                float(probs[i][0]), float(probs[i][1]),
                float(max(probs[i][0], probs[i][1])),
                bool(pred_labels[i] == true_labels[i])
            ])
    
    # 7. SAVE METRICS
    metrics_path = V23_RESULT_DIR / "final_holdout_metrics.json"
    metrics_data = {
        "protocol_version": "practice_3_v2.3",
        "winner_run_id": expected_winner,
        "winner_lock_hash": lock_data["lock_hash"],
        "checkpoint_path": lock_data["checkpoint_path"],
        "checkpoint_hash": lock_data["checkpoint_hash"],
        "holdout_size": len(holdout_dataset),
        "holdout_loss": float(loss),
        "holdout_accuracy": float(acc),
        "holdout_precision": float(prec),
        "holdout_recall": float(rec),
        "holdout_f1": float(f1),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "diagnostics": {
            "brier_score": float(brier),
            "prediction_entropy": float(entropy),
            "mean_confidence": float(mean_conf),
            "wrong_prediction_confidence": float(wrong_conf),
            "expected_calibration_error": float(ece)
        },
        "split_fingerprint": lock_data["split_fingerprint"],
        "evaluation_count": 1,
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "official_test_loaded": False
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    
    # 8. VALIDATION VS HOLDOUT COMPARISON
    gap_path = V23_RESULT_DIR / "validation_vs_holdout_comparison.json"
    v_loss = lock_data["best_val_loss"]
    v_acc = lock_data["best_val_accuracy"]
    v_f1 = lock_data["best_val_f1"]
    
    gap_data = {
        "validation_loss": v_loss,
        "holdout_loss": float(loss),
        "loss_difference": float(loss - v_loss),
        "validation_accuracy": v_acc,
        "holdout_accuracy": float(acc),
        "accuracy_gap": float(v_acc - acc),
        "validation_f1": v_f1,
        "holdout_f1": float(f1),
        "f1_gap": float(v_f1 - f1)
    }
    gap_path.write_text(json.dumps(gap_data, indent=2), encoding="utf-8")
    
    # 9. MANIFEST
    manifest_path = V23_RESULT_DIR / "final_holdout_artifact_manifest.json"
    manifest_data = {
        "protocol_version": "practice_3_v2.3",
        "winner_run_id": expected_winner,
        "winner_lock_hash": lock_data["lock_hash"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": {
            "holdout_predictions.csv": sha256_payload(csv_path.read_text(encoding="utf-8")),
            "final_holdout_metrics.json": sha256_payload(metrics_path.read_text(encoding="utf-8")),
            "validation_vs_holdout_comparison.json": sha256_payload(gap_path.read_text(encoding="utf-8"))
        }
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    
    # 10. MARKDOWN EVALUATION REPORT
    md_path = V23_RESULT_DIR / "final_holdout_evaluation.md"
    md_text = f"""# Final Holdout Evaluation
**Protocol**: practice_3_v2.3

## A. Locked Model
- **Winner Run ID**: `{expected_winner}`
- **Checkpoint Hash**: `{lock_data['checkpoint_hash'][:16]}...`
- **Lock Hash**: `{lock_data['lock_hash'][:16]}...`

## B. Why Holdout is Evaluated Only Now
The model selection process relied purely on the Validation split (where `E4` was chosen). The Holdout was sequestered and intentionally excluded from hyperparameter ranking. This one-time execution serves as an unbiased estimate of real-world generalization performance.

## C. Holdout Metrics
- **Loss**: {loss:.4f}
- **Accuracy**: {acc*100:.2f}%
- **Precision**: {prec*100:.2f}%
- **Recall**: {rec*100:.2f}%
- **F1-Score**: {f1*100:.2f}%

*Confusion Counts*: TP={tp}, TN={tn}, FP={fp}, FN={fn}

## D. Validation vs Holdout Comparison
- **Loss Diff**: {gap_data["loss_difference"]:+.4f} (Val: {v_loss:.4f}, Holdout: {loss:.4f})
- **Acc Gap**: {gap_data["accuracy_gap"]*100:+.2f}% (Val: {v_acc*100:.2f}%, Holdout: {acc*100:.2f}%)
- **F1 Gap**: {gap_data["f1_gap"]*100:+.2f}% (Val: {v_f1*100:.2f}%, Holdout: {f1*100:.2f}%)

## E. Generalization Interpretation
{"The model generalizes exceptionally well to the Holdout split, as metrics are comparable to Validation performance." if abs(gap_data["accuracy_gap"]) < 0.05 else "The model shows some performance drop on the Holdout split, implying partial overfitting to the Validation distribution."}

## F. Statement of Finality
**No further hyperparameter tuning is permitted.** The model configuration is permanently locked. The Holdout split has been evaluated.
"""
    md_path.write_text(md_text, encoding="utf-8")
    
    # 11. PROCESS LOG
    log_path = PRACTICE_ROOT / "docs" / "save_process_proceduce_own_phase_refactor&fix" / "practice_3_v2_3_one_time_holdout_evaluation_process_2026-08-16.md"
    log_text = """# One-Time Holdout Evaluation Process Log
- **Preconditions**: Winner locked (`E4`), Checkpoint hash matched exactly, Holdout flag was strictly `False`.
- **Holdout Claim**: Successfully wrote `CLAIMED` lock via attempt UUID.
- **Evaluation Execution**: One pass of `Trainer.predict()` executed. No gradients were computed. `Trainer.train()` was NOT called.
- **Artifacts Created**: `holdout_predictions.csv`, `final_holdout_metrics.json`, `validation_vs_holdout_comparison.json`, manifest, and markdown reports.
- **Transition**: Holdout explicitly marked `EVALUATED_FINAL` (count=1). Official Test protected and untouched.
"""
    log_path.write_text(log_text, encoding="utf-8")
    
    # 12. UPDATE HOLDOUT REGISTRY STATE
    registry["holdout_status"] = "EVALUATED_FINAL"
    registry["holdout_evaluation_count"] = 1
    registry["holdout_winner_run"] = expected_winner
    registry["holdout_checkpoint_hash"] = lock_data["checkpoint_hash"]
    registry["holdout_metrics_manifest"] = manifest_data["files"]["final_holdout_metrics.json"]
    reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    
    # Also update winner lock to reflect it was accessed
    lock_data["holdout_accessed"] = True
    lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

    print("\nHoldout completely processed and successfully locked.")

if __name__ == "__main__":
    main()
