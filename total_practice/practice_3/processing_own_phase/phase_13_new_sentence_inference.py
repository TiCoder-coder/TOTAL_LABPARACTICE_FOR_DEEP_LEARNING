"""Phase 13: guarded inference for new custom sentences only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BASE_MODEL_CHECKPOINT, MAX_TOKEN_LENGTH, RESULT_DIR
from .phase_01_environment import get_device
from .phase_06_preprocessing import get_data_collator


PHASE_09_SELECTED_PATH = RESULT_DIR / "phase_09_training" / "selected_checkpoint.json"
PHASE_11_MANIFEST_PATH = RESULT_DIR / "phase_11_evaluation" / "phase_11_evaluation_manifest.json"
PHASE_11_CHECKPOINT_PATH = RESULT_DIR / "phase_11_evaluation" / "checkpoint_verification.json"
PHASE_13_ARTIFACT_PATH = RESULT_DIR / "phase_13_inference_examples.json"

EXPECTED_CHECKPOINT_NAME = "checkpoint-1068"
EXPECTED_EPOCH = 2.0
EXPECTED_STEP = 1068
EXPECTED_ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
EXPECTED_LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}
PROBABILITY_SUM_TOLERANCE = 1e-6
CONFIDENCE_TOLERANCE = 1e-8
DETERMINISM_TOLERANCE = 1e-7

DEFAULT_CUSTOM_SENTENCES = [
    "The performances were warm, convincing, and deeply moving.",
    "The story was tedious, predictable, and painfully slow.",
    "It is not a bad movie at all.",
    "The acting is excellent, but the plot is disappointingly shallow.",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if _load_json(path) != json.loads(serialized):
        raise RuntimeError(f"JSON read-back failed: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_inputs(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        normalized = [texts]
    elif isinstance(texts, Sequence) and not isinstance(texts, (bytes, bytearray)):
        normalized = list(texts)
    else:
        raise TypeError("texts must be one string or an ordered sequence of strings")
    if not normalized:
        raise ValueError("texts must contain at least one sentence")
    for index, text in enumerate(normalized):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] is not a string")
        if not text.strip():
            raise ValueError(f"texts[{index}] is empty or whitespace-only")
    return normalized


def _normalized_id2label(mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in mapping.items()}


def verify_authoritative_checkpoint() -> dict[str, Any]:
    """Verify the frozen Phase 9/11 checkpoint identity without reranking."""
    selected = _load_json(PHASE_09_SELECTED_PATH)
    manifest = _load_json(PHASE_11_MANIFEST_PATH)
    phase_11_checkpoint = _load_json(PHASE_11_CHECKPOINT_PATH)
    checkpoint = Path(selected["checkpoint"]).resolve()
    manifest_checkpoint = Path(manifest["checkpoint"]).resolve()
    verified_checkpoint = Path(phase_11_checkpoint["checkpoint"]).resolve()
    weight_path = checkpoint / "model.safetensors"
    if not weight_path.is_file():
        fallback = checkpoint / "pytorch_model.bin"
        weight_path = fallback if fallback.is_file() else weight_path
    if not checkpoint.is_dir() or not weight_path.is_file() or weight_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Authoritative checkpoint is incomplete: {checkpoint}")
    weight_sha256 = _sha256_file(weight_path)
    selection_sha256 = _sha256_json(selected)
    checks = {
        "phase_11_status_final_test_complete": manifest.get("status") == "FINAL_TEST_COMPLETE",
        "checkpoint_paths_agree": checkpoint == manifest_checkpoint == verified_checkpoint,
        "checkpoint_name_1068": checkpoint.name == EXPECTED_CHECKPOINT_NAME,
        "selected_epoch_2": float(selected.get("epoch", -1)) == EXPECTED_EPOCH
        and float(manifest.get("selected_epoch", -1)) == EXPECTED_EPOCH,
        "selected_step_1068": int(selected.get("step", -1)) == EXPECTED_STEP
        and int(manifest.get("selected_step", -1)) == EXPECTED_STEP,
        "weight_sha256_matches_phase_11": weight_sha256
        == manifest.get("checkpoint_weight_sha256")
        == phase_11_checkpoint.get("weight_sha256"),
        "selection_sha256_matches_phase_11": selection_sha256
        == manifest.get("selection_record_sha256"),
        "test_evaluation_count_is_one": manifest.get("test_evaluation_count") == 1,
        "test_not_used_for_selection": manifest.get("test_used_for_selection") is False,
        "checkpoint_not_changed_after_test": manifest.get("checkpoint_changed_after_test") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 13 checkpoint provenance verification failed: {checks}")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "epoch": EXPECTED_EPOCH,
        "step": EXPECTED_STEP,
        "weight_file": str(weight_path.resolve()),
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": weight_sha256,
        "selection_record_sha256": selection_sha256,
        "checks": checks,
    }


def _load_inference_components(provenance: dict[str, Any]) -> tuple[Any, Any, Any, torch.device]:
    checkpoint = provenance["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    id2label = _normalized_id2label(model.config.id2label)
    label2id = {str(key): int(value) for key, value in model.config.label2id.items()}
    config_checks = {
        "model_class_distilbert_sequence_classifier": model.__class__.__name__
        == "DistilBertForSequenceClassification",
        "model_type_distilbert": model.config.model_type == "distilbert",
        "num_labels_2": int(model.config.num_labels) == 2,
        "id2label_correct": id2label == EXPECTED_ID2LABEL,
        "label2id_correct": label2id == EXPECTED_LABEL2ID,
        "model_eval_mode": model.training is False,
        "tokenizer_contract": tokenizer.name_or_path == checkpoint,
    }
    if not all(config_checks.values()):
        raise RuntimeError(f"Model/tokenizer contract verification failed: {config_checks}")
    device = torch.device(get_device())
    model.to(device)
    provenance["model_config"] = {
        "model_class": model.__class__.__name__,
        "model_type": model.config.model_type,
        "num_labels": int(model.config.num_labels),
        "id2label": id2label,
        "label2id": label2id,
        "checks": config_checks,
    }
    return model, tokenizer, get_data_collator(tokenizer), device


def _prepare_batch(texts: list[str], tokenizer: Any, data_collator: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=MAX_TOKEN_LENGTH,
    )
    features = [
        {key: encoded[key][index] for key in ("input_ids", "attention_mask")}
        for index in range(len(texts))
    ]
    original_lengths = [len(feature["input_ids"]) for feature in features]
    batch = data_collator(features)
    shape = list(batch["input_ids"].shape)
    padding_counts = [int((row == 0).sum().item()) for row in batch["attention_mask"]]
    checks = {
        "input_ids_present": "input_ids" in batch,
        "attention_mask_present": "attention_mask" in batch,
        "batch_count_preserved": shape[0] == len(texts),
        "id_mask_shapes_match": batch["input_ids"].shape == batch["attention_mask"].shape,
        "sequence_length_within_80": shape[1] <= MAX_TOKEN_LENGTH,
        "dynamic_padding_counts_valid": padding_counts
        == [shape[1] - length for length in original_lengths],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 13 preprocessing verification failed: {checks}")
    metadata = {
        "tokenizer_checkpoint": f"distilbert/{BASE_MODEL_CHECKPOINT}",
        "max_length": MAX_TOKEN_LENGTH,
        "truncation": True,
        "per_example_padding": False,
        "batch_padding": "DataCollatorWithPadding_dynamic",
        "original_token_lengths": original_lengths,
        "batch_input_ids_shape": shape,
        "batch_attention_mask_shape": list(batch["attention_mask"].shape),
        "padding_counts": padding_counts,
        "checks": checks,
    }
    return {key: value.to(device) for key, value in batch.items()}, metadata


def _forward_probabilities(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(**batch).logits
        probabilities = torch.softmax(logits, dim=-1)
    return logits.detach().cpu(), probabilities.detach().cpu()


def infer_new_sentences(texts: str | Sequence[str]) -> dict[str, Any]:
    """Run verified inference on custom text without any dataset/Test access."""
    normalized = _normalize_inputs(texts)
    provenance = verify_authoritative_checkpoint()
    model, tokenizer, data_collator, device = _load_inference_components(provenance)
    batch, preprocessing = _prepare_batch(normalized, tokenizer, data_collator, device)
    logits_first, probabilities_first = _forward_probabilities(model, batch)
    logits_second, probabilities_second = _forward_probabilities(model, batch)
    expected_shape = (len(normalized), 2)
    labels_first = probabilities_first.argmax(dim=-1)
    labels_second = probabilities_second.argmax(dim=-1)
    row_sums = probabilities_first.sum(dim=-1)
    numerical_checks = {
        "logits_shape_B_2": tuple(logits_first.shape) == expected_shape,
        "logits_finite": bool(torch.isfinite(logits_first).all().item()),
        "probabilities_finite": bool(torch.isfinite(probabilities_first).all().item()),
        "probabilities_in_0_1": bool(
            ((probabilities_first >= 0.0) & (probabilities_first <= 1.0)).all().item()
        ),
        "probability_rows_sum_to_one": bool(
            torch.allclose(row_sums, torch.ones_like(row_sums), atol=PROBABILITY_SUM_TOLERANCE, rtol=0.0)
        ),
        "predicted_labels_deterministic": bool(torch.equal(labels_first, labels_second)),
        "probabilities_deterministic": bool(
            torch.allclose(
                probabilities_first,
                probabilities_second,
                atol=DETERMINISM_TOLERANCE,
                rtol=0.0,
            )
        ),
        "model_remains_in_eval_mode": model.training is False,
    }
    results = []
    for index, text in enumerate(normalized):
        predicted_label = int(labels_first[index].item())
        negative_probability = float(probabilities_first[index, 0].item())
        positive_probability = float(probabilities_first[index, 1].item())
        confidence = float(probabilities_first[index, predicted_label].item())
        record_checks = {
            "label_matches_argmax": predicted_label
            == int(torch.argmax(probabilities_first[index]).item()),
            "confidence_matches_predicted_class": math.isclose(
                confidence,
                (negative_probability, positive_probability)[predicted_label],
                abs_tol=CONFIDENCE_TOLERANCE,
                rel_tol=0.0,
            ),
        }
        if not all(record_checks.values()):
            raise RuntimeError(f"Prediction record verification failed at input {index}")
        results.append({
            "input_index": index,
            "text": text,
            "predicted_label": predicted_label,
            "predicted_label_name": EXPECTED_ID2LABEL[predicted_label],
            "negative_probability": negative_probability,
            "positive_probability": positive_probability,
            "confidence": confidence,
            "checks": record_checks,
        })
    weight_sha256_after = _sha256_file(Path(provenance["weight_file"]))
    verification_checks = {
        **numerical_checks,
        "output_count_equals_input_count": len(results) == len(normalized),
        "input_order_preserved": [row["text"] for row in results] == normalized,
        "all_record_checks_pass": all(all(row["checks"].values()) for row in results),
        "checkpoint_fingerprint_unchanged": weight_sha256_after == provenance["weight_sha256"],
        "single_input_contract_verified": len(_normalize_inputs(normalized[0])) == 1,
        "batch_input_contract_verified": len(_normalize_inputs(normalized)) == len(normalized),
    }
    if not all(verification_checks.values()):
        raise RuntimeError(f"Phase 13 inference verification failed: {verification_checks}")
    return {
        "status": "PASS",
        "executed_at_utc": _utc_now(),
        "input_provenance": "custom_authored_not_test",
        "input_signature_sha256": _sha256_json(normalized),
        "input_count": len(normalized),
        "device": str(device),
        "checkpoint": provenance,
        "preprocessing": preprocessing,
        "logits_shape": list(logits_first.shape),
        "results": results,
        "verification": {
            "status": "PASS",
            "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
            "confidence_tolerance": CONFIDENCE_TOLERANCE,
            "determinism_tolerance": DETERMINISM_TOLERANCE,
            "checks": verification_checks,
        },
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
        "test_accessed": False,
        "test_evaluated": False,
        "test_evaluation_count": 1,
        "checkpoint_changed": False,
        "phase_14_started": False,
    }


def _valid_existing_artifact(texts: list[str]) -> dict[str, Any] | None:
    if not PHASE_13_ARTIFACT_PATH.is_file():
        return None
    artifact = _load_json(PHASE_13_ARTIFACT_PATH)
    try:
        provenance = verify_authoritative_checkpoint()
    except (FileNotFoundError, RuntimeError, KeyError, TypeError, ValueError):
        return None
    valid = (
        artifact.get("status") == "PASS"
        and artifact.get("verification", {}).get("status") == "PASS"
        and all(artifact.get("verification", {}).get("checks", {}).values())
        and artifact.get("input_signature_sha256") == _sha256_json(texts)
        and artifact.get("input_count") == len(texts)
        and len(artifact.get("results", [])) == len(texts)
        and artifact.get("checkpoint", {}).get("weight_sha256") == provenance["weight_sha256"]
        and artifact.get("checkpoint", {}).get("step") == EXPECTED_STEP
        and artifact.get("preprocessing", {}).get("max_length") == MAX_TOKEN_LENGTH
        and artifact.get("input_provenance") == "custom_authored_not_test"
        and artifact.get("training_performed") is False
        and artifact.get("test_accessed") is False
        and artifact.get("test_evaluated") is False
        and artifact.get("test_evaluation_count") == 1
        and artifact.get("checkpoint_changed") is False
        and artifact.get("phase_14_started") is False
    )
    return artifact if valid else None


def run_or_load_phase_13(
    texts: str | Sequence[str] = DEFAULT_CUSTOM_SENTENCES,
) -> dict[str, Any]:
    """Load matching verified demo evidence or perform new-text inference."""
    normalized = _normalize_inputs(texts)
    existing = _valid_existing_artifact(normalized)
    if existing is not None:
        return {
            **existing,
            "guard_action": "loaded_verified_phase_13_artifact_no_model_inference",
            "artifact_path": str(PHASE_13_ARTIFACT_PATH.resolve()),
        }
    result = infer_new_sentences(normalized)
    _atomic_json(PHASE_13_ARTIFACT_PATH, result)
    return {
        **result,
        "guard_action": "executed_new_sentence_inference",
        "artifact_path": str(PHASE_13_ARTIFACT_PATH.resolve()),
    }


def load_phase_13_report() -> dict[str, Any]:
    """Load the verified default-demo artifact without loading the model."""
    normalized = _normalize_inputs(DEFAULT_CUSTOM_SENTENCES)
    artifact = _valid_existing_artifact(normalized)
    if artifact is None:
        raise RuntimeError("No verified Phase 13 default-demo artifact is available")
    return {
        **artifact,
        "guard_action": "loaded_verified_phase_13_artifact_no_model_inference",
        "artifact_path": str(PHASE_13_ARTIFACT_PATH.resolve()),
    }
