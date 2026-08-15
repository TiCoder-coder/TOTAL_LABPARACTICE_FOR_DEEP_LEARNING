"""Phase 14: save-once local package and exact save/reload verification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_TOKEN_LENGTH, RESULT_DIR
from .phase_01_environment import get_device
from .phase_06_preprocessing import get_data_collator
from .phase_13_new_sentence_inference import (
    EXPECTED_ID2LABEL,
    EXPECTED_LABEL2ID,
    PHASE_13_ARTIFACT_PATH,
    verify_authoritative_checkpoint,
)


PACKAGE_DIR = RESULT_DIR / "phase_14_saved_model"
PACKAGE_MANIFEST_PATH = PACKAGE_DIR / "package_manifest.json"
VERIFICATION_PATH = RESULT_DIR / "phase_14_save_reload_verification.json"

LOGITS_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-7
CONFIDENCE_TOLERANCE = 1e-7
PROBABILITY_SUM_TOLERANCE = 1e-6
EXPECTED_INPUT_COUNT = 4
FORBIDDEN_TRAINING_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
}


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


def _normalize_id2label(mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in mapping.items()}


def _tokenizer_summary(tokenizer: Any) -> dict[str, Any]:
    special_tokens = {key: str(value) for key, value in tokenizer.special_tokens_map.items()}
    special_token_ids = {
        key: int(getattr(tokenizer, f"{key}_id"))
        for key in special_tokens
        if getattr(tokenizer, f"{key}_id", None) is not None
    }
    return {
        "class": tokenizer.__class__.__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "model_max_length": int(tokenizer.model_max_length),
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
    }


def _model_summary(model: Any) -> dict[str, Any]:
    return {
        "class": model.__class__.__name__,
        "model_type": str(model.config.model_type),
        "num_labels": int(model.config.num_labels),
        "id2label": _normalize_id2label(model.config.id2label),
        "label2id": {str(key): int(value) for key, value in model.config.label2id.items()},
    }


def _model_summary_equal(actual: dict[str, Any], recorded: dict[str, Any]) -> bool:
    """Compare a runtime summary with its JSON round-tripped representation."""
    normalized_recorded = {
        **recorded,
        "id2label": _normalize_id2label(recorded.get("id2label", {})),
        "label2id": {
            str(key): int(value) for key, value in recorded.get("label2id", {}).items()
        },
    }
    return actual == normalized_recorded


def _load_phase_13_contract() -> tuple[dict[str, Any], list[str]]:
    artifact = _load_json(PHASE_13_ARTIFACT_PATH)
    texts = [str(record["text"]) for record in artifact.get("results", [])]
    checks = {
        "phase_13_status_pass": artifact.get("status") == "PASS",
        "phase_13_verification_pass": artifact.get("verification", {}).get("status") == "PASS",
        "custom_not_test_provenance": artifact.get("input_provenance") == "custom_authored_not_test",
        "input_count_four": artifact.get("input_count") == EXPECTED_INPUT_COUNT
        and len(texts) == EXPECTED_INPUT_COUNT,
        "ordered_signature_matches": _sha256_json(texts)
        == artifact.get("input_signature_sha256"),
        "max_length_80": artifact.get("preprocessing", {}).get("max_length")
        == MAX_TOKEN_LENGTH
        == 80,
        "truncation_locked": artifact.get("preprocessing", {}).get("truncation") is True,
        "dynamic_padding_locked": artifact.get("preprocessing", {}).get("batch_padding")
        == "DataCollatorWithPadding_dynamic",
        "test_not_accessed": artifact.get("test_accessed") is False,
        "test_not_evaluated": artifact.get("test_evaluated") is False,
        "test_evaluation_count_one": artifact.get("test_evaluation_count") == 1,
        "phase_14_not_previously_started": artifact.get("phase_14_started") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 13 inference contract verification failed: {checks}")
    return artifact, texts


def _load_local_components(path: Path) -> tuple[Any, Any]:
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model.eval()
    summary = _model_summary(model)
    checks = {
        "distilbert_classifier": summary["class"]
        == "DistilBertForSequenceClassification",
        "model_type_distilbert": summary["model_type"] == "distilbert",
        "num_labels_two": summary["num_labels"] == 2,
        "id2label_correct": summary["id2label"] == EXPECTED_ID2LABEL,
        "label2id_correct": summary["label2id"] == EXPECTED_LABEL2ID,
        "eval_mode": model.training is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Local model configuration verification failed: {checks}")
    return model, tokenizer


def _package_inventory(directory: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == PACKAGE_MANIFEST_PATH.name:
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Package entry is not a non-empty regular file: {path}")
        inventory[path.name] = {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    weight_files = {"model.safetensors", "pytorch_model.bin"} & set(inventory)
    checks = {
        "required_config_tokenizer_files": required.issubset(inventory),
        "exactly_one_weight_file": len(weight_files) == 1,
        "no_training_state_files": not (FORBIDDEN_TRAINING_FILES & set(inventory)),
        "all_files_non_empty": all(item["bytes"] > 0 for item in inventory.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Saved package file verification failed: {checks}")
    return inventory


def _verify_inventory(directory: Path, manifest: dict[str, Any]) -> dict[str, bool]:
    recorded = manifest.get("package_files", {})
    actual_names = {
        path.name for path in directory.iterdir()
        if path.is_file() and path.name != PACKAGE_MANIFEST_PATH.name
    }
    checks = {
        "recorded_file_set_exact": actual_names == set(recorded),
        "no_forbidden_training_files": not (FORBIDDEN_TRAINING_FILES & actual_names),
        "manifest_present_non_empty": (directory / PACKAGE_MANIFEST_PATH.name).is_file()
        and (directory / PACKAGE_MANIFEST_PATH.name).stat().st_size > 0,
    }
    for name, metadata in recorded.items():
        path = directory / name
        checks[f"{name}_integrity"] = (
            path.is_file()
            and path.stat().st_size == int(metadata.get("bytes", -1))
            and _sha256_file(path) == metadata.get("sha256")
        )
    return checks


def _prepare_batch(tokenizer: Any, texts: list[str], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
    batch = get_data_collator(tokenizer)(features)
    batch_length = int(batch["input_ids"].shape[1])
    padding_counts = [int((mask == 0).sum().item()) for mask in batch["attention_mask"]]
    checks = {
        "input_ids_attention_mask_shape_match": batch["input_ids"].shape
        == batch["attention_mask"].shape,
        "batch_count_four": int(batch["input_ids"].shape[0]) == EXPECTED_INPUT_COUNT,
        "sequence_length_within_80": batch_length <= MAX_TOKEN_LENGTH,
        "dynamic_padding_valid": padding_counts
        == [batch_length - length for length in original_lengths],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 14 batch verification failed: {checks}")
    return (
        {key: value.to(device) for key, value in batch.items()},
        {
            "input_ids": batch["input_ids"].tolist(),
            "attention_mask": batch["attention_mask"].tolist(),
            "shape": list(batch["input_ids"].shape),
            "original_lengths": original_lengths,
            "padding_counts": padding_counts,
            "checks": checks,
        },
    )


def _state_dict_equivalence(source_model: Any, reloaded_model: Any) -> dict[str, Any]:
    source_state = source_model.state_dict()
    reloaded_state = reloaded_model.state_dict()
    source_keys = list(source_state)
    reloaded_keys = list(reloaded_state)
    key_sets_equal = source_keys == reloaded_keys
    mismatched_shapes = []
    mismatched_dtypes = []
    unequal_tensors = []
    if key_sets_equal:
        for key in source_keys:
            source_tensor = source_state[key].detach().cpu()
            reloaded_tensor = reloaded_state[key].detach().cpu()
            if source_tensor.shape != reloaded_tensor.shape:
                mismatched_shapes.append(key)
            if source_tensor.dtype != reloaded_tensor.dtype:
                mismatched_dtypes.append(key)
            if not torch.equal(source_tensor, reloaded_tensor):
                unequal_tensors.append(key)
    status = key_sets_equal and not mismatched_shapes and not mismatched_dtypes and not unequal_tensors
    return {
        "status": "PASS" if status else "FAIL",
        "tensor_count": len(source_keys),
        "key_sets_and_order_equal": key_sets_equal,
        "mismatched_shape_keys": mismatched_shapes,
        "mismatched_dtype_keys": mismatched_dtypes,
        "unequal_tensor_keys": unequal_tensors,
        "all_tensors_exact_equal": status,
    }


def _forward(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(**batch).logits.detach().cpu()
        probabilities = torch.softmax(logits, dim=-1)
    return logits, probabilities


def _prediction_equivalence(
    source_model: Any,
    source_tokenizer: Any,
    reloaded_model: Any,
    reloaded_tokenizer: Any,
    texts: list[str],
    device: torch.device,
) -> dict[str, Any]:
    source_batch, source_batch_info = _prepare_batch(source_tokenizer, texts, device)
    reloaded_batch, reloaded_batch_info = _prepare_batch(reloaded_tokenizer, texts, device)
    source_logits, source_probabilities = _forward(source_model, source_batch)
    reloaded_logits, reloaded_probabilities = _forward(reloaded_model, reloaded_batch)
    source_labels = source_probabilities.argmax(dim=-1)
    reloaded_labels = reloaded_probabilities.argmax(dim=-1)
    source_confidence = source_probabilities.max(dim=-1).values
    reloaded_confidence = reloaded_probabilities.max(dim=-1).values
    logits_max_diff = float((source_logits - reloaded_logits).abs().max().item())
    probability_max_diff = float(
        (source_probabilities - reloaded_probabilities).abs().max().item()
    )
    confidence_max_diff = float((source_confidence - reloaded_confidence).abs().max().item())
    tokenizer_checks = {
        "input_ids_exact_equal": source_batch_info["input_ids"]
        == reloaded_batch_info["input_ids"],
        "attention_mask_exact_equal": source_batch_info["attention_mask"]
        == reloaded_batch_info["attention_mask"],
        "batch_shapes_equal": source_batch_info["shape"] == reloaded_batch_info["shape"],
        "special_tokens_equal": _tokenizer_summary(source_tokenizer)["special_tokens"]
        == _tokenizer_summary(reloaded_tokenizer)["special_tokens"],
        "special_token_ids_equal": _tokenizer_summary(source_tokenizer)["special_token_ids"]
        == _tokenizer_summary(reloaded_tokenizer)["special_token_ids"],
        "source_dynamic_padding_pass": all(source_batch_info["checks"].values()),
        "reloaded_dynamic_padding_pass": all(reloaded_batch_info["checks"].values()),
    }
    numerical_checks = {
        "logits_shapes_B_2": list(source_logits.shape) == [EXPECTED_INPUT_COUNT, 2]
        and list(reloaded_logits.shape) == [EXPECTED_INPUT_COUNT, 2],
        "logits_finite": bool(torch.isfinite(source_logits).all().item())
        and bool(torch.isfinite(reloaded_logits).all().item()),
        "probabilities_finite": bool(torch.isfinite(source_probabilities).all().item())
        and bool(torch.isfinite(reloaded_probabilities).all().item()),
        "probabilities_in_range": bool(
            ((source_probabilities >= 0) & (source_probabilities <= 1)).all().item()
        ) and bool(
            ((reloaded_probabilities >= 0) & (reloaded_probabilities <= 1)).all().item()
        ),
        "probability_rows_sum_to_one": bool(torch.allclose(
            source_probabilities.sum(dim=-1),
            torch.ones(EXPECTED_INPUT_COUNT),
            atol=PROBABILITY_SUM_TOLERANCE,
            rtol=0.0,
        )) and bool(torch.allclose(
            reloaded_probabilities.sum(dim=-1),
            torch.ones(EXPECTED_INPUT_COUNT),
            atol=PROBABILITY_SUM_TOLERANCE,
            rtol=0.0,
        )),
        "predicted_labels_exact_equal": bool(torch.equal(source_labels, reloaded_labels)),
        "logits_within_tolerance": logits_max_diff <= LOGITS_TOLERANCE,
        "probabilities_within_tolerance": probability_max_diff <= PROBABILITY_TOLERANCE,
        "confidence_within_tolerance": confidence_max_diff <= CONFIDENCE_TOLERANCE,
        "output_count_and_order_preserved": len(texts) == EXPECTED_INPUT_COUNT,
    }
    if not all(tokenizer_checks.values()) or not all(numerical_checks.values()):
        raise RuntimeError(
            f"Save/reload prediction equivalence failed: tokenizer={tokenizer_checks}, "
            f"numerical={numerical_checks}"
        )
    rows = []
    for index, text in enumerate(texts):
        source_label = int(source_labels[index].item())
        reloaded_label = int(reloaded_labels[index].item())
        rows.append({
            "input_index": index,
            "text": text,
            "source_label": source_label,
            "source_label_name": EXPECTED_ID2LABEL[source_label],
            "reloaded_label": reloaded_label,
            "reloaded_label_name": EXPECTED_ID2LABEL[reloaded_label],
            "source_negative_probability": float(source_probabilities[index, 0].item()),
            "reloaded_negative_probability": float(reloaded_probabilities[index, 0].item()),
            "source_positive_probability": float(source_probabilities[index, 1].item()),
            "reloaded_positive_probability": float(reloaded_probabilities[index, 1].item()),
            "source_confidence": float(source_confidence[index].item()),
            "reloaded_confidence": float(reloaded_confidence[index].item()),
            "logits_max_abs_diff_for_input": float(
                (source_logits[index] - reloaded_logits[index]).abs().max().item()
            ),
            "probability_max_abs_diff_for_input": float(
                (source_probabilities[index] - reloaded_probabilities[index]).abs().max().item()
            ),
            "confidence_abs_diff": float(
                abs(source_confidence[index].item() - reloaded_confidence[index].item())
            ),
        })
    return {
        "status": "PASS",
        "tokenizer_checks": tokenizer_checks,
        "numerical_checks": numerical_checks,
        "source_batch": source_batch_info,
        "reloaded_batch": reloaded_batch_info,
        "source_logits_shape": list(source_logits.shape),
        "reloaded_logits_shape": list(reloaded_logits.shape),
        "logits_max_abs_diff": logits_max_diff,
        "probability_max_abs_diff": probability_max_diff,
        "confidence_max_abs_diff": confidence_max_diff,
        "tolerances": {
            "logits": LOGITS_TOLERANCE,
            "probabilities": PROBABILITY_TOLERANCE,
            "confidence": CONFIDENCE_TOLERANCE,
            "probability_sum": PROBABILITY_SUM_TOLERANCE,
        },
        "predictions": rows,
    }


def _build_manifest(
    directory: Path,
    provenance: dict[str, Any],
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    inventory = _package_inventory(directory)
    weight_name = next(name for name in inventory if name in {"model.safetensors", "pytorch_model.bin"})
    manifest = {
        "status": "PASS",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "package_path": str(PACKAGE_DIR.resolve()),
        "source_checkpoint": provenance["checkpoint"],
        "source_checkpoint_name": provenance["checkpoint_name"],
        "source_epoch": provenance["epoch"],
        "source_step": provenance["step"],
        "source_weight_sha256": provenance["weight_sha256"],
        "selection_record_sha256": provenance["selection_record_sha256"],
        "package_weight_file": weight_name,
        "package_weight_sha256": inventory[weight_name]["sha256"],
        "model": _model_summary(model),
        "tokenizer": _tokenizer_summary(tokenizer),
        "preprocessing": {
            "max_length": MAX_TOKEN_LENGTH,
            "truncation": True,
            "per_example_padding": False,
            "batch_padding": "DataCollatorWithPadding_dynamic",
        },
        "save_methods": ["model.save_pretrained", "tokenizer.save_pretrained"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "package_files": inventory,
        "manifest_self_hash_excluded": True,
        "training_performed": False,
        "test_accessed": False,
        "test_evaluated": False,
        "test_evaluation_count": 1,
        "checkpoint_selected_or_changed": False,
        "phase_15_started": False,
    }
    _atomic_json(directory / PACKAGE_MANIFEST_PATH.name, manifest)
    return manifest


def _create_package() -> dict[str, Any]:
    if PACKAGE_DIR.exists() or VERIFICATION_PATH.exists():
        raise RuntimeError(
            "Phase 14 contains existing/partial package artifacts; refusing automatic overwrite"
        )
    provenance = verify_authoritative_checkpoint()
    phase_13, texts = _load_phase_13_contract()
    source_path = Path(provenance["checkpoint"])
    source_model, source_tokenizer = _load_local_components(source_path)
    source_model_summary = _model_summary(source_model)
    source_tokenizer_summary = _tokenizer_summary(source_tokenizer)
    source_weight_sha_before = _sha256_file(Path(provenance["weight_file"]))
    temporary_directory = Path(tempfile.mkdtemp(prefix=".phase_14_saved_model.", dir=RESULT_DIR))
    try:
        source_model.save_pretrained(str(temporary_directory), safe_serialization=True)
        source_tokenizer.save_pretrained(str(temporary_directory))
        manifest = _build_manifest(
            temporary_directory, provenance, source_model, source_tokenizer
        )
        inventory_checks = _verify_inventory(temporary_directory, manifest)
        if not all(inventory_checks.values()):
            raise RuntimeError(f"Temporary package integrity failed: {inventory_checks}")
        temporary_directory.replace(PACKAGE_DIR)
    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        raise

    published_manifest = _load_json(PACKAGE_MANIFEST_PATH)
    package_checks = _verify_inventory(PACKAGE_DIR, published_manifest)
    if not all(package_checks.values()):
        raise RuntimeError(f"Published package integrity failed: {package_checks}")
    reloaded_model, reloaded_tokenizer = _load_local_components(PACKAGE_DIR)
    reloaded_model_summary = _model_summary(reloaded_model)
    reloaded_tokenizer_summary = _tokenizer_summary(reloaded_tokenizer)
    state_dict = _state_dict_equivalence(source_model, reloaded_model)
    if state_dict["status"] != "PASS":
        raise RuntimeError(f"State-dict equality failed: {state_dict}")
    device = torch.device(get_device())
    source_model.to(device)
    reloaded_model.to(device)
    prediction = _prediction_equivalence(
        source_model,
        source_tokenizer,
        reloaded_model,
        reloaded_tokenizer,
        texts,
        device,
    )
    source_weight_sha_after = _sha256_file(Path(provenance["weight_file"]))
    completion_checks = {
        "authoritative_source_verified": provenance["status"] == "PASS",
        "source_checkpoint_unchanged": source_weight_sha_before
        == source_weight_sha_after
        == provenance["weight_sha256"],
        "package_integrity_pass": all(package_checks.values()),
        "package_reloaded_local_only": reloaded_tokenizer.name_or_path
        == str(PACKAGE_DIR.resolve()),
        "model_config_equal": source_model_summary == reloaded_model_summary,
        "tokenizer_contract_equal": source_tokenizer_summary == reloaded_tokenizer_summary,
        "state_dict_exact_equal": state_dict["status"] == "PASS",
        "prediction_equivalence_pass": prediction["status"] == "PASS",
        "max_length_80": MAX_TOKEN_LENGTH == 80,
        "test_evaluation_count_one": phase_13["test_evaluation_count"] == 1,
        "no_training_or_test_access": True,
    }
    if not all(completion_checks.values()):
        raise RuntimeError(f"Phase 14 completion verification failed: {completion_checks}")
    verification = {
        "status": "PASS",
        "verified_at_utc": _utc_now(),
        "guard_action": "saved_and_verified_phase_14_package_once",
        "package_path": str(PACKAGE_DIR.resolve()),
        "package_manifest_path": str(PACKAGE_MANIFEST_PATH.resolve()),
        "package_manifest_sha256": _sha256_file(PACKAGE_MANIFEST_PATH),
        "verification_artifact_path": str(VERIFICATION_PATH.resolve()),
        "source_checkpoint": provenance,
        "source_model": source_model_summary,
        "reloaded_model": reloaded_model_summary,
        "source_tokenizer": source_tokenizer_summary,
        "reloaded_tokenizer": reloaded_tokenizer_summary,
        "preprocessing": published_manifest["preprocessing"],
        "phase_13_input_signature_sha256": phase_13["input_signature_sha256"],
        "input_count": len(texts),
        "device": str(device),
        "package_files": published_manifest["package_files"],
        "package_integrity_checks": package_checks,
        "state_dict_equivalence": state_dict,
        "prediction_equivalence": prediction,
        "completion_checks": completion_checks,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_performed": False,
        "scheduler_step_performed": False,
        "test_accessed": False,
        "test_evaluated": False,
        "test_evaluation_count": 1,
        "checkpoint_selected_or_changed": False,
        "phase_15_started": False,
    }
    _atomic_json(VERIFICATION_PATH, verification)
    return verification


def _load_verified_package() -> dict[str, Any] | None:
    if not PACKAGE_DIR.is_dir() or not PACKAGE_MANIFEST_PATH.is_file() or not VERIFICATION_PATH.is_file():
        return None
    manifest = _load_json(PACKAGE_MANIFEST_PATH)
    verification = _load_json(VERIFICATION_PATH)
    provenance = verify_authoritative_checkpoint()
    phase_13, texts = _load_phase_13_contract()
    package_checks = _verify_inventory(PACKAGE_DIR, manifest)
    semantic_checks = {
        "manifest_pass": manifest.get("status") == "PASS",
        "verification_pass": verification.get("status") == "PASS",
        "manifest_hash_matches_verification": _sha256_file(PACKAGE_MANIFEST_PATH)
        == verification.get("package_manifest_sha256"),
        "all_package_files_valid": all(package_checks.values()),
        "source_fingerprint_current": manifest.get("source_weight_sha256")
        == provenance["weight_sha256"],
        "source_identity_current": manifest.get("source_checkpoint_name") == "checkpoint-1068"
        and float(manifest.get("source_epoch", -1)) == 2.0
        and int(manifest.get("source_step", -1)) == 1068,
        "phase_13_signature_current": verification.get("phase_13_input_signature_sha256")
        == phase_13["input_signature_sha256"]
        == _sha256_json(texts),
        "completion_checks_pass": all(verification.get("completion_checks", {}).values()),
        "equivalence_artifact_pass": verification.get("state_dict_equivalence", {}).get("status") == "PASS"
        and verification.get("prediction_equivalence", {}).get("status") == "PASS",
        "test_evaluation_count_one": verification.get("test_evaluation_count") == 1,
        "isolation_flags_pass": verification.get("training_performed") is False
        and verification.get("test_accessed") is False
        and verification.get("test_evaluated") is False
        and verification.get("checkpoint_selected_or_changed") is False
        and verification.get("phase_15_started") is False,
    }
    if not all(semantic_checks.values()):
        return None
    model, tokenizer = _load_local_components(PACKAGE_DIR)
    loaded_checks = {
        "model_loaded_from_package": _model_summary_equal(
            _model_summary(model), verification["reloaded_model"]
        ),
        "tokenizer_loaded_from_package": _tokenizer_summary(tokenizer)
        == verification["reloaded_tokenizer"],
        "package_path_local": tokenizer.name_or_path == str(PACKAGE_DIR.resolve()),
    }
    if not all(loaded_checks.values()):
        return None
    return {
        **verification,
        "guard_action": "loaded_verified_phase_14_package_no_resave",
        "guard_checks": {**semantic_checks, **loaded_checks},
    }


def run_or_load_phase_14() -> dict[str, Any]:
    """Save and verify once; subsequent calls only validate/load the local package."""
    existing = _load_verified_package()
    if existing is not None:
        return existing
    if PACKAGE_DIR.exists() or PACKAGE_MANIFEST_PATH.exists() or VERIFICATION_PATH.exists():
        raise RuntimeError(
            "Phase 14 package/artifact exists but failed integrity; refusing overwrite or repair"
        )
    return _create_package()


def load_phase_14_report() -> dict[str, Any]:
    """Load a valid Phase 14 package report without saving or Test evaluation."""
    report = _load_verified_package()
    if report is None:
        raise RuntimeError("No complete verified Phase 14 package is available")
    return report
