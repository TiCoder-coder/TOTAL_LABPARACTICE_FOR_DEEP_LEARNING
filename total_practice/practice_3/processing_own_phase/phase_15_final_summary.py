"""Phase 15: artifact-only final summary for Practice 3.

This module intentionally uses only the Python standard library. It does not
load a model, tokenizer, dataset, Trainer, checkpoint tensor, or Test provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, RESULT_DIR


SUMMARY_JSON_PATH = RESULT_DIR / "phase_15_final_summary.json"
SUMMARY_MARKDOWN_PATH = RESULT_DIR / "phase_15_final_summary.md"

OVERVIEW_PLAN_PATH = PROJECT_ROOT / "docs" / "plan-doc" / "plan_overview" / "plan.md"
PHASE_00_03_LOG_PATH = (
    PROJECT_ROOT
    / "docs"
    / "save_process_proceduce_own_phase_refactor&fix"
    / "phase_00_03_priority_1_fix_log_2026-08-12.md"
)

ENVIRONMENT_PATH = RESULT_DIR / "2026-08-10_phase01-environment-log.json"
DATASET_PATH = RESULT_DIR / "phase_04_dataset_summary.json"
EDA_PATH = RESULT_DIR / "phase_05_eda_summary.json"
TOKEN_STATS_PATH = RESULT_DIR / "phase_05_token_length_statistics.json"
PREPROCESSING_PATH = RESULT_DIR / "phase_06_preprocessing_verification.json"
MODEL_PATH = RESULT_DIR / "phase_07_model_verification.json"
CONFIGURATION_PATH = RESULT_DIR / "phase_08_metrics_training_configuration_verification.json"

PHASE_09_DIR = RESULT_DIR / "phase_09_training"
PHASE_09_MANIFEST_PATH = PHASE_09_DIR / "phase_09_training_manifest.json"
TRAINING_CONFIGURATION_PATH = PHASE_09_DIR / "training_configuration.json"
TRAINING_HISTORY_PATH = PHASE_09_DIR / "training_history.json"
VALIDATION_HISTORY_PATH = PHASE_09_DIR / "validation_metrics_by_epoch.json"
CHECKPOINT_RECORDS_PATH = PHASE_09_DIR / "checkpoint_records.json"
SELECTED_CHECKPOINT_PATH = PHASE_09_DIR / "selected_checkpoint.json"

LEARNING_ANALYSIS_PATH = RESULT_DIR / "phase_10_learning_curve_analysis.json"
LOSS_CURVES_PATH = RESULT_DIR / "phase_10_loss_curves.png"
VALIDATION_CURVES_PATH = RESULT_DIR / "phase_10_validation_metrics.png"

PHASE_11_DIR = RESULT_DIR / "phase_11_evaluation"
VALIDATION_EVALUATION_PATH = PHASE_11_DIR / "validation_evaluation.json"
TEST_EVALUATION_PATH = PHASE_11_DIR / "test_evaluation.json"
PHASE_11_MANIFEST_PATH = PHASE_11_DIR / "phase_11_evaluation_manifest.json"

CONFUSION_PATH = RESULT_DIR / "phase_12_confusion_matrix.json"
CONFUSION_FIGURE_PATH = RESULT_DIR / "phase_12_confusion_matrix.png"
ERROR_ANALYSIS_PATH = RESULT_DIR / "phase_12_error_analysis.json"
ERROR_SAMPLES_PATH = RESULT_DIR / "phase_12_error_samples.json"

INFERENCE_PATH = RESULT_DIR / "phase_13_inference_examples.json"
SAVE_RELOAD_PATH = RESULT_DIR / "phase_14_save_reload_verification.json"
PACKAGE_MANIFEST_PATH = RESULT_DIR / "phase_14_saved_model" / "package_manifest.json"

METRICS = ("loss", "accuracy", "precision", "recall", "f1")
EXPECTED_COUNTS = {"train": 8530, "validation": 1066, "test": 1066}
EXPECTED_WEIGHT_SHA256 = "22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660"

SOURCE_PATHS = (
    OVERVIEW_PLAN_PATH,
    PHASE_00_03_LOG_PATH,
    ENVIRONMENT_PATH,
    DATASET_PATH,
    EDA_PATH,
    TOKEN_STATS_PATH,
    PREPROCESSING_PATH,
    MODEL_PATH,
    CONFIGURATION_PATH,
    PHASE_09_MANIFEST_PATH,
    TRAINING_CONFIGURATION_PATH,
    TRAINING_HISTORY_PATH,
    VALIDATION_HISTORY_PATH,
    CHECKPOINT_RECORDS_PATH,
    SELECTED_CHECKPOINT_PATH,
    LEARNING_ANALYSIS_PATH,
    LOSS_CURVES_PATH,
    VALIDATION_CURVES_PATH,
    VALIDATION_EVALUATION_PATH,
    TEST_EVALUATION_PATH,
    PHASE_11_MANIFEST_PATH,
    CONFUSION_PATH,
    CONFUSION_FIGURE_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SAMPLES_PATH,
    INFERENCE_PATH,
    SAVE_RELOAD_PATH,
    PACKAGE_MANIFEST_PATH,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 0–14 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 0–14 document is missing: {path}")
    return path.read_text(encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    if path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Text read-back verification failed: {path}")
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    _atomic_text(path, serialized)
    if _load_json(path) != json.loads(serialized):
        raise RuntimeError(f"JSON read-back verification failed: {path}")
    return path


def _source_metadata(paths: tuple[Path, ...] = SOURCE_PATHS) -> dict[str, dict[str, Any]]:
    metadata = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required non-empty source is missing: {path}")
        relative = str(path.relative_to(PROJECT_ROOT))
        metadata[relative] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return metadata


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    if set(METRICS) - set(metrics):
        return False
    loss = float(metrics["loss"])
    scores = [float(metrics[name]) for name in METRICS if name != "loss"]
    return (
        math.isfinite(loss)
        and loss >= 0.0
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores)
    )


def _verify_package_files(package_manifest: dict[str, Any]) -> dict[str, bool]:
    package_dir = Path(package_manifest["package_path"])
    checks = {}
    for name, metadata in package_manifest.get("package_files", {}).items():
        path = package_dir / name
        checks[f"package_{name}_integrity"] = (
            path.is_file()
            and path.stat().st_size == int(metadata["bytes"])
            and _sha256_file(path) == metadata["sha256"]
        )
    forbidden = {
        "optimizer.pt", "scheduler.pt", "rng_state.pth",
        "trainer_state.json", "training_args.bin",
    }
    checks["package_has_no_training_state"] = not any(
        (package_dir / name).exists() for name in forbidden
    )
    return checks


def _load_sources() -> dict[str, Any]:
    return {
        "overview": _read_text(OVERVIEW_PLAN_PATH),
        "phase_00_03_log": _read_text(PHASE_00_03_LOG_PATH),
        "environment": _load_json(ENVIRONMENT_PATH),
        "dataset": _load_json(DATASET_PATH),
        "eda": _load_json(EDA_PATH),
        "token_stats": _load_json(TOKEN_STATS_PATH),
        "preprocessing": _load_json(PREPROCESSING_PATH),
        "model": _load_json(MODEL_PATH),
        "configuration": _load_json(CONFIGURATION_PATH),
        "phase_09_manifest": _load_json(PHASE_09_MANIFEST_PATH),
        "training_configuration": _load_json(TRAINING_CONFIGURATION_PATH),
        "training_history": _load_json(TRAINING_HISTORY_PATH),
        "validation_history": _load_json(VALIDATION_HISTORY_PATH),
        "checkpoint_records": _load_json(CHECKPOINT_RECORDS_PATH),
        "selected": _load_json(SELECTED_CHECKPOINT_PATH),
        "learning": _load_json(LEARNING_ANALYSIS_PATH),
        "validation": _load_json(VALIDATION_EVALUATION_PATH),
        "test": _load_json(TEST_EVALUATION_PATH),
        "phase_11_manifest": _load_json(PHASE_11_MANIFEST_PATH),
        "confusion": _load_json(CONFUSION_PATH),
        "error_analysis": _load_json(ERROR_ANALYSIS_PATH),
        "error_samples": _load_json(ERROR_SAMPLES_PATH),
        "inference": _load_json(INFERENCE_PATH),
        "save_reload": _load_json(SAVE_RELOAD_PATH),
        "package_manifest": _load_json(PACKAGE_MANIFEST_PATH),
    }


def _validate_sources(source: dict[str, Any]) -> tuple[dict[str, str], dict[str, bool]]:
    environment = source["environment"]
    dataset = source["dataset"]
    eda = source["eda"]
    preprocessing = source["preprocessing"]
    model = source["model"]
    configuration = source["configuration"]
    phase_09 = source["phase_09_manifest"]
    selected = source["selected"]
    learning = source["learning"]
    validation = source["validation"]
    test = source["test"]
    phase_11 = source["phase_11_manifest"]
    confusion = source["confusion"]
    error_analysis = source["error_analysis"]
    error_samples = source["error_samples"]
    inference = source["inference"]
    save_reload = source["save_reload"]
    package = source["package_manifest"]

    phase_status = {
        "phase_00": "PASS" if "Practice Overview" in source["overview"] else "FAIL",
        "phase_01": "PASS" if environment.get("seed") == 42 else "FAIL",
        "phase_02": "PASS" if "Phase 2 pretrained inference: PASS" in source["phase_00_03_log"] else "FAIL",
        "phase_03": "PASS" if "Phase 3 decode sanity check: PASS" in source["phase_00_03_log"] else "FAIL",
        "phase_04": "PASS" if dataset.get("contract", {}).get("splits_exist") is True else "FAIL",
        "phase_05": "PASS" if eda.get("all_pass") is True else "FAIL",
        "phase_06": "PASS" if preprocessing.get("all_pass") is True
        and preprocessing.get("dynamic_padding", {}).get("all_pass") is True else "FAIL",
        "phase_07": str(model.get("status", "FAIL")),
        "phase_08": str(configuration.get("status", "FAIL")),
        "phase_09": str(phase_09.get("status", "FAIL")),
        "phase_10": str(learning.get("status", "FAIL")),
        "phase_11": "PASS" if phase_11.get("status") == "FINAL_TEST_COMPLETE" else "FAIL",
        "phase_12": "PASS" if confusion.get("status") == error_analysis.get("status") == "PASS" else "FAIL",
        "phase_13": str(inference.get("status", "FAIL")),
        "phase_14": "PASS" if save_reload.get("status") == package.get("status") == "PASS" else "FAIL",
    }

    counts = dataset.get("contract", {}).get("split_sizes", {})
    label_sets = dataset.get("contract", {}).get("label_sets", {})
    selected_path = Path(selected.get("checkpoint", ""))
    learning_selected = learning.get("selected_checkpoint", {})
    phase_11_checkpoint = Path(phase_11.get("checkpoint", ""))
    inference_checkpoint = inference.get("checkpoint", {})
    save_checkpoint = save_reload.get("source_checkpoint", {})
    checkpoint_paths = {
        str(selected_path.resolve()),
        str(Path(learning_selected.get("checkpoint", "")).resolve()),
        str(phase_11_checkpoint.resolve()),
        str(Path(inference_checkpoint.get("checkpoint", "")).resolve()),
        str(Path(save_checkpoint.get("checkpoint", "")).resolve()),
    }
    epoch_two_record = next(
        (record for record in source["validation_history"] if float(record["epoch"]) == 2.0),
        None,
    )
    checkpoint_weight_path = Path(save_checkpoint.get("weight_file", ""))
    package_checks = _verify_package_files(package)
    metric_comparisons = confusion.get("phase_11_metric_comparisons", [])
    counts_cm = confusion.get("counts", {})
    direction_counts = error_analysis.get("error_direction_counts", {})
    checks = {
        "all_phases_00_14_pass": all(value == "PASS" for value in phase_status.values()),
        "environment_packages_no_error": all(
            item.get("status") != "ERROR" for item in environment.get("packages", {}).values()
        ),
        "dataset_counts_8530_1066_1066": counts == EXPECTED_COUNTS,
        "all_split_label_sets_0_1": all(
            label_sets.get(split) == [0, 1] for split in EXPECTED_COUNTS
        ),
        "eda_max_length_80": eda.get("selected_max_length") == 80
        and source["token_stats"].get("selected_max_length") == 80,
        "preprocessing_contract_locked": preprocessing.get("max_length") == 80
        and preprocessing.get("all_pass") is True
        and preprocessing.get("dynamic_padding", {}).get("all_pass") is True,
        "phase_07_model_contract": model.get("model_type") == "distilbert"
        and model.get("num_labels") == 2
        and model.get("label_mapping_pass") is True,
        "phase_08_metrics_configuration_pass": configuration.get("status") == "PASS"
        and configuration.get("training_arguments_verification_pass") is True,
        "phase_09_training_contract": phase_09.get("train_samples") == EXPECTED_COUNTS["train"]
        and phase_09.get("validation_samples") == EXPECTED_COUNTS["validation"]
        and phase_09.get("completed_epochs") == 3.0
        and phase_09.get("test_accessed") is False,
        "three_epoch_records": len(source["validation_history"]) == 3
        and len(source["checkpoint_records"]) == 3
        and len(learning.get("epoch_records", [])) == 3,
        "selected_checkpoint_identity_consistent": len(checkpoint_paths) == 1
        and selected_path.name == "checkpoint-1068"
        and float(selected.get("epoch", -1)) == 2.0
        and int(selected.get("step", -1)) == 1068
        and float(learning_selected.get("epoch", -1)) == 2.0
        and int(learning_selected.get("step", -1)) == 1068
        and float(phase_11.get("selected_epoch", -1)) == 2.0
        and int(phase_11.get("selected_step", -1)) == 1068,
        "selected_by_minimum_validation_loss": selected.get("primary_criterion")
        == "minimum eval_loss"
        and float(selected["eval_loss"])
        == min(float(record["eval_loss"]) for record in source["checkpoint_records"]),
        "validation_reload_matches_epoch_two": epoch_two_record is not None
        and all(
            abs(float(validation["metrics"][name]) - float(epoch_two_record[f"eval_{name}"]))
            <= (1e-5 if name == "loss" else 1e-6)
            for name in METRICS
        ),
        "validation_metrics_valid": validation.get("status") == "PASS"
        and _finite_metrics(validation.get("metrics", {})),
        "test_metrics_valid": test.get("status") == "PASS"
        and _finite_metrics(test.get("metrics", {})),
        "one_time_test_policy": phase_11.get("test_evaluation_count") == 1
        and phase_11.get("test_used_for_selection") is False
        and phase_11.get("checkpoint_changed_after_test") is False,
        "confusion_total_1066": sum(int(value) for value in counts_cm.values()) == 1066
        and confusion.get("total") == 1066
        and confusion.get("correct", 0) + confusion.get("incorrect", 0) == 1066,
        "confusion_metrics_match_phase_11": len(metric_comparisons) == 4
        and all(item.get("pass") is True for item in metric_comparisons),
        "error_directions_match_fp_fn": direction_counts.get("NEGATIVE_TO_POSITIVE")
        == counts_cm.get("FP")
        and direction_counts.get("POSITIVE_TO_NEGATIVE") == counts_cm.get("FN")
        and error_samples.get("total_errors") == confusion.get("incorrect"),
        "phase_13_artifact_only_evidence": inference.get("input_count") == 4
        and inference.get("input_provenance") == "custom_authored_not_test"
        and inference.get("verification", {}).get("status") == "PASS"
        and inference.get("training_performed") is False
        and inference.get("test_accessed") is False
        and inference.get("test_evaluated") is False,
        "phase_14_exact_equivalence": save_reload.get("state_dict_equivalence", {}).get("status") == "PASS"
        and save_reload.get("state_dict_equivalence", {}).get("all_tensors_exact_equal") is True
        and save_reload.get("prediction_equivalence", {}).get("status") == "PASS"
        and save_reload.get("prediction_equivalence", {}).get("logits_max_abs_diff", 1.0) <= 1e-6
        and save_reload.get("prediction_equivalence", {}).get("probability_max_abs_diff", 1.0) <= 1e-7
        and save_reload.get("prediction_equivalence", {}).get("confidence_max_abs_diff", 1.0) <= 1e-7,
        "authoritative_source_hash_unchanged": checkpoint_weight_path.is_file()
        and _sha256_file(checkpoint_weight_path) == EXPECTED_WEIGHT_SHA256
        == phase_11.get("checkpoint_weight_sha256")
        == inference_checkpoint.get("weight_sha256")
        == save_checkpoint.get("weight_sha256")
        == package.get("source_weight_sha256"),
        "phase_14_package_hashes_unchanged": all(package_checks.values()),
        "test_count_remains_one_through_phase_14": all(
            artifact.get("test_evaluation_count") == 1
            for artifact in (phase_11, error_analysis, inference, save_reload, package)
        ),
    }
    checks.update(package_checks)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 15 source/cross-artifact verification failed: {failed}")
    return phase_status, checks


def _pipeline() -> list[dict[str, Any]]:
    blocks = [
        ("0–1", "Overview & Environment", "Define the two-exercise workflow and reproducible runtime", "Environment/package contract"),
        ("2", "Pretrained Inference", "Run the already fine-tuned SST-2 sentiment model", "Sentiment labels and confidence examples"),
        ("3", "Tokenization Investigation", "Inspect tokens, IDs, masks and decode behavior", "Verified tokenizer understanding"),
        ("4", "Dataset Loading", "Load and verify Rotten Tomatoes fixed splits", "Train/Validation/Test contract"),
        ("5", "EDA & Sanity Checks", "Inspect text quality, labels, duplicates, overlaps and lengths", "Evidence for maximum token length"),
        ("6", "Preprocessing", "Tokenize all splits with truncation and dynamic padding", "Verified model-ready inputs"),
        ("7", "Model Construction", "Attach a two-label classification head to generic DistilBERT", "Forward-pass-verified baseline model"),
        ("8", "Metrics & Configuration", "Define metrics, training arguments and checkpoint policy", "Locked baseline configuration"),
        ("9", "Fine-Tuning", "Train for three epochs using Train and Validation only", "History and three checkpoints"),
        ("10", "Learning Curves", "Analyze artifact-only training/Validation trajectories", "Selected epoch and cautious generalization evidence"),
        ("11", "Validation & Final Test", "Verify checkpoint reload on Validation, then evaluate Test once", "Frozen final metrics and predictions"),
        ("12", "Error Analysis", "Derive confusion matrix and deterministic errors from frozen predictions", "Class/error evidence without reevaluation"),
        ("13", "New-Sentence Inference", "Run guarded inference on four custom sentences", "Verified custom prediction artifact"),
        ("14", "Save & Reload", "Save a reusable local package and prove exact equivalence", "Integrity-verified local model/tokenizer package"),
    ]
    return [
        {"phase": phase, "name": name, "processing": processing, "verified_output": output, "status": "PASS"}
        for phase, name, processing, output in blocks
    ]


def _build_summary(source: dict[str, Any], phase_status: dict[str, str], checks: dict[str, bool]) -> dict[str, Any]:
    environment = source["environment"]
    dataset = source["dataset"]
    eda = source["eda"]
    preprocessing = source["preprocessing"]
    model = source["model"]
    configuration = source["configuration"]
    phase_09 = source["phase_09_manifest"]
    selected = source["selected"]
    learning = source["learning"]
    validation = source["validation"]
    test = source["test"]
    phase_11 = source["phase_11_manifest"]
    confusion = source["confusion"]
    error_analysis = source["error_analysis"]
    inference = source["inference"]
    save_reload = source["save_reload"]
    package = source["package_manifest"]
    inference_rows = [
        {
            "input_index": record["input_index"],
            "text": record["text"],
            "predicted_label": record["predicted_label_name"],
            "confidence": record["confidence"],
        }
        for record in inference["results"]
    ]
    return {
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "practice": {
            "name": "Practice 3 — Hugging Face Transformers",
            "task": "English binary sentiment classification",
            "dataset": "cornell-movie-review-data/rotten_tomatoes",
            "phase_range": "0–15",
            "completed_phases": "0–14 results summarized by Phase 15",
        },
        "phase_status": phase_status,
        "pipeline": _pipeline(),
        "environment_and_reproducibility": {
            "python_version": environment["python_version"],
            "device_recorded_by_latest_run": environment["device"],
            "seed": environment["seed"],
            "package_versions": {
                name: details["installed"] for name, details in environment["packages"].items()
            },
        },
        "dataset_contract": {
            "identifier": "cornell-movie-review-data/rotten_tomatoes",
            "split_counts": dataset["contract"]["split_sizes"],
            "label_sets": dataset["contract"]["label_sets"],
            "label_mapping": {"0": "NEGATIVE", "1": "POSITIVE"},
            "balanced": eda["eda_conclusion"]["balanced"],
            "null_empty_whitespace_found": eda["eda_conclusion"]["missing_text_found"],
            "duplicate_or_cross_split_overlap_found": eda["eda_conclusion"]["duplicate_or_overlap_found"],
        },
        "preprocessing_contract": {
            "tokenizer": "distilbert/distilbert-base-uncased",
            "max_length": preprocessing["max_length"],
            "truncation": True,
            "dynamic_padding": "DataCollatorWithPadding",
            "token_length_evidence": eda["token_length_recommendation"],
            "justification": eda["max_length_justification"],
            "verification_pass": preprocessing["all_pass"]
            and preprocessing["dynamic_padding"]["all_pass"],
        },
        "final_model": {
            "architecture": "DistilBERT for sequence classification",
            "model_class": package["model"]["class"],
            "num_labels": package["model"]["num_labels"],
            "label_mapping": package["model"]["id2label"],
            "selected_checkpoint": Path(selected["checkpoint"]).name,
            "selected_checkpoint_path": selected["checkpoint"],
            "epoch": selected["epoch"],
            "step": selected["step"],
            "selection_reason": selected["primary_criterion"],
            "tie_breaker": selected["tie_breaker"],
            "remaining_tie": selected["remaining_tie"],
            "selected_validation_loss": selected["eval_loss"],
            "selected_validation_f1": selected["eval_f1"],
            "test_used_for_selection": selected["test_used_for_selection"],
            "authoritative_weight_sha256": phase_11["checkpoint_weight_sha256"],
        },
        "training_configuration": {
            "epochs": phase_09["completed_epochs"],
            "train_samples": phase_09["train_samples"],
            "validation_samples": phase_09["validation_samples"],
            "seed": phase_09["seed"],
            "effective": source["training_configuration"],
        },
        "training_and_generalization": {
            "epoch_records": learning["epoch_records"],
            "epoch_deltas": learning["epoch_deltas"],
            "selected_checkpoint": learning["selected_checkpoint"],
            "interpretation": learning["generalization_interpretation"],
            "evidence_limit": "Three epochs and one seeded baseline run are limited evidence; no broad conclusion is claimed.",
            "figures": {
                "loss_curves": str(LOSS_CURVES_PATH.resolve()),
                "validation_metrics": str(VALIDATION_CURVES_PATH.resolve()),
            },
        },
        "validation_metrics": validation["metrics"],
        "final_test_metrics": test["metrics"],
        "final_test_policy": {
            "validation_verified_before_test": phase_11["validation_verified_before_test"],
            "test_used_for_selection": phase_11["test_used_for_selection"],
            "test_evaluation_count": phase_11["test_evaluation_count"],
            "checkpoint_changed_after_test": phase_11["checkpoint_changed_after_test"],
        },
        "confusion_matrix": {
            "counts": confusion["counts"],
            "total": confusion["total"],
            "correct": confusion["correct"],
            "incorrect": confusion["incorrect"],
            "class_level": confusion["class_level"],
            "figure": str(CONFUSION_FIGURE_PATH.resolve()),
            "metrics_match_phase_11": all(
                item["pass"] for item in confusion["phase_11_metric_comparisons"]
            ),
        },
        "error_analysis": {
            "total_errors": confusion["incorrect"],
            "direction_counts": error_analysis["error_direction_counts"],
            "representative_count": error_analysis["representative_count"],
            "interpretation": error_analysis["interpretation"],
            "causal_claim_made": False,
        },
        "custom_inference": {
            "status": inference["status"],
            "input_provenance": inference["input_provenance"],
            "input_count": inference["input_count"],
            "results": inference_rows,
            "verification_status": inference["verification"]["status"],
            "test_accessed": inference["test_accessed"],
        },
        "saved_package": {
            "status": save_reload["status"],
            "path": save_reload["package_path"],
            "files": package["package_files"],
            "state_dict_exact_equal": save_reload["state_dict_equivalence"]["all_tensors_exact_equal"],
            "state_tensor_count": save_reload["state_dict_equivalence"]["tensor_count"],
            "logits_max_abs_diff": save_reload["prediction_equivalence"]["logits_max_abs_diff"],
            "probability_max_abs_diff": save_reload["prediction_equivalence"]["probability_max_abs_diff"],
            "confidence_max_abs_diff": save_reload["prediction_equivalence"]["confidence_max_abs_diff"],
            "package_weight_sha256": package["package_weight_sha256"],
            "package_resaved_by_phase_15": False,
        },
        "limitations": [
            "The baseline was trained for only three epochs.",
            "Fine-tuning evidence comes from one DistilBERT baseline configuration and one full seeded run.",
            "No broad hyperparameter search or multi-seed robustness study was performed.",
            "The implemented task is English binary sentiment classification only.",
            "Evaluation evidence is tied to the Rotten Tomatoes dataset and its fixed splits.",
            "Mixed sentiment, negation and high-confidence errors remain observable challenges; descriptive flags do not establish causes.",
            "The project reports aggregate evaluation and artifact-based error analysis, not production monitoring.",
            "A local package was verified, but no production API, serving infrastructure, load/latency test, drift monitoring, security review or deployment was implemented.",
        ],
        "source_artifacts": _source_metadata(),
        "verification_checks": checks,
        "artifact_only_summary": True,
        "model_loaded": False,
        "dataset_loaded": False,
        "training_performed": False,
        "inference_performed": False,
        "validation_evaluated": False,
        "test_accessed": False,
        "test_evaluated": False,
        "test_evaluation_count": 1,
        "checkpoint_changed": False,
        "package_resaved": False,
        "production_ready_claimed": False,
        "sota_claimed": False,
        "final_conclusion": "Phase 0–14 evidence is internally consistent and PASS; Practice 3 implementation is complete without new model execution in Phase 15.",
    }


def _metric_table(validation: dict[str, Any], test: dict[str, Any]) -> str:
    rows = ["| Metric | Validation | Final Test |", "|---|---:|---:|"]
    for metric in METRICS:
        rows.append(f"| {metric.title()} | {validation[metric]:.10f} | {test[metric]:.10f} |")
    return "\n".join(rows)


def _render_markdown(summary: dict[str, Any]) -> str:
    model = summary["final_model"]
    confusion = summary["confusion_matrix"]
    errors = summary["error_analysis"]
    package = summary["saved_package"]
    lines = [
        "# Practice 3 — Final Summary",
        "",
        f"**Status:** {summary['status']} — PRACTICE 3 COMPLETE",
        "",
        "## Pipeline",
        "",
        "| Phase | Block | Processing | Verified output | Status |",
        "|---|---|---|---|---|",
    ]
    for block in summary["pipeline"]:
        lines.append(
            f"| {block['phase']} | {block['name']} | {block['processing']} | "
            f"{block['verified_output']} | {block['status']} |"
        )
    lines.extend([
        "",
        "## Final Model",
        "",
        f"- Architecture: {model['architecture']}",
        f"- Selected checkpoint: `{model['selected_checkpoint']}`",
        f"- Epoch / step: {model['epoch']:.0f} / {model['step']}",
        f"- Selection reason: {model['selection_reason']}",
        f"- Selected Validation loss: {model['selected_validation_loss']:.10f}",
        f"- Selected Validation F1: {model['selected_validation_f1']:.10f}",
        f"- Test used for selection: {model['test_used_for_selection']}",
        f"- Authoritative SHA-256: `{model['authoritative_weight_sha256']}`",
        "",
        "## Validation and Final Test Metrics",
        "",
        _metric_table(summary["validation_metrics"], summary["final_test_metrics"]),
        "",
        "The official Test split was evaluated exactly once after the Validation reload gate passed; it was not used for checkpoint selection.",
        "",
        "## Training and Generalization",
        "",
        summary["training_and_generalization"]["interpretation"],
        "",
        f"Evidence limit: {summary['training_and_generalization']['evidence_limit']}",
        "",
        f"- [Loss curves]({Path(summary['training_and_generalization']['figures']['loss_curves']).name})",
        f"- [Validation metrics]({Path(summary['training_and_generalization']['figures']['validation_metrics']).name})",
        "",
        "## Confusion Matrix and Error Analysis",
        "",
        f"- TN={confusion['counts']['TN']}, FP={confusion['counts']['FP']}, "
        f"FN={confusion['counts']['FN']}, TP={confusion['counts']['TP']}",
        f"- Total={confusion['total']}, correct={confusion['correct']}, incorrect={confusion['incorrect']}",
        f"- NEGATIVE→POSITIVE errors: {errors['direction_counts']['NEGATIVE_TO_POSITIVE']}",
        f"- POSITIVE→NEGATIVE errors: {errors['direction_counts']['POSITIVE_TO_NEGATIVE']}",
        "- [Confusion matrix figure](phase_12_confusion_matrix.png)",
        "",
        errors["interpretation"],
        "",
        "## Custom Inference Evidence",
        "",
        "| Text | Prediction | Confidence |",
        "|---|---|---:|",
    ])
    for row in summary["custom_inference"]["results"]:
        safe_text = row["text"].replace("|", "\\|")
        lines.append(f"| {safe_text} | {row['predicted_label']} | {row['confidence']:.6f} |")
    lines.extend([
        "",
        "## Save/Reload Evidence",
        "",
        f"- Local package: `{package['path']}`",
        f"- Exact-equal state tensors: {package['state_tensor_count']}",
        f"- Maximum logits difference: {package['logits_max_abs_diff']}",
        f"- Maximum probability difference: {package['probability_max_abs_diff']}",
        f"- Maximum confidence difference: {package['confidence_max_abs_diff']}",
        f"- Package SHA-256: `{package['package_weight_sha256']}`",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.extend([
        "",
        "## Final Conclusion",
        "",
        summary["final_conclusion"],
        "",
        "Phase 15 was artifact-only: no model/dataset loading, training, inference, Validation/Test evaluation, checkpoint change or package resave occurred.",
        "",
    ])
    return "\n".join(lines)


def build_final_summary() -> dict[str, Any]:
    """Validate Phase 0–14 artifacts and create JSON/Markdown summaries only."""
    source = _load_sources()
    phase_status, checks = _validate_sources(source)
    summary = _build_summary(source, phase_status, checks)
    if not all(summary["verification_checks"].values()):
        raise RuntimeError("Phase 15 final verification is not PASS")
    markdown = _render_markdown(summary)
    _atomic_json(SUMMARY_JSON_PATH, summary)
    _atomic_text(SUMMARY_MARKDOWN_PATH, markdown)
    if _render_markdown(_load_json(SUMMARY_JSON_PATH)) != SUMMARY_MARKDOWN_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("Markdown does not match canonical Phase 15 JSON")
    return {
        **summary,
        "guard_action": "built_artifact_only_phase_15_summary",
        "summary_json_path": str(SUMMARY_JSON_PATH.resolve()),
        "summary_markdown_path": str(SUMMARY_MARKDOWN_PATH.resolve()),
    }


def _valid_existing_summary() -> dict[str, Any] | None:
    if not SUMMARY_JSON_PATH.is_file() or not SUMMARY_MARKDOWN_PATH.is_file():
        return None
    summary = _load_json(SUMMARY_JSON_PATH)
    if not (
        summary.get("status") == "PASS"
        and summary.get("artifact_only_summary") is True
        and summary.get("model_loaded") is False
        and summary.get("dataset_loaded") is False
        and summary.get("training_performed") is False
        and summary.get("inference_performed") is False
        and summary.get("test_accessed") is False
        and summary.get("test_evaluated") is False
        and summary.get("test_evaluation_count") == 1
        and summary.get("checkpoint_changed") is False
        and summary.get("package_resaved") is False
        and all(summary.get("verification_checks", {}).values())
    ):
        return None
    for relative, metadata in summary.get("source_artifacts", {}).items():
        path = PROJECT_ROOT / relative
        if not (
            path.is_file()
            and path.stat().st_size == int(metadata.get("bytes", -1))
            and _sha256_file(path) == metadata.get("sha256")
        ):
            return None
    if _render_markdown(summary) != SUMMARY_MARKDOWN_PATH.read_text(encoding="utf-8"):
        return None
    return summary


def run_or_load_phase_15() -> dict[str, Any]:
    """Load a current summary, or rebuild it only from changed verified artifacts."""
    existing = _valid_existing_summary()
    if existing is not None:
        return {
            **existing,
            "guard_action": "loaded_verified_phase_15_artifact_summary",
            "summary_json_path": str(SUMMARY_JSON_PATH.resolve()),
            "summary_markdown_path": str(SUMMARY_MARKDOWN_PATH.resolve()),
        }
    return build_final_summary()


def load_phase_15_report() -> dict[str, Any]:
    """Load the valid canonical final summary without any model operation."""
    summary = _valid_existing_summary()
    if summary is None:
        raise RuntimeError("No current verified Phase 15 final summary is available")
    return {
        **summary,
        "guard_action": "loaded_verified_phase_15_artifact_summary",
        "summary_json_path": str(SUMMARY_JSON_PATH.resolve()),
        "summary_markdown_path": str(SUMMARY_MARKDOWN_PATH.resolve()),
    }
