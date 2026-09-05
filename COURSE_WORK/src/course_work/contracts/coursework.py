import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from course_work.utils.artifacts import (
    canonical_json_bytes,
    get_project_root,
    read_json,
    sha256_file,
    write_bytes_once_or_verify,
    write_json_once_or_verify,
)


EXPECTED_OPTION_IDS = {
    "feature_sets": {"FS0", "FS1", "FS2"},
    "time_features": {"TF0", "TF1"},
    "target_scaling": {"YS0", "YS1"},
    "lookbacks": {"L36", "L72", "L144"},
    "pooling": {"P0", "P1"},
    "activation": {"A0", "A1"},
    "batch_size": {"B32", "B64"},
    "learning_rate": {"LR1", "LR2", "LR3"},
    "weight_decay": {"WD0", "WD1", "WD2", "WD3"},
    "dropout": {"DR01", "DR02", "DR03"},
    "d_model": {"D32", "D64"},
    "heads": {"H2", "H4"},
    "layers": {"N1", "N2"},
    "ffn": {"F64", "F128", "F256"},
    "loss": {"L0", "L1"},
    "epoch_cap": {"E50", "E100"},
    "gradient_clip": {"GC0", "GC1"},
    "revin": {"RN0", "RN1"},
    "boundary": {"WB0", "WB1"},
}


def load_coursework_contract(path: Path | None = None) -> dict[str, Any]:
    contract_path = path or get_project_root() / "configs/base/coursework_contract.json"
    value = read_json(contract_path)
    if not isinstance(value, dict):
        raise TypeError("Coursework contract must be a JSON object")
    return value


def validate_coursework_contract(contract: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if contract.get("contract_version") != "COURSEWORK-CONTRACT-v1":
        errors.append("contract_version must be COURSEWORK-CONTRACT-v1")
    problem = contract.get("problem", {})
    expected_problem = {
        "task": "multivariate_time_series_regression",
        "dataset": "UCI Appliances Energy Prediction",
        "target": "Appliances",
        "target_unit": "Wh",
        "sampling_minutes": 10,
        "forecast_horizon_steps": 1,
        "forecast_horizon_minutes": 10,
    }
    for key, expected in expected_problem.items():
        if problem.get(key) != expected:
            errors.append(f"problem.{key} must equal {expected!r}")
    if contract.get("lookbacks") != {"options": [36, 72, 144], "primary": 144}:
        errors.append("lookbacks must register 36, 72, 144 with primary 144")
    split = contract.get("split", {})
    if split.get("type") != "chronological":
        errors.append("split.type must be chronological")
    fractions = [split.get("train"), split.get("validation"), split.get("test")]
    if not all(isinstance(value, (int, float)) for value in fractions):
        errors.append("split fractions must be numeric")
    elif abs(sum(fractions) - 1.0) > 1e-12:
        errors.append("split fractions must sum to 1.0")
    if fractions != [0.7, 0.15, 0.15]:
        errors.append("split fractions must be 0.70, 0.15, 0.15")
    if split.get("membership_basis") != "target_timestamp":
        errors.append("split membership must use target_timestamp")
    metrics = contract.get("metrics", {})
    if metrics.get("selection") != "validation_rmse":
        errors.append("selection metric must be validation_rmse")
    if metrics.get("final") != ["mae", "rmse", "r2"]:
        errors.append("final metrics must be mae, rmse, r2")
    if metrics.get("report_scale") != "original_wh":
        errors.append("final metric scale must be original_wh")
    if contract.get("models") != ["persistence", "lstm", "transformer_encoder"]:
        errors.append("models must be persistence, lstm, transformer_encoder")
    if contract.get("final_seeds") != [42, 123, 2026]:
        errors.append("final seeds must be 42, 123, 2026")
    registry = contract.get("option_registry", {})
    for group, expected_ids in EXPECTED_OPTION_IDS.items():
        options = registry.get(group)
        if not isinstance(options, list):
            errors.append(f"option_registry.{group} must be a list")
            continue
        actual_ids = {item.get("id") for item in options if isinstance(item, dict)}
        if actual_ids != expected_ids:
            errors.append(f"option_registry.{group} IDs are incomplete or invalid")
    research_questions = contract.get("research_questions", [])
    question_ids = {item.get("id") for item in research_questions if isinstance(item, dict)}
    if question_ids != {f"RQ{index}" for index in range(1, 8)}:
        errors.append("research_questions must contain RQ1 through RQ7")
    for field in (
        "validation_policy",
        "test_policy",
        "anti_leakage_rules",
        "fair_comparison_rules",
        "attention_rules",
        "protocol_violations",
        "definition_of_done",
        "experiment_registry_fields",
    ):
        value = contract.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{field} must be a non-empty list")
    baseline = contract.get("baseline_transformer", {})
    if baseline.get("lookback") != 144 or baseline.get("horizon") != 1:
        errors.append("baseline_transformer must use lookback 144 and horizon 1")
    d_model = baseline.get("d_model")
    heads = baseline.get("heads")
    if not isinstance(d_model, int) or not isinstance(heads, int) or heads == 0 or d_model % heads:
        errors.append("baseline_transformer d_model must be divisible by heads")
    return tuple(errors)


def coursework_contract_fingerprint(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


def materialize_phase_0(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    config_path = root / "configs/base/coursework_contract.json"
    artifact_root = root / "artifacts/contracts"
    artifact_path = artifact_root / "coursework_contract.json"
    checksum_path = artifact_root / "coursework_contract.sha256"
    signoff_path = artifact_root / "phase_0_signoff.json"
    contract = load_coursework_contract(config_path)
    errors = validate_coursework_contract(contract)
    if errors:
        raise ValueError("; ".join(errors))
    contract_bytes = canonical_json_bytes(contract)
    config_checksum = sha256_file(config_path)
    artifact_checksum = hashlib.sha256(contract_bytes).hexdigest()
    write_bytes_once_or_verify(artifact_path, contract_bytes)
    checksum_content = f"{artifact_checksum}  coursework_contract.json\n".encode("utf-8")
    write_bytes_once_or_verify(checksum_path, checksum_content)
    if sha256_file(artifact_path) != artifact_checksum:
        raise RuntimeError("Coursework artifact checksum verification failed")
    if signoff_path.exists():
        signoff = read_json(signoff_path)
        if signoff.get("status") != "PASS":
            raise RuntimeError("Existing Phase 0 sign-off is not PASS")
        if signoff.get("config_fingerprint") != artifact_checksum:
            raise RuntimeError("Existing Phase 0 sign-off fingerprint mismatch")
        return signoff
    signoff = {
        "artifact_version": "COURSEWORK-CONTRACT-v1",
        "phase_id": 0,
        "phase_version": "PHASE-0-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment_id": None,
        "dataset_revision": None,
        "input_paths": ["configs/base/coursework_contract.json"],
        "input_checksums": {"configs/base/coursework_contract.json": config_checksum},
        "output_paths": [
            "artifacts/contracts/coursework_contract.json",
            "artifacts/contracts/coursework_contract.sha256",
        ],
        "output_checksums": {
            "artifacts/contracts/coursework_contract.json": artifact_checksum,
            "artifacts/contracts/coursework_contract.sha256": hashlib.sha256(checksum_content).hexdigest(),
        },
        "config_fingerprint": artifact_checksum,
        "status": "PASS",
        "tests": [
            "contract_schema",
            "option_registry_completeness",
            "chronological_split_sum",
            "forecast_target_unambiguous",
            "data_independence",
        ],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return signoff
