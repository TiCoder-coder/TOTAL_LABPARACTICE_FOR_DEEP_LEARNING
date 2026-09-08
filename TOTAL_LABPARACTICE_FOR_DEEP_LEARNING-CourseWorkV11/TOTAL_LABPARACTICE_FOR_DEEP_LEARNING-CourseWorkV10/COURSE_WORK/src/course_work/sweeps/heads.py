from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from course_work.experiments.phase_execution import (
    plan_phase_resume,
    resolve_phase_conditions,
    validate_condition_request,
)
from course_work.models.transformer_regressor import TransformerRegressor, validate_transformer_config
from course_work.sweeps.dropout import inspect_dropout_scope
from course_work.utils.artifacts import canonical_json_bytes, get_project_root, read_json, sha256_bytes, sha256_file
from course_work.utils.reproducibility import set_seed


PHASE_ID = 34
PHASE_VERSION = "PHASE-34-v1"
SWEEP_ID = "S12_HEADS"
SWEEP_VERSION = "SWEEP_S12_HEADS-v1"
REFERENCE_CONDITION_ID = "H4"
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_33_ROOT = Path("artifacts/sweeps/S11_d_model")
PHASE_33_SIGNOFF_PATH = PHASE_33_ROOT / "phase_33_signoff.json"
PHASE_33_WINNER_PATH = PHASE_33_ROOT / "s11_d_model_winner.json"
PHASE_33_REFERENCE_PATH = PHASE_33_ROOT / "s11_reference_update.json"
PHASE_33_LOG_PATH = Path("docs/save_log_in_processing/phase_33_s11_d_model_log.json")
REGISTRY_PATH = Path("artifacts/experiments/experiment_registry.jsonl")
HISTORICAL_REFERENCE_MODE = "HISTORICAL_REFERENCE_WITH_INCOMPLETE_ARTIFACT_RETENTION"
HISTORICAL_REFERENCE_STATUS = "PASS_WITH_WARNING"
HISTORICAL_REFERENCE_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"
H4_MISSING_ARTIFACT_SUFFIXES = {
    "checkpoints/best_checkpoint.pt": "3ccf735488336340275a4dccf040fcd17b98a4fa4746d3e665cf14f97428d75d",
    "training_history.csv": "635e00dfcd032c3664c5e031b13f38d94caa3f73c4c6bf142eb0d6883d9eb43b",
    "training.log": "f709354699e8fb103b226fbae64ac2d2611fd80854ae1342246e86c94415d5d7",
    "predictions/best_validation_predictions.csv": "9bb212edb469f8fac02cf9179a401fec04d27487131a1e17a5e14179b94a5a4e",
}


@dataclass(frozen=True)
class HeadCondition:
    condition_id: str
    num_heads: int
    execution_mode: str


CONDITIONS = (
    HeadCondition("H2", 2, "TRAIN_NEW"),
    HeadCondition("H4", 4, "REUSE_REFERENCE"),
)


def _load_object(root: Path, relative_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative_path
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return None, "INVALID_JSON"
    if not isinstance(value, dict):
        return None, "INVALID_OBJECT"
    return value, None


def _status(record: dict[str, Any]) -> str | None:
    value = record.get("status", record.get("overall_status"))
    return str(value).upper() if value is not None else None


def _is_pass(record: dict[str, Any]) -> bool:
    return _status(record) in {"PASS", "PASS_WITH_WARNING"}


def _test_is_locked(value: Any) -> bool:
    return value is not None and str(value).upper() in {
        "FORBIDDEN",
        "LOCKED",
        "NOT_ACCESSED",
        "UNTOUCHED",
        "LOCKED_UNTIL_PHASE_47",
    }


def _number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _string(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _integer_id(value: Any, prefix: str) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith(prefix) and value[len(prefix):].isdigit():
        return int(value[len(prefix):])
    return None


def _add_issue(issues: list[dict[str, str]], path: Path | str, reason: str) -> None:
    issue = {"path": str(path), "reason": reason}
    if issue not in issues:
        issues.append(issue)


def _field_matches(
    issues: list[dict[str, str]],
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
) -> None:
    left_value = left.get(field)
    right_value = right.get(field)
    if left_value is None or right_value is None:
        _add_issue(issues, field, "FROZEN_FIELD_MISSING")
    elif left_value != right_value:
        _add_issue(issues, field, "FROZEN_FIELD_MISMATCH")


def _validate_declared_outputs(root: Path, signoff: dict[str, Any], issues: list[dict[str, str]]) -> None:
    output_paths = signoff.get("output_paths")
    output_checksums = signoff.get("output_checksums")
    if not isinstance(output_paths, list) or not isinstance(output_checksums, dict):
        _add_issue(issues, PHASE_33_SIGNOFF_PATH, "INVALID_OUTPUT_DECLARATION")
        return
    for relative_path in output_paths:
        if not isinstance(relative_path, str):
            _add_issue(issues, str(relative_path), "INVALID_PATH")
            continue
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            _add_issue(issues, relative_path, "PATH_OUTSIDE_PROJECT")
            continue
        if not path.is_file():
            _add_issue(issues, relative_path, "MISSING")
            continue
        expected = output_checksums.get(relative_path)
        if not isinstance(expected, str):
            _add_issue(issues, relative_path, "CHECKSUM_MISSING")
        elif sha256_file(path) != expected:
            _add_issue(issues, relative_path, "CHECKSUM_MISMATCH")


def _state_fingerprint(model: TransformerRegressor) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def inspect_head_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    config = validate_transformer_config(model_config)
    set_seed(SEED)
    model = TransformerRegressor(config)
    model.eval()
    sample = torch.randn(2, 36, config.input_size, dtype=torch.float32)
    prediction = model(sample)
    inspected_prediction, attention_maps = model.forward_with_attention(sample)
    loss = inspected_prediction.square().mean()
    loss.backward()
    state_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    parameter_schema = {
        name: {
            "shape": list(parameter.shape),
            "requires_grad": parameter.requires_grad,
            "dtype": str(parameter.dtype),
        }
        for name, parameter in model.named_parameters()
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_attention_shape = [2, config.num_heads, 36, 36]
    attention_shapes = [list(value.shape) for value in attention_maps]
    attention_finite = all(torch.isfinite(value).all().item() for value in attention_maps)
    attention_nonnegative = all((value >= 0).all().item() for value in attention_maps)
    attention_row_sums = all(
        torch.allclose(value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)), atol=1e-5, rtol=1e-5)
        for value in attention_maps
    )
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    prediction_paths_equal = torch.allclose(
        prediction,
        inspected_prediction,
        atol=1e-6,
        rtol=1e-6,
    )
    status_checks = (
        config.d_model % config.num_heads == 0,
        prediction.shape == (2, 1),
        inspected_prediction.shape == (2, 1),
        len(attention_maps) == config.num_layers,
        all(shape == expected_attention_shape for shape in attention_shapes),
        attention_finite,
        attention_nonnegative,
        attention_row_sums,
        gradients_finite,
        prediction_paths_equal,
    )
    geometry = {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "head_dim": config.d_model // config.num_heads,
        "num_layers": config.num_layers,
        "ffn_dim": config.ffn_dim,
        "output_shape": list(prediction.shape),
        "attention_shapes": attention_shapes,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "state_shapes": state_shapes,
        "parameter_schema": parameter_schema,
    }
    return {
        "geometry_version": "HEAD-GEOMETRY-v1",
        **geometry,
        "geometry_fingerprint": sha256_bytes(canonical_json_bytes(geometry)),
        "initial_state_fingerprint": _state_fingerprint(model),
        "divisible_by_heads": config.d_model % config.num_heads == 0,
        "attention_finite": attention_finite,
        "attention_nonnegative": attention_nonnegative,
        "attention_row_sums": attention_row_sums,
        "gradients_finite": gradients_finite,
        "prediction_paths_equal": prediction_paths_equal,
        "status": "PASS" if all(status_checks) else "FAIL",
    }


def compare_head_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for condition in CONDITIONS:
        candidate_config = dict(model_config)
        candidate_config["num_heads"] = condition.num_heads
        candidates.append(
            {
                "condition_id": condition.condition_id,
                **inspect_head_geometry(candidate_config),
            }
        )
    by_id = {item["condition_id"]: item for item in candidates}
    h2 = by_id["H2"]
    h4 = by_id["H4"]
    parameter_schema_equal = h2["parameter_schema"] == h4["parameter_schema"]
    state_shapes_equal = h2["state_shapes"] == h4["state_shapes"]
    parameter_count_equal = h2["parameter_count"] == h4["parameter_count"]
    trainable_parameter_count_equal = (
        h2["trainable_parameter_count"] == h4["trainable_parameter_count"]
    )
    initial_state_equal = h2["initial_state_fingerprint"] == h4["initial_state_fingerprint"]
    config_h2 = validate_transformer_config({**model_config, "num_heads": 2}).to_dict()
    config_h4 = validate_transformer_config({**model_config, "num_heads": 4}).to_dict()
    config_delta = sorted(key for key in config_h2 if config_h2[key] != config_h4[key])
    status_checks = (
        all(item["status"] == "PASS" for item in candidates),
        parameter_schema_equal,
        state_shapes_equal,
        parameter_count_equal,
        trainable_parameter_count_equal,
        initial_state_equal,
        config_delta == ["num_heads"],
    )
    return {
        "comparison_version": "HEAD-COMPARISON-v1",
        "candidates": candidates,
        "parameter_schema_equal": parameter_schema_equal,
        "state_shapes_equal": state_shapes_equal,
        "parameter_count_equal": parameter_count_equal,
        "trainable_parameter_count_equal": trainable_parameter_count_equal,
        "initial_state_equal": initial_state_equal,
        "config_delta": config_delta,
        "external_output_contract_equal": h2["output_shape"] == h4["output_shape"] == [2, 1],
        "status": "PASS" if all(status_checks) else "FAIL",
    }


def _load_registry_record(root: Path, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    path = root / REGISTRY_PATH
    if not path.is_file():
        return None, "MISSING"
    matches = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict) and value.get("run_id") == run_id:
                    matches.append(value)
    except (OSError, json.JSONDecodeError):
        return None, "INVALID_JSONL"
    if len(matches) != 1:
        return None, "RUN_ID_NOT_UNIQUE" if matches else "RUN_ID_MISSING"
    return matches[0], None


def _validate_reference_run(
    root: Path,
    run_id: str,
    config_fingerprint: str | None,
    population_fingerprint: str | None,
    signoff: dict[str, Any],
    winner: dict[str, Any],
    reference: dict[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    record, error = _load_registry_record(root, run_id)
    if error is not None:
        _add_issue(issues, REGISTRY_PATH, error)
        return None
    if record.get("status") != "COMPLETED":
        _add_issue(issues, run_id, "RUN_NOT_COMPLETED")
    if record.get("test_access_authorized") is not False:
        _add_issue(issues, run_id, "TEST_FIREWALL_NOT_CONFIRMED")
    if record.get("config_fingerprint") != config_fingerprint:
        _add_issue(issues, run_id, "CONFIG_FINGERPRINT_MISMATCH")
    expected_missing_artifacts = {
        f"artifacts/runs/{run_id}/{suffix}": checksum
        for suffix, checksum in H4_MISSING_ARTIFACT_SUFFIXES.items()
    }
    required_types = {"CONFIG", "STATUS", "METRICS"}
    present_types = {
        item.get("artifact_type")
        for item in record.get("artifacts", [])
        if item.get("required") is True
    }
    if not required_types.issubset(present_types):
        _add_issue(issues, run_id, "REQUIRED_RUN_ARTIFACT_MISSING")
    for artifact in record.get("artifacts", []):
        if artifact.get("required") is not True:
            continue
        relative_path = artifact.get("artifact_path")
        expected = artifact.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            _add_issue(issues, run_id, "INVALID_RUN_ARTIFACT_DECLARATION")
            continue
        path = root / relative_path
        if not path.is_file():
            if relative_path not in expected_missing_artifacts:
                _add_issue(issues, relative_path, "MISSING")
        elif sha256_file(path) != expected:
            _add_issue(issues, relative_path, "CHECKSUM_MISMATCH")
    metrics = {}
    for metric in record.get("metrics", []):
        if metric.get("split_id") != "VALIDATION" or metric.get("status") != "PASS":
            continue
        if metric.get("population_fingerprint") != population_fingerprint:
            _add_issue(issues, run_id, "POPULATION_MISMATCH")
        name = metric.get("metric_name")
        value = metric.get("metric_value")
        if name in {"rmse_wh", "mae_wh", "r2"} and isinstance(value, (int, float)):
            metrics[name] = float(value)
    if set(metrics) != {"rmse_wh", "mae_wh", "r2"}:
        _add_issue(issues, run_id, "VALIDATION_METRICS_INCOMPLETE")
    run_root = root / "artifacts/runs" / run_id
    missing_artifacts = []
    for relative_path, expected_sha256 in expected_missing_artifacts.items():
        path = root / relative_path
        if path.is_file():
            _add_issue(issues, relative_path, "EXPECTED_MISSING_ARTIFACT_PRESENT")
        else:
            missing_artifacts.append(
                {
                    "artifact_path": relative_path,
                    "expected_sha256": expected_sha256,
                    "retention_status": "MISSING_UNRECOVERABLE",
                }
            )
    unexpected_test_artifacts = [
        path.relative_to(root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    if unexpected_test_artifacts:
        _add_issue(issues, run_id, "TEST_ARTIFACT_DETECTED")
    status_payload, status_error = _load_object(root, Path("artifacts/runs") / run_id / "status.json")
    metrics_payload, metrics_error = _load_object(
        root,
        Path("artifacts/runs") / run_id / "metrics/best_validation_metrics.json",
    )
    config_payload, config_error = _load_object(root, Path("artifacts/runs") / run_id / "config.json")
    for path, error_value in (
        (Path("artifacts/runs") / run_id / "status.json", status_error),
        (Path("artifacts/runs") / run_id / "metrics/best_validation_metrics.json", metrics_error),
        (Path("artifacts/runs") / run_id / "config.json", config_error),
    ):
        if error_value is not None:
            _add_issue(issues, path, error_value)
    status_payload = status_payload or {}
    metrics_payload = metrics_payload or {}
    config_payload = config_payload or {}
    metric_result = metrics_payload.get("metric_result") if metrics_payload is not None else None
    if not isinstance(metric_result, dict):
        _add_issue(issues, run_id, "BEST_VALIDATION_METRICS_INVALID")
        metric_result = {}
    config = config_payload.get("config") if config_payload is not None else None
    if not isinstance(config, dict):
        _add_issue(issues, run_id, "CONFIG_MISSING")
        config = {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    lineage = config.get("lineage") if isinstance(config.get("lineage"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    registry_epochs = {
        metric.get("epoch_or_checkpoint")
        for metric in record.get("metrics", [])
        if metric.get("split_id") == "VALIDATION"
    }
    registry_metric_versions = {
        metric.get("metric_version")
        for metric in record.get("metrics", [])
        if metric.get("split_id") == "VALIDATION"
    }
    registry_populations = {
        metric.get("population_fingerprint")
        for metric in record.get("metrics", [])
        if metric.get("split_id") == "VALIDATION"
    }
    best_epoch = status_payload.get("best_epoch") if status_payload is not None else None
    expected_epoch_token = f"epoch_{best_epoch}"
    checks = (
        (record.get("run_id") == status_payload.get("run_id") == metric_result.get("run_id") == run_id, "RUN_ID_EVIDENCE_MISMATCH"),
        (config_payload.get("run_id") == run_id and config_payload.get("config_fingerprint") == config_fingerprint, "CONFIG_IDENTITY_MISMATCH"),
        (record.get("config") == config, "REGISTRY_CONFIG_MISMATCH"),
        (model.get("d_model") == winner.get("winner_d_model") == reference.get("selected_d_model") == signoff.get("winner_d_model"), "DMODEL_EVIDENCE_MISMATCH"),
        (model.get("num_heads") == winner.get("num_heads") == reference.get("current_num_heads") == 4, "HEAD_COUNT_EVIDENCE_MISMATCH"),
        (model.get("num_layers") == winner.get("num_layers") == reference.get("num_layers") == 2, "LAYER_COUNT_EVIDENCE_MISMATCH"),
        (model.get("ffn_dim") == winner.get("ffn_dim") == reference.get("ffn_dim") == 128, "FFN_DIM_EVIDENCE_MISMATCH"),
        (lineage.get("population_fingerprint") == population_fingerprint == metric_result.get("population_fingerprint") == signoff.get("population_fingerprint") and registry_populations == {population_fingerprint}, "POPULATION_EVIDENCE_MISMATCH"),
        (lineage.get("metric_version") == winner.get("metric_version") == metric_result.get("metric_version") == signoff.get("metric_version") and registry_metric_versions == {winner.get("metric_version")}, "METRIC_VERSION_EVIDENCE_MISMATCH"),
        (record.get("best_epoch") == best_epoch and registry_epochs == {expected_epoch_token}, "BEST_EPOCH_EVIDENCE_MISMATCH"),
        (metrics.get("rmse_wh") == metric_result.get("rmse_wh") == winner.get("winner_rmse_wh") == reference.get("winner_rmse_wh") == signoff.get("winner_rmse_wh") == status_payload.get("best_validation_rmse_wh"), "RMSE_EVIDENCE_MISMATCH"),
        (metrics.get("mae_wh") == metric_result.get("mae_wh") == winner.get("winner_mae_wh"), "MAE_EVIDENCE_MISMATCH"),
        (metrics.get("r2") == metric_result.get("r2") == winner.get("winner_r2"), "R2_EVIDENCE_MISMATCH"),
        (data.get("target_access_mode") == "VALIDATION" and metric_result.get("split_id") == "VALIDATION", "VALIDATION_ONLY_EVIDENCE_MISMATCH"),
        (record.get("test_access_authorized") is False and _test_is_locked(winner.get("test_status")) and _test_is_locked(reference.get("test_status")) and _test_is_locked(signoff.get("test_status")), "TEST_FIREWALL_NOT_CONFIRMED"),
    )
    for passed, reason in checks:
        if not passed:
            _add_issue(issues, run_id, reason)
    return {
        "record": record,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "evidence_mode": HISTORICAL_REFERENCE_MODE,
        "evidence_status": HISTORICAL_REFERENCE_STATUS,
        "warnings": [HISTORICAL_REFERENCE_WARNING],
        "missing_artifacts": missing_artifacts,
    }


def inspect_phase_33_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff, signoff_error = _load_object(root, PHASE_33_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_33_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_33_REFERENCE_PATH)
    processing_log, processing_log_error = _load_object(root, PHASE_33_LOG_PATH)
    issues: list[dict[str, str]] = []
    for path, error in (
        (PHASE_33_SIGNOFF_PATH, signoff_error),
        (PHASE_33_WINNER_PATH, winner_error),
        (PHASE_33_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)
    if signoff is not None:
        if not _is_pass(signoff):
            _add_issue(issues, PHASE_33_SIGNOFF_PATH, "SIGNOFF_NOT_PASS")
        if signoff.get("approved_for_phase34") is not True:
            _add_issue(issues, PHASE_33_SIGNOFF_PATH, "PHASE_34_NOT_APPROVED")
        if not _test_is_locked(signoff.get("test_status")):
            _add_issue(issues, PHASE_33_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
        _validate_declared_outputs(root, signoff, issues)
    if winner is not None:
        if not _is_pass(winner):
            _add_issue(issues, PHASE_33_WINNER_PATH, "WINNER_NOT_PASS")
        if not _test_is_locked(winner.get("test_status")):
            _add_issue(issues, PHASE_33_WINNER_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
    if reference is not None:
        if reference.get("approved_for_phase34") is not True:
            _add_issue(issues, PHASE_33_REFERENCE_PATH, "PHASE_34_NOT_APPROVED")
        if not _test_is_locked(reference.get("test_status")):
            _add_issue(issues, PHASE_33_REFERENCE_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    selected_d_model = None
    winner_run_id = None
    source_config_path = None
    source_config_fingerprint = None
    geometry_comparison = None
    dropout_scope = None
    reference_evidence = None
    if winner is not None and reference is not None:
        winner_run_id = _string(winner, "winner_run_id")
        reference_run_id = _string(reference, "winner_run_id", "current_reference_run_id")
        if winner_run_id is None or reference_run_id is None:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISSING")
        elif winner_run_id != reference_run_id:
            _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")
        winner_d_model = _number(winner, "winner_d_model", "d_model")
        reference_d_model = _number(reference, "selected_d_model", "d_model")
        if winner_d_model is None or reference_d_model is None:
            _add_issue(issues, "d_model", "SELECTED_DMODEL_MISSING")
        elif not math.isclose(winner_d_model, reference_d_model, rel_tol=0.0, abs_tol=0.0):
            _add_issue(issues, "d_model", "SELECTED_DMODEL_MISMATCH")
        else:
            selected_d_model = int(winner_d_model)
        for field in (
            "feature_variant_id",
            "target_scaling_id",
            "lookback_id",
            "pooling_id",
            "activation_id",
            "batch_id",
            "learning_rate",
            "weight_decay",
            "dropout",
            "d_model",
            "num_heads",
            "num_layers",
            "ffn_dim",
            "population_fingerprint",
        ):
            _field_matches(issues, winner, reference, field)
        if reference.get("num_heads") != 4 or reference.get("current_num_heads") != 4:
            _add_issue(issues, PHASE_33_REFERENCE_PATH, "REFERENCE_HEAD_COUNT_MISMATCH")
        if selected_d_model is not None:
            if selected_d_model % 2 != 0 or selected_d_model % 4 != 0:
                _add_issue(issues, "d_model", "HEAD_DIVISIBILITY_INVALID")
            expected_head_dim = selected_d_model // 4
            if reference.get("current_head_dim") != expected_head_dim:
                _add_issue(issues, PHASE_33_REFERENCE_PATH, "REFERENCE_HEAD_DIM_MISMATCH")
        if winner_run_id is not None:
            source_config_path = Path("artifacts/runs") / winner_run_id / "config.json"
            source_payload, source_error = _load_object(root, source_config_path)
            if source_error is not None:
                _add_issue(issues, source_config_path, source_error)
            elif source_payload is not None:
                source_config_fingerprint = _string(source_payload, "config_fingerprint")
                if source_payload.get("run_id") != winner_run_id:
                    _add_issue(issues, source_config_path, "RUN_ID_MISMATCH")
                if source_config_fingerprint != _string(reference, "winner_config_fingerprint"):
                    _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISMATCH")
                config = source_payload.get("config")
                if not isinstance(config, dict):
                    _add_issue(issues, source_config_path, "CONFIG_MISSING")
                else:
                    data = config.get("data")
                    model = config.get("model")
                    training = config.get("training")
                    reproducibility = config.get("reproducibility")
                    lineage = config.get("lineage")
                    if not all(
                        isinstance(value, dict)
                        for value in (data, model, training, reproducibility, lineage)
                    ):
                        _add_issue(issues, source_config_path, "CONFIG_STRUCTURE_INVALID")
                    else:
                        expected_lookback = _integer_id(winner.get("lookback_id"), "L")
                        expected_batch = _integer_id(winner.get("batch_id"), "B")
                        checks = (
                            (data.get("feature_variant_id"), winner.get("feature_variant_id"), "FEATURE_VARIANT_DRIFT"),
                            (data.get("target_scaling_option"), winner.get("target_scaling_id"), "TARGET_SCALING_DRIFT"),
                            (data.get("lookback_steps"), expected_lookback, "LOOKBACK_DRIFT"),
                            (model.get("pooling"), winner.get("pooling_id"), "POOLING_DRIFT"),
                            (model.get("activation"), winner.get("activation_id"), "ACTIVATION_DRIFT"),
                            (model.get("d_model"), selected_d_model, "DMODEL_DRIFT"),
                            (model.get("num_heads"), 4, "HEAD_COUNT_DRIFT"),
                            (model.get("num_layers"), 2, "LAYER_COUNT_DRIFT"),
                            (model.get("ffn_dim"), 128, "FFN_DIM_DRIFT"),
                            (training.get("batch_size"), expected_batch, "BATCH_DRIFT"),
                            (training.get("optimizer_name"), "AdamW", "OPTIMIZER_IDENTITY_DRIFT"),
                            (training.get("loss_name"), "MSE", "LOSS_DRIFT"),
                            (training.get("max_epochs"), 50, "EPOCH_BUDGET_DRIFT"),
                            (training.get("early_stopping_patience"), 10, "PATIENCE_DRIFT"),
                            (training.get("early_stopping_min_delta", 0), 0, "MIN_DELTA_DRIFT"),
                            (training.get("gradient_clipping_enabled"), True, "GRADIENT_CLIP_DRIFT"),
                            (training.get("gradient_clip_max_norm"), 1.0, "GRADIENT_CLIP_DRIFT"),
                            (training.get("scheduler_name"), None, "SCHEDULER_DRIFT"),
                            (reproducibility.get("seed"), SEED, "SEED_DRIFT"),
                            (lineage.get("population_fingerprint"), winner.get("population_fingerprint"), "POPULATION_MISMATCH"),
                        )
                        for actual, expected, reason in checks:
                            if actual != expected:
                                _add_issue(issues, source_config_path, reason)
                        for actual, expected, reason in (
                            (model.get("dropout"), winner.get("dropout"), "DROPOUT_DRIFT"),
                            (training.get("learning_rate"), winner.get("learning_rate"), "LEARNING_RATE_DRIFT"),
                            (training.get("weight_decay"), winner.get("weight_decay"), "WEIGHT_DECAY_DRIFT"),
                        ):
                            if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
                                _add_issue(issues, source_config_path, reason)
                            elif not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                                _add_issue(issues, source_config_path, reason)
                        try:
                            geometry_comparison = compare_head_geometry(model)
                            dropout_scope = inspect_dropout_scope(model)
                        except (TypeError, ValueError, RuntimeError):
                            _add_issue(issues, source_config_path, "HEAD_PREFLIGHT_INVALID")
                        else:
                            if geometry_comparison["status"] != "PASS":
                                _add_issue(issues, source_config_path, "HEAD_GEOMETRY_INVALID")
                            if dropout_scope["status"] != "PASS":
                                _add_issue(issues, source_config_path, "DROPOUT_SCOPE_INVALID")
                reference_evidence = _validate_reference_run(
                    root,
                    winner_run_id,
                    source_config_fingerprint,
                    winner.get("population_fingerprint"),
                    signoff or {},
                    winner,
                    reference,
                    issues,
                )

    frozen_configuration = None
    if winner is not None and reference is not None and not issues:
        frozen_configuration = {
            "feature_variant_id": winner["feature_variant_id"],
            "target_scaling_id": winner["target_scaling_id"],
            "lookback_id": winner["lookback_id"],
            "pooling_id": winner["pooling_id"],
            "activation_id": winner["activation_id"],
            "batch_id": winner["batch_id"],
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "dropout": winner["dropout"],
            "d_model": selected_d_model,
            "population_fingerprint": winner["population_fingerprint"],
            "optimizer": "AdamW",
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 128,
            "max_epochs": 50,
            "patience": 10,
            "min_delta": 0,
            "gradient_clip_norm": 1.0,
            "seed": SEED,
            "scheduler": "NONE",
            "warmup": "NONE",
            "gradient_accumulation": 1,
            "loss": "MSE",
            "revin_enabled": False,
            "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
            "test_access": TEST_ACCESS,
            "source_config_path": str(source_config_path),
            "source_config_fingerprint": source_config_fingerprint,
        }

    return {
        "valid": not issues,
        "issues": issues,
        "source_paths": [
            str(PHASE_33_SIGNOFF_PATH),
            str(PHASE_33_WINNER_PATH),
            str(PHASE_33_REFERENCE_PATH),
        ],
        "selected_d_model": selected_d_model,
        "winner_run_id": winner_run_id,
        "frozen_configuration": frozen_configuration,
        "geometry_comparison": geometry_comparison,
        "dropout_scope": dropout_scope,
        "reference_evidence": reference_evidence,
        "warnings": reference_evidence.get("warnings", []) if reference_evidence is not None else [],
        "status": HISTORICAL_REFERENCE_STATUS if not issues else "BLOCKED",
        "processing_log_observation": {
            "path": str(PHASE_33_LOG_PATH),
            "available": processing_log_error is None,
            "status": processing_log.get("status") if processing_log is not None else processing_log_error,
            "authoritative": False,
        },
    }


def build_phase_34_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    decision = plan_phase_resume(PHASE_ID, root)
    handoff = inspect_phase_33_handoff(root)
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selected_d_model": handoff["selected_d_model"],
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "decision": decision,
        "ready": handoff["valid"] and decision["readiness"]["ready"],
    }


def prepare_phase_34_condition(
    condition_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {condition.condition_id: condition for condition in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 34")
    handoff = inspect_phase_33_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 34 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    head_dim = handoff["selected_d_model"] // condition.num_heads
    if condition.execution_mode == "REUSE_REFERENCE":
        evidence = resolve_phase_conditions(PHASE_ID, root)
        verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
        if condition_id not in verified:
            raise RuntimeError(f"Phase 34 reference evidence is invalid: {evidence}")
        return {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "condition_id": condition.condition_id,
            "execution_mode": condition.execution_mode,
            "num_heads": condition.num_heads,
            "head_dim": head_dim,
            "reference_run_id": handoff["winner_run_id"],
            "reference_evidence": verified[condition_id],
            "frozen_configuration": handoff["frozen_configuration"],
            "geometry_comparison": handoff["geometry_comparison"],
            "test_access": TEST_ACCESS,
        }
    gate = validate_condition_request(PHASE_ID, condition_id, root)
    if not gate["allowed"]:
        raise RuntimeError(f"Condition execution blocked: {gate}")
    return {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "condition_id": condition.condition_id,
        "execution_mode": condition.execution_mode,
        "num_heads": condition.num_heads,
        "head_dim": head_dim,
        "reference_run_id": handoff["winner_run_id"],
        "frozen_configuration": handoff["frozen_configuration"],
        "geometry_comparison": handoff["geometry_comparison"],
        "test_access": TEST_ACCESS,
    }
