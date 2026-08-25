from __future__ import annotations

import hashlib
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
from course_work.utils.artifacts import canonical_json_bytes, get_project_root, read_json, sha256_bytes, sha256_file
from course_work.utils.reproducibility import set_seed


PHASE_ID = 35
PHASE_VERSION = "PHASE-35-v1"
SWEEP_ID = "S13_LAYERS"
SWEEP_VERSION = "SWEEP_S13_LAYERS-v1"
REFERENCE_CONDITION_ID = "N2"
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_34_ROOT = Path("artifacts/sweeps/S12_heads")
PHASE_34_SIGNOFF_PATH = PHASE_34_ROOT / "phase_34_signoff.json"
PHASE_34_WINNER_PATH = PHASE_34_ROOT / "s12_head_winner.json"
PHASE_34_REFERENCE_PATH = PHASE_34_ROOT / "s12_reference_update.json"
INHERITED_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


@dataclass(frozen=True)
class LayerCondition:
    condition_id: str
    num_layers: int
    execution_mode: str


CONDITIONS = (
    LayerCondition("N1", 1, "TRAIN_NEW"),
    LayerCondition("N2", 2, "REUSE_REFERENCE"),
)


def _load_object(root: Path, relative_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative_path
    if not path.is_file():
        return None, "MISSING"
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError):
        return None, "INVALID_JSON"
    return (value, None) if isinstance(value, dict) else (None, "INVALID_OBJECT")


def _add_issue(issues: list[dict[str, str]], path: Path | str, reason: str) -> None:
    issue = {"path": str(path), "reason": reason}
    if issue not in issues:
        issues.append(issue)


def _test_locked(value: Any) -> bool:
    return str(value).upper() in {"FORBIDDEN", "LOCKED", "NOT_ACCESSED", "UNTOUCHED"}


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def inspect_layer_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    base = validate_transformer_config(model_config).to_dict()
    candidates: list[dict[str, Any]] = []
    models: dict[str, TransformerRegressor] = {}
    for condition in CONDITIONS:
        config = {**base, "num_layers": condition.num_layers}
        set_seed(SEED)
        model = TransformerRegressor(config)
        model.eval()
        sample = torch.randn(2, 36, config["input_size"], dtype=torch.float32)
        prediction, attention_maps = model.forward_with_attention(sample)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
        model_parameter_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        optimizer_parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_duplicates = len(optimizer_parameter_ids) - len(set(optimizer_parameter_ids))
        optimizer_missing = len(model_parameter_ids - set(optimizer_parameter_ids))
        optimizer_coverage = (
            set(optimizer_parameter_ids) == model_parameter_ids
            and optimizer_duplicates == 0
        )
        prediction_standard = model(sample)
        attention_finite = all(bool(torch.isfinite(value).all().item()) for value in attention_maps)
        attention_nonnegative = all(bool((value >= 0).all().item()) for value in attention_maps)
        row_sum_error = max(
            (
                float((value.sum(dim=-1) - torch.ones_like(value.sum(dim=-1))).abs().max().item())
                for value in attention_maps
            ),
            default=0.0,
        )
        prediction_paths_equal = bool(
            torch.allclose(prediction_standard, prediction, atol=1e-6, rtol=1e-6)
        )
        parameter_schema = {
            name: {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "numel": parameter.numel(),
            }
            for name, parameter in model.named_parameters()
        }
        state_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
        state_digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            tensor = value.detach().cpu().contiguous()
            state_digest.update(name.encode("utf-8"))
            state_digest.update(str(tensor.dtype).encode("utf-8"))
            state_digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            state_digest.update(tensor.numpy().tobytes())
        models[condition.condition_id] = model
        candidates.append(
            {
                "condition_id": condition.condition_id,
                "num_layers": condition.num_layers,
                "d_model": config["d_model"],
                "num_heads": config["num_heads"],
                "head_dim": config["d_model"] // config["num_heads"],
                "ffn_dim": config["ffn_dim"],
                "prediction_shape": list(prediction.shape),
                "attention_shapes": [list(value.shape) for value in attention_maps],
                "attention_finite": attention_finite,
                "attention_nonnegative": attention_nonnegative,
                "attention_row_sum_error_max": row_sum_error,
                "prediction_paths_equal": prediction_paths_equal,
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameter_count": sum(p.numel() for p in model.parameters()),
                "parameter_schema": parameter_schema,
                "parameter_schema_fingerprint": sha256_bytes(canonical_json_bytes(parameter_schema)),
                "state_shapes": state_shapes,
                "state_shape_fingerprint": sha256_bytes(canonical_json_bytes(state_shapes)),
                "initial_state_fingerprint": state_digest.hexdigest(),
                "optimizer_parameter_reference_count": len(optimizer_parameter_ids),
                "unique_optimizer_parameter_count": len(set(optimizer_parameter_ids)),
                "optimizer_missing_parameters": optimizer_missing,
                "optimizer_duplicate_parameters": optimizer_duplicates,
                "layer0_covered": all(
                    id(parameter) in set(optimizer_parameter_ids)
                    for parameter in model.encoder.layers[0].parameters()
                    if parameter.requires_grad
                ),
                "layer1_covered_if_applicable": (
                    all(
                        id(parameter) in set(optimizer_parameter_ids)
                        for parameter in model.encoder.layers[1].parameters()
                        if parameter.requires_grad
                    )
                    if condition.num_layers > 1
                    else None
                ),
                "optimizer_coverage": optimizer_coverage,
                "status": "PASS" if all(
                    (
                        len(attention_maps) == condition.num_layers,
                        optimizer_coverage,
                        attention_finite,
                        attention_nonnegative,
                        row_sum_error <= 1e-5,
                        prediction_paths_equal,
                    )
                ) else "FAIL",
            }
        )
    n1, n2 = models["N1"], models["N2"]
    n1_keys = set(n1.state_dict())
    n2_keys = set(n2.state_dict())
    extra_keys = sorted(n2_keys - n1_keys)
    shared_shapes_equal = all(n1.state_dict()[key].shape == n2.state_dict()[key].shape for key in n1_keys)
    layers_independent = all(
        left is not right and left.data_ptr() != right.data_ptr()
        for left, right in zip(n2.encoder.layers[0].parameters(), n2.encoder.layers[1].parameters())
    )
    by_id = {item["condition_id"]: item for item in candidates}
    parameter_delta = by_id["N2"]["trainable_parameter_count"] - by_id["N1"]["trainable_parameter_count"]
    n2_only_key_numel = sum(n2.state_dict()[key].numel() for key in extra_keys)
    state_key_rows = []
    for key in sorted(n1_keys | n2_keys):
        present_n1 = key in n1_keys
        present_n2 = key in n2_keys
        shape_n1 = list(n1.state_dict()[key].shape) if present_n1 else None
        shape_n2 = list(n2.state_dict()[key].shape) if present_n2 else None
        shared_key = present_n1 and present_n2
        n2_only_expected = present_n2 and not present_n1 and key.startswith("encoder.layers.1.")
        unexpected = (shared_key and shape_n1 != shape_n2) or (present_n2 and not present_n1 and not n2_only_expected) or (present_n1 and not present_n2)
        state_key_rows.append(
            {
                "key": key,
                "present_n1": present_n1,
                "present_n2": present_n2,
                "shape_n1": shape_n1,
                "shape_n2": shape_n2,
                "shared_key": shared_key,
                "n2_only_expected": n2_only_expected,
                "unexpected_difference": unexpected,
                "status": "FAIL" if unexpected else "PASS",
            }
        )
    shared_schema_rows = []
    for key in sorted(set(dict(n1.named_parameters())) & set(dict(n2.named_parameters()))):
        left = dict(n1.named_parameters())[key]
        right = dict(n2.named_parameters())[key]
        shape_equal = tuple(left.shape) == tuple(right.shape)
        dtype_equal = left.dtype == right.dtype
        trainable_equal = left.requires_grad == right.requires_grad
        shared_schema_rows.append(
            {
                "semantic_parameter": key,
                "n1_key": key,
                "n2_key": key,
                "shape_n1": list(left.shape),
                "shape_n2": list(right.shape),
                "dtype_equal": dtype_equal,
                "trainable_equal": trainable_equal,
                "shape_equal": shape_equal,
                "status": "PASS" if shape_equal and dtype_equal and trainable_equal else "FAIL",
            }
        )
    independence_rows = []
    left_parameters = dict(n2.encoder.layers[0].named_parameters())
    right_parameters = dict(n2.encoder.layers[1].named_parameters())
    for name in sorted(left_parameters):
        left = left_parameters[name]
        right = right_parameters[name]
        same_object = left is right
        same_storage = left.data_ptr() == right.data_ptr()
        independence_rows.append(
            {
                "n2_layer0_parameter": f"encoder.layers.0.{name}",
                "n2_layer1_parameter": f"encoder.layers.1.{name}",
                "same_shape": tuple(left.shape) == tuple(right.shape),
                "same_object": same_object,
                "same_storage": same_storage,
                "shared_parameter_detected": same_object or same_storage,
                "status": "FAIL" if same_object or same_storage else "PASS",
            }
        )
    stack_rows = []
    for condition_id in ("N1", "N2"):
        model = models[condition_id]
        config = model.config
        for stack_index in (0, 1):
            present = stack_index < config.num_layers
            stack_rows.append(
                {
                    "layer_id": condition_id,
                    "stack_index": stack_index,
                    "present": present,
                    "embed_dim": config.d_model if present else None,
                    "num_heads": config.num_heads if present else None,
                    "head_dim": config.d_model // config.num_heads if present else None,
                    "ffn_dim": config.ffn_dim if present else None,
                    "dropout": config.dropout if present else None,
                    "activation": config.activation if present else None,
                    "norm_policy": "POST_NORM" if present else None,
                    "independent_parameter_instance": True if present else None,
                    "status": "PASS",
                }
            )
    status = "PASS" if all(
        (
            by_id["N1"]["status"] == "PASS",
            by_id["N2"]["status"] == "PASS",
            by_id["N1"]["prediction_shape"] == [2, 1],
            by_id["N2"]["prediction_shape"] == [2, 1],
            n1_keys < n2_keys,
            bool(extra_keys),
            all(key.startswith("encoder.layers.1.") for key in extra_keys),
            shared_shapes_equal,
            layers_independent,
            parameter_delta > 0,
            parameter_delta == n2_only_key_numel,
            all(row["status"] == "PASS" for row in state_key_rows),
            all(row["status"] == "PASS" for row in shared_schema_rows),
            all(row["status"] == "PASS" for row in independence_rows),
        )
    ) else "FAIL"
    return {
        "status": status,
        "config_delta": ["num_layers"],
        "candidates": candidates,
        "state_dict_subset": n1_keys < n2_keys,
        "extra_state_keys": extra_keys,
        "shared_parameter_shapes_equal": shared_shapes_equal,
        "layers_independent": layers_independent,
        "parameter_delta": parameter_delta,
        "n2_only_key_numel": n2_only_key_numel,
        "state_key_rows": state_key_rows,
        "shared_schema_rows": shared_schema_rows,
        "independence_rows": independence_rows,
        "stack_rows": stack_rows,
    }


def inspect_phase_34_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff, signoff_error = _load_object(root, PHASE_34_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_34_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_34_REFERENCE_PATH)
    issues: list[dict[str, str]] = []
    for path, error in (
        (PHASE_34_SIGNOFF_PATH, signoff_error),
        (PHASE_34_WINNER_PATH, winner_error),
        (PHASE_34_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)
    if signoff is None or winner is None or reference is None:
        return {
            "valid": False,
            "issues": issues,
            "source_paths": [str(PHASE_34_SIGNOFF_PATH), str(PHASE_34_WINNER_PATH), str(PHASE_34_REFERENCE_PATH)],
            "warnings": [],
            "status": "BLOCKED",
            "winner_run_id": None,
            "selected_d_model": None,
            "selected_num_heads": None,
            "selected_head_dim": None,
            "frozen_configuration": None,
            "reference_evidence": None,
            "geometry_comparison": None,
        }

    if signoff.get("status", signoff.get("overall_status")) != "PASS_WITH_WARNING":
        _add_issue(issues, PHASE_34_SIGNOFF_PATH, "PHASE_34_STATUS_MISMATCH")
    if signoff.get("approved_for_phase35") is not True or reference.get("approved_for_phase35") is not True:
        _add_issue(issues, PHASE_34_SIGNOFF_PATH, "PHASE_35_NOT_APPROVED")
    if any(not _test_locked(item.get("test_status")) for item in (signoff, winner, reference)):
        _add_issue(issues, PHASE_34_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")

    expected_checksums = signoff.get("output_checksums", {})
    for relative_path in (PHASE_34_WINNER_PATH, PHASE_34_REFERENCE_PATH):
        expected = expected_checksums.get(str(relative_path))
        if not isinstance(expected, str) or sha256_file(root / relative_path) != expected:
            _add_issue(issues, relative_path, "SIGNED_HANDOFF_CHECKSUM_MISMATCH")

    run_ids = {
        winner.get("winner_run_id"),
        reference.get("winner_run_id"),
        reference.get("current_reference_run_id"),
        signoff.get("winner_run_id"),
        signoff.get("h4_reference_run_id"),
    }
    if len(run_ids) != 1 or None in run_ids:
        _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")
    winner_run_id = next(iter(run_ids)) if len(run_ids) == 1 else None

    exact_fields = (
        (winner.get("winner_head_id"), "H4", "WINNER_CONDITION_MISMATCH"),
        (winner.get("winner_num_heads"), 4, "HEAD_COUNT_MISMATCH"),
        (reference.get("selected_num_heads"), 4, "HEAD_COUNT_MISMATCH"),
        (signoff.get("winner_num_heads"), 4, "HEAD_COUNT_MISMATCH"),
        (winner.get("winner_head_dim"), 16, "HEAD_DIM_MISMATCH"),
        (reference.get("selected_head_dim"), 16, "HEAD_DIM_MISMATCH"),
        (signoff.get("winner_head_dim"), 16, "HEAD_DIM_MISMATCH"),
        (winner.get("d_model"), 64, "DMODEL_MISMATCH"),
        (reference.get("selected_d_model"), 64, "DMODEL_MISMATCH"),
        (signoff.get("selected_d_model"), 64, "DMODEL_MISMATCH"),
        (winner.get("num_layers"), 2, "LAYER_COUNT_MISMATCH"),
        (reference.get("current_num_layers"), 2, "LAYER_COUNT_MISMATCH"),
        (winner.get("ffn_dim"), 128, "FFN_DIM_MISMATCH"),
    )
    for actual, expected, reason in exact_fields:
        if actual != expected:
            _add_issue(issues, PHASE_34_WINNER_PATH, reason)

    for field in (
        "feature_variant_id", "target_scaling_id", "lookback_id", "pooling_id", "activation_id",
        "batch_id", "population_fingerprint", "winner_config_fingerprint",
        "winner_optimizer_config_fingerprint",
    ):
        values = {winner.get(field), reference.get(field)}
        if len(values) != 1 or None in values:
            _add_issue(issues, field, "FROZEN_FIELD_MISMATCH")
    for field in ("learning_rate", "weight_decay", "dropout", "winner_rmse_wh"):
        if not _same_number(winner.get(field), reference.get(field)):
            _add_issue(issues, field, "FROZEN_FIELD_MISMATCH")

    source_config_path = Path("artifacts/runs") / str(winner_run_id) / "config.json"
    source_payload, source_error = _load_object(root, source_config_path)
    geometry = None
    if source_error is not None:
        _add_issue(issues, source_config_path, source_error)
    else:
        config = source_payload.get("config")
        if source_payload.get("config_fingerprint") != winner.get("winner_config_fingerprint"):
            _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISMATCH")
        if not isinstance(config, dict):
            _add_issue(issues, source_config_path, "CONFIG_MISSING")
        else:
            model = config.get("model", {})
            training = config.get("training", {})
            reproducibility = config.get("reproducibility", {})
            expected = (
                (model.get("d_model"), 64, "DMODEL_DRIFT"),
                (model.get("num_heads"), 4, "HEAD_COUNT_DRIFT"),
                (model.get("num_layers"), 2, "LAYER_COUNT_DRIFT"),
                (model.get("ffn_dim"), 128, "FFN_DIM_DRIFT"),
                (training.get("max_epochs"), 50, "EPOCH_BUDGET_DRIFT"),
                (training.get("early_stopping_patience"), 10, "PATIENCE_DRIFT"),
                (training.get("gradient_clip_max_norm"), 1.0, "GRADIENT_CLIP_DRIFT"),
                (reproducibility.get("seed"), SEED, "SEED_DRIFT"),
            )
            for actual, wanted, reason in expected:
                if actual != wanted:
                    _add_issue(issues, source_config_path, reason)
            try:
                geometry = inspect_layer_geometry(model)
            except (RuntimeError, TypeError, ValueError):
                _add_issue(issues, source_config_path, "LAYER_PREFLIGHT_INVALID")
            else:
                if geometry["status"] != "PASS":
                    _add_issue(issues, source_config_path, "LAYER_GEOMETRY_INVALID")

    evidence = resolve_phase_conditions(PHASE_ID, root)
    verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
    reference_evidence = verified.get(REFERENCE_CONDITION_ID)
    if reference_evidence is None:
        _add_issue(issues, str(winner_run_id), "N2_REFERENCE_EVIDENCE_INVALID")
    elif reference_evidence.get("run_id") != winner_run_id:
        _add_issue(issues, str(winner_run_id), "N2_REFERENCE_RUN_MISMATCH")

    frozen = None
    if not issues:
        frozen = {
            "feature_variant_id": winner["feature_variant_id"],
            "target_scaling_id": winner["target_scaling_id"],
            "lookback_id": winner["lookback_id"],
            "pooling_id": winner["pooling_id"],
            "activation_id": winner["activation_id"],
            "batch_id": winner["batch_id"],
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "dropout": winner["dropout"],
            "d_model": winner["d_model"],
            "num_heads": winner["winner_num_heads"],
            "head_dim": winner["winner_head_dim"],
            "num_layers": 2,
            "ffn_dim": winner["ffn_dim"],
            "population_fingerprint": winner["population_fingerprint"],
            "optimizer": "AdamW",
            "max_epochs": 50,
            "patience": 10,
            "min_delta": 0,
            "gradient_clip_norm": 1.0,
            "seed": SEED,
            "scheduler": "NONE",
            "warmup": "NONE",
            "gradient_accumulation": 1,
            "loss": "MSE",
            "test_access": TEST_ACCESS,
            "source_config_path": str(source_config_path),
            "source_config_fingerprint": winner["winner_config_fingerprint"],
        }
    warnings = list(dict.fromkeys([*winner.get("inherited_warnings", []), *reference.get("inherited_warnings", [])]))
    return {
        "valid": not issues,
        "issues": issues,
        "source_paths": [str(PHASE_34_SIGNOFF_PATH), str(PHASE_34_WINNER_PATH), str(PHASE_34_REFERENCE_PATH)],
        "winner_run_id": winner_run_id,
        "selected_d_model": winner.get("d_model"),
        "selected_num_heads": winner.get("winner_num_heads"),
        "selected_head_dim": winner.get("winner_head_dim"),
        "frozen_configuration": frozen,
        "reference_evidence": reference_evidence,
        "geometry_comparison": geometry,
        "warnings": warnings,
        "status": "PASS_WITH_WARNING" if not issues and INHERITED_WARNING in warnings else ("PASS" if not issues else "BLOCKED"),
    }


def build_phase_35_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    handoff = inspect_phase_34_handoff(root)
    decision = plan_phase_resume(PHASE_ID, root)
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selection_metric": "VALIDATION_RMSE_WH",
        "tie_rule": "N1_ON_EXACT_RMSE_TIE",
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "decision": decision,
        "ready": handoff["valid"] and decision["readiness"]["ready"],
    }


def prepare_phase_35_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {item.condition_id: item for item in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 35")
    handoff = inspect_phase_34_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 35 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    if condition.execution_mode == "REUSE_REFERENCE":
        return {
            "phase_id": PHASE_ID,
            "sweep_id": SWEEP_ID,
            "condition_id": condition.condition_id,
            "execution_mode": condition.execution_mode,
            "num_layers": condition.num_layers,
            "reference_run_id": handoff["winner_run_id"],
            "reference_evidence": handoff["reference_evidence"],
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
        "num_layers": condition.num_layers,
        "reference_run_id": handoff["winner_run_id"],
        "frozen_configuration": handoff["frozen_configuration"],
        "geometry_comparison": handoff["geometry_comparison"],
        "test_access": TEST_ACCESS,
    }
