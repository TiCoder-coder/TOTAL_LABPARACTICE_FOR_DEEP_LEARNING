from __future__ import annotations

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
from course_work.sweeps.sweep_results import validate_sweep_signoff
from course_work.utils.artifacts import get_project_root, read_json, sha256_file
from course_work.utils.reproducibility import set_seed


PHASE_ID = 36
PHASE_VERSION = "PHASE-36-v1"
SWEEP_ID = "S14_FFN"
SWEEP_VERSION = "SWEEP_S14_FFN-v1"
REFERENCE_CONDITION_ID = "F128"
SEED = 42
TEST_ACCESS = "FORBIDDEN"
PHASE_35_ROOT = Path("artifacts/sweeps/S13_layers")
PHASE_35_SIGNOFF_PATH = PHASE_35_ROOT / "phase_35_signoff.json"
PHASE_35_WINNER_PATH = PHASE_35_ROOT / "s13_layer_winner.json"
PHASE_35_REFERENCE_PATH = PHASE_35_ROOT / "s13_reference_update.json"
INHERITED_WARNING = "H4_SOURCE_ARTIFACT_RETENTION_INCOMPLETE"


@dataclass(frozen=True)
class FFNCondition:
    condition_id: str
    ffn_dim: int
    execution_mode: str


CONDITIONS = (
    FFNCondition("F64", 64, "TRAIN_NEW"),
    FFNCondition("F128", 128, "REUSE_REFERENCE"),
    FFNCondition("F256", 256, "TRAIN_NEW"),
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


def _contains_signed_registry_prefix(path: Path, expected_sha256: object) -> bool:
    """Accept a signed JSONL registry input only when it is an exact byte prefix."""
    import hashlib

    if not path.is_file() or not isinstance(expected_sha256, str):
        return False
    payload = path.read_bytes()
    for newline_index, value in enumerate(payload):
        if value == 10 and hashlib.sha256(payload[: newline_index + 1]).hexdigest() == expected_sha256:
            return True
    return hashlib.sha256(payload).hexdigest() == expected_sha256


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    )


def _optimizer_coverage(model: torch.nn.Module) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    model_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    unique_ids = set(optimizer_ids)
    missing = len(model_ids - unique_ids)
    unexpected = len(unique_ids - model_ids)
    duplicates = len(optimizer_ids) - len(unique_ids)
    return {
        "model_trainable_parameter_references": len(model_ids),
        "optimizer_parameter_references": len(optimizer_ids),
        "unique_optimizer_parameter_references": len(unique_ids),
        "missing_parameters": missing,
        "unexpected_parameters": unexpected,
        "duplicate_parameters": duplicates,
        "status": "PASS" if missing == 0 and unexpected == 0 and duplicates == 0 else "FAIL",
    }


def _guarded_sanity(model: TransformerRegressor, config: dict[str, Any]) -> dict[str, Any]:
    model.train()
    sample = torch.randn(2, 36, config["input_size"], dtype=torch.float32)
    target = torch.zeros(2, 1, dtype=torch.float32)
    prediction = model(sample)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        standard_prediction = model(sample)
        inspection_prediction, attention_maps = model.forward_with_attention(sample)
    attention_shapes = [list(value.shape) for value in attention_maps]
    expected_attention_shape = [2, config["num_heads"], 36, 36]
    attention_nonnegative = all(bool((value >= 0).all().item()) for value in attention_maps)
    attention_row_sum_error_max = max(
        (float((value.sum(dim=-1) - 1.0).abs().max().item()) for value in attention_maps),
        default=0.0,
    )
    status = all(
        (
            list(prediction.shape) == [2, 1],
            bool(torch.isfinite(prediction).all().item()),
            bool(torch.isfinite(loss).item()),
            gradients_finite,
            len(attention_maps) == config["num_layers"],
            all(shape == expected_attention_shape for shape in attention_shapes),
            all(bool(torch.isfinite(value).all().item()) for value in attention_maps),
            attention_nonnegative,
            attention_row_sum_error_max <= 1e-5,
            bool(torch.allclose(standard_prediction, inspection_prediction, atol=1e-6, rtol=1e-6)),
        )
    )
    return {
        "prediction_shape": list(prediction.shape),
        "forward_finite": bool(torch.isfinite(prediction).all().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "backward_finite": gradients_finite,
        "attention_shapes": attention_shapes,
        "attention_finite": all(bool(torch.isfinite(value).all().item()) for value in attention_maps),
        "attention_nonnegative": attention_nonnegative,
        "attention_row_sum_error_max": attention_row_sum_error_max,
        "prediction_paths_equal": bool(
            torch.allclose(standard_prediction, inspection_prediction, atol=1e-6, rtol=1e-6)
        ),
        "optimizer_step_executed": False,
        "training_engine_called": False,
        "status": "PASS" if status else "FAIL",
    }


def inspect_ffn_geometry(model_config: dict[str, Any]) -> dict[str, Any]:
    base = validate_transformer_config(model_config).to_dict()
    candidates: list[dict[str, Any]] = []
    state_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    parameter_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for condition in CONDITIONS:
        config = {**base, "ffn_dim": condition.ffn_dim}
        set_seed(SEED)
        model = TransformerRegressor(config)
        coverage = _optimizer_coverage(model)
        sanity = _guarded_sanity(model, config)
        layer_widths = [
            {
                "layer_index": index,
                "linear1_out_features": layer.linear1.out_features,
                "linear2_in_features": layer.linear2.in_features,
                "status": (
                    "PASS"
                    if layer.linear1.out_features == condition.ffn_dim
                    and layer.linear2.in_features == condition.ffn_dim
                    else "FAIL"
                ),
            }
            for index, layer in enumerate(model.encoder.layers)
        ]
        state_shapes[condition.condition_id] = {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        }
        parameter_shapes[condition.condition_id] = {
            name: tuple(value.shape) for name, value in model.named_parameters()
        }
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        candidates.append(
            {
                "condition_id": condition.condition_id,
                "ffn_dim": condition.ffn_dim,
                "execution_mode": condition.execution_mode,
                "d_model": config["d_model"],
                "num_heads": config["num_heads"],
                "head_dim": config["d_model"] // config["num_heads"],
                "num_layers": config["num_layers"],
                "expansion_ratio": condition.ffn_dim / config["d_model"],
                "trainable_parameter_count": trainable_count,
                "layer_widths": layer_widths,
                "optimizer_coverage": coverage,
                "sanity": sanity,
                "status": (
                    "PASS"
                    if coverage["status"] == "PASS"
                    and sanity["status"] == "PASS"
                    and all(item["status"] == "PASS" for item in layer_widths)
                    else "FAIL"
                ),
            }
        )
    by_id = {item["condition_id"]: item for item in candidates}
    key_sets_equal = all(set(state_shapes[item]) == set(state_shapes["F128"]) for item in state_shapes)
    changed_shape_rows = []
    non_ffn_invariant = True
    ffn_only_delta = True
    for key in sorted(state_shapes["F128"]):
        shapes = {condition_id: list(values[key]) for condition_id, values in state_shapes.items()}
        changed = len({tuple(value) for value in shapes.values()}) > 1
        ffn_key = ".linear1." in key or ".linear2." in key
        if changed and not ffn_key:
            ffn_only_delta = False
            non_ffn_invariant = False
        changed_shape_rows.append(
            {
                "key": key,
                "shapes": shapes,
                "shape_changes": changed,
                "ffn_width_dependent": ffn_key,
                "status": "PASS" if not changed or ffn_key else "FAIL",
            }
        )
    counts = [by_id[item]["trainable_parameter_count"] for item in ("F64", "F128", "F256")]
    monotonic = counts[0] < counts[1] < counts[2]
    d_model = int(base["d_model"])
    num_layers = int(base["num_layers"])
    expected_delta_64_128 = num_layers * (2 * d_model + 1) * (128 - 64)
    expected_delta_128_256 = num_layers * (2 * d_model + 1) * (256 - 128)
    actual_delta_64_128 = counts[1] - counts[0]
    actual_delta_128_256 = counts[2] - counts[1]
    delta_valid = (
        actual_delta_64_128 == expected_delta_64_128
        and actual_delta_128_256 == expected_delta_128_256
    )
    mha_keys = [key for key in parameter_shapes["F128"] if ".self_attn." in key]
    mha_invariant = all(
        parameter_shapes["F64"][key]
        == parameter_shapes["F128"][key]
        == parameter_shapes["F256"][key]
        for key in mha_keys
    )
    optimizer_coverage = all(item["optimizer_coverage"]["status"] == "PASS" for item in candidates)
    sanity_status = {
        condition_id: by_id[condition_id]["sanity"]["status"] for condition_id in ("F64", "F256")
    }
    status = "PASS" if all(
        (
            key_sets_equal,
            ffn_only_delta,
            non_ffn_invariant,
            monotonic,
            delta_valid,
            mha_invariant,
            optimizer_coverage,
            all(item["status"] == "PASS" for item in candidates),
        )
    ) else "FAIL"
    return {
        "status": status,
        "config_delta": ["ffn_dim"],
        "candidates": candidates,
        "state_dict_key_sets_equal": key_sets_equal,
        "state_dict_shape_rows": changed_shape_rows,
        "ffn_only_shape_delta": ffn_only_delta,
        "mha_invariance": "PASS" if mha_invariant else "FAIL",
        "non_ffn_invariance": "PASS" if non_ffn_invariant else "FAIL",
        "parameter_monotonicity": "PASS" if monotonic else "FAIL",
        "parameter_counts": {item: by_id[item]["trainable_parameter_count"] for item in by_id},
        "parameter_deltas": {
            "F64_TO_F128": actual_delta_64_128,
            "F128_TO_F256": actual_delta_128_256,
        },
        "expected_parameter_deltas": {
            "F64_TO_F128": expected_delta_64_128,
            "F128_TO_F256": expected_delta_128_256,
        },
        "parameter_delta_audit": "PASS" if delta_valid else "FAIL",
        "optimizer_coverage": "PASS" if optimizer_coverage else "FAIL",
        "sanity": sanity_status,
    }


def inspect_phase_35_handoff(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    signoff_validation = validate_sweep_signoff(35, root)
    signoff, signoff_error = _load_object(root, PHASE_35_SIGNOFF_PATH)
    winner, winner_error = _load_object(root, PHASE_35_WINNER_PATH)
    reference, reference_error = _load_object(root, PHASE_35_REFERENCE_PATH)
    issues: list[dict[str, str]] = []
    for path, error in (
        (PHASE_35_SIGNOFF_PATH, signoff_error),
        (PHASE_35_WINNER_PATH, winner_error),
        (PHASE_35_REFERENCE_PATH, reference_error),
    ):
        if error is not None:
            _add_issue(issues, path, error)
    if not signoff_validation["valid"]:
        for issue in signoff_validation["issues"]:
            if (
                issue["path"] == "artifacts/experiments/experiment_registry.jsonl"
                and issue["reason"] == "CHECKSUM_MISMATCH"
                and signoff is not None
                and _contains_signed_registry_prefix(
                    root / issue["path"],
                    signoff.get("input_checksums", {}).get(issue["path"]),
                )
            ):
                continue
            _add_issue(issues, issue["path"], issue["reason"])
    if signoff is None or winner is None or reference is None:
        return {
            "valid": False,
            "status": "BLOCKED",
            "issues": issues,
            "source_paths": [str(PHASE_35_SIGNOFF_PATH), str(PHASE_35_WINNER_PATH), str(PHASE_35_REFERENCE_PATH)],
            "warnings": [],
            "winner_run_id": None,
            "frozen_configuration": None,
            "reference_evidence": None,
            "geometry_comparison": None,
        }
    if signoff.get("approved_for_phase36") is not True or reference.get("approved_for_phase36") is not True:
        _add_issue(issues, PHASE_35_SIGNOFF_PATH, "PHASE_36_NOT_APPROVED")
    if any(not _test_locked(item.get("test_status")) for item in (signoff, winner, reference)):
        _add_issue(issues, PHASE_35_SIGNOFF_PATH, "TEST_FIREWALL_NOT_CONFIRMED")
    expected_policy = {"F64": "TRAIN_NEW", "F128": "REUSE_REFERENCE", "F256": "TRAIN_NEW"}
    if reference.get("phase36_condition_policy") != expected_policy:
        _add_issue(issues, PHASE_35_REFERENCE_PATH, "PHASE_36_CONDITION_POLICY_MISMATCH")
    run_ids = {
        winner.get("winner_run_id"),
        reference.get("winner_run_id"),
        reference.get("current_reference_run_id"),
        signoff.get("winner_run_id"),
        signoff.get("n2_reference_run_id"),
    }
    if len(run_ids) != 1 or None in run_ids:
        _add_issue(issues, "winner_run_id", "REFERENCE_RUN_ID_MISMATCH")
    winner_run_id = next(iter(run_ids)) if len(run_ids) == 1 else None
    for field in (
        "feature_variant_id", "target_scaling_id", "lookback_id", "pooling_id", "activation_id",
        "batch_id", "population_fingerprint", "winner_config_fingerprint",
        "winner_optimizer_config_fingerprint",
    ):
        values = {winner.get(field), reference.get(field), signoff.get(field)}
        values.discard(None)
        if len(values) != 1:
            _add_issue(issues, field, "FROZEN_FIELD_MISMATCH")
    numeric_fields = (
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("dropout_probability", "dropout_probability"),
        ("d_model", "d_model"),
        ("num_heads", "num_heads"),
        ("selected_head_dim", "selected_head_dim"),
        ("winner_num_layers", "selected_num_layers"),
        ("ffn_dim", "current_ffn_dim"),
        ("winner_rmse_wh", "winner_rmse_wh"),
    )
    for winner_field, reference_field in numeric_fields:
        left = winner.get(winner_field)
        right = reference.get(reference_field)
        if not _same_number(left, right):
            _add_issue(issues, winner_field, "FROZEN_FIELD_MISMATCH")
    if reference.get("current_ffn_dim") != 128 or winner.get("ffn_dim") != 128:
        _add_issue(issues, PHASE_35_REFERENCE_PATH, "F128_REFERENCE_MISMATCH")
    source_config_path = Path("artifacts/runs") / str(winner_run_id) / "config.json"
    source_payload, source_error = _load_object(root, source_config_path)
    geometry = None
    frozen = None
    if source_error is not None:
        _add_issue(issues, source_config_path, source_error)
    else:
        config = source_payload.get("config")
        if source_payload.get("config_fingerprint") != reference.get("winner_config_fingerprint"):
            _add_issue(issues, source_config_path, "CONFIG_FINGERPRINT_MISMATCH")
        if not isinstance(config, dict):
            _add_issue(issues, source_config_path, "CONFIG_MISSING")
        else:
            model = config.get("model", {})
            training = config.get("training", {})
            reproducibility = config.get("reproducibility", {})
            expected = (
                (model.get("d_model"), reference.get("d_model"), "DMODEL_DRIFT"),
                (model.get("num_heads"), reference.get("num_heads"), "HEAD_COUNT_DRIFT"),
                (model.get("num_layers"), reference.get("selected_num_layers"), "LAYER_COUNT_DRIFT"),
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
                geometry = inspect_ffn_geometry(model)
            except (RuntimeError, TypeError, ValueError):
                _add_issue(issues, source_config_path, "FFN_PREFLIGHT_INVALID")
            else:
                if geometry["status"] != "PASS":
                    _add_issue(issues, source_config_path, "FFN_GEOMETRY_INVALID")
            if not issues:
                frozen = {
                    "feature_variant_id": reference["feature_variant_id"],
                    "target_scaling_id": reference["target_scaling_id"],
                    "lookback_id": reference["lookback_id"],
                    "pooling_id": reference["pooling_id"],
                    "activation_id": reference["activation_id"],
                    "batch_id": reference["batch_id"],
                    "learning_rate": reference["learning_rate"],
                    "weight_decay": reference["weight_decay"],
                    "dropout": reference["dropout_probability"],
                    "d_model": reference["d_model"],
                    "num_heads": reference["num_heads"],
                    "head_dim": reference["selected_head_dim"],
                    "num_layers": reference["selected_num_layers"],
                    "ffn_dim": reference["current_ffn_dim"],
                    "population_fingerprint": reference["population_fingerprint"],
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
                    "source_config_fingerprint": reference["winner_config_fingerprint"],
                }
    evidence = resolve_phase_conditions(PHASE_ID, root)
    verified = {item["condition_id"]: item for item in evidence["verified_conditions"]}
    reference_evidence = verified.get(REFERENCE_CONDITION_ID)
    if reference_evidence is None:
        _add_issue(issues, str(winner_run_id), "F128_REFERENCE_EVIDENCE_INVALID")
    elif reference_evidence.get("run_id") != winner_run_id:
        _add_issue(issues, str(winner_run_id), "F128_REFERENCE_RUN_MISMATCH")
    warnings = list(
        dict.fromkeys(
            [
                *winner.get("inherited_warnings", []),
                *reference.get("inherited_warnings", []),
                *signoff.get("inherited_warnings", []),
            ]
        )
    )
    return {
        "valid": not issues,
        "issues": issues,
        "source_paths": [str(PHASE_35_SIGNOFF_PATH), str(PHASE_35_WINNER_PATH), str(PHASE_35_REFERENCE_PATH)],
        "winner_run_id": winner_run_id,
        "selected_d_model": reference.get("d_model"),
        "selected_num_heads": reference.get("num_heads"),
        "selected_head_dim": reference.get("selected_head_dim"),
        "selected_num_layers": reference.get("selected_num_layers"),
        "selected_ffn_dim": reference.get("current_ffn_dim"),
        "frozen_configuration": frozen,
        "reference_evidence": reference_evidence,
        "geometry_comparison": geometry,
        "warnings": warnings,
        "status": (
            "PASS_WITH_WARNING"
            if not issues and INHERITED_WARNING in warnings
            else "PASS" if not issues else "BLOCKED"
        ),
    }


def build_phase_36_preflight(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    handoff = inspect_phase_35_handoff(root)
    decision = plan_phase_resume(PHASE_ID, root)
    geometry = handoff.get("geometry_comparison") or {}
    return {
        "phase_id": PHASE_ID,
        "phase_version": PHASE_VERSION,
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_condition_id": REFERENCE_CONDITION_ID,
        "selection_metric": "VALIDATION_RMSE_WH",
        "selection_direction": "MIN",
        "tie_rule": "SMALLEST_FFN_ON_EXACT_RMSE_TIE",
        "test_access": TEST_ACCESS,
        "handoff": handoff,
        "decision": decision,
        "parameter_monotonicity": geometry.get("parameter_monotonicity", "FAIL"),
        "parameter_delta_audit": geometry.get("parameter_delta_audit", "FAIL"),
        "ffn_only_delta": "PASS" if geometry.get("ffn_only_shape_delta") else "FAIL",
        "mha_invariance": geometry.get("mha_invariance", "FAIL"),
        "non_ffn_invariance": geometry.get("non_ffn_invariance", "FAIL"),
        "optimizer_coverage": geometry.get("optimizer_coverage", "FAIL"),
        "sanity": geometry.get("sanity", {}),
        "ready": handoff["valid"] and geometry.get("status") == "PASS" and decision["readiness"]["ready"],
    }


def prepare_phase_36_condition(condition_id: str, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    condition_by_id = {item.condition_id: item for item in CONDITIONS}
    if condition_id not in condition_by_id:
        raise ValueError(f"Condition {condition_id} is not registered for Phase 36")
    handoff = inspect_phase_35_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 36 handoff is invalid: {handoff['issues']}")
    condition = condition_by_id[condition_id]
    payload = {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "condition_id": condition.condition_id,
        "execution_mode": condition.execution_mode,
        "ffn_dim": condition.ffn_dim,
        "expansion_ratio": condition.ffn_dim / handoff["frozen_configuration"]["d_model"],
        "reference_run_id": handoff["winner_run_id"],
        "reference_evidence": handoff["reference_evidence"],
        "frozen_configuration": handoff["frozen_configuration"],
        "geometry_comparison": handoff["geometry_comparison"],
        "warnings": handoff["warnings"],
        "test_access": TEST_ACCESS,
    }
    if condition.execution_mode == "REUSE_REFERENCE":
        return payload
    gate = validate_condition_request(PHASE_ID, condition_id, root)
    if not gate["allowed"]:
        raise RuntimeError(f"Condition execution blocked: {gate}")
    if condition.condition_id == "F256":
        f64_verification = verify_phase_36_f64_best(project_root=root)
        if f64_verification.get("status") != "VERIFIED":
            raise RuntimeError("F256 is blocked until F64 strict BEST verification passes")
        payload["f64_verification_gate"] = {
            "status": "PASS",
            "run_id": f64_verification["run_id"],
            "best_checkpoint_sha256": f64_verification["best_checkpoint_sha256"],
            "recomputed_validation_rmse_wh": f64_verification["recomputed_validation_metrics"]["rmse_wh"],
            "population_fingerprint": f64_verification["population_fingerprint"],
            "test_access": f64_verification["test_access"],
        }
    return payload


def _registry_record(root: Path, run_id: str) -> dict[str, Any]:
    registry_path = root / "artifacts/experiments/experiment_registry.jsonl"
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("run_id") == run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Phase 36 expected one registry record for {run_id}, found {len(matches)}")
    return matches[0]


def _config_differences(left: object, right: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((*path, str(key)))
            else:
                differences.extend(_config_differences(left[key], right[key], (*path, str(key))))
        return differences
    return [] if left == right else [path]


def _verify_phase_36_trained_best(
    condition_id: str,
    run_id: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Strictly verify one retained fresh S14 BEST run without training or Test access."""
    import numpy as np
    import pandas as pd

    from course_work.data.datasets import build_train_validation_loaders
    from course_work.data.scaling import inverse_transform_target, load_validated_target_scaler
    from course_work.evaluation.metrics import EvaluationMode, compute_regression_metrics

    root = Path(project_root or get_project_root()).resolve()
    expected_ffn_dim = {"F64": 64, "F256": 256}.get(condition_id)
    if expected_ffn_dim is None:
        raise ValueError(f"Strict Phase 36 verification is not defined for {condition_id}")
    record = _registry_record(root, run_id)
    run_root = root / "artifacts/runs" / run_id
    paths = {
        "config": run_root / "config.json",
        "status": run_root / "status.json",
        "checkpoint": run_root / "checkpoints/best_checkpoint.pt",
        "history": run_root / "training_history.csv",
        "metrics": run_root / "metrics/best_validation_metrics.json",
        "predictions": run_root / "predictions/best_validation_predictions.csv",
    }
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase 36 {condition_id} evidence is missing: {missing}")
    if record.get("status") != "COMPLETED":
        raise RuntimeError(f"Phase 36 {condition_id} registry status is not COMPLETED")
    if record.get("experiment_family") != SWEEP_ID or record.get("sweep_stage") != "S14":
        raise RuntimeError(f"Phase 36 {condition_id} registry family/sweep identity is invalid")
    if record.get("test_access_authorized") is not False:
        raise RuntimeError(f"Phase 36 {condition_id} registry does not preserve the Test firewall")
    if any(metric.get("split_id") != "VALIDATION" for metric in record.get("metrics", [])):
        raise RuntimeError(f"Phase 36 {condition_id} registry contains non-Validation metrics")
    if any("test" in str(item.get("artifact_path", "")).lower() for item in record.get("artifacts", [])):
        raise RuntimeError(f"Phase 36 {condition_id} registry contains Test artifacts")
    if any("test" in path.name.lower() for path in run_root.rglob("*") if path.is_file()):
        raise RuntimeError(f"Phase 36 {condition_id} run directory contains Test evidence")
    for artifact in record.get("artifacts", []):
        if artifact.get("required") is not True:
            continue
        artifact_path = root / str(artifact.get("artifact_path"))
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
                raise RuntimeError(f"Phase 36 {condition_id} artifact checksum failed: {artifact_path.relative_to(root)}")

    config_payload = read_json(paths["config"])
    config = config_payload.get("config")
    if not isinstance(config, dict) or config_payload.get("config_fingerprint") != record.get("config_fingerprint"):
        raise RuntimeError(f"Phase 36 {condition_id} config identity/fingerprint is invalid")
    handoff = inspect_phase_35_handoff(root)
    if not handoff["valid"]:
        raise RuntimeError(f"Phase 35 handoff is invalid: {handoff['issues']}")
    reference_config = read_json(root / handoff["frozen_configuration"]["source_config_path"])["config"]
    if _config_differences(config, reference_config) != [("model", "ffn_dim")]:
        raise RuntimeError(f"Phase 36 {condition_id} frozen configuration drift exceeds model.ffn_dim")
    model_config = config.get("model", {})
    if (
        model_config.get("ffn_dim") != expected_ffn_dim
        or model_config.get("d_model") != 64
        or model_config.get("num_heads") != 4
        or model_config.get("num_layers") != 2
        or config.get("reproducibility", {}).get("seed") != SEED
    ):
        raise RuntimeError(f"Phase 36 {condition_id} model/seed contract is invalid")
    if config.get("data", {}).get("target_access_mode") != "VALIDATION":
        raise RuntimeError(f"Phase 36 {condition_id} data access is not Validation-only")

    status = read_json(paths["status"])
    metrics = read_json(paths["metrics"])["metric_result"]
    history = pd.read_csv(paths["history"])
    predictions = pd.read_csv(paths["predictions"])
    if status.get("status") != "COMPLETED" or status.get("best_epoch") != record.get("best_epoch"):
        raise RuntimeError(f"Phase 36 {condition_id} status/BEST epoch evidence mismatch")
    if history.empty or int(history["epoch"].max()) != len(history):
        raise RuntimeError(f"Phase 36 {condition_id} epoch evidence mismatch")
    best_rows = history.loc[history["is_best"].astype(bool)]
    if best_rows.empty or int(best_rows.iloc[-1]["epoch"]) != int(status["best_epoch"]):
        raise RuntimeError(f"Phase 36 {condition_id} BEST history marker mismatch")
    stored_values = {field: float(metrics[field]) for field in ("rmse_wh", "mae_wh", "r2")}
    if not all(math.isfinite(value) for value in stored_values.values()):
        raise RuntimeError(f"Phase 36 {condition_id} stored metrics are non-finite")
    if (
        metrics.get("split_id") != "VALIDATION"
        or metrics.get("metric_version") != "METRICS-v1"
        or metrics.get("finite_status") != "PASS"
        or float(metrics["rmse_wh"]) != float(record["best_validation_rmse_wh"])
    ):
        raise RuntimeError(f"Phase 36 {condition_id} stored BEST metrics are invalid")
    if len(predictions) != int(metrics["n_samples"]):
        raise RuntimeError(f"Phase 36 {condition_id} prediction population size mismatch")
    for field in ("sample_idx", "y_true_wh", "y_pred_wh", "residual_wh"):
        if not predictions[field].map(lambda value: math.isfinite(float(value))).all():
                raise RuntimeError(f"Phase 36 {condition_id} predictions contain non-finite {field}")
    if predictions["run_id"].nunique() != 1 or predictions["run_id"].iloc[0] != run_id:
        raise RuntimeError(f"Phase 36 {condition_id} prediction run identity mismatch")
    sample_ids = predictions["sample_idx"].astype(int)
    expected_start = int(config["data"]["train_sample_count"])
    expected_count = int(config["data"]["validation_sample_count"])
    if (
        len(sample_ids) != expected_count
        or not sample_ids.is_monotonic_increasing
        or not sample_ids.is_unique
        or sample_ids.iloc[0] != expected_start
        or sample_ids.iloc[-1] != expected_start + expected_count - 1
    ):
        raise RuntimeError(f"Phase 36 {condition_id} ordered Validation population is invalid")
    residual_error = (
        predictions["residual_wh"] - (predictions["y_true_wh"] - predictions["y_pred_wh"])
    ).abs().max()
    if float(residual_error) > 1e-9:
        raise RuntimeError(f"Phase 36 {condition_id} residual convention is invalid")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError(f"Phase 36 {condition_id} BEST checkpoint payload is invalid")
    set_seed(SEED)
    model = TransformerRegressor(model_config)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Phase 36 {condition_id} BEST strict-load returned incompatible keys")
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint_metadata = checkpoint.get("checkpoint_metadata", {})
    if checkpoint_metadata.get("config") != model.config.to_dict():
        raise RuntimeError(f"Phase 36 {condition_id} BEST model-config provenance mismatch")
    if checkpoint_metadata.get("trainable_parameters") != trainable_parameters:
        raise RuntimeError(f"Phase 36 {condition_id} BEST parameter-count provenance mismatch")
    if checkpoint.get("best_epoch") != int(status["best_epoch"]):
        raise RuntimeError(f"Phase 36 {condition_id} BEST epoch provenance mismatch")
    layer_widths = [
        (layer.linear1.in_features, layer.linear1.out_features, layer.linear2.in_features, layer.linear2.out_features)
        for layer in model.encoder.layers
    ]
    expected_layer_shape = (64, expected_ffn_dim, expected_ffn_dim, 64)
    if layer_widths != [expected_layer_shape, expected_layer_shape]:
        raise RuntimeError(f"Phase 36 {condition_id} active Encoder FFN geometry is invalid")
    coverage = _optimizer_coverage(model)
    if coverage["status"] != "PASS":
        raise RuntimeError(f"Phase 36 {condition_id} optimizer parameter coverage failed")

    model.eval()
    loaders = build_train_validation_loaders(
        project_root=root,
        variant_id=config["data"]["feature_variant_id"],
        lookback=int(config["data"]["lookback_steps"]),
        target_option=config["data"]["target_scaling_option"],
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["reproducibility"]["seed"]),
        num_workers=0,
        device_type="cpu",
    )
    validation_loader = loaders["VALIDATION"][0]
    target_scaler = (
        load_validated_target_scaler(root)
        if config["data"]["target_scaling_option"] == "YS1"
        else None
    )
    recomputed_sample_ids: list[int] = []
    recomputed_y_true: list[float] = []
    recomputed_y_pred: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            model_predictions = model(batch["x"])
            if model_predictions.shape != batch["y_model"].shape:
                    raise RuntimeError(f"Phase 36 {condition_id} strict verification prediction shape mismatch")
            if config["data"]["target_scaling_option"] == "YS1":
                prediction_wh = inverse_transform_target(
                    model_predictions.cpu().numpy().reshape(-1, 1), "YS1", target_scaler
                ).reshape(-1)
            else:
                prediction_wh = model_predictions.cpu().numpy().reshape(-1)
            recomputed_sample_ids.extend(batch["sample_idx"].cpu().numpy().tolist())
            recomputed_y_true.extend(batch["y_raw_wh"].cpu().numpy().reshape(-1).tolist())
            recomputed_y_pred.extend(prediction_wh.tolist())
    recomputed = compute_regression_metrics(
        np.asarray(recomputed_y_true, dtype=np.float64),
        np.asarray(recomputed_y_pred, dtype=np.float64),
        np.asarray(recomputed_sample_ids, dtype=np.int64),
        "VALIDATION",
        EvaluationMode.VALIDATION.value,
        config["lineage"]["population_fingerprint"],
        run_id,
        config["model"].get("model_name", config["model"]["model_family"]),
        lookback_steps=int(config["data"]["lookback_steps"]),
        horizon_steps=int(config["data"]["horizon_steps"]),
        target_scaling_option=config["data"]["target_scaling_option"],
        project_root=root,
    )
    recomputed_values = {
        "rmse_wh": float(recomputed.rmse_wh),
        "mae_wh": float(recomputed.mae_wh),
        "r2": float(recomputed.r2),
    }
    tolerance = 1e-6
    if any(
        not math.isclose(recomputed_values[field], stored_values[field], rel_tol=tolerance, abs_tol=tolerance)
        for field in stored_values
    ):
        raise RuntimeError(
            f"Phase 36 {condition_id} strict Validation metrics differ from stored BEST evidence: "
            f"stored={stored_values}, recomputed={recomputed_values}"
        )
    if recomputed.n_samples != int(metrics["n_samples"]):
        raise RuntimeError(f"Phase 36 {condition_id} strict Validation population count mismatch")
    if recomputed.population_fingerprint != metrics["population_fingerprint"]:
        raise RuntimeError(f"Phase 36 {condition_id} strict Validation population fingerprint mismatch")
    if not np.array_equal(np.asarray(recomputed_sample_ids, dtype=np.int64), sample_ids.to_numpy(dtype=np.int64)):
        raise RuntimeError(f"Phase 36 {condition_id} strict Validation sample order mismatch")
    if not np.allclose(
        np.asarray(recomputed_y_true, dtype=np.float64),
        predictions["y_true_wh"].to_numpy(dtype=np.float64),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise RuntimeError(f"Phase 36 {condition_id} strict Validation targets differ from retained predictions")
    prediction_max_abs_delta = float(
        np.max(
            np.abs(
                np.asarray(recomputed_y_pred, dtype=np.float64)
                - predictions["y_pred_wh"].to_numpy(dtype=np.float64)
            )
        )
    )

    geometry = inspect_ffn_geometry(reference_config["model"])
    condition_geometry = next(
        item for item in geometry["candidates"] if item["condition_id"] == condition_id
    )
    if geometry["status"] != "PASS" or condition_geometry["trainable_parameter_count"] != trainable_parameters:
        raise RuntimeError(f"Phase 36 {condition_id} parameter geometry/monotonicity failed")
    terminal_log = root / "artifacts/sweeps/logs" / f"phase_36_{condition_id.lower()}_{run_id}_terminal.log"
    if not terminal_log.is_file():
        raise RuntimeError(f"Phase 36 {condition_id} terminal log is missing: {terminal_log.relative_to(root)}")
    return {
        "phase_id": PHASE_ID,
        "sweep_id": SWEEP_ID,
        "condition_id": condition_id,
        "run_id": run_id,
        "status": "VERIFIED",
        "epochs_executed": len(history),
        "best_epoch": int(status["best_epoch"]),
        "runtime_seconds": float(history["epoch_seconds"].sum()) if "epoch_seconds" in history else 0.0,
        "checkpoint_size_bytes": paths["checkpoint"].stat().st_size,
        "stored_validation_metrics": stored_values,
        "recomputed_validation_metrics": recomputed_values,
        "metric_delta": abs(recomputed_values["rmse_wh"] - stored_values["rmse_wh"]),
        "prediction_max_abs_delta_wh": prediction_max_abs_delta,
        "n_samples": recomputed.n_samples,
        "population_fingerprint": recomputed.population_fingerprint,
        "strict_load_verification": "PASS",
        "strict_best_verification": "PASS",
        "config_verification": "PASS",
        "ffn_geometry": "PASS",
        "parameter_count": trainable_parameters,
        "parameter_monotonicity": geometry["parameter_monotonicity"],
        "optimizer_coverage": coverage["status"],
        "population_verification": "PASS",
        "test_access": TEST_ACCESS,
        "test_evidence": "ABSENT",
        "terminal_log": str(terminal_log.relative_to(root)),
        "terminal_log_sha256": sha256_file(terminal_log),
        "best_checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "training_history_sha256": sha256_file(paths["history"]),
        "best_validation_metrics_sha256": sha256_file(paths["metrics"]),
        "best_validation_predictions_sha256": sha256_file(paths["predictions"]),
        "verification_tolerance": tolerance,
    }


def verify_phase_36_f64_best(
    run_id: str = "RUN_TR_S14_0022_AA048302",
    project_root: Path | None = None,
) -> dict[str, Any]:
    return _verify_phase_36_trained_best("F64", run_id, project_root)


def verify_phase_36_f256_best(
    run_id: str = "RUN_TR_S14_0023_A711A9B8",
    project_root: Path | None = None,
) -> dict[str, Any]:
    return _verify_phase_36_trained_best("F256", run_id, project_root)


def materialize_phase_36_post_f64_log(
    verification: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    """Record the read-only F64 gate result in the canonical processing JSON log."""
    from course_work.reporting.phase_summary import build_phase_resume_log, save_phase_resume_log

    if verification.get("status") != "VERIFIED" or verification.get("test_access") != TEST_ACCESS:
        raise RuntimeError("Only a successful Validation-only F64 verification may update Phase 36 state")
    root = Path(project_root or get_project_root()).resolve()
    log = build_phase_resume_log(PHASE_ID, root, allow_execution=False)
    log["summary"]["F64 verification"] = "PASS"
    log["summary"]["F64 run"] = verification["run_id"]
    log["summary"]["F64 full-precision Validation RMSE Wh"] = verification[
        "recomputed_validation_metrics"
    ]["rmse_wh"]
    log["summary"]["F256 authorization gate"] = "READY"
    log["technical_details"]["f64_verification"] = verification
    return save_phase_resume_log(log, root)


def materialize_phase_36_final_log(project_root: Path | None = None) -> Path:
    """Materialize the finalized Phase 36 processing snapshot from canonical artifacts."""
    from course_work.reporting.phase_summary import build_phase_resume_log, save_phase_resume_log

    root = Path(project_root or get_project_root()).resolve()
    validation = validate_sweep_signoff(PHASE_ID, root)
    if not validation["valid"]:
        raise RuntimeError(f"Phase 36 sign-off is invalid: {validation['issues']}")
    signoff = validation["record"]
    winner = read_json(root / "artifacts/sweeps/S14_ffn/s14_ffn_winner.json")
    reference = read_json(root / "artifacts/sweeps/S14_ffn/s14_reference_update.json")
    f64 = verify_phase_36_f64_best(signoff["f64_run_id"], root)
    f256 = verify_phase_36_f256_best(signoff["f256_run_id"], root)
    log = build_phase_resume_log(PHASE_ID, root, allow_execution=False)
    log["status"] = signoff["status"]
    log["summary"].update(
        {
            "F64 verification": "PASS",
            "F64 run": f64["run_id"],
            "F64 Validation RMSE Wh": f64["stored_validation_metrics"]["rmse_wh"],
            "F128 reference run": signoff["f128_reference_run_id"],
            "F128 Validation RMSE Wh": 58.08190056355405,
            "F256 verification": "PASS",
            "F256 run": f256["run_id"],
            "F256 Validation RMSE Wh": f256["stored_validation_metrics"]["rmse_wh"],
            "Winner": winner["winner_ffn_id"],
            "Selected ffn_dim": winner["winner_ffn_dim"],
            "Phase 37 approved": reference["approved_for_phase37"],
            "Phase 37 policy": reference["phase37_condition_policy"],
            "Test access": TEST_ACCESS,
        }
    )
    log["result"] = {
        "winner": winner,
        "phase37_reference": reference,
        "status": signoff["status"],
    }
    log["technical_details"]["f64_verification"] = f64
    log["technical_details"]["f256_verification"] = f256
    log["technical_details"]["phase_36_signoff"] = signoff
    log["discrepancies"] = []
    return save_phase_resume_log(log, root)
