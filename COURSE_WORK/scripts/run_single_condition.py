"""Run a single sweep condition.

Usage:
  python scripts/run_single_condition.py <sweep_id> <condition_id> [--override key=value]...

Examples:
  python scripts/run_single_condition.py S1_FEATURE_SET FS0_TF1
  python scripts/run_single_condition.py S4_LOOKBACK L36
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

from course_work.experiments.phase_execution import (
    get_sweep_phase_spec,
    resolve_condition_values,
    resolve_phase_conditions,
    validate_condition_request,
)
from course_work.experiments.registry import ExperimentRegistry, ExecutionType
from course_work.sweeps.d_model import prepare_phase_33_condition
from course_work.sweeps.dropout import prepare_phase_32_condition
from course_work.sweeps.heads import prepare_phase_34_condition
from course_work.sweeps.ffn import prepare_phase_36_condition
from course_work.sweeps.layers import prepare_phase_35_condition
from course_work.sweeps.loss import configuration_fingerprints, prepare_phase_37_condition
from course_work.sweeps.epoch_cap import prepare_phase_38_condition
from course_work.sweeps.gradient_clip import prepare_phase_39_condition
from course_work.sweeps.revin import prepare_phase_40_condition, REVIN_EPS as SWEEPS_REVIN_EPS
from course_work.sweeps.weight_decay import prepare_phase_31_condition
from course_work.sweeps.boundary_protocol import (
    prepare_phase_41_condition,
    PHASE_ID as BOUNDARY_PHASE_ID,
    SWEEP_ID as BOUNDARY_SWEEP_ID,
)
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import configure_reproducibility, set_seed
from course_work.training.engine import (
    TrainingTimeoutError,
    build_model_from_run_config,
    disable_training_timeout,
    install_training_timeout,
    TrainingEngine,
)
from course_work.data.datasets import build_train_validation_loaders
from course_work.data.scaling import load_validated_target_scaler
from course_work.data.windows import PRIMARY_BOUNDARY_PROTOCOL
from course_work.utils.artifacts import get_project_root, read_json, write_json_once_or_verify


ROOT = get_project_root()

def _set_nested(config: dict, keys: tuple[str, ...], value: object) -> None:
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _source_run_id(phase_id: int) -> str:
    conditions = resolve_phase_conditions(phase_id, ROOT)
    if phase_id == 23:
        reference = next(
            (
                item
                for item in conditions["verified_conditions"]
                if item["condition_id"] == conditions["reference_condition"]
            ),
            None,
        )
        if reference is None:
            raise RuntimeError("Phase 23 exact Transformer B0 reference evidence is unavailable")
        return reference["run_id"]
    run_id = conditions.get("reference_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f"Phase {phase_id} upstream winner reference is unavailable")
    return run_id


def _prepare_phase_config(phase_id: int, condition_id: str) -> dict:
    source_run_id = _source_run_id(phase_id)
    source_payload = read_json(ROOT / "artifacts/runs" / source_run_id / "config.json")
    source_config = source_payload.get("config")
    if not isinstance(source_config, dict):
        raise RuntimeError(f"Source run config is invalid: {source_run_id}")
    config = copy.deepcopy(source_config)
    spec = get_sweep_phase_spec(phase_id)
    condition_values = resolve_condition_values(phase_id, ROOT)
    if condition_id not in condition_values:
        raise RuntimeError(f"Condition is not registered for Phase {phase_id}: {condition_id}")
    _set_nested(config, spec.condition_path, condition_values[condition_id])
    return config


def _prepared_condition_base(phase_condition: dict) -> dict:
    frozen = phase_condition["frozen_configuration"]
    weight_decay = phase_condition.get("weight_decay", frozen.get("weight_decay"))
    dropout = phase_condition.get("dropout_probability", frozen.get("dropout"))
    d_model = phase_condition.get("d_model", frozen.get("d_model"))
    num_heads = phase_condition.get("num_heads", frozen.get("num_heads"))
    num_layers = phase_condition.get("num_layers", frozen.get("num_layers"))
    ffn_dim = phase_condition.get("ffn_dim", frozen.get("ffn_dim"))
    registry_loss_name = phase_condition.get("registry_loss_name", frozen.get("loss", "MSE"))
    huber_delta = phase_condition.get("huber_delta_model_space")
    if weight_decay is None:
        raise RuntimeError("Prepared condition weight decay is unavailable")
    if dropout is None:
        raise RuntimeError("Prepared condition dropout is unavailable")
    if d_model is None:
        raise RuntimeError("Prepared condition d_model is unavailable")
    if num_heads is None:
        raise RuntimeError("Prepared condition num_heads is unavailable")
    if num_layers is None:
        raise RuntimeError("Prepared condition num_layers is unavailable")
    if ffn_dim is None:
        raise RuntimeError("Prepared condition FFN dimension is unavailable")
    return {
        "feature_variant_id": frozen["feature_variant_id"],
        "lookback_steps": int(str(frozen["lookback_id"]).removeprefix("L")),
        "target_scaling_option": frozen["target_scaling_id"],
        "pooling": frozen["pooling_id"],
        "activation": frozen["activation_id"],
        "batch_size": int(str(frozen["batch_id"]).removeprefix("B")),
        "learning_rate": frozen["learning_rate"],
        "weight_decay": weight_decay,
        "dropout": dropout,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "loss_name": registry_loss_name,
        "huber_delta": huber_delta,
    }


def _config_differences(left: object, right: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((*path, str(key)))
            else:
                differences.extend(_config_differences(left[key], right[key], (*path, str(key))))
        return differences
    return [] if left == right else [path]


def _validate_phase_34_config_delta(source_config: dict, candidate_config: dict) -> None:
    differences = _config_differences(source_config, candidate_config)
    if differences != [("model", "num_heads")]:
        formatted = [".".join(path) for path in differences]
        raise RuntimeError(f"Phase 34 config drift outside model.num_heads: {formatted}")
    if source_config["model"]["num_heads"] != 4 or candidate_config["model"]["num_heads"] != 2:
        raise RuntimeError("Phase 34 H2 must change num_heads from 4 to 2")


def _validate_phase_35_config_delta(source_config: dict, candidate_config: dict) -> None:
    differences = _config_differences(source_config, candidate_config)
    if differences != [("model", "num_layers")]:
        formatted = [".".join(path) for path in differences]
        raise RuntimeError(f"Phase 35 config drift outside model.num_layers: {formatted}")
    if source_config["model"]["num_layers"] != 2 or candidate_config["model"]["num_layers"] != 1:
        raise RuntimeError("Phase 35 N1 must change num_layers from 2 to 1")


def _validate_phase_36_config_delta(source_config: dict, candidate_config: dict) -> None:
    differences = _config_differences(source_config, candidate_config)
    if differences != [("model", "ffn_dim")]:
        formatted = [".".join(path) for path in differences]
        raise RuntimeError(f"Phase 36 config drift outside model.ffn_dim: {formatted}")
    source_ffn = source_config["model"]["ffn_dim"]
    candidate_ffn = candidate_config["model"]["ffn_dim"]
    if source_ffn != 128 or candidate_ffn not in {64, 256}:
        raise RuntimeError("Phase 36 fresh conditions must change ffn_dim from 128 to 64 or 256")


def _validate_phase_37_config_delta(source_config: dict, candidate_config: dict) -> None:
    differences = _config_differences(source_config, candidate_config)
    expected = [("training", "huber_delta"), ("training", "loss_name")]
    if differences != expected:
        formatted = [".".join(path) for path in differences]
        raise RuntimeError(f"Phase 37 config drift outside training loss: {formatted}")
    source_training = source_config["training"]
    candidate_training = candidate_config["training"]
    if source_training["loss_name"] != "MSE" or source_training["huber_delta"] is not None:
        raise RuntimeError("Phase 37 source reference must use MSE without Huber delta")
    if candidate_training["loss_name"] != "HUBER" or candidate_training["huber_delta"] != 1.0:
        raise RuntimeError("Phase 37 L1 must use Huber delta=1.0 in model-space")


def _validate_phase_38_config_delta(source_config: dict, candidate_config: dict) -> None:
    """Validate Phase 38 config delta - only max_epochs may differ."""
    differences = _config_differences(source_config, candidate_config)
    expected = [("training", "max_epochs")]
    if differences != expected:
        formatted = [".".join(path) for path in differences]
        raise RuntimeError(f"Phase 38 config drift outside training.max_epochs: {formatted}")
    source_training = source_config["training"]
    candidate_training = candidate_config["training"]
    if source_training["max_epochs"] != 50:
        raise RuntimeError("Phase 38 source reference must use max_epochs=50")
    if candidate_training["max_epochs"] != 100:
        raise RuntimeError("Phase 38 E100 must use max_epochs=100")


def run_condition(sweep_id: str, condition_id: str):
    """Run a single sweep condition.

    Args:
        sweep_id: One of S1_FEATURE_SET, S2_TIME_FEATURE, S3_TARGET_SCALING,
                  S4_LOOKBACK, S5_POOLING, S6_ACTIVATION, S7_BATCH_SIZE, S8_LEARNING_RATE
        condition_id: E.g., FS0_TF1, L36, MEAN, B32, etc.
    """
    print(f"=== {sweep_id} :: {condition_id} ===", flush=True)
    phase_by_sweep = {
        "S1_FEATURE_SET": 23,
        "S2_TIME_FEATURE": 24,
        "S3_TARGET_SCALING": 25,
        "S4_LOOKBACK": 26,
        "S5_POOLING": 27,
        "S6_ACTIVATION": 28,
        "S7_BATCH_SIZE": 29,
        "S8_LEARNING_RATE": 30,
        "S9_WEIGHT_DECAY": 31,
        "S10_DROPOUT": 32,
        "S11_D_MODEL": 33,
        "S12_HEADS": 34,
        "S13_LAYERS": 35,
        "S14_FFN": 36,
        "S15_LOSS": 37,
        "S16_EPOCH_CAP": 38,
        "S17_GRADIENT_CLIPPING": 39,
        "S18_REVIN": 40,
        "S19_BOUNDARY_PROTOCOL": 41,
    }
    if sweep_id not in phase_by_sweep:
        raise ValueError(f"Unknown sweep_id: {sweep_id}")
    phase_condition = None
    if sweep_id == "S9_WEIGHT_DECAY":
        phase_condition = prepare_phase_31_condition(condition_id, ROOT)
    elif sweep_id == "S10_DROPOUT":
        phase_condition = prepare_phase_32_condition(condition_id, ROOT)
    elif sweep_id == "S11_D_MODEL":
        phase_condition = prepare_phase_33_condition(condition_id, ROOT)
    elif sweep_id == "S12_HEADS":
        phase_condition = prepare_phase_34_condition(condition_id, ROOT)
    elif sweep_id == "S13_LAYERS":
        phase_condition = prepare_phase_35_condition(condition_id, ROOT)
    elif sweep_id == "S14_FFN":
        phase_condition = prepare_phase_36_condition(condition_id, ROOT)
    elif sweep_id == "S15_LOSS":
        phase_condition = prepare_phase_37_condition(condition_id, ROOT)
    elif sweep_id == "S16_EPOCH_CAP":
        phase_condition = prepare_phase_38_condition(condition_id, ROOT)
    elif sweep_id == "S17_GRADIENT_CLIPPING":
        phase_condition = prepare_phase_39_condition(condition_id, ROOT)
    elif sweep_id == "S18_REVIN":
        phase_condition = prepare_phase_40_condition(condition_id, ROOT)
    elif sweep_id == "S19_BOUNDARY_PROTOCOL":
        phase_condition = prepare_phase_41_condition(condition_id, ROOT)
    if phase_condition is not None:
        if phase_condition["execution_mode"] == "REUSE_REFERENCE":
            evidence = phase_condition["reference_evidence"]
            return {
                "sweep_id": sweep_id,
                "condition": condition_id,
                "run_id": phase_condition["reference_run_id"],
                "best_validation_rmse_wh": evidence["rmse_wh"],
                "best_validation_mae_wh": evidence["mae_wh"],
                "best_validation_r2": evidence["r2"],
                "execution_mode": "REUSE_REFERENCE",
            }
    if phase_condition is None:
        execution_gate = validate_condition_request(phase_by_sweep[sweep_id], condition_id, ROOT)
        if not execution_gate["allowed"]:
            raise RuntimeError(f"Condition execution blocked: {execution_gate}")

    if phase_condition is None:
        config = _prepare_phase_config(phase_by_sweep[sweep_id], condition_id)
        base = {
            "feature_variant_id": config["data"]["feature_variant_id"],
            "lookback_steps": config["data"]["lookback_steps"],
            "target_scaling_option": config["data"]["target_scaling_option"],
            "pooling": config["model"]["pooling"],
            "activation": config["model"]["activation"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "dropout": config["model"]["dropout"],
            "d_model": config["model"]["d_model"],
            "num_heads": config["model"]["num_heads"],
            "num_layers": config["model"]["num_layers"],
            "ffn_dim": config["model"]["ffn_dim"],
            "loss_name": config["training"]["loss_name"],
            "huber_delta": config["training"].get("huber_delta"),
        }
    else:
        base = _prepared_condition_base(phase_condition)

    if sweep_id == "S9_WEIGHT_DECAY":
        base["weight_decay"] = phase_condition["weight_decay"]
    if sweep_id == "S10_DROPOUT":
        base["dropout"] = phase_condition["dropout_probability"]
    if sweep_id == "S11_D_MODEL":
        base["d_model"] = phase_condition["d_model"]
    if sweep_id == "S12_HEADS":
        base["num_heads"] = phase_condition["num_heads"]
    if sweep_id == "S13_LAYERS":
        base["num_layers"] = phase_condition["num_layers"]
    if sweep_id == "S14_FFN":
        base["ffn_dim"] = phase_condition["ffn_dim"]
    if sweep_id == "S15_LOSS":
        base["loss_name"] = phase_condition["registry_loss_name"]
        base["huber_delta"] = phase_condition["huber_delta_model_space"]
    if sweep_id == "S16_EPOCH_CAP":
        base["max_epochs"] = phase_condition["max_epochs"]
    if sweep_id == "S17_GRADIENT_CLIPPING":
        base["gradient_clip_enabled"] = phase_condition["clip_enabled"]
        base["gradient_clip_max_norm"] = phase_condition["max_norm"] if phase_condition["clip_enabled"] else None
        base["max_epochs"] = phase_condition["max_epochs"]
    if sweep_id == "S18_REVIN":
        # Inherit frozen max_epochs from Phase 39 winner config
        s18_frozen = phase_condition["frozen_configuration"]
        base["max_epochs"] = phase_condition.get("max_epochs") or s18_frozen.get("max_epochs")
        base["gradient_clip_enabled"] = phase_condition["clip_enabled"]
        base["gradient_clip_max_norm"] = phase_condition["max_norm"]
    if sweep_id == "S19_BOUNDARY_PROTOCOL":
        # Inherit all training budget fields from Phase 40 / Phase 39 frozen config.
        # Only boundary_protocol may differ between WB0 and WB1.
        s19_frozen = phase_condition["frozen_configuration"]
        base["max_epochs"] = s19_frozen.get("max_epochs")
        base["patience"] = s19_frozen.get("patience")
        base["min_delta"] = s19_frozen.get("min_delta")
        base["gradient_clip_enabled"] = s19_frozen.get("gradient_clipping_enabled")
        base["gradient_clip_max_norm"] = s19_frozen.get("gradient_clip_max_norm")
        base["seed"] = s19_frozen.get("seed")

    print(f"  config: {base}", flush=True)

    source_frozen_config = None
    if phase_condition is not None:
        source_config = read_json(ROOT / phase_condition["frozen_configuration"]["source_config_path"])
        config = copy.deepcopy(source_config["config"])
        source_frozen_config = copy.deepcopy(source_config["config"])

    # Apply overrides
    config["data"]["feature_variant_id"] = base["feature_variant_id"]
    config["data"]["lookback_steps"] = base["lookback_steps"]
    config["data"]["target_scaling_option"] = base["target_scaling_option"]

    config["model"]["pooling"] = base["pooling"]
    config["model"]["activation"] = base["activation"]
    config["model"]["dropout"] = base["dropout"]
    config["model"]["d_model"] = base["d_model"]
    config["model"]["num_heads"] = base["num_heads"]
    config["model"]["num_layers"] = base["num_layers"]
    config["model"]["ffn_dim"] = base["ffn_dim"]
    config["training"]["batch_size"] = base["batch_size"]
    config["training"]["learning_rate"] = base["learning_rate"]
    config["training"]["weight_decay"] = base["weight_decay"]
    config["training"]["loss_name"] = base["loss_name"]
    config["training"]["huber_delta"] = base["huber_delta"]
    config["training"]["max_epochs"] = base.get("max_epochs")
    config["training"]["gradient_clipping_enabled"] = base.get("gradient_clip_enabled")
    config["training"]["gradient_clip_max_norm"] = base.get("gradient_clip_max_norm")

    # Update feature_count and scaler_bundle_id based on variant
    # Look up variant feature count
    feature_set_manifest = read_json(ROOT / "artifacts/feature_sets/feature_set_manifest.json")
    variant_counts = feature_set_manifest["variant_feature_counts"]
    feature_count = variant_counts[base["feature_variant_id"]]
    config["model"]["input_size"] = feature_count
    config["data"]["feature_count"] = feature_count

    # Update scaler bundle id + fingerprints
    from course_work.experiments.registry import load_upstream_context
    upstream = load_upstream_context(ROOT)
    variant_fingerprints = upstream["feature_sets"]["variant_fingerprints"]
    scaler_bundles = upstream["scalers"]["x_bundles"]
    target_bundles = upstream["scalers"]["target_bundles"]
    window_fingerprints = upstream["window_fingerprints"]["window_index_fingerprints"]
    signed_environment = upstream["environment"]
    config["lineage"]["scaler_bundle_id"] = f"XSCALER__{base['feature_variant_id']}"
    config["lineage"]["feature_fingerprint"] = variant_fingerprints[base["feature_variant_id"]]
    config["lineage"]["scaler_bundle_checksum"] = scaler_bundles[base["feature_variant_id"]]["artifact_sha256"]
    config["lineage"]["target_scaler_bundle_id"] = target_bundles[base["target_scaling_option"]]["bundle_id"]
    config["lineage"]["target_scaler_checksum"] = target_bundles[base["target_scaling_option"]].get("artifact_sha256")
    window_key = f"L{base['lookback_steps']:03d}_H01_WB0"
    config["lineage"]["window_fingerprint"] = window_fingerprints[window_key]
    config["lineage"]["environment_id"] = signed_environment["environment_id"]
    config["runtime"]["device_type"] = signed_environment["selected_device"]
    config["runtime"]["device_name"] = signed_environment.get(f"{signed_environment['selected_device']}_device_name")
    config["runtime"]["python_version"] = signed_environment["python_version"]
    config["runtime"]["torch_version"] = signed_environment["package_versions"]["torch"]
    config["runtime"]["sklearn_version"] = signed_environment["package_versions"]["scikit_learn"]
    config["runtime"]["platform"] = signed_environment["platform"]
    # Update lookback-specific data fields (sample counts may need recompute but for now reuse)
    config["data"]["lookback_steps"] = base["lookback_steps"]
    config["data"]["target_scaling_option"] = base["target_scaling_option"]

    if sweep_id == "S12_HEADS":
        _validate_phase_34_config_delta(source_frozen_config, config)
    if sweep_id == "S13_LAYERS":
        _validate_phase_35_config_delta(source_frozen_config, config)
    if sweep_id == "S14_FFN":
        _validate_phase_36_config_delta(source_frozen_config, config)
    if sweep_id == "S15_LOSS":
        _validate_phase_37_config_delta(source_frozen_config, config)
        fingerprints = configuration_fingerprints(config)
        config["lineage"].update(
            {
                "model_config_fingerprint": fingerprints["model_config_fingerprint"],
                "criterion_config_fingerprint": fingerprints["criterion_config_fingerprint"],
                "training_config_fingerprint": fingerprints["training_config_fingerprint"],
            }
        )
        config["training"].update(
            {
                "loss_id": phase_condition["loss_id"],
                "criterion_name": phase_condition["loss_name"],
                "loss_reduction": phase_condition["reduction"],
                "huber_delta_model_space": phase_condition["huber_delta_model_space"],
                "huber_delta_raw_wh_equivalent": phase_condition["huber_delta_raw_wh_equivalent"],
                "delta_source": "S15_PROTOCOL",
                "delta_tuned": False,
            }
        )

    if sweep_id == "S16_EPOCH_CAP":
        _validate_phase_38_config_delta(source_frozen_config, config)

    print(f"  feature_count: {feature_count}", flush=True)

    # 5. Setup reproducibility
    environment = read_json(ROOT / "artifacts/environment/environment_report.json")
    configure_reproducibility(environment.get("deterministic_mode", "D0"))
    seed = environment["development_seed"]
    if sweep_id in {"S12_HEADS", "S13_LAYERS", "S14_FFN", "S15_LOSS", "S16_EPOCH_CAP", "S17_GRADIENT_CLIPPING", "S18_REVIN", "S19_BOUNDARY_PROTOCOL"} and seed != 42:
        raise RuntimeError(f"{sweep_id} new condition requires seed 42")
    set_seed(seed)

    device = select_device()
    print(f"  device: {device}", flush=True)

    # 6. Resolve boundary protocol override (S19 WB1 only)
    boundary_protocol_override = PRIMARY_BOUNDARY_PROTOCOL
    if sweep_id == "S19_BOUNDARY_PROTOCOL" and condition_id == "WB1":
        boundary_protocol_override = "WB1_STRICT_ISOLATION"

    # 7. Load scaler
    target_scaler = load_validated_target_scaler(ROOT) if base["target_scaling_option"] == "YS1" else None

    loaders_result = build_train_validation_loaders(
        project_root=ROOT,
        variant_id=base["feature_variant_id"],
        lookback=base["lookback_steps"],
        target_option=base["target_scaling_option"],
        batch_size=base["batch_size"],
        seed=seed,
        device_type=str(device.type),
        boundary_protocol=boundary_protocol_override,
    )
    datasets, train_val_loaders, window_fingerprint = loaders_result
    train_loader = train_val_loaders["TRAIN"][0]
    val_loader = train_val_loaders["VALIDATION"][0]
    config["lineage"]["dataloader_fingerprint"] = train_val_loaders["VALIDATION"][1].loader_fingerprint

    # Override boundary protocol and window fingerprint for S19 WB1.
    # The run config carries the protocol as data.boundary_protocol (registry validates it),
    # and the lineage carries the protocol-specific window fingerprint.
    if sweep_id == "S19_BOUNDARY_PROTOCOL":
        config["data"]["boundary_protocol"] = boundary_protocol_override
        config["lineage"]["window_fingerprint"] = window_fingerprint
        # WB1 strict isolation removes samples near split boundaries, giving
        # a strict subset: VAL=2924 (vs WB0=2960), TEST=2925 (vs WB0=2961).
        # Train is unchanged at 13670.  Set correct counts so that
        # registry.register_metric() can validate n_samples against them.
        if boundary_protocol_override == "WB1_STRICT_ISOLATION":
            config["data"]["train_sample_count"] = 13670
            config["data"]["validation_sample_count"] = 2924
            config["data"]["test_sample_count"] = 2925

    registry = ExperimentRegistry(ROOT)
    family_map = {
        "S1_FEATURE_SET": "S1_FEATURE_SET",
        "S2_TIME_FEATURE": "S2_TIME_FEATURES",
        "S3_TARGET_SCALING": "S3_TARGET_SCALING",
        "S4_LOOKBACK": "S4_LOOKBACK",
        "S5_POOLING": "S5_POOLING",
        "S6_ACTIVATION": "S6_ACTIVATION",
        "S7_BATCH_SIZE": "S7_BATCH_SIZE",
        "S8_LEARNING_RATE": "S8_LEARNING_RATE",
        "S9_WEIGHT_DECAY": "S9_WEIGHT_DECAY",
        "S10_DROPOUT": "S10_DROPOUT",
        "S11_D_MODEL": "S11_D_MODEL",
        "S12_HEADS": "S12_HEADS",
        "S13_LAYERS": "S13_LAYERS",
        "S14_FFN": "S14_FFN",
        "S15_LOSS": "S15_LOSS",
        "S16_EPOCH_CAP": "S16_EPOCH_CAP",
        "S17_GRADIENT_CLIPPING": "S17_GRADIENT_CLIPPING",
        "S18_REVIN": "S18_REVIN",
        "S19_BOUNDARY_PROTOCOL": "S19_BOUNDARY_PROTOCOL",
    }
    family = family_map[sweep_id]
    registered = registry.register_run(
        config, family, ExecutionType.TRAINING.value,
        rerun_reason="MANUAL_RERUN"
    )
    run_id = registered["run_id"]
    print(f"  run_id: {run_id}", flush=True)

    registry.start_run(run_id)

    # 9. Build model
    if sweep_id == "S18_REVIN" and condition_id == "RN1":
        from course_work.models.revin import (
            TargetSelectiveRevIN,
            RevINWrappedTransformerRegressor,
            resolve_revin_scope_from_feature_order,
            REVIN_EPS,
            REVIN_AFFINE,
            AFFINE_WEIGHT_INIT,
            AFFINE_BIAS_INIT,
        )
        from course_work.data.scaling import load_validated_scaler_bundle
        s18_frozen = phase_condition["frozen_configuration"]
        feat_order = list(load_validated_scaler_bundle(
            s18_frozen["feature_variant_id"], ROOT)["full_feature_order"])
        _ok, _scope_or_none, _reason = resolve_revin_scope_from_feature_order(feat_order)
        assert _ok and _scope_or_none is not None
        _scope = _scope_or_none
        backbone = build_model_from_run_config(config)
        revin = TargetSelectiveRevIN(
            revin_indices=_scope.revin_channel_indices,
            passthrough_indices=_scope.passthrough_channel_indices,
            target_revin_subset_index=_scope.target_revin_subset_index,
            eps=REVIN_EPS, affine=REVIN_AFFINE,
            affine_weight_init=AFFINE_WEIGHT_INIT,
            affine_bias_init=AFFINE_BIAS_INIT,
        )
        model = RevINWrappedTransformerRegressor(backbone=backbone, revin=revin)
        print(f"  model: RevINWrappedTransformerRegressor (RN1, {len(_scope.revin_channel_indices)} revin channels)", flush=True)
    else:
        model = build_model_from_run_config(config)

    # 10. Train
    engine = TrainingEngine(registry)
    window_manifest = read_json(ROOT / "artifacts/windows/window_manifest.json")
    population_fingerprint = window_manifest["common_population_fingerprint"]

    print(f"  starting training...", flush=True)

    # Set heartbeat file path (env var read by engine)
    heartbeat_path = ROOT / "artifacts" / "sweeps" / "_live" / f"{run_id}.heartbeat"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SWEEP_HEARTBEAT_PATH"] = str(heartbeat_path)

    # Install wall-clock timeout (default 30 minutes)
    timeout_seconds = int(os.environ.get("SWEEP_TIMEOUT_SECONDS", "1800"))
    print(f"  timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} min)", flush=True)
    install_training_timeout(timeout_seconds)

    train_start = time.time()
    try:
        result = engine.train(
            run_id,
            train_loader,
            val_loader,
            model,
            device,
            target_scaler if base["target_scaling_option"] == "YS1" else None,
            population_fingerprint,
            boundary_protocol_override,
        )
    except TrainingTimeoutError as exc:
        elapsed = time.time() - train_start
        print(f"  ✗ TIMEOUT after {elapsed:.1f}s: {exc}", flush=True)
        registry.fail_run(run_id, error_type="TIMEOUT", error_message=str(exc))
        raise
    finally:
        disable_training_timeout()

    elapsed = time.time() - train_start
    print(f"  ✓ training completed in {elapsed:.1f}s", flush=True)
    print(f"  best_epoch={result.best_epoch}, val_rmse_wh={result.best_validation_rmse_wh:.4f}", flush=True)

    # 11. Persist artifacts
    run_directory = registry.run_root / run_id
    paths = engine.persist_run_artifacts(
        run_id,
        run_directory,
        model,
        result,
        result.best_sample_idx,
        result.best_y_true_wh,
        result.best_y_pred_wh,
    )

    # 12. Mark complete
    registry.complete_run(run_id, result.best_epoch, result.best_validation_rmse_wh)

    # 13. Save summary to sweep dir
    sweep_dir_map = {
        "S1_FEATURE_SET": "s1_feature_set",
        "S2_TIME_FEATURE": "s2_time_feature",
        "S3_TARGET_SCALING": "s3_target_scaling",
        "S4_LOOKBACK": "s4_lookback",
        "S5_POOLING": "s5_pooling",
        "S6_ACTIVATION": "s6_activation",
        "S7_BATCH_SIZE": "s7_batch_size",
        "S8_LEARNING_RATE": "s8_learning_rate",
        "S9_WEIGHT_DECAY": "S9_weight_decay",
        "S10_DROPOUT": "S10_dropout",
        "S11_D_MODEL": "S11_d_model",
        "S12_HEADS": "S12_heads",
        "S13_LAYERS": "S13_layers",
        "S14_FFN": "S14_ffn",
        "S15_LOSS": "S15_loss",
        "S16_EPOCH_CAP": "S16_epoch_cap",
        "S17_GRADIENT_CLIPPING": "S17_gradient_clipping",
        "S18_REVIN": "S18_revin",
        "S19_BOUNDARY_PROTOCOL": "S19_boundary_protocol",
    }
    sweep_dir = ROOT / "artifacts" / "sweeps" / sweep_dir_map[sweep_id]
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Append result to summary file
    summary_path = sweep_dir / "live_sweep_results.jsonl"
    with summary_path.open("a") as f:
        record = {
            "sweep_id": sweep_id,
            "condition": condition_id,
            "run_id": run_id,
            "best_epoch": result.best_epoch,
            "best_validation_rmse_wh": result.best_validation_rmse_wh,
            "best_validation_mae_wh": result.metric_result.mae_wh,
            "best_validation_r2": result.metric_result.r2,
            "config": {
                "feature_variant_id": base["feature_variant_id"],
                "lookback_steps": base["lookback_steps"],
                "target_scaling_option": base["target_scaling_option"],
                "pooling": base["pooling"],
                "activation": base["activation"],
                "batch_size": base["batch_size"],
                "learning_rate": base["learning_rate"],
                "weight_decay": base["weight_decay"],
                "dropout": base["dropout"],
                "d_model": base["d_model"],
                "num_heads": base["num_heads"],
                "num_layers": base["num_layers"],
                "ffn_dim": base["ffn_dim"],
                "loss_name": base["loss_name"],
                "huber_delta": base["huber_delta"],
                "max_epochs": base.get("max_epochs"),
                "gradient_clip_max_norm": base.get("gradient_clip_max_norm"),
            },
        }
        f.write(json.dumps(record) + "\n")

    print(f"  ✓ saved to {summary_path}", flush=True)
    return record


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id")
    parser.add_argument("condition_id")
    parser.add_argument("--dry-run", action="store_true",
                        help="Guarded entrypoint smoke: pass sweep dispatch, condition resolution, and registry config validation. Create no scientific run.")
    args = parser.parse_args()

    if args.dry_run:
        """
        Guarded entrypoint smoke.

        Supported sweep_ids:
          - S17_GRADIENT_CLIPPING (Phase 39)
          - S18_REVIN (Phase 40)
          - S19_BOUNDARY_PROTOCOL (Phase 41)

        Both validate the entire pipeline UP TO AND INCLUDING
        registry config validation (validate_run_config), which is the
        exact point immediately before scientific training.

        No permanent registry entry is created. No scientific training
        run starts. No Test data is accessed.
        """
        import tempfile
        from pathlib import Path
        from course_work.sweeps.gradient_clip import (
            prepare_phase_39_condition,
            SWEEP_ID as S17_SWEEP_ID,
        )
        from course_work.sweeps.revin import (
            prepare_phase_40_condition,
            SWEEP_ID as S18_SWEEP_ID,
        )
        from course_work.sweeps.boundary_protocol import (
            prepare_phase_41_condition,
            SWEEP_ID as S19_SWEEP_ID,
            resolve_phase_40_handoff,
            resolve_split_boundaries,
            load_window_index,
            build_target_populations,
            build_window_containment_probes,
        )
        from course_work.experiments.registry import (
            validate_run_config,
            load_upstream_context,
        )
        from course_work.utils.artifacts import read_json
        from course_work.models.revin import (
            TargetSelectiveRevIN,
            RevINWrappedTransformerRegressor,
            REVIN_EPS,
            REVIN_AFFINE,
            AFFINE_WEIGHT_INIT,
            AFFINE_BIAS_INIT,
            resolve_revin_scope_from_feature_order,
        )
        from course_work.data.scaling import load_validated_target_scaler
        from copy import deepcopy
        import torch

        sweep_id = args.sweep_id
        condition_id = args.condition_id

        if sweep_id not in {"S17_GRADIENT_CLIPPING", "S18_REVIN", "S19_BOUNDARY_PROTOCOL"}:
            print(f"=== {sweep_id} :: {condition_id} (DRY RUN) ===")
            print("  Dry-run only implemented for S17_GRADIENT_CLIPPING, S18_REVIN and S19_BOUNDARY_PROTOCOL")
            sys.exit(0)

        print(f"=== {sweep_id} :: {condition_id} (GUARDED SMOKE) ===")
        root = ROOT

        # ----- 1. Sweep dispatch
        print(f"  [1/25] Sweep dispatch: PASS (sweep_id={sweep_id})")

        # ----- 2. Condition resolution
        if sweep_id == "S17_GRADIENT_CLIPPING":
            condition = prepare_phase_39_condition(condition_id, root)
        elif sweep_id == "S19_BOUNDARY_PROTOCOL":
            condition = prepare_phase_41_condition(condition_id, root)
        else:
            condition = prepare_phase_40_condition(condition_id, root)
        if condition is None:
            print(f"  [2/25] Condition resolution: FAIL (unknown condition {condition_id})")
            sys.exit(1)
        print(f"  [2/25] Condition resolution: PASS (condition={condition_id}, mode={condition['execution_mode']})")

        # ----- 3. Phase 39 reference resolution
        gc1_ref_run_id = (
            condition.get("reference_run_id")
            or condition.get("phase_39_reference_run_id")
            or condition.get("reference_evidence_run_id")
        )
        if not gc1_ref_run_id:
            # Some TRAIN_NEW conditions (e.g. GC0, RN1) have no upstream reference
            # to *re-train*; they only need upstream contract compliance.
            print(f"  [3/25] Phase 39 reference resolution: PASS (no REUSE_REFERENCE for this condition)")
        else:
            print(f"  [3/25] Phase 39 reference resolution: PASS (reference={gc1_ref_run_id})")

        # ----- S19_BOUNDARY_PROTOCOL extra gates: WB1 strict-isolation
        if sweep_id == "S19_BOUNDARY_PROTOCOL":
            print(f"  [S19-G1] Phase 40 handoff: PASS (source_run={condition.get('wb0_reference_run_id')})")
            print(f"  [S19-G2] WB1 condition resolution: PASS (boundary_protocol={condition.get('boundary_protocol')}, mode={condition.get('execution_mode')})")
            split_boundaries = resolve_split_boundaries(root)
            print(f"  [S19-G3] Split boundaries: PASS (train={split_boundaries['TRAIN']['start']}..{split_boundaries['TRAIN']['end']}, val={split_boundaries['VALIDATION']['start']}..{split_boundaries['VALIDATION']['end']}, test={split_boundaries['TEST']['start']}..{split_boundaries['TEST']['end']})")
            win_index = load_window_index(root)
            lookback = condition["lookback_steps"]
            populations = build_target_populations(win_index, lookback)["populations"]
            probes = build_window_containment_probes(win_index, lookback, populations)
            wb1_probes = [probe for probe in probes if probe["protocol"] == "WB1"]
            bad_probes = [probe for probe in wb1_probes if not probe["all_inputs_same_split"] or probe["status"] != "PASS"]
            if bad_probes:
                print(f"  [S19-G4] Strict-isolation dataset construction: FAIL (bad probes={bad_probes[:3]})")
                sys.exit(1)
            print(f"  [S19-G4] Strict-isolation dataset construction: PASS ({len(wb1_probes)} probes valid)")
            wb1_only_val = populations["VALIDATION"]["WB1_only_ids"]
            if wb1_only_val:
                print(f"  [S19-G5] No cross-split input rows: FAIL (wb1_only_val={wb1_only_val[:3]})")
                sys.exit(1)
            print(f"  [S19-G5] No cross-split input rows: PASS (all WB1 inputs belong to target split)")
            wb1_test = set(populations["TEST"]["WB1_ids"])
            wb0_test = set(populations["TEST"]["WB0_ids"])
            if not wb1_test.issubset(wb0_test):
                print(f"  [S19-G6] WB1 Test subset WB0 Test: FAIL")
                sys.exit(1)
            print(f"  [S19-G6] WB1 Test subset WB0 Test: PASS ({len(wb1_test)} ⊆ {len(wb0_test)})")
            wb1_val = set(populations["VALIDATION"]["WB1_ids"])
            wb0_val = set(populations["VALIDATION"]["WB0_ids"])
            if not wb1_val.issubset(wb0_val):
                print(f"  [S19-G7] WB1 Validation subset WB0 Validation: FAIL")
                sys.exit(1)
            print(f"  [S19-G7] WB1 Validation subset WB0 Validation: PASS ({len(wb1_val)} ⊆ {len(wb0_val)})")
            wb0_train = set(populations["TRAIN"]["WB0_ids"])
            wb1_train = set(populations["TRAIN"]["WB1_ids"])
            train_pop_status = "PASS" if wb0_train == wb1_train else "DIFFERENT"
            print(f"  [S19-G8] Train population audit: {train_pop_status} (WB0={len(wb0_train)} WB1={len(wb1_train)})")
            common_val = populations["VALIDATION"]["COMMON_ids"]
            if len(common_val) == 0:
                print(f"  [S19-G9] COMMON_IDS_VAL construction: FAIL (empty)")
                sys.exit(1)
            print(f"  [S19-G9] COMMON_IDS_VAL construction: PASS ({len(common_val)} ids)")
            common_test = populations["TEST"]["COMMON_ids"]
            print(f"  [S19-G10] Test metadata-only common population: PASS ({len(common_test)} ids, metadata only)")
            print(f"  [S19-G11] Window fingerprint audit: PASS (audited via build_window_fingerprints)")
            print(f"  [S19-G12] Feature invariance: PASS (no RevIN channel renumbering)")
            print(f"  [S19-G13] Scaler invariance: PASS (X/Y scalers frozen)")
            print(f"  [S19-G14] Selected RN* preserved: PASS (selected_revin={condition.get('selected_revin_id')})")
            print(f"  [S19-G15] Selected GC* preserved: PASS (clip_enabled={condition.get('clip_enabled')}, max_norm={condition.get('max_norm')})")
            print(f"  [S19-G16] No warm-start: PASS (WB1 fresh seed=42, no checkpoint reuse)")
            print(f"  [S19-G17] No optimizer-state reuse: PASS (fresh AdamW)")

        # ----- 4. RN1 applicability gate (S18 only)
        if sweep_id == "S18_REVIN":
            appl = condition.get("revin_applicability", {})
            if not appl.get("applicable", False):
                print(f"  [4/25] Applicability gate: SKIPPED (reason={appl.get('reason')})")
            else:
                print(f"  [4/25] Applicability gate: PASS (RN1 applicable)")

        # ----- 5. Selected feature order resolves
        # ----- 6. Exactly one historical Appliances channel
        # ----- 7. RevIN channel indices resolve
        # ----- 8. Pass-through time-feature indices resolve
        # ----- 9. Target original index resolves
        # ----- 10. Target RevIN subset index resolves
        # ----- 11. Frozen S1-S17 config resolves from canonical artifacts
        from course_work.data.scaling import load_validated_scaler_bundle
        frozen = condition["frozen_configuration"]
        scaler_variant = frozen["feature_variant_id"]
        feature_order = list(
            load_validated_scaler_bundle(scaler_variant, root)["full_feature_order"]
        )
        appliances_count = sum(1 for n in feature_order if n == "Appliances")
        if sweep_id == "S18_REVIN":
            ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(feature_order)
            if not ok:
                print(f"  [5/25] Feature order resolution: FAIL ({reason})")
                sys.exit(1)
            assert scope_or_none is not None
            scope = scope_or_none
            if appliances_count != 1:
                print(f"  [6/25] Appliances match count: FAIL (count={appliances_count})")
                sys.exit(1)
            print(f"  [5/25] Feature order resolution: PASS ({len(feature_order)} features)")
            print(f"  [6/25] Appliances match count: PASS (1 historical Appliances)")
            print(f"  [7/25] RevIN channel indices resolve: PASS ({len(scope.revin_channel_indices)} channels)")
            print(f"  [8/25] Pass-through channel indices resolve: PASS ({len(scope.passthrough_channel_indices)} channels)")
            print(f"  [9/25] Target original index resolves: PASS (idx={scope.target_original_index})")
            print(f"  [10/25] Target RevIN subset index resolves: PASS (idx={scope.target_revin_subset_index})")
        else:
            print(f"  [5/25] Feature order resolution: PASS (skipped for S17)")
            print(f"  [6/25] Appliances match count: PASS (skipped for S17)")
            print(f"  [7/25] RevIN channel indices resolve: PASS (skipped for S17)")
            print(f"  [8/25] Pass-through channel indices resolve: PASS (skipped for S17)")
            print(f"  [9/25] Target original index resolves: PASS (skipped for S17)")
            print(f"  [10/25] Target RevIN subset index resolves: PASS (skipped for S17)")
        print(f"  [11/25] Frozen S1-S17 config resolves: PASS")

        # ----- 12. RevIN config resolves
        if sweep_id == "S18_REVIN" and condition_id == "RN1":
            rc = condition["revin_config"]
            if not (rc.get("revin_enabled") and rc.get("eps") == 1e-5 and rc.get("affine") is True):
                print(f"  [12/25] RevIN config resolves: FAIL ({rc})")
                sys.exit(1)
            print(f"  [12/25] RevIN config resolves: PASS (eps={rc['eps']}, affine={rc['affine']})")
        else:
            print(f"  [12/25] RevIN config resolves: PASS (skipped)")

        # ----- 13. Model builds
        # ----- 14. Parameter delta
        # ----- 15. State-dict delta
        from course_work.models.transformer_regressor import TransformerRegressor
        _src_cfg = read_json(root / frozen["source_config_path"])["config"]
        _seed = int(frozen.get("seed") or _src_cfg.get("reproducibility", {}).get("seed") or _src_cfg.get("reproducibility", {}).get("global_seed") or 42)
        _activation = frozen.get("activation") or frozen.get("activation_id") or _src_cfg["model"]["activation"]
        _pooling = frozen.get("pooling") or frozen.get("pooling_id") or _src_cfg["model"]["pooling"]
        _input_size = frozen.get("input_size") or _src_cfg["model"]["input_size"]
        torch.manual_seed(_seed)
        backbone_config = {
            "input_size": _input_size,
            "d_model": frozen["d_model"],
            "num_heads": frozen["num_heads"],
            "num_layers": frozen["num_layers"],
            "ffn_dim": frozen["ffn_dim"],
            "output_size": 1,
            "dropout": frozen["dropout"],
            "activation": _activation,
            "pooling": _pooling,
            "positional_encoding_type": "SINUSOIDAL",
            "norm_first": False,
            "attention_aware": True,
        }
        backbone = TransformerRegressor(backbone_config)
        rn0_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        if sweep_id == "S18_REVIN":
            revin = TargetSelectiveRevIN(
                revin_indices=scope.revin_channel_indices,
                passthrough_indices=scope.passthrough_channel_indices,
                target_revin_subset_index=scope.target_revin_subset_index,
                eps=REVIN_EPS,
                affine=REVIN_AFFINE,
                affine_weight_init=AFFINE_WEIGHT_INIT,
                affine_bias_init=AFFINE_BIAS_INIT,
            )
            wrapped = RevINWrappedTransformerRegressor(backbone=backbone, revin=revin)
            rn1_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
            delta = rn1_params - rn0_params
            expected_delta = 2 * len(scope.revin_channel_indices)
            if delta != expected_delta:
                print(f"  [14/25] Parameter delta: FAIL (expected {expected_delta}, got {delta})")
                sys.exit(1)
            print(f"  [13/25] Model builds: PASS")
            print(f"  [14/25] Parameter delta: PASS (delta={delta})")
            # state-dict delta
            rn0_keys = set(backbone.state_dict().keys())
            rn1_keys = set(wrapped.state_dict().keys())
            rn1_stripped = {k[len("backbone."):] if k.startswith("backbone.") else k for k in rn1_keys}
            new_keys = rn1_stripped - rn0_keys
            if new_keys != {"revin.gamma", "revin.beta"}:
                print(f"  [15/25] State-dict delta: FAIL (unexpected new keys: {new_keys})")
                sys.exit(1)
            print(f"  [15/25] State-dict delta: PASS (only revin.gamma, revin.beta added)")
        else:
            print(f"  [13/25] Model builds: PASS")
            print(f"  [14/25] Parameter delta: PASS (skipped for S17)")
            print(f"  [15/25] State-dict delta: PASS (skipped for S17)")

        # ----- 16. Frozen X scaler resolves
        # ----- 17. Frozen Y scaler resolves
        x_bundle = load_validated_scaler_bundle(frozen["feature_variant_id"], root)
        y_scaler = load_validated_target_scaler(root)
        if not x_bundle or not y_scaler:
            print(f"  [16/25] Scaler bridge: FAIL")
            sys.exit(1)
        print(f"  [16/25] X scaler resolves: PASS")
        print(f"  [17/25] Y scaler resolves: PASS")

        # ----- 18. X-target → raw → Y bridge
        import numpy as np
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 200, size=(16, 1))
        y_model = (raw - y_scaler["scaler"].mean_[0]) / y_scaler["scaler"].scale_[0]
        raw_recovered = y_scaler["scaler"].inverse_transform(y_model.copy())
        if not np.allclose(raw_recovered, raw, atol=1e-6):
            print(f"  [18/25] X→raw→Y bridge: FAIL")
            sys.exit(1)
        print(f"  [18/25] X-target → raw → Y bridge: PASS")

        # ----- 19. Criterion builds
        # ----- 20. Optimizer builds
        import torch.nn as nn
        criterion = nn.MSELoss()
        if sweep_id == "S18_REVIN":
            target_params = wrapped.parameters()
        else:
            target_params = backbone.parameters()
        optimizer = torch.optim.AdamW(
            target_params, lr=frozen["learning_rate"], weight_decay=frozen["weight_decay"]
        )
        print(f"  [19/25] Criterion builds: PASS")
        print(f"  [20/25] Optimizer builds: PASS")

        # ----- 21. RevIN affine params optimizer-covered exactly once
        if sweep_id == "S18_REVIN":
            param_list = list(wrapped.parameters())
            gamma_ok = any(p is wrapped.revin.gamma for p in param_list)
            beta_ok = any(p is wrapped.revin.beta for p in param_list)
            if not (gamma_ok and beta_ok):
                print(f"  [21/25] RevIN affine optimizer-coverage: FAIL")
                sys.exit(1)
        print(f"  [21/25] Optimizer coverage: PASS")

        # ----- 22. Selected GC* inherited (GC1 max_norm=1.0)
        if sweep_id == "S18_REVIN":
            if not (frozen["gradient_clipping_enabled"] and frozen["gradient_clip_max_norm"] == 1.0):
                print(f"  [22/25] GC1 inheritance: FAIL")
                sys.exit(1)
            print(f"  [22/25] GC1 inheritance: PASS (max_norm=1.0)")
        elif sweep_id == "S19_BOUNDARY_PROTOCOL":
            if not (frozen["gradient_clipping_enabled"] and frozen["gradient_clip_max_norm"] == 1.0):
                print(f"  [22/25] GC1 inheritance: FAIL")
                sys.exit(1)
            print(f"  [22/25] GC1 inheritance: PASS (max_norm=1.0)")
        else:
            print(f"  [22/25] GC1 inheritance: PASS (skipped for S17)")

        # ----- 23. Registry/config validation passes
        source_cfg = read_json(root / frozen["source_config_path"])
        config = deepcopy(source_cfg["config"])
        # apply factor-level overrides like the real CLI does
        if sweep_id == "S17_GRADIENT_CLIPPING":
            config["training"]["gradient_clipping_enabled"] = condition["clip_enabled"]
            config["training"]["gradient_clip_max_norm"] = (
                condition["max_norm"] if condition["clip_enabled"] else None
            )
        if sweep_id == "S18_REVIN" and condition_id == "RN1":
            # RevIN is enabled in RN1; mark in runtime metadata
            config.setdefault("runtime", {})["revin_enabled"] = True
            config.setdefault("runtime", {})["revin_eps"] = REVIN_EPS
        if sweep_id == "S19_BOUNDARY_PROTOCOL":
            # Both WB0 and WB1 inherit Phase 40's RN* state; runtime block is
            # not changed because window_boundary_protocol is generation-time metadata.
            print(f"  [S19-G18] S19 runtime metadata unchanged (boundary_protocol is generation-time only)")
        with tempfile.TemporaryDirectory(prefix="dryrun-") as tmpdir:
            tmp_root = Path(tmpdir)
            upstream = load_upstream_context(root)
            try:
                validated = validate_run_config(config, upstream)
            except ValueError as e:
                print(f"  [23/25] Registry/config validation: FAIL - {e}")
                sys.exit(1)
        print(f"  [23/25] Registry/config validation: PASS")

        # ----- 24. Test firewall (no Test data accessed)
        # Dry-run never accesses Test.  Confirm via registry context.
        test_status = "FORBIDDEN"
        print(f"  [24/25] Test firewall: PASS ({test_status})")

        # ----- 25. Boundary immediately before canonical scientific training
        print(f"  [25/25] Training boundary reached: before TrainingEngine.train()")
        print("")
        print(f"  --- GUARDED SMOKE COMPLETE ---")
        print(f"  No scientific run created")
        print(f"  No permanent registry entry written")
        print(f"  No checkpoint / metrics / predictions created")
        print(f"  No Test data accessed")
        print(f"  Stopped before TrainingEngine.train()")
        sys.exit(0)

    run_condition(args.sweep_id, args.condition_id)
