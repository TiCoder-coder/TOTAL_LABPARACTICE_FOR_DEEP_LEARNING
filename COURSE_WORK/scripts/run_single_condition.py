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
from course_work.sweeps.weight_decay import prepare_phase_31_condition
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
    if weight_decay is None:
        raise RuntimeError("Prepared condition weight decay is unavailable")
    if dropout is None:
        raise RuntimeError("Prepared condition dropout is unavailable")
    if d_model is None:
        raise RuntimeError("Prepared condition d_model is unavailable")
    if num_heads is None:
        raise RuntimeError("Prepared condition num_heads is unavailable")
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
    }


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

    print(f"  config: {base}", flush=True)

    if phase_condition is not None:
        source_config = read_json(ROOT / phase_condition["frozen_configuration"]["source_config_path"])
        config = copy.deepcopy(source_config["config"])

    # Apply overrides
    config["data"]["feature_variant_id"] = base["feature_variant_id"]
    config["data"]["lookback_steps"] = base["lookback_steps"]
    config["data"]["target_scaling_option"] = base["target_scaling_option"]
    config["model"]["pooling"] = base["pooling"]
    config["model"]["activation"] = base["activation"]
    config["model"]["dropout"] = base["dropout"]
    config["model"]["d_model"] = base["d_model"]
    config["model"]["num_heads"] = base["num_heads"]
    config["training"]["batch_size"] = base["batch_size"]
    config["training"]["learning_rate"] = base["learning_rate"]
    config["training"]["weight_decay"] = base["weight_decay"]

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

    print(f"  feature_count: {feature_count}", flush=True)

    # 5. Setup reproducibility
    environment = read_json(ROOT / "artifacts/environment/environment_report.json")
    configure_reproducibility(environment.get("deterministic_mode", "D0"))
    seed = environment["development_seed"]
    set_seed(seed)

    device = select_device()
    print(f"  device: {device}", flush=True)

    # 6. Load scaler
    target_scaler = load_validated_target_scaler(ROOT) if base["target_scaling_option"] == "YS1" else None

    loaders = build_train_validation_loaders(
        project_root=ROOT,
        variant_id=base["feature_variant_id"],
        lookback=base["lookback_steps"],
        target_option=base["target_scaling_option"],
        batch_size=base["batch_size"],
        seed=seed,
        device_type=str(device.type),
    )
    train_loader = loaders["TRAIN"][0]
    val_loader = loaders["VALIDATION"][0]
    config["lineage"]["dataloader_fingerprint"] = loaders["VALIDATION"][1].loader_fingerprint

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
            },
        }
        f.write(json.dumps(record) + "\n")

    print(f"  ✓ saved to {summary_path}", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_id")
    parser.add_argument("condition_id")
    args = parser.parse_args()
    run_condition(args.sweep_id, args.condition_id)
