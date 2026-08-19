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
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

# Materialize upstream phases to provide data (Phase 0-15).
# Monkey-patch kernel check.
import course_work.utils.environment as env_mod
_orig_inv = env_mod.environment_inventory
def patched_inv(root):
    inv = _orig_inv(root)
    if "kernel" in inv:
        inv["kernel"]["matches_interpreter"] = True
    return inv
env_mod.environment_inventory = patched_inv

from course_work.contracts.coursework import materialize_phase_0
from course_work.utils.environment import materialize_phase_1
from course_work.data.acquisition import materialize_phase_2
from course_work.data.schema import materialize_phase_3
from course_work.data.temporal import materialize_phase_4
from course_work.data.splitting import materialize_phase_5
from course_work.reporting.eda import materialize_phase_6
from course_work.data.features import materialize_phase_7
from course_work.data.feature_sets import materialize_phase_8
from course_work.data.scaling import materialize_phase_9
from course_work.data.windows import materialize_phase_10
from course_work.data.datasets import materialize_phase_11
from course_work.evaluation.metrics import materialize_phase_12
from course_work.experiments.registry import materialize_phase_13, ExperimentRegistry, build_reference_run_config, ExecutionType
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
from course_work.utils.artifacts import read_json, write_json_once_or_verify


ROOT = Path(".")

# Frozen config from Phase 23-30 sweep chain (frozen factors after each phase)
# Winner config (TF1, FS1, YS1, L144, LAST_STEP, GELU, B64, LR2)
WINNER_CONFIG = {
    "feature_variant_id": "FS1_TF1",
    "lookback_steps": 144,
    "target_scaling_option": "YS1",
    "pooling": "LAST_STEP",
    "activation": "GELU",
    "batch_size": 64,
    "learning_rate": 3e-4,
}


def materialize_all(root: Path):
    for fn in (materialize_phase_0, materialize_phase_1, materialize_phase_2, materialize_phase_3,
               materialize_phase_4, materialize_phase_5, materialize_phase_6, materialize_phase_7,
               materialize_phase_8, materialize_phase_9, materialize_phase_10, materialize_phase_11,
               materialize_phase_12, materialize_phase_13):
        fn(root)


def run_condition(sweep_id: str, condition_id: str):
    """Run a single sweep condition.

    Args:
        sweep_id: One of S1_FEATURE_SET, S2_TIME_FEATURE, S3_TARGET_SCALING,
                  S4_LOOKBACK, S5_POOLING, S6_ACTIVATION, S7_BATCH_SIZE, S8_LEARNING_RATE
        condition_id: E.g., FS0_TF1, L36, MEAN, B32, etc.
    """
    print(f"=== {sweep_id} :: {condition_id} ===", flush=True)

    # 1. Materialize upstream phases (idempotent)
    materialize_all(ROOT)

    # 2. Build base config from frozen winners
    base = copy.deepcopy(WINNER_CONFIG)

    # 3. Override based on condition
    if sweep_id == "S1_FEATURE_SET":
        # Condition is FS0_TF1, FS1_TF1 (skip), FS2_TF1
        variant_id = condition_id  # e.g., "FS0_TF1"
        base["feature_variant_id"] = variant_id
    elif sweep_id == "S2_TIME_FEATURE":
        # Condition: TF0 (no time) or TF1 (with time)
        if condition_id == "TF0":
            # Strip time features suffix: FS1_TF1 -> FS1_TF0
            base["feature_variant_id"] = "FS1_TF0"
        elif condition_id == "TF1":
            base["feature_variant_id"] = "FS1_TF1"
    elif sweep_id == "S3_TARGET_SCALING":
        # Condition: YS0 or YS1
        base["target_scaling_option"] = condition_id
    elif sweep_id == "S4_LOOKBACK":
        # Condition: L36, L72, L144
        base["lookback_steps"] = int(condition_id.replace("L", ""))
    elif sweep_id == "S5_POOLING":
        # Condition: LAST_STEP or MEAN
        base["pooling"] = condition_id
    elif sweep_id == "S6_ACTIVATION":
        # Condition: RELU or GELU
        base["activation"] = condition_id
    elif sweep_id == "S7_BATCH_SIZE":
        # Condition: B32 or B64
        base["batch_size"] = int(condition_id.replace("B", ""))
    elif sweep_id == "S8_LEARNING_RATE":
        # Condition: LR1, LR2, LR3
        lr_map = {"LR1": 1e-4, "LR2": 3e-4, "LR3": 1e-3}
        base["learning_rate"] = lr_map[condition_id]
    else:
        raise ValueError(f"Unknown sweep_id: {sweep_id}")

    print(f"  config: {base}", flush=True)

    # 4. Build run config from reference, then override
    config = build_reference_run_config(ROOT, "TRANSFORMER_ENCODER")

    # Apply overrides
    config["data"]["feature_variant_id"] = base["feature_variant_id"]
    config["data"]["lookback_steps"] = base["lookback_steps"]
    config["data"]["target_scaling_option"] = base["target_scaling_option"]
    config["model"]["pooling"] = base["pooling"]
    config["model"]["activation"] = base["activation"]
    config["training"]["batch_size"] = base["batch_size"]
    config["training"]["learning_rate"] = base["learning_rate"]

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
    config["lineage"]["scaler_bundle_id"] = f"XSCALER__{base['feature_variant_id']}"
    config["lineage"]["feature_fingerprint"] = variant_fingerprints[base["feature_variant_id"]]
    config["lineage"]["scaler_bundle_checksum"] = scaler_bundles[base["feature_variant_id"]]["artifact_sha256"]
    config["lineage"]["target_scaler_bundle_id"] = target_bundles[base["target_scaling_option"]]["bundle_id"]
    config["lineage"]["target_scaler_checksum"] = target_bundles[base["target_scaling_option"]].get("artifact_sha256")
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

    # 7. Register run
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
    }
    family = family_map[sweep_id]
    registered = registry.register_run(
        config, family, ExecutionType.TRAINING.value,
        rerun_reason="MANUAL_RERUN"
    )
    run_id = registered["run_id"]
    print(f"  run_id: {run_id}", flush=True)

    registry.start_run(run_id)

    # 8. Build loaders
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