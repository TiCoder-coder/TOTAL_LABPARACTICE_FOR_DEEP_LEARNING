"""Read-only training authorization gate for Practice 3 v2 Part 3.1."""

from __future__ import annotations

import json
import shutil
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

from .config import PROJECT_ROOT, RESULT_DIR
from .dataset_protocol_v2 import validate_split_manifest
from .experiment_protocol_v2 import (
    LEARNING_RATES,
    PART_2_1_PLAN,
    PROTOCOL_VERSION,
    RUN_IDS,
    V2_RESULT_DIR,
    sha256_file,
    sha256_payload,
    validate_protocol_manifest,
    validate_run_configs,
    optimizer_compatibility_preflight,
)
from .experiment_registry_v2 import validate_registry
from .holdout_guard_v2 import guarded_holdout_request, validate_initial_holdout_state


EXPECTED_EXECUTION_HASH = "fdfcbb87b20d0bc618890a51c70a9689d618682b5a7cdf8edf05b638fa347055"
EXPECTED_SPLIT_HASH = "4d22ccf19a61c37bad6138fcc12d40102603407fcfbc6e19a3f5e634803cbb13"
EXPECTED_V1_HASH = "22fcd24d3db7bfebc8ea1dd4f6c0665be5176e34de005e168a747ee83acfa660"
AUTHORIZATION_PATH = V2_RESULT_DIR / "part_03_1_training_authorization.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _load(path)
        if existing == payload:
            return
        raise FileExistsError(f"Refusing to overwrite training authorization: {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _directory_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".write-check-", delete=True):
            pass
        return True
    except OSError:
        return False


def _execution_components(
    protocol: dict[str, Any],
    split: dict[str, Any],
    configs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "part_2_1_plan_hash": sha256_file(PART_2_1_PLAN),
        "protocol_manifest_hash": protocol["protocol_manifest_hash"],
        "dataset_split_manifest_hash": split["split_manifest_hash"],
        "run_config_hashes": {
            config["run_id"]: config["config_hash"] for config in configs
        },
        "ranking": protocol["ranking"],
    }


def create_training_authorization() -> dict[str, Any]:
    """Authorize Part 3.2 only after every zero-training pre-run check passes."""
    protocol = _load(V2_RESULT_DIR / "protocol_manifest.json")
    split = _load(V2_RESULT_DIR / "dataset_split_manifest.json")
    registry = _load(V2_RESULT_DIR / "experiment_registry.json")
    state = _load(V2_RESULT_DIR / "holdout_access_state.json")
    readiness = _load(V2_RESULT_DIR / "pretraining_readiness_manifest.json")
    final_audit = _load(V2_RESULT_DIR / "part_02_final_audit.json")
    configs = [
        _load(V2_RESULT_DIR / "experiments" / run_id / "run_config.json")
        for run_id in RUN_IDS
    ]
    validate_protocol_manifest(protocol)
    validate_split_manifest(split)
    validate_run_configs(configs)
    validate_registry(registry, configs, require_planned=True)
    validate_initial_holdout_state(state)

    components = _execution_components(protocol, split, configs)
    recomputed_execution_hash = sha256_payload(components)
    optimizer = optimizer_compatibility_preflight()
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    checkpoint = RESULT_DIR / "phase_09_training" / "checkpoints" / "checkpoint-1068"
    sample_checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    )
    worst_case_checkpoint_bytes = sample_checkpoint_bytes * 30
    required_bytes = max(30 * 1024**3, int(worst_case_checkpoint_bytes * 1.25))

    paths = {}
    path_sets = {"output": set(), "checkpoint": set(), "history": set(), "summary": set(), "tensorboard": set()}
    stale_files = []
    for config in configs:
        run_dir = V2_RESULT_DIR / "experiments" / config["run_id"]
        run_paths = {
            "output": str(run_dir.relative_to(PROJECT_ROOT)),
            "checkpoint": str((run_dir / "checkpoints").relative_to(PROJECT_ROOT)),
            "history": str((run_dir / "training_history.json").relative_to(PROJECT_ROOT)),
            "summary": str((run_dir / "run_summary.json").relative_to(PROJECT_ROOT)),
            "tensorboard": config["tensorboard_log_dir"],
        }
        paths[config["run_id"]] = run_paths
        for key, value in run_paths.items():
            path_sets[key].add(value)
        for item in run_dir.rglob("*"):
            if item.is_file() and item.name != "run_config.json":
                stale_files.append(str(item.relative_to(PROJECT_ROOT)))
    tensorboard_root = PROJECT_ROOT / "runs" / "practice_3_v2"
    tensorboard_events = list(tensorboard_root.glob("**/events.out.tfevents*"))

    provider_calls = {"count": 0}

    def forbidden_provider() -> None:
        provider_calls["count"] += 1

    prelock_rejected = False
    try:
        guarded_holdout_request(
            V2_RESULT_DIR / "holdout_access_state.json",
            "winner-absent",
            forbidden_provider,
        )
    except PermissionError:
        prelock_rejected = True

    result_fields = (
        "stopped_epoch", "best_epoch", "best_val_loss", "best_val_accuracy",
        "best_val_precision", "best_val_recall", "best_val_f1",
        "best_checkpoint", "runtime_seconds",
    )
    v1_manifest = _load(RESULT_DIR / "phase_11_evaluation" / "phase_11_evaluation_manifest.json")
    v1_checkpoint = checkpoint / "model.safetensors"
    v1_package = RESULT_DIR / "phase_14_saved_model" / "model.safetensors"

    checks = {
        "protocol_hash_unchanged": (
            recomputed_execution_hash == EXPECTED_EXECUTION_HASH
            and readiness["execution_protocol_hash"] == EXPECTED_EXECUTION_HASH
            and final_audit["execution_protocol_hash"] == EXPECTED_EXECUTION_HASH
        ),
        "protocol_drift_false": final_audit["protocol_drift"] is False,
        "three_runs_planned": (
            [record["run_id"] for record in registry["runs"]] == list(RUN_IDS)
            and all(record["status"] == "PLANNED" for record in registry["runs"])
        ),
        "only_learning_rate_differs": tuple(config["learning_rate"] for config in configs) == LEARNING_RATES,
        "no_real_results": all(
            all(record[field] is None for field in result_fields)
            for record in registry["runs"]
        ),
        "dataset_boundary": (
            split["split_counts"] == {"train": 7676, "validation": 960, "holdout": 960}
            and split["split_manifest_hash"] == EXPECTED_SPLIT_HASH
            and split["official_test_loaded"] is False
        ),
        "holdout_sealed": (
            protocol["holdout_status"] == "SEALED"
            and state["holdout_access_allowed"] is False
            and state["holdout_evaluation_count"] == 0
            and readiness["holdout_materialized"] is False
        ),
        "device_ready": optimizer["device"] in {"mps", "cuda", "cpu"},
        "mps_ready": optimizer["device"] == "mps" and torch.backends.mps.is_available(),
        "optimizer_ready": optimizer["status"] == "SUPPORTED",
        "tensorboard_dependency": find_spec("tensorboard") is not None,
        "project_writable": _directory_writable(PROJECT_ROOT),
        "result_writable": _directory_writable(V2_RESULT_DIR),
        "runs_writable": _directory_writable(tensorboard_root),
        "disk_sufficient": free_bytes >= required_bytes,
        "output_paths_isolated": all(len(values) == 3 for values in path_sets.values()),
        "output_paths_clean": not stale_files and not tensorboard_events,
        "tensorboard_ready": (
            all((tensorboard_root / run_id).is_dir() for run_id in RUN_IDS)
            and not tensorboard_events
        ),
        "fresh_state_runner_contract": (
            all(report["fresh_model_constructed"] for report in readiness["trainer_reports"])
            and all(report["fresh_optimizer_constructed"] for report in readiness["trainer_reports"])
            and all(report["fresh_scheduler_constructed"] for report in readiness["trainer_reports"])
            and [config["run_id"] for config in configs] == list(RUN_IDS)
        ),
        "winner_absent": not (V2_RESULT_DIR / "winner_manifest.json").exists(),
        "holdout_prelock_rejected": prelock_rejected and provider_calls["count"] == 0,
        "v1_test_count_one": v1_manifest["test_evaluation_count"] == 1,
        "v1_checkpoint_unchanged": (
            sha256_file(v1_checkpoint) == EXPECTED_V1_HASH
            and sha256_file(v1_package) == EXPECTED_V1_HASH
        ),
        "no_training": (
            readiness["training_performed"] is False
            and readiness["trainer_train_called"] is False
            and readiness["backward_called"] is False
            and readiness["optimizer_step_called"] is False
        ),
    }
    authorized = all(checks.values())
    authorization = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_protocol_hash": recomputed_execution_hash,
        "split_manifest_hash": split["split_manifest_hash"],
        "run_config_hashes": components["run_config_hashes"],
        "registry_states": {
            record["run_id"]: record["status"] for record in registry["runs"]
        },
        "execution_order": list(RUN_IDS),
        "device": optimizer["device"],
        "optimizer": "ADAMW_TORCH_FUSED",
        "optimizer_status": optimizer["status"],
        "optimizer_step_called": False,
        "disk_status": "PASS" if checks["disk_sufficient"] else "BLOCKED",
        "disk_free_gib": round(free_bytes / 1024**3, 2),
        "disk_required_gib": round(required_bytes / 1024**3, 2),
        "checkpoint_size_basis_gib": round(sample_checkpoint_bytes / 1024**3, 3),
        "output_paths": paths,
        "output_paths_clean": checks["output_paths_clean"],
        "stale_artifacts": stale_files,
        "tensorboard_ready": checks["tensorboard_ready"],
        "tensorboard_command": "tensorboard --logdir runs/practice_3_v2",
        "tensorboard_required_scalars": protocol["tensorboard"]["required_scalars"],
        "tensorboard_events_present": bool(tensorboard_events),
        "runner_contract": [
            "verify_protocol_hash",
            "verify_run_config",
            "PLANNED_to_RUNNING",
            "reset_seed",
            "fresh_pretrained_distilbert",
            "fresh_optimizer_scheduler",
            "Trainer.train_in_Part_3_2_only",
            "save_isolated_artifacts",
            "COMPLETED_or_FAILED",
        ],
        "part_3_2_runner_execution_implemented": False,
        "holdout_status": "SEALED",
        "holdout_evaluation_count": 0,
        "official_test_loaded": False,
        "winner_manifest_exists": False,
        "training_performed": False,
        "trainer_train_called": False,
        "backward_called": False,
        "checks": checks,
        "training_authorized": authorized,
    }
    authorization["authorization_hash"] = sha256_payload(authorization)
    _write_json(AUTHORIZATION_PATH, authorization)
    if not authorized:
        raise RuntimeError(f"Training is not authorized: {checks}")
    return authorization


if __name__ == "__main__":
    result = create_training_authorization()
    print({
        "device": result["device"],
        "disk_status": result["disk_status"],
        "output_paths_clean": result["output_paths_clean"],
        "training_authorized": result["training_authorized"],
    })

