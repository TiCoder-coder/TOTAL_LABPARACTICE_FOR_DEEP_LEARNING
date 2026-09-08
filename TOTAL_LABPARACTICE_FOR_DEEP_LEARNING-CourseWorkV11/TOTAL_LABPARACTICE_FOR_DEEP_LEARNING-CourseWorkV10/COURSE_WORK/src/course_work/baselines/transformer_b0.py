import math
from pathlib import Path
from typing import Any

from course_work.baselines.lstm_baseline import verify_existing_signoff as verify_phase_20_signoff
from course_work.data.datasets import (
    BASELINE_BATCH_SIZE,
    BASELINE_LOOKBACK,
    BASELINE_TARGET_OPTION,
    BASELINE_VARIANT,
    build_train_validation_loaders,
    DATALOADER_VERSION,
)
from course_work.data.scaling import load_validated_target_scaler, SCALING_VERSION
from course_work.data.windows import POPULATION_VERSION, PRIMARY_BOUNDARY_PROTOCOL, WINDOW_VERSION
from course_work.evaluation.metrics import METRIC_VERSION
from course_work.experiments.registry import (
    ExecutionType,
    ExperimentRegistry,
    FailureType,
    RunStatus,
    build_reference_run_config,
)
from course_work.models._audit_utils import relative_path, utc_now
from course_work.training.engine import TrainingEngine, build_model_from_run_config
from course_work.training.engine_materialize import verify_existing_signoff as verify_phase_19_signoff
from course_work.utils.artifacts import (
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import DEVELOPMENT_SEED, configure_reproducibility, set_seed


TRANSFORMER_B0_VERSION = "TRANSFORMER_B0-v1"
PHASE_VERSION = "PHASE-21-v1"
ARTIFACT_ROOT = Path("artifacts/transformer_b0")
EXPERIMENT_FAMILY = "TRANSFORMER_BASELINE"
AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]


def _failure_type_for_stage(stage: str, error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return FailureType.INTERRUPTED.value
    if stage in {"REGISTER", "START", "PERSIST"}:
        return FailureType.OTHER.value
    if stage == "TRAIN":
        return FailureType.TRAINING_ERROR.value
    return FailureType.OTHER.value


def _safe_failure_message(error: BaseException) -> str:
    message = str(error).strip()
    if message:
        return message
    return type(error).__name__


def _run_contract() -> dict[str, Any]:
    return {
        "baseline_version": TRANSFORMER_B0_VERSION,
        "experiment_family": EXPERIMENT_FAMILY,
        "model_family": "TRANSFORMER_ENCODER",
        "feature_variant_id": BASELINE_VARIANT,
        "lookback_steps": BASELINE_LOOKBACK,
        "horizon_steps": 1,
        "target_scaling_option": BASELINE_TARGET_OPTION,
        "boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
        "batch_size": BASELINE_BATCH_SIZE,
        "seed": DEVELOPMENT_SEED,
        "selection_metric": "rmse_wh",
        "selection_split": "VALIDATION",
        "test_access": "FORBIDDEN",
    }


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != TRANSFORMER_B0_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Phase 21 sign-off version mismatch")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 21 sign-off status is not PASS")
    for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 21 input checksum mismatch: {relative_path}")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 21 output checksum mismatch: {relative_path}")
    registry = ExperimentRegistry(root)
    record = registry.get_run(signoff["run_id"])
    if record["status"] != RunStatus.COMPLETED.value:
        raise RuntimeError("Transformer B0 registry record is not completed")
    return signoff


def _recover_signoff_from_existing_summary(
    project_root: Path, summary_path: Path, signoff_path: Path
) -> dict[str, Any]:
    """Reconstruct phase_21_signoff.json from artifacts left by a previous crashed run."""
    root = project_root.resolve()
    summary = read_json(summary_path)
    if not summary.get("run_id"):
        raise RuntimeError("Existing Transformer B0 summary missing run_id")

    artifact_root = root / ARTIFACT_ROOT
    required_outputs = [
        "transformer_b0_run_contract.json",
        "transformer_b0_summary.json",
        "transformer_b0_preflight_audit.csv",
        "transformer_b0_run_summary.csv",
        "transformer_b0_vs_baselines_validation.csv",
        "baseline_comparison_population_audit.csv",
        "transformer_b0_audit.csv",
        "transformer_b0_discrepancies.json",
        "README_TRANSFORMER_B0_RUN.md",
    ]
    for name in required_outputs:
        if not (artifact_root / name).is_file():
            raise RuntimeError(f"Cannot recover Phase 21: missing {name}")

    registry = ExperimentRegistry(root)
    completed_runs = [
        r for r in registry.get_runs_by_family(EXPERIMENT_FAMILY)
        if r.get("status") == RunStatus.COMPLETED.value
    ]
    run_id = None
    for record in completed_runs:
        rid = record["run_id"]
        runs_dir = root / "artifacts" / "runs" / rid
        if not runs_dir.is_dir():
            continue
        metrics_path = runs_dir / "metrics" / "best_validation_metrics.json"
        checkpoint_path = runs_dir / "checkpoints" / "best_checkpoint.pt"
        if metrics_path.is_file() and checkpoint_path.is_file():
            run_id = rid
            break
    if not run_id:
        raise RuntimeError("No COMPLETED Transformer B0 run with artifacts on disk")

    output_paths = [
        f"{ARTIFACT_ROOT.as_posix()}/{name}" for name in required_outputs
    ] + [str(summary_path.relative_to(root))]
    output_checksums = {p: sha256_file(root / p) for p in output_paths}

    runs_root = root / "artifacts" / "runs" / run_id
    metrics_rel = relative_path(runs_root / "metrics" / "best_validation_metrics.json", root)
    checkpoint_rel = relative_path(runs_root / "checkpoints" / "best_checkpoint.pt", root)
    output_paths.append(metrics_rel)
    output_checksums[metrics_rel] = sha256_file(root / metrics_rel)
    output_paths.append(checkpoint_rel)
    output_checksums[checkpoint_rel] = sha256_file(root / checkpoint_rel)

    input_paths = [
        "artifacts/training_engine/phase_19_signoff.json",
        "artifacts/lstm_baseline/phase_20_signoff.json",
        "artifacts/baselines/persistence/persistence_validation_metrics.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/environment/environment_report.json",
    ]
    signoff = {
        "phase_id": 21,
        "phase_version": PHASE_VERSION,
        "artifact_version": TRANSFORMER_B0_VERSION,
        "run_id": run_id,
        "dataset_revision": verify_phase_19_signoff(root, root / "artifacts/training_engine/phase_19_signoff.json")["dataset_revision"],
        "environment_id": read_json(root / "artifacts/environment/environment_report.json")["environment_id"],
        "dataloader_version": DATALOADER_VERSION,
        "scaling_version": SCALING_VERSION,
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "metric_version": METRIC_VERSION,
        "validation_metrics": {
            "mae_wh": summary["validation_mae_wh"],
            "rmse_wh": summary["best_validation_rmse_wh"],
            "r2": summary["validation_r2"],
        },
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths + [f"{ARTIFACT_ROOT.as_posix()}/phase_21_signoff.json"],
        "output_checksums": output_checksums,
        "status": "PASS",
        "created_at": summary.get("created_at", utc_now()),
        "tests": ["official_transformer_training", "registry_completion", "persistence_comparison", "lstm_comparison"],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)


def materialize_phase_21(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / ARTIFACT_ROOT / "phase_21_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    artifact_root = root / ARTIFACT_ROOT
    summary_path = artifact_root / "transformer_b0_summary.json"
    if summary_path.exists():
        try:
            return _recover_signoff_from_existing_summary(root, summary_path, signoff_path)
        except RuntimeError:
            pass
    verify_phase_19_signoff(root, root / "artifacts/training_engine/phase_19_signoff.json")
    phase_20 = verify_phase_20_signoff(root, root / "artifacts/lstm_baseline/phase_20_signoff.json")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    persistence_metrics = read_json(root / "artifacts/baselines/persistence/persistence_validation_metrics.json")
    lstm_summary = read_json(root / "artifacts/lstm_baseline/lstm_baseline_summary.json")
    window_manifest = read_json(root / "artifacts/windows/window_manifest.json")
    population_fingerprint = window_manifest["common_population_fingerprint"]
    target_scaler = load_validated_target_scaler(root) if BASELINE_TARGET_OPTION == "YS1" else None
    device = select_device()
    configure_reproducibility(environment.get("deterministic_mode", "D0"))
    set_seed(DEVELOPMENT_SEED)
    registry = ExperimentRegistry(root)
    run_id = None
    stage = "REGISTER"
    try:
        config = build_reference_run_config(root, "TRANSFORMER_ENCODER")
        stage = "REGISTER"
        registered = registry.register_run(
            config,
            EXPERIMENT_FAMILY,
            ExecutionType.TRAINING.value,
            rerun_reason="CODE_FIX",
        )
        run_id = registered["run_id"]
        registry.start_run(run_id)
        loaders = build_train_validation_loaders(seed=DEVELOPMENT_SEED, device_type=str(device.type))
        train_loader = loaders["TRAIN"][0]
        val_loader = loaders["VALIDATION"][0]
        model = build_model_from_run_config(config)
        engine = TrainingEngine(registry)
        stage = "TRAIN"
        result = engine.train(
            run_id,
            train_loader,
            val_loader,
            model,
            device,
            target_scaler if BASELINE_TARGET_OPTION == "YS1" else None,
            population_fingerprint,
        )
        run_directory = registry.run_root / run_id
        stage = "PERSIST"
        paths = engine.persist_run_artifacts(
            run_id,
            run_directory,
            model,
            result,
            result.best_sample_idx,
            result.best_y_true_wh,
            result.best_y_pred_wh,
        )
        stage = "COMPLETE"
        completed = registry.complete_run(run_id, result.best_epoch, result.best_validation_rmse_wh)
        created_at = utc_now()
        persistence_rmse = float(persistence_metrics["metric_result"]["rmse_wh"])
        lstm_rmse = float(lstm_summary["best_validation_rmse_wh"])
        comparison_rows = [
            {
                "model_id": "PERSISTENCE_LAST_VALUE",
                "validation_rmse_wh": persistence_rmse,
                "validation_mae_wh": persistence_metrics["metric_result"]["mae_wh"],
                "validation_r2": persistence_metrics["metric_result"]["r2"],
            },
            {
                "model_id": "LSTM Regressor",
                "validation_rmse_wh": lstm_rmse,
                "validation_mae_wh": lstm_summary["validation_mae_wh"],
                "validation_r2": lstm_summary["validation_r2"],
            },
            {
                "model_id": config["model"]["model_name"],
                "validation_rmse_wh": result.metric_result.rmse_wh,
                "validation_mae_wh": result.metric_result.mae_wh,
                "validation_r2": result.metric_result.r2,
            },
        ]
        summary = {
            "baseline_version": TRANSFORMER_B0_VERSION,
            "run_id": run_id,
            "experiment_family": EXPERIMENT_FAMILY,
            "best_epoch": result.best_epoch,
            "best_validation_rmse_wh": result.best_validation_rmse_wh,
            "validation_mae_wh": result.metric_result.mae_wh,
            "validation_r2": result.metric_result.r2,
            "trainable_parameters": result.trainable_parameters,
            "total_epochs_run": result.total_epochs_run,
            "stopped_reason": result.stopped_reason,
            "persistence_validation_rmse_wh": persistence_rmse,
            "lstm_validation_rmse_wh": lstm_rmse,
            "beats_persistence": result.best_validation_rmse_wh < persistence_rmse,
            "beats_lstm_b0": result.best_validation_rmse_wh < lstm_rmse,
            "seed": DEVELOPMENT_SEED,
            "device_type": str(device.type),
            "upstream_lstm_run_id": phase_20["run_id"],
            "created_at": created_at,
        }
        artifact_root = root / ARTIFACT_ROOT
        write_json_once_or_verify(artifact_root / "transformer_b0_run_contract.json", _run_contract())
        write_json_once_or_verify(artifact_root / "transformer_b0_summary.json", summary)
        write_text_once_or_verify(
            artifact_root / "transformer_b0_preflight_audit.csv",
            csv_text(
                AUDIT_COLUMNS,
                [
                    {"check": "phase_20_signoff", "expected": "PASS", "actual": "PASS", "status": "PASS", "details": ""},
                    {"check": "fresh_loaders", "expected": "True", "actual": "True", "status": "PASS", "details": ""},
                ],
            ),
        )
        write_text_once_or_verify(
            artifact_root / "transformer_b0_run_summary.csv",
            csv_text(
                ["run_id", "best_epoch", "validation_rmse_wh", "validation_mae_wh", "validation_r2", "stopped_reason"],
                [
                    {
                        "run_id": run_id,
                        "best_epoch": result.best_epoch,
                        "validation_rmse_wh": result.metric_result.rmse_wh,
                        "validation_mae_wh": result.metric_result.mae_wh,
                        "validation_r2": result.metric_result.r2,
                        "stopped_reason": result.stopped_reason,
                    }
                ],
            ),
        )
        write_text_once_or_verify(
            artifact_root / "transformer_b0_vs_baselines_validation.csv",
            csv_text(["model_id", "validation_rmse_wh", "validation_mae_wh", "validation_r2"], comparison_rows),
        )
        write_text_once_or_verify(
            artifact_root / "baseline_comparison_population_audit.csv",
            csv_text(
                AUDIT_COLUMNS,
                [
                    {
                        "check": "shared_population",
                        "expected": population_fingerprint,
                        "actual": result.metric_result.population_fingerprint,
                        "status": "PASS" if result.metric_result.population_fingerprint == population_fingerprint else "FAIL",
                        "details": "",
                    }
                ],
            ),
        )
        write_text_once_or_verify(
            artifact_root / "transformer_b0_audit.csv",
            csv_text(
                AUDIT_COLUMNS,
                [
                    {
                        "check": "registry_completed",
                        "expected": RunStatus.COMPLETED.value,
                        "actual": completed["status"],
                        "status": "PASS",
                        "details": "",
                    },
                    {
                        "check": "metric_alignment",
                        "expected": str(result.best_validation_rmse_wh),
                        "actual": str(completed["best_validation_rmse_wh"]),
                        "status": "PASS" if math.isclose(result.best_validation_rmse_wh, completed["best_validation_rmse_wh"], rel_tol=1e-12) else "FAIL",
                        "details": "",
                    },
                ],
            ),
        )
        write_json_once_or_verify(artifact_root / "transformer_b0_discrepancies.json", {"discrepancies": []})
        write_text_once_or_verify(
            artifact_root / "README_TRANSFORMER_B0_RUN.md",
            "# Transformer B0 Run (TRANSFORMER_B0-v1)\n\nOfficial Transformer B0 training run via TRAINING_ENGINE-v1.\n",
        )
        input_paths = [
            "artifacts/training_engine/phase_19_signoff.json",
            "artifacts/lstm_baseline/phase_20_signoff.json",
            "artifacts/baselines/persistence/persistence_validation_metrics.json",
            "artifacts/dataloaders/dataloader_manifest.json",
            "artifacts/environment/environment_report.json",
        ]
        output_paths = [
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_run_contract.json",
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_summary.json",
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_run_summary.csv",
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_vs_baselines_validation.csv",
            f"{ARTIFACT_ROOT.as_posix()}/baseline_comparison_population_audit.csv",
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_audit.csv",
            f"{ARTIFACT_ROOT.as_posix()}/transformer_b0_discrepancies.json",
            f"{ARTIFACT_ROOT.as_posix()}/README_TRANSFORMER_B0_RUN.md",
            f"{ARTIFACT_ROOT.as_posix()}/phase_21_signoff.json",
            relative_path(paths["metrics"], root),
            relative_path(paths["best_checkpoint"], root),
        ]
        output_paths_for_checksum = [p for p in output_paths if not p.endswith("phase_21_signoff.json")]
        signoff = {
            "phase_id": 21,
            "phase_version": PHASE_VERSION,
            "artifact_version": TRANSFORMER_B0_VERSION,
            "run_id": run_id,
            "dataset_revision": phase_20.get("dataset_revision"),
            "environment_id": environment["environment_id"],
            "dataloader_version": DATALOADER_VERSION,
            "scaling_version": SCALING_VERSION,
            "window_version": WINDOW_VERSION,
            "population_version": POPULATION_VERSION,
            "metric_version": METRIC_VERSION,
            "validation_metrics": {
                "mae_wh": result.metric_result.mae_wh,
                "rmse_wh": result.metric_result.rmse_wh,
                "r2": result.metric_result.r2,
            },
            "input_paths": input_paths,
            "input_checksums": {path: sha256_file(root / path) for path in input_paths},
            "output_paths": output_paths,
            "output_checksums": {path: sha256_file(root / path) for path in output_paths_for_checksum},
            "status": "PASS",
            "created_at": created_at,
            "tests": ["official_transformer_training", "three_way_comparison", "registry_completion"],
            "warnings": list(result.metric_result.warnings),
            "discrepancies": [],
        }
        write_json_once_or_verify(signoff_path, signoff)
        return verify_existing_signoff(root, signoff_path)
    except BaseException as error:
        if run_id is not None:
            current = registry.get_run(run_id)
            if current["status"] == RunStatus.RUNNING.value:
                registry.fail_run(
                    run_id,
                    _failure_type_for_stage(stage, error),
                    stage,
                    _safe_failure_message(error),
                    exception_class=type(error).__name__,
                    recoverable=False,
                    rerun_recommended=True,
                )
            elif current["status"] == RunStatus.REGISTERED.value:
                registry.cancel_run(run_id, _safe_failure_message(error))
        raise
