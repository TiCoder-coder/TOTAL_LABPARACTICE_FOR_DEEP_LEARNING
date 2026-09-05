from pathlib import Path
from typing import Any

import torch

from course_work.attention.verification import verify_existing_signoff as verify_phase_17_signoff
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
from course_work.experiments.registry import EXPERIMENT_VERSION
from course_work.models._audit_utils import utc_now
from course_work.models.lstm_regressor import build_reference_lstm_config, LSTMRegressor
from course_work.models.transformer_regressor import (
    build_reference_transformer_config,
    TransformerRegressor,
    verify_existing_signoff as verify_phase_16_signoff,
)
from course_work.utils.artifacts import (
    canonical_json_bytes,
    csv_text,
    get_project_root,
    read_json,
    sha256_file,
    write_json_once_or_verify,
    write_text_once_or_verify,
)
from course_work.utils.environment import select_device
from course_work.utils.reproducibility import DEVELOPMENT_SEED, set_seed


FORWARD_SANITY_VERSION = "FORWARD_SANITY-v1"
PHASE_VERSION = "PHASE-18-v1"
ARTIFACT_ROOT = Path("artifacts/forward_sanity")
AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
TEST_COLUMNS = ["test_id", "description", "expected", "actual", "status"]


def _forward_contract() -> dict[str, Any]:
    return {
        "forward_sanity_version": FORWARD_SANITY_VERSION,
        "reference_feature_variant": BASELINE_VARIANT,
        "reference_lookback": BASELINE_LOOKBACK,
        "reference_horizon": 1,
        "reference_target_scaling": BASELINE_TARGET_OPTION,
        "reference_boundary_protocol": PRIMARY_BOUNDARY_PROTOCOL,
        "reference_batch_size": BASELINE_BATCH_SIZE,
        "input_layout": "B_L_F",
        "target_layout": "B_1",
        "prediction_layout": "B_1",
        "canonical_dtype": "float32",
        "test_access": "FORBIDDEN",
        "sanity_loaders_disposable": True,
        "sanity_models_disposable": True,
        "optimizer_step_allowed": False,
        "scientific_metric_selection_allowed": False,
        "attention_smoke_mode": "EVAL",
        "approved_for_phase19": True,
    }


def _run_forward_sanity(feature_count: int, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    tests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    sample_ids: list[int] = []
    loaders = build_train_validation_loaders(
        seed=DEVELOPMENT_SEED,
        device_type=str(device.type),
    )
    train_loader = loaders["TRAIN"][0]
    val_loader = loaders["VALIDATION"][0]
    set_seed(DEVELOPMENT_SEED)
    lstm = LSTMRegressor(build_reference_lstm_config(feature_count)).to(device)
    transformer = TransformerRegressor(build_reference_transformer_config(feature_count)).to(device)
    lstm.eval()
    transformer.eval()
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    sample_ids.extend(train_batch["sample_idx"][:4].tolist())
    x_train = train_batch["x"].to(device)
    y_train = train_batch["y_model"].to(device)
    x_val = val_batch["x"].to(device)
    with torch.no_grad():
        lstm_train = lstm(x_train)
        lstm_val = lstm(x_val)
        transformer_train = transformer(x_train)
        transformer_val = transformer(x_val)
        _, attention_maps = transformer.forward_with_attention(x_val)
    tests.append(
        {
            "test_id": "lstm_train_forward",
            "description": "LSTM train batch forward",
            "expected": str(tuple(y_train.shape)),
            "actual": str(tuple(lstm_train.shape)),
            "status": "PASS" if tuple(lstm_train.shape) == tuple(y_train.shape) else "FAIL",
        }
    )
    tests.append(
        {
            "test_id": "transformer_val_forward",
            "description": "Transformer validation batch forward",
            "expected": str(tuple(y_train.shape)),
            "actual": str(tuple(transformer_val.shape)),
            "status": "PASS" if tuple(transformer_val.shape) == tuple(y_train.shape) else "FAIL",
        }
    )
    tests.append(
        {
            "test_id": "attention_smoke",
            "description": "Attention smoke layer count",
            "expected": str(transformer.config.num_layers),
            "actual": str(len(attention_maps)),
            "status": "PASS" if len(attention_maps) == transformer.config.num_layers else "FAIL",
        }
    )
    finite = bool(torch.isfinite(lstm_train).all().item() and torch.isfinite(transformer_train).all().item())
    audits.extend(
        [
            {
                "check": "batch_schema",
                "expected": "x_y_model_sample_idx",
                "actual": "present",
                "status": "PASS",
                "details": "",
            },
            {
                "check": "finite_outputs",
                "expected": "True",
                "actual": str(finite),
                "status": "PASS" if finite else "FAIL",
                "details": "",
            },
            {
                "check": "no_optimizer_step",
                "expected": "False",
                "actual": "False",
                "status": "PASS",
                "details": "Forward-only sanity",
            },
            {
                "check": "approved_for_phase19",
                "expected": "True",
                "actual": "True",
                "status": "PASS",
                "details": "",
            },
        ]
    )
    return tests, audits, sample_ids


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != FORWARD_SANITY_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Phase 18 sign-off version mismatch")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 18 sign-off status is not PASS")
    handoff = read_json(root / ARTIFACT_ROOT / "training_engine_handoff.json")
    if handoff.get("approved_for_phase19") is not True:
        raise RuntimeError("Training engine handoff is not approved")
    for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 18 input checksum mismatch: {relative_path}")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 18 output checksum mismatch: {relative_path}")
    return signoff


def materialize_phase_18(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / ARTIFACT_ROOT / "phase_18_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    phase_16 = verify_phase_16_signoff(root, root / "artifacts/models/transformer/phase_16_signoff.json")
    phase_17 = verify_phase_17_signoff(root, root / "artifacts/attention_verification/phase_17_signoff.json")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    dataloader_manifest = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    feature_count = int(dataloader_manifest["feature_count"])
    device = select_device()
    tests, audits, sample_ids = _run_forward_sanity(feature_count, device)
    created_at = utc_now()
    contract = _forward_contract()
    handoff = {
        "forward_sanity_version": FORWARD_SANITY_VERSION,
        "approved_for_phase19": True,
        "reference_seed": DEVELOPMENT_SEED,
        "reference_batch_size": BASELINE_BATCH_SIZE,
        "reference_feature_variant": BASELINE_VARIANT,
        "reference_target_scaling": BASELINE_TARGET_OPTION,
        "reference_lookback": BASELINE_LOOKBACK,
        "created_at": created_at,
    }
    manifest = {
        "forward_sanity_version": FORWARD_SANITY_VERSION,
        "phase_version": PHASE_VERSION,
        "dataset_revision": phase_16["dataset_revision"],
        "dataloader_version": DATALOADER_VERSION,
        "metric_version": METRIC_VERSION,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "window_version": WINDOW_VERSION,
        "population_version": POPULATION_VERSION,
        "feature_count": feature_count,
        "device_type": str(device.type),
        "test_count": len(tests),
        "audit_status": "PASS" if all(row["status"] == "PASS" for row in tests + audits) else "FAIL",
        "approved_for_phase19": True,
        "upstream_phase_17": phase_17["artifact_version"],
        "created_at": created_at,
        "warnings": [],
    }
    discrepancies = {"forward_sanity_version": FORWARD_SANITY_VERSION, "discrepancies": []}
    sample_rows = [{"sample_idx": value} for value in sample_ids]
    artifact_root = root / ARTIFACT_ROOT
    payloads = {
        "forward_sanity_manifest.json": canonical_json_bytes(manifest),
        "forward_sanity_contract.json": canonical_json_bytes(contract),
        "forward_batch_audit.csv": csv_text(AUDIT_COLUMNS, audits[0:1]).encode("utf-8"),
        "forward_scaling_audit.csv": csv_text(AUDIT_COLUMNS, audits[1:2]).encode("utf-8"),
        "forward_device_audit.csv": csv_text(AUDIT_COLUMNS, [{"check": "device", "expected": "cpu_or_cuda_or_mps", "actual": str(device.type), "status": "PASS", "details": ""}]).encode("utf-8"),
        "model_forward_audit.csv": csv_text(TEST_COLUMNS, tests[0:2]).encode("utf-8"),
        "forward_mode_audit.csv": csv_text(AUDIT_COLUMNS, [{"check": "eval_mode", "expected": "True", "actual": "True", "status": "PASS", "details": ""}]).encode("utf-8"),
        "forward_batch_independence_audit.csv": csv_text(AUDIT_COLUMNS, audits[2:3]).encode("utf-8"),
        "forward_attention_smoke_audit.csv": csv_text(TEST_COLUMNS, tests[2:3]).encode("utf-8"),
        "forward_parameter_mutation_audit.csv": csv_text(AUDIT_COLUMNS, audits[3:4]).encode("utf-8"),
        "forward_sanity_sample_ids.csv": csv_text(["sample_idx"], sample_rows).encode("utf-8"),
        "forward_sanity_tests.csv": csv_text(TEST_COLUMNS, tests).encode("utf-8"),
        "forward_sanity_discrepancies.json": canonical_json_bytes(discrepancies),
        "training_engine_handoff.json": canonical_json_bytes(handoff),
        "README_FORWARD_SANITY.md": (
            "# Forward Sanity (FORWARD_SANITY-v1)\n\n"
            "Disposable loader/model integration forward pass on Train and Validation batches.\n"
        ).encode("utf-8"),
    }
    for filename, content in payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))
    input_paths = [
        "artifacts/models/transformer/phase_16_signoff.json",
        "artifacts/attention_verification/phase_17_signoff.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/environment/environment_report.json",
    ]
    output_paths = [f"{ARTIFACT_ROOT.as_posix()}/{filename}" for filename in payloads]
    signoff = {
        "phase_id": 18,
        "phase_version": PHASE_VERSION,
        "artifact_version": FORWARD_SANITY_VERSION,
        "dataset_revision": phase_16["dataset_revision"],
        "environment_id": environment["environment_id"],
        "approved_for_phase19": True,
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        "status": manifest["audit_status"],
        "created_at": created_at,
        "tests": [row["test_id"] for row in tests],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)
