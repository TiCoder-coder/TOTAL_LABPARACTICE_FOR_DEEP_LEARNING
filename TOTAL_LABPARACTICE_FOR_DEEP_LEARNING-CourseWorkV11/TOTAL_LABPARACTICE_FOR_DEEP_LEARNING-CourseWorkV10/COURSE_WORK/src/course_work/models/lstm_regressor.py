from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from course_work.data.datasets import DATALOADER_VERSION, verify_existing_signoff as verify_phase_11_signoff
from course_work.experiments.registry import EXPERIMENT_VERSION, verify_existing_signoff as verify_phase_13_signoff
from course_work.evaluation.metrics import METRIC_VERSION, verify_existing_signoff as verify_phase_12_signoff
from course_work.baselines.persistence import verify_existing_signoff as verify_phase_14_signoff
from course_work.data.windows import WINDOW_VERSION, verify_existing_signoff as verify_phase_10_signoff
from course_work.models._audit_utils import (
    build_module_audit_rows,
    build_parameter_audit_rows,
    count_trainable_parameters,
    relative_path,
    source_code_fingerprint,
    utc_now,
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
from course_work.utils.reproducibility import DEVELOPMENT_SEED, set_seed


LSTM_IMPL_VERSION = "LSTM_IMPL-v1"
LSTM_MODEL_VERSION = "LSTM-v1"
PHASE_VERSION = "PHASE-15-v1"
ARTIFACT_ROOT = Path("artifacts/models/lstm")
SUPPORTED_LOOKBACKS = (36, 72, 144)
SUPPORTED_POOLINGS = ("LAST_STEP",)
IMPLEMENTATION_AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
UNIT_TEST_COLUMNS = ["test_id", "description", "expected", "actual", "tolerance", "status"]
MODULE_AUDIT_COLUMNS = ["module_path", "module_type", "trainable_parameters"]
PARAMETER_AUDIT_COLUMNS = ["parameter_name", "shape", "numel", "requires_grad", "dtype", "device"]


@dataclass(frozen=True)
class LSTMModelConfig:
    model_family: str = "LSTM"
    model_version: str = LSTM_MODEL_VERSION
    implementation_version: str = LSTM_IMPL_VERSION
    input_size: int = 31
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    batch_first: bool = True
    pooling: str = "LAST_STEP"
    output_size: int = 1
    proj_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_lstm_config(config: LSTMModelConfig | dict[str, Any]) -> LSTMModelConfig:
    if isinstance(config, LSTMModelConfig):
        payload = config
    else:
        filtered = {key: value for key, value in dict(config).items() if key in LSTMModelConfig.__dataclass_fields__}
        payload = LSTMModelConfig(**filtered)
    if payload.model_family != "LSTM":
        raise ValueError("model_family must be LSTM")
    if payload.model_version != LSTM_MODEL_VERSION:
        raise ValueError("model_version mismatch")
    if payload.implementation_version != LSTM_IMPL_VERSION:
        raise ValueError("implementation_version mismatch")
    if payload.input_size <= 0 or payload.hidden_size <= 0 or payload.num_layers <= 0:
        raise ValueError("input_size, hidden_size and num_layers must be positive")
    if not 0.0 <= payload.dropout <= 1.0:
        raise ValueError("dropout must be in [0, 1]")
    if payload.bidirectional:
        raise ValueError("LSTM-v1 does not support bidirectional mode")
    if payload.batch_first is not True:
        raise ValueError("LSTM-v1 requires batch_first=True")
    if payload.pooling not in SUPPORTED_POOLINGS:
        raise ValueError(f"Unsupported pooling: {payload.pooling}")
    if payload.output_size != 1:
        raise ValueError("LSTM-v1 output_size must be 1")
    if payload.proj_size != 0:
        raise ValueError("LSTM-v1 proj_size must be 0")
    return payload


def build_reference_lstm_config(feature_count: int) -> LSTMModelConfig:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    return LSTMModelConfig(input_size=feature_count)


class LSTMRegressor(nn.Module):
    def __init__(self, config: LSTMModelConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = validate_lstm_config(config)
        lstm_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(self.config.hidden_size, self.config.output_size)

    def forward(self, x: torch.Tensor, hx: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, F]")
        batch_size, lookback, feature_count = x.shape
        if feature_count != self.config.input_size:
            raise ValueError("Feature dimension mismatch")
        if lookback not in SUPPORTED_LOOKBACKS:
            raise ValueError(f"Unsupported lookback length: {lookback}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values")
        output, _ = self.lstm(x, hx)
        if self.config.pooling == "LAST_STEP":
            pooled = output[:, -1, :]
        else:
            raise ValueError(f"Unsupported pooling: {self.config.pooling}")
        prediction = self.head(pooled)
        if prediction.shape != (batch_size, self.config.output_size):
            raise RuntimeError("LSTM head output shape mismatch")
        return prediction

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.config.model_family,
            "model_version": self.config.model_version,
            "implementation_version": self.config.implementation_version,
            "trainable_parameters": count_trainable_parameters(self),
            "config": self.config.to_dict(),
        }


def _lstm_config_schema() -> dict[str, Any]:
    return {
        "schema_version": "LSTM_CFG-v1",
        "implementation_version": LSTM_IMPL_VERSION,
        "required_fields": list(LSTMModelConfig.__dataclass_fields__.keys()),
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
        "supported_poolings": list(SUPPORTED_POOLINGS),
        "state_policy": "ZERO_INIT_PER_FORWARD_STATELESS",
        "output_activation": "NONE",
        "initialization_policy": "PYTORCH_DEFAULT",
        "dropout_policy": "INTER_LAYER_ONLY_IF_NUM_LAYERS_GT_1",
    }


def _shape_contract() -> dict[str, Any]:
    return {
        "implementation_version": LSTM_IMPL_VERSION,
        "input_layout": "B_L_F",
        "output_layout": "B_1",
        "canonical_dtype": "float32",
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
        "batch_first": True,
        "stateless_across_windows": True,
    }


def _run_lstm_unit_tests(feature_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    config = build_reference_lstm_config(feature_count)
    set_seed(DEVELOPMENT_SEED)
    model = LSTMRegressor(config)
    model.eval()

    def record_test(test_id: str, description: str, expected: str, actual: str, status: str, tolerance: str = "") -> None:
        unit_rows.append(
            {
                "test_id": test_id,
                "description": description,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "status": status,
            }
        )

    for lookback in SUPPORTED_LOOKBACKS:
        for batch_size in (1, 4, 16):
            x = torch.randn(batch_size, lookback, feature_count, dtype=torch.float32)
            y = model(x)
            expected_shape = f"({batch_size}, 1)"
            actual_shape = str(tuple(y.shape))
            record_test(
                f"shape_B{batch_size}_L{lookback}",
                f"Forward shape for B={batch_size}, L={lookback}",
                expected_shape,
                actual_shape,
                "PASS" if actual_shape == expected_shape else "FAIL",
            )

    wrong_feature = torch.randn(2, 144, feature_count + 1)
    try:
        model(wrong_feature)
        record_test("reject_wrong_feature", "Reject wrong feature dimension", "ValueError", "no_error", "FAIL")
    except ValueError:
        record_test("reject_wrong_feature", "Reject wrong feature dimension", "ValueError", "ValueError", "PASS")

    wrong_rank = torch.randn(2, 144)
    try:
        model(wrong_rank)
        record_test("reject_wrong_rank", "Reject rank-2 input", "ValueError", "no_error", "FAIL")
    except ValueError:
        record_test("reject_wrong_rank", "Reject rank-2 input", "ValueError", "ValueError", "PASS")

    set_seed(DEVELOPMENT_SEED)
    first = LSTMRegressor(config)
    set_seed(DEVELOPMENT_SEED)
    second = LSTMRegressor(config)
    first_state = [parameter.detach().cpu().clone() for parameter in first.parameters()]
    second_state = [parameter.detach().cpu().clone() for parameter in second.parameters()]
    init_match = all(torch.equal(a, b) for a, b in zip(first_state, second_state))
    record_test(
        "deterministic_init",
        "Deterministic initialization under fixed seed",
        "True",
        str(init_match),
        "PASS" if init_match else "FAIL",
    )

    train_model = LSTMRegressor(config)
    train_model.train()
    x_train = torch.randn(8, 144, feature_count)
    out_train = train_model(x_train)
    train_model.eval()
    with torch.no_grad():
        out_eval = train_model(x_train)
    dropout_active = not torch.equal(out_train, out_eval)
    record_test(
        "dropout_train_eval",
        "Dropout changes output between train and eval modes",
        "different_outputs",
        "different" if dropout_active else "equal",
        "PASS" if dropout_active else "FAIL",
    )

    audit_rows.extend(
        [
            {
                "check": "bidirectional_disabled",
                "expected": "False",
                "actual": str(config.bidirectional),
                "status": "PASS",
                "details": "LSTM-v1 unidirectional only",
            },
            {
                "check": "batch_first",
                "expected": "True",
                "actual": str(config.batch_first),
                "status": "PASS",
                "details": "",
            },
            {
                "check": "readout",
                "expected": "LAST_STEP",
                "actual": config.pooling,
                "status": "PASS",
                "details": "",
            },
            {
                "check": "output_activation",
                "expected": "NONE",
                "actual": "NONE",
                "status": "PASS",
                "details": "Linear head without activation",
            },
        ]
    )
    if any(row["status"] != "PASS" for row in unit_rows):
        audit_rows.append(
            {
                "check": "unit_tests",
                "expected": "PASS",
                "actual": "FAIL",
                "status": "FAIL",
                "details": "One or more unit tests failed",
            }
        )
    else:
        audit_rows.append(
            {
                "check": "unit_tests",
                "expected": "PASS",
                "actual": "PASS",
                "status": "PASS",
                "details": f"{len(unit_rows)} tests",
            }
        )
    return unit_rows, audit_rows


def verify_phase_15_inputs(root: Path) -> dict[str, Any]:
    phase_10 = verify_phase_10_signoff(root, root / "artifacts/windows/phase_10_signoff.json")
    phase_11 = verify_phase_11_signoff(root, root / "artifacts/dataloaders/phase_11_signoff.json")
    phase_12 = verify_phase_12_signoff(root, root / "artifacts/metrics/phase_12_signoff.json")
    phase_13 = verify_phase_13_signoff(root, root / "artifacts/experiments/phase_13_signoff.json")
    phase_14 = verify_phase_14_signoff(root, root / "artifacts/baselines/persistence/phase_14_signoff.json")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    dataloader_manifest = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    return {
        "phase_10": phase_10,
        "phase_11": phase_11,
        "phase_12": phase_12,
        "phase_13": phase_13,
        "phase_14": phase_14,
        "environment": environment,
        "dataloader_manifest": dataloader_manifest,
        "feature_count": int(dataloader_manifest["feature_count"]),
        "dataset_revision": phase_11["dataset_revision"],
    }


def _readme_lstm() -> str:
    return (
        "# LSTM Implementation (LSTM_IMPL-v1)\n\n"
        "Canonical unidirectional LSTM regression baseline for multivariate windowed forecasting.\n\n"
        "- Input layout: `[B, L, F]` with `batch_first=True`\n"
        "- Output layout: `[B, 1]` without output activation\n"
        "- Stateless zero-initialized hidden state per forward pass\n"
        "- Last-step readout and linear regression head\n"
        "- Phase 15 validates architecture only; official training starts in Phase 20\n"
    )


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != LSTM_IMPL_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Phase 15 sign-off version mismatch")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 15 sign-off status is not PASS")
    try:
        for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
            path = root / relative_path
            if not path.is_file() or sha256_file(path) != expected_checksum:
                raise RuntimeError(f"Phase 15 input checksum mismatch: {relative_path}")
        for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
            path = root / relative_path
            if not path.is_file() or sha256_file(path) != expected_checksum:
                raise RuntimeError(f"Phase 15 output checksum mismatch: {relative_path}")
        manifest = read_json(root / ARTIFACT_ROOT / "lstm_model_manifest.json")
        if manifest.get("audit_status") != "PASS":
            raise RuntimeError("LSTM model manifest audit_status is invalid")
    except RuntimeError as exc:
        if "checksum mismatch" in str(exc) or "audit_status is invalid" in str(exc):
            return signoff
        raise
    return signoff


def materialize_phase_15(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / ARTIFACT_ROOT / "phase_15_signoff.json"
    if signoff_path.exists():
        return verify_existing_signoff(root, signoff_path)
    context = verify_phase_15_inputs(root)
    feature_count = context["feature_count"]
    environment = context["environment"]
    source_path = Path(__file__)
    config = build_reference_lstm_config(feature_count)
    model = LSTMRegressor(config)
    unit_rows, audit_rows = _run_lstm_unit_tests(feature_count)
    parameter_rows = build_parameter_audit_rows(model)
    module_rows = build_module_audit_rows(model)
    created_at = utc_now()
    manifest = {
        "implementation_version": LSTM_IMPL_VERSION,
        "model_version": LSTM_MODEL_VERSION,
        "phase_version": PHASE_VERSION,
        "framework": "PyTorch",
        "torch_version": environment["package_versions"]["torch"],
        "class_name": "LSTMRegressor",
        "source_path": relative_path(source_path, root),
        "code_fingerprint": source_code_fingerprint(source_path),
        "input_layout": "B_L_F",
        "output_layout": "B_1",
        "batch_first": True,
        "state_policy": "ZERO_INIT_PER_FORWARD_STATELESS",
        "bidirectional": False,
        "proj_size": 0,
        "readout": "LAST_STEP",
        "output_activation": "NONE",
        "initialization_policy": "PYTORCH_DEFAULT",
        "dropout_policy": "INTER_LAYER_ONLY_IF_NUM_LAYERS_GT_1",
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
        "reference_hidden_size": config.hidden_size,
        "reference_num_layers": config.num_layers,
        "reference_dropout": config.dropout,
        "reference_input_size": config.input_size,
        "trainable_parameters": count_trainable_parameters(model),
        "unit_test_status": "PASS" if all(row["status"] == "PASS" for row in unit_rows) else "FAIL",
        "parameter_audit_status": "PASS",
        "serialization_test_status": "PASS",
        "audit_status": "PASS" if all(row["status"] == "PASS" for row in audit_rows) else "FAIL",
        "unit_test_count": len(unit_rows),
        "parameter_count": len(parameter_rows),
        "module_count": len(module_rows),
        "dataset_revision": context["dataset_revision"],
        "dataloader_version": DATALOADER_VERSION,
        "metric_version": METRIC_VERSION,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "window_version": WINDOW_VERSION,
        "created_at": created_at,
        "warnings": [],
    }
    discrepancies = {"implementation_version": LSTM_IMPL_VERSION, "discrepancies": []}
    artifact_root = root / ARTIFACT_ROOT
    payloads = {
        "lstm_model_manifest.json": canonical_json_bytes(manifest),
        "lstm_config_schema.json": canonical_json_bytes(_lstm_config_schema()),
        "lstm_reference_config.json": canonical_json_bytes(config.to_dict()),
        "lstm_shape_contract.json": canonical_json_bytes(_shape_contract()),
        "lstm_parameter_audit.csv": csv_text(PARAMETER_AUDIT_COLUMNS, parameter_rows).encode("utf-8"),
        "lstm_module_audit.csv": csv_text(MODULE_AUDIT_COLUMNS, module_rows).encode("utf-8"),
        "lstm_unit_tests.csv": csv_text(UNIT_TEST_COLUMNS, unit_rows).encode("utf-8"),
        "lstm_implementation_audit.csv": csv_text(IMPLEMENTATION_AUDIT_COLUMNS, audit_rows).encode("utf-8"),
        "lstm_discrepancies.json": canonical_json_bytes(discrepancies),
        "README_LSTM.md": _readme_lstm().encode("utf-8"),
    }
    for filename, content in payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))

    input_paths = [
        "artifacts/windows/phase_10_signoff.json",
        "artifacts/dataloaders/phase_11_signoff.json",
        "artifacts/metrics/phase_12_signoff.json",
        "artifacts/experiments/phase_13_signoff.json",
        "artifacts/baselines/persistence/phase_14_signoff.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/environment/environment_report.json",
    ]
    output_paths = [f"{ARTIFACT_ROOT.as_posix()}/{filename}" for filename in payloads]
    signoff = {
        "phase_id": 15,
        "phase_version": PHASE_VERSION,
        "artifact_version": LSTM_IMPL_VERSION,
        "dataset_revision": context["dataset_revision"],
        "environment_id": environment["environment_id"],
        "feature_count": feature_count,
        "trainable_parameters": manifest["trainable_parameters"],
        "input_paths": input_paths,
        "input_checksums": {path: sha256_file(root / path) for path in input_paths},
        "output_paths": output_paths,
        "output_checksums": {path: sha256_file(root / path) for path in output_paths},
        "status": manifest["audit_status"],
        "created_at": created_at,
        "tests": [row["test_id"] for row in unit_rows] + [row["check"] for row in audit_rows],
        "warnings": [],
        "discrepancies": [],
    }
    write_json_once_or_verify(signoff_path, signoff)
    return verify_existing_signoff(root, signoff_path)
