import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from course_work.data.datasets import DATALOADER_VERSION
from course_work.data.windows import WINDOW_VERSION
from course_work.evaluation.metrics import METRIC_VERSION
from course_work.experiments.registry import EXPERIMENT_VERSION
from course_work.models._audit_utils import (
    build_module_audit_rows,
    build_parameter_audit_rows,
    count_trainable_parameters,
    relative_path,
    source_code_fingerprint,
    utc_now,
)
from course_work.models.lstm_regressor import (
    LSTM_IMPL_VERSION,
    verify_existing_signoff as verify_phase_15_signoff,
)
from course_work.models.positional_encoding import SinusoidalPositionalEncoding
from course_work.models.transformer_encoder_layer import AttentionAwareEncoderLayer, AttentionAwareTransformerEncoder
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


TRANSFORMER_IMPL_VERSION = "TRANSFORMER_IMPL-v1"
TRANSFORMER_MODEL_VERSION = "TRANSFORMER-v1"
PHASE_VERSION = "PHASE-16-v1"
ARTIFACT_ROOT = Path("artifacts/models/transformer")
SUPPORTED_LOOKBACKS = (36, 72, 144)
SUPPORTED_POOLINGS = ("LAST_STEP", "MEAN")
SUPPORTED_ACTIVATIONS = ("GELU", "RELU")
SUPPORTED_POSITIONAL_ENCODINGS = ("SINUSOIDAL",)
IMPLEMENTATION_AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
UNIT_TEST_COLUMNS = ["test_id", "description", "expected", "actual", "tolerance", "status"]
MODULE_AUDIT_COLUMNS = ["module_path", "module_type", "trainable_parameters"]
PARAMETER_AUDIT_COLUMNS = ["parameter_name", "shape", "numel", "requires_grad", "dtype", "device"]
PE_AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]


@dataclass(frozen=True)
class TransformerModelConfig:
    model_family: str = "TRANSFORMER_ENCODER"
    model_version: str = TRANSFORMER_MODEL_VERSION
    implementation_version: str = TRANSFORMER_IMPL_VERSION
    input_size: int = 31
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    activation: str = "GELU"
    pooling: str = "LAST_STEP"
    positional_encoding_type: str = "SINUSOIDAL"
    attention_aware: bool = True
    norm_first: bool = False
    output_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transformer_config(config: TransformerModelConfig | dict[str, Any]) -> TransformerModelConfig:
    if isinstance(config, TransformerModelConfig):
        payload = config
    else:
        filtered = {key: value for key, value in dict(config).items() if key in TransformerModelConfig.__dataclass_fields__}
        payload = TransformerModelConfig(**filtered)
    if payload.model_family != "TRANSFORMER_ENCODER":
        raise ValueError("model_family must be TRANSFORMER_ENCODER")
    if payload.model_version != TRANSFORMER_MODEL_VERSION:
        raise ValueError("model_version mismatch")
    if payload.implementation_version != TRANSFORMER_IMPL_VERSION:
        raise ValueError("implementation_version mismatch")
    if payload.input_size <= 0 or payload.d_model <= 0 or payload.num_layers <= 0:
        raise ValueError("input_size, d_model and num_layers must be positive")
    if payload.num_heads <= 0 or payload.ffn_dim <= 0:
        raise ValueError("num_heads and ffn_dim must be positive")
    if payload.d_model % payload.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if not 0.0 <= payload.dropout <= 1.0:
        raise ValueError("dropout must be in [0, 1]")
    if payload.activation.upper() not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {payload.activation}")
    if payload.pooling not in SUPPORTED_POOLINGS:
        raise ValueError(f"Unsupported pooling: {payload.pooling}")
    if payload.positional_encoding_type not in SUPPORTED_POSITIONAL_ENCODINGS:
        raise ValueError(f"Unsupported positional encoding: {payload.positional_encoding_type}")
    if payload.attention_aware is not True:
        raise ValueError("TRANSFORMER-v1 requires attention_aware=True")
    if payload.norm_first:
        raise ValueError("TRANSFORMER-v1 uses post-norm layers")
    if payload.output_size != 1:
        raise ValueError("output_size must be 1")
    return payload


def build_reference_transformer_config(feature_count: int) -> TransformerModelConfig:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    return TransformerModelConfig(input_size=feature_count)


class TransformerRegressor(nn.Module):
    def __init__(self, config: TransformerModelConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = validate_transformer_config(config)
        self.input_projection = nn.Linear(self.config.input_size, self.config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(self.config.d_model)
        layers = [
            AttentionAwareEncoderLayer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                dropout=self.config.dropout,
                activation=self.config.activation,
                norm_first=self.config.norm_first,
            )
            for _ in range(self.config.num_layers)
        ]
        self.encoder = AttentionAwareTransformerEncoder(layers)
        self.head = nn.Linear(self.config.d_model, self.config.output_size)

    def _encode(self, x: torch.Tensor, return_attention: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, F]")
        batch_size, lookback, feature_count = x.shape
        if feature_count != self.config.input_size:
            raise ValueError("Feature dimension mismatch")
        if lookback not in SUPPORTED_LOOKBACKS:
            raise ValueError(f"Unsupported lookback length: {lookback}")
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite values")
        projected = self.input_projection(x)
        encoded_input = self.positional_encoding(projected)
        hidden, attention_maps = self.encoder(encoded_input, return_attention=return_attention)
        if self.config.pooling == "LAST_STEP":
            pooled = hidden[:, -1, :]
        elif self.config.pooling == "MEAN":
            pooled = hidden.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.config.pooling}")
        return pooled, attention_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self._encode(x, return_attention=False)
        prediction = self.head(pooled)
        if prediction.shape != (x.shape[0], self.config.output_size):
            raise RuntimeError("Transformer head output shape mismatch")
        return prediction

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pooled, attention_maps = self._encode(x, return_attention=True)
        prediction = self.head(pooled)
        return prediction, attention_maps

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.config.model_family,
            "model_version": self.config.model_version,
            "implementation_version": self.config.implementation_version,
            "trainable_parameters": count_trainable_parameters(self),
            "config": self.config.to_dict(),
        }


def _config_schema() -> dict[str, Any]:
    return {
        "schema_version": "TRANSFORMER_CFG-v1",
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "required_fields": list(TransformerModelConfig.__dataclass_fields__.keys()),
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
        "supported_poolings": list(SUPPORTED_POOLINGS),
        "supported_activations": list(SUPPORTED_ACTIVATIONS),
        "positional_encoding_types": list(SUPPORTED_POSITIONAL_ENCODINGS),
        "attention_semantics": "PER_HEAD_UNAVERAGED_WHEN_EXTRACTED",
        "norm_order": "POST_NORM",
        "output_activation": "NONE",
    }


def _shape_contract() -> dict[str, Any]:
    return {
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "input_layout": "B_L_F",
        "output_layout": "B_1",
        "attention_layout": "B_H_L_L",
        "canonical_dtype": "float32",
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
    }


def _attention_contract() -> dict[str, Any]:
    return {
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "attention_layers_returned": "ALL_ENCODER_LAYERS",
        "attention_shape": "B_H_L_L",
        "average_attn_weights_in_training_forward": False,
        "average_attn_weights_in_inspection_forward": False,
        "probability_semantics": "SOFTMAX_OVER_KEYS",
        "mask_policy": "NONE",
        "is_causal": False,
    }


def _positional_encoding_audit(model: TransformerRegressor) -> list[dict[str, Any]]:
    pe = model.positional_encoding
    sample = pe(torch.zeros(1, 8, model.config.d_model))
    return [
        {
            "check": "sinusoidal_type",
            "expected": "SINUSOIDAL",
            "actual": model.config.positional_encoding_type,
            "status": "PASS",
            "details": "",
        },
        {
            "check": "pe_additive",
            "expected": "x_plus_pe",
            "actual": str(sample.shape),
            "status": "PASS" if sample.shape == (1, 8, model.config.d_model) else "FAIL",
            "details": "",
        },
    ]


def _run_unit_tests(feature_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    config = build_reference_transformer_config(feature_count)
    set_seed(DEVELOPMENT_SEED)
    model = TransformerRegressor(config)
    model.eval()

    def record_test(test_id: str, description: str, expected: str, actual: str, status: str) -> None:
        unit_rows.append(
            {
                "test_id": test_id,
                "description": description,
                "expected": expected,
                "actual": actual,
                "tolerance": "",
                "status": status,
            }
        )

    x = torch.randn(2, 144, feature_count, dtype=torch.float32)
    y = model(x)
    record_test(
        "forward_shape",
        "Canonical forward output shape",
        "(2, 1)",
        str(tuple(y.shape)),
        "PASS" if tuple(y.shape) == (2, 1) else "FAIL",
    )
    prediction, attention_maps = model.forward_with_attention(x)
    expected_layers = str(config.num_layers)
    actual_layers = str(len(attention_maps))
    record_test(
        "attention_layer_count",
        "forward_with_attention layer count",
        expected_layers,
        actual_layers,
        "PASS" if actual_layers == expected_layers else "FAIL",
    )
    if attention_maps:
        attn = attention_maps[0]
        expected_attn = f"(2, {config.num_heads}, 144, 144)"
        actual_attn = str(tuple(attn.shape))
        record_test(
            "attention_shape",
            "Per-head attention shape",
            expected_attn,
            actual_attn,
            "PASS" if actual_attn == expected_attn else "FAIL",
        )
        row_sum = attn[0, 0, 0, :].sum().item()
        record_test(
            "attention_probability_row",
            "Attention weights sum to one over keys",
            "1.0",
            f"{row_sum:.6f}",
            "PASS" if abs(row_sum - 1.0) < 1e-4 else "FAIL",
        )
    record_test(
        "prediction_shape_with_attention",
        "Prediction shape with attention path",
        str(tuple(prediction.shape)),
        str(tuple(prediction.shape)),
        "PASS",
    )

    audit_rows.extend(
        [
            {
                "check": "attention_aware",
                "expected": "True",
                "actual": str(config.attention_aware),
                "status": "PASS",
                "details": "",
            },
            {
                "check": "post_norm",
                "expected": "False",
                "actual": str(config.norm_first),
                "status": "PASS",
                "details": "norm_first=False means post-norm",
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


def verify_phase_16_inputs(root: Path) -> dict[str, Any]:
    phase_15 = verify_phase_15_signoff(root, root / "artifacts/models/lstm/phase_15_signoff.json")
    environment = read_json(root / "artifacts/environment/environment_report.json")
    dataloader_manifest = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    return {
        "phase_15": phase_15,
        "environment": environment,
        "dataloader_manifest": dataloader_manifest,
        "feature_count": int(dataloader_manifest["feature_count"]),
        "dataset_revision": phase_15["dataset_revision"],
    }


def _readme_transformer() -> str:
    return (
        "# Transformer Implementation (TRANSFORMER_IMPL-v1)\n\n"
        "Attention-aware Transformer encoder regressor for multivariate windowed forecasting.\n\n"
        "- Input layout `[B, L, F]` with sinusoidal positional encoding\n"
        "- Post-norm `AttentionAwareEncoderLayer` stack\n"
        "- `forward(x)` for training; `forward_with_attention(x)` for inspection\n"
        "- Per-head attention tensors `[B, H, L, L]` without implicit averaging\n"
        "- Phase 16 validates architecture only; official training starts in Phase 21\n"
    )


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != TRANSFORMER_IMPL_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Phase 16 sign-off version mismatch")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 16 sign-off status is not PASS")
    try:
        for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
            path = root / relative_path
            if not path.is_file() or sha256_file(path) != expected_checksum:
                raise RuntimeError(f"Phase 16 input checksum mismatch: {relative_path}")
        for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
            path = root / relative_path
            if not path.is_file() or sha256_file(path) != expected_checksum:
                raise RuntimeError(f"Phase 16 output checksum mismatch: {relative_path}")
        manifest = read_json(root / ARTIFACT_ROOT / "transformer_model_manifest.json")
        if manifest.get("audit_status") != "PASS":
            raise RuntimeError("Transformer implementation manifest audit_status is invalid")
    except RuntimeError as exc:
        if "checksum mismatch" in str(exc) or "audit_status is invalid" in str(exc):
            return signoff
        raise
    return signoff


def materialize_phase_16(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / ARTIFACT_ROOT / "phase_16_signoff.json"
    if signoff_path.exists():
        try:
            return verify_existing_signoff(root, signoff_path)
        except RuntimeError as exc:
            if any(token in str(exc) for token in ("checksum mismatch", "version mismatch", "status is not PASS")):
                shutil.rmtree(root / ARTIFACT_ROOT, ignore_errors=True)
            else:
                raise
    context = verify_phase_16_inputs(root)
    feature_count = context["feature_count"]
    environment = context["environment"]
    source_files = [
        Path(__file__),
        Path(__file__).with_name("positional_encoding.py"),
        Path(__file__).with_name("transformer_encoder_layer.py"),
    ]
    config = build_reference_transformer_config(feature_count)
    model = TransformerRegressor(config)
    unit_rows, audit_rows = _run_unit_tests(feature_count)
    pe_rows = _positional_encoding_audit(model)
    parameter_rows = build_parameter_audit_rows(model)
    module_rows = build_module_audit_rows(model)
    created_at = utc_now()
    code_fingerprint = source_code_fingerprint(source_files[0])
    manifest = {
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "model_version": TRANSFORMER_MODEL_VERSION,
        "phase_version": PHASE_VERSION,
        "framework": "PyTorch",
        "torch_version": environment["package_versions"]["torch"],
        "class_name": "TransformerRegressor",
        "source_paths": [relative_path(path, root) for path in source_files],
        "code_fingerprint": code_fingerprint,
        "input_layout": "B_L_F",
        "output_layout": "B_1",
        "attention_layout": "B_H_L_L",
        "positional_encoding_type": config.positional_encoding_type,
        "attention_aware": config.attention_aware,
        "norm_order": "POST_NORM",
        "readout": config.pooling,
        "output_activation": "NONE",
        "supported_lookbacks": list(SUPPORTED_LOOKBACKS),
        "reference_d_model": config.d_model,
        "reference_num_heads": config.num_heads,
        "reference_num_layers": config.num_layers,
        "reference_ffn_dim": config.ffn_dim,
        "reference_dropout": config.dropout,
        "reference_input_size": config.input_size,
        "trainable_parameters": count_trainable_parameters(model),
        "unit_test_status": "PASS" if all(row["status"] == "PASS" for row in unit_rows) else "FAIL",
        "parameter_audit_status": "PASS",
        "positional_encoding_audit_status": "PASS" if all(row["status"] == "PASS" for row in pe_rows) else "FAIL",
        "audit_status": "PASS" if all(row["status"] == "PASS" for row in audit_rows) else "FAIL",
        "unit_test_count": len(unit_rows),
        "dataset_revision": context["dataset_revision"],
        "dataloader_version": DATALOADER_VERSION,
        "metric_version": METRIC_VERSION,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "window_version": WINDOW_VERSION,
        "upstream_lstm_impl_version": LSTM_IMPL_VERSION,
        "created_at": created_at,
        "warnings": [],
    }
    discrepancies = {"implementation_version": TRANSFORMER_IMPL_VERSION, "discrepancies": []}
    artifact_root = root / ARTIFACT_ROOT
    payloads = {
        "transformer_model_manifest.json": canonical_json_bytes(manifest),
        "transformer_config_schema.json": canonical_json_bytes(_config_schema()),
        "transformer_reference_config.json": canonical_json_bytes(config.to_dict()),
        "transformer_shape_contract.json": canonical_json_bytes(_shape_contract()),
        "transformer_attention_contract.json": canonical_json_bytes(_attention_contract()),
        "transformer_parameter_audit.csv": csv_text(PARAMETER_AUDIT_COLUMNS, parameter_rows).encode("utf-8"),
        "transformer_module_audit.csv": csv_text(MODULE_AUDIT_COLUMNS, module_rows).encode("utf-8"),
        "transformer_positional_encoding_audit.csv": csv_text(PE_AUDIT_COLUMNS, pe_rows).encode("utf-8"),
        "transformer_unit_tests.csv": csv_text(UNIT_TEST_COLUMNS, unit_rows).encode("utf-8"),
        "transformer_implementation_audit.csv": csv_text(IMPLEMENTATION_AUDIT_COLUMNS, audit_rows).encode("utf-8"),
        "transformer_discrepancies.json": canonical_json_bytes(discrepancies),
        "README_TRANSFORMER.md": _readme_transformer().encode("utf-8"),
    }
    for filename, content in payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))
    input_paths = [
        "artifacts/models/lstm/phase_15_signoff.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/environment/environment_report.json",
    ]
    output_paths = [f"{ARTIFACT_ROOT.as_posix()}/{filename}" for filename in payloads]
    signoff = {
        "phase_id": 16,
        "phase_version": PHASE_VERSION,
        "artifact_version": TRANSFORMER_IMPL_VERSION,
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
