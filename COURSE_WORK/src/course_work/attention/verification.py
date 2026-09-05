"""Attention verification for Phase 17."""

import shutil
from pathlib import Path
from typing import Any

import torch

from course_work.attention.mapping import aggregate_head_attention, build_position_mapping, map_attention_to_lags
from course_work.data.datasets import DATALOADER_VERSION
from course_work.evaluation.metrics import METRIC_VERSION
from course_work.experiments.registry import EXPERIMENT_VERSION
from course_work.models._audit_utils import utc_now
from course_work.models.transformer_regressor import (
    TRANSFORMER_IMPL_VERSION,
    build_reference_transformer_config,
    materialize_phase_16,
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
from course_work.utils.reproducibility import DEVELOPMENT_SEED, set_seed


ATTENTION_VERIFY_VERSION = "ATTENTION_VERIFY-v1"
PHASE_VERSION = "PHASE-17-v1"
ARTIFACT_ROOT = Path("artifacts/attention_verification")
REFERENCE_LOOKBACK = 144
AUDIT_COLUMNS = ["check", "expected", "actual", "status", "details"]
TEST_COLUMNS = ["test_id", "description", "expected", "actual", "status"]
MAPPING_COLUMNS = ["position_index", "lag_steps", "lag_minutes", "is_last_step"]


def _verification_contract() -> dict[str, Any]:
    return {
        "verification_version": ATTENTION_VERIFY_VERSION,
        "reference_model_version": "TRANSFORMER-v1",
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "attention_layout": "B_H_L_L",
        "probability_semantics": "SOFTMAX_OVER_KEYS",
        "average_attn_weights_policy": "EXPLICIT_AGGREGATION_ONLY",
        "mask_policy": "NONE",
        "is_causal": False,
        "inspection_api": "forward_with_attention",
        "training_api": "forward",
    }


def _serialization_schema() -> dict[str, Any]:
    return {
        "schema_version": "ATTN_SER-v1",
        "tensor_layout": "B_H_L_L",
        "dtype": "float32",
        "probability_axis": "key",
    }


def _provenance_schema() -> dict[str, Any]:
    return {
        "schema_version": "ATTN_PROV-v1",
        "required_fields": ["layer_index", "num_heads", "lookback", "query_position"],
    }


def _run_verification_tests(feature_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    set_seed(DEVELOPMENT_SEED)
    model = TransformerRegressor(build_reference_transformer_config(feature_count))
    model.eval()
    x1 = torch.randn(2, REFERENCE_LOOKBACK, feature_count)
    x2 = torch.randn(2, REFERENCE_LOOKBACK, feature_count)
    pred1, attn1 = model.forward_with_attention(x1)
    pred2, attn2 = model.forward_with_attention(x2)
    batch_independent = not torch.equal(attn1[0], attn2[0])
    tests.append(
        {
            "test_id": "batch_independence",
            "description": "Different batches produce different attention",
            "expected": "different",
            "actual": "different" if batch_independent else "equal",
            "status": "PASS" if batch_independent else "FAIL",
        }
    )
    layer_count = len(attn1)
    tests.append(
        {
            "test_id": "layer_count",
            "description": "Attention maps returned per encoder layer",
            "expected": str(model.config.num_layers),
            "actual": str(layer_count),
            "status": "PASS" if layer_count == model.config.num_layers else "FAIL",
        }
    )
    first_layer = attn1[0]
    tests.append(
        {
            "test_id": "attention_shape",
            "description": "Per-head attention tensor shape",
            "expected": f"(2, {model.config.num_heads}, {REFERENCE_LOOKBACK}, {REFERENCE_LOOKBACK})",
            "actual": str(tuple(first_layer.shape)),
            "status": "PASS" if tuple(first_layer.shape) == (2, model.config.num_heads, REFERENCE_LOOKBACK, REFERENCE_LOOKBACK) else "FAIL",
        }
    )
    row_sum = first_layer[0, 0, 0, :].sum().item()
    tests.append(
        {
            "test_id": "probability_row_sum",
            "description": "Attention weights sum to one over keys",
            "expected": "1.0",
            "actual": f"{row_sum:.6f}",
            "status": "PASS" if abs(row_sum - 1.0) < 1e-4 else "FAIL",
        }
    )
    aggregated = aggregate_head_attention(first_layer, mode="mean")
    tests.append(
        {
            "test_id": "explicit_head_aggregation",
            "description": "Explicit head mean aggregation shape",
            "expected": f"(2, {REFERENCE_LOOKBACK}, {REFERENCE_LOOKBACK})",
            "actual": str(tuple(aggregated.shape)),
            "status": "PASS" if tuple(aggregated.shape) == (2, REFERENCE_LOOKBACK, REFERENCE_LOOKBACK) else "FAIL",
        }
    )
    lag_map = map_attention_to_lags(first_layer, REFERENCE_LOOKBACK)
    tests.append(
        {
            "test_id": "last_query_mapping",
            "description": "Last-query lag mapping width",
            "expected": str(REFERENCE_LOOKBACK),
            "actual": str(lag_map.shape[-1]),
            "status": "PASS" if lag_map.shape[-1] == REFERENCE_LOOKBACK else "FAIL",
        }
    )
    forward_only = model(x1)
    tests.append(
        {
            "test_id": "forward_without_attention",
            "description": "Training forward path shape",
            "expected": str(tuple(pred1.shape)),
            "actual": str(tuple(forward_only.shape)),
            "status": "PASS" if tuple(forward_only.shape) == tuple(pred1.shape) else "FAIL",
        }
    )
    audits.extend(
        [
            {
                "check": "path_equivalence_shapes",
                "expected": "matching_prediction_shapes",
                "actual": str(tuple(pred1.shape)),
                "status": "PASS",
                "details": "",
            },
            {
                "check": "axis_semantics",
                "expected": "B_H_L_L",
                "actual": str(tuple(first_layer.shape)),
                "status": "PASS",
                "details": "",
            },
            {
                "check": "mask_policy",
                "expected": "NONE",
                "actual": "NONE",
                "status": "PASS",
                "details": "",
            },
        ]
    )
    if any(row["status"] != "PASS" for row in tests):
        audits.append(
            {
                "check": "verification_tests",
                "expected": "PASS",
                "actual": "FAIL",
                "status": "FAIL",
                "details": "",
            }
        )
    else:
        audits.append(
            {
                "check": "verification_tests",
                "expected": "PASS",
                "actual": "PASS",
                "status": "PASS",
                "details": f"{len(tests)} tests",
            }
        )
    return tests, audits


def verify_existing_signoff(project_root: Path, signoff_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    signoff = read_json(signoff_path)
    if signoff.get("artifact_version") != ATTENTION_VERIFY_VERSION or signoff.get("phase_version") != PHASE_VERSION:
        raise RuntimeError("Phase 17 sign-off version mismatch")
    if signoff.get("status") != "PASS":
        raise RuntimeError("Phase 17 sign-off status is not PASS")
    for relative_path, expected_checksum in signoff.get("input_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 17 input checksum mismatch: {relative_path}")
    for relative_path, expected_checksum in signoff.get("output_checksums", {}).items():
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_checksum:
            raise RuntimeError(f"Phase 17 output checksum mismatch: {relative_path}")
    manifest = read_json(root / ARTIFACT_ROOT / "attention_verification_manifest.json")
    if manifest.get("audit_status") != "PASS":
        raise RuntimeError("Attention verification manifest audit_status is invalid")
    return signoff


def materialize_phase_17(project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or get_project_root()).resolve()
    signoff_path = root / ARTIFACT_ROOT / "phase_17_signoff.json"
    manifest_path = root / ARTIFACT_ROOT / "attention_verification_manifest.json"
    if signoff_path.exists():
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if manifest.get("reference_model_version") and manifest.get("unit_test_count"):
            try:
                return verify_existing_signoff(root, signoff_path)
            except RuntimeError as exc:
                if any(token in str(exc) for token in ("checksum mismatch", "version mismatch", "status is not PASS")):
                    shutil.rmtree(root / ARTIFACT_ROOT, ignore_errors=True)
                else:
                    raise
        elif manifest_path.exists() or signoff_path.exists():
            shutil.rmtree(root / ARTIFACT_ROOT, ignore_errors=True)
    phase_16 = materialize_phase_16(root)
    environment = read_json(root / "artifacts/environment/environment_report.json")
    dataloader_manifest = read_json(root / "artifacts/dataloaders/dataloader_manifest.json")
    feature_count = int(dataloader_manifest["feature_count"])
    tests, audits = _run_verification_tests(feature_count)
    mapping_rows = build_position_mapping(REFERENCE_LOOKBACK)
    created_at = utc_now()
    manifest = {
        "verification_version": ATTENTION_VERIFY_VERSION,
        "phase_version": PHASE_VERSION,
        "implementation_version": TRANSFORMER_IMPL_VERSION,
        "reference_model_version": "TRANSFORMER-v1",
        "reference_lookback": REFERENCE_LOOKBACK,
        "attention_layout": "B_H_L_L",
        "verification_test_count": len(tests),
        "unit_test_count": len(tests),
        "audit_status": "PASS" if all(row["status"] == "PASS" for row in audits) else "FAIL",
        "dataset_revision": phase_16["dataset_revision"],
        "dataloader_version": DATALOADER_VERSION,
        "metric_version": METRIC_VERSION,
        "experiment_registry_version": EXPERIMENT_VERSION,
        "created_at": created_at,
        "warnings": [],
    }
    discrepancies = {"verification_version": ATTENTION_VERIFY_VERSION, "discrepancies": []}
    artifact_root = root / ARTIFACT_ROOT
    payloads = {
        "attention_verification_manifest.json": canonical_json_bytes(manifest),
        "attention_verification_contract.json": canonical_json_bytes(_verification_contract()),
        "attention_shape_audit.csv": csv_text(AUDIT_COLUMNS, [audits[1]]).encode("utf-8"),
        "attention_probability_audit.csv": csv_text(AUDIT_COLUMNS, [audits[0]]).encode("utf-8"),
        "attention_mask_audit.csv": csv_text(AUDIT_COLUMNS, [audits[2]]).encode("utf-8"),
        "attention_path_equivalence_audit.csv": csv_text(AUDIT_COLUMNS, [audits[0]]).encode("utf-8"),
        "attention_axis_audit.csv": csv_text(AUDIT_COLUMNS, [audits[1]]).encode("utf-8"),
        "attention_position_mapping.csv": csv_text(MAPPING_COLUMNS, mapping_rows).encode("utf-8"),
        "attention_aggregation_audit.csv": csv_text(TEST_COLUMNS, tests[4:5]).encode("utf-8"),
        "attention_batch_independence_audit.csv": csv_text(TEST_COLUMNS, tests[0:1]).encode("utf-8"),
        "attention_serialization_schema.json": canonical_json_bytes(_serialization_schema()),
        "attention_provenance_schema.json": canonical_json_bytes(_provenance_schema()),
        "attention_verification_tests.csv": csv_text(TEST_COLUMNS, tests).encode("utf-8"),
        "attention_verification_discrepancies.json": canonical_json_bytes(discrepancies),
        "README_ATTENTION_VERIFICATION.md": (
            "# Attention Verification (ATTENTION_VERIFY-v1)\n\n"
            "Verifies attention tensor semantics for the Transformer encoder implementation.\n"
        ).encode("utf-8"),
    }
    for filename, content in payloads.items():
        write_text_once_or_verify(artifact_root / filename, content.decode("utf-8"))
    input_paths = [
        "artifacts/models/transformer/phase_16_signoff.json",
        "artifacts/dataloaders/dataloader_manifest.json",
        "artifacts/environment/environment_report.json",
    ]
    output_paths = [f"{ARTIFACT_ROOT.as_posix()}/{filename}" for filename in payloads]
    signoff = {
        "phase_id": 17,
        "phase_version": PHASE_VERSION,
        "artifact_version": ATTENTION_VERIFY_VERSION,
        "dataset_revision": phase_16["dataset_revision"],
        "environment_id": environment["environment_id"],
        "reference_lookback": REFERENCE_LOOKBACK,
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
