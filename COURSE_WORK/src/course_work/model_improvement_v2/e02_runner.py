"""E02 controlled FS1_TF1 versus accepted E01 FS2_TF1 ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.data.feature_sets import (
    EXPECTED_FEATURE_COUNTS,
    compute_feature_fingerprint,
    get_feature_list,
)
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)

EXPERIMENT_ID = "E02"
SUPPORTED_SEED = 42
CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
CONTROL_VARIANT = "FS2_TF1"
CHALLENGER_VARIANT = "FS1_TF1"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E02"
REGISTRY_NAMESPACE = "V2_E02"
CONFIG_RELATIVE_PATH = Path(
    "artifacts/model_improvement_v2/experiments/E02/e02_config_snapshot.json"
)
E01_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E01")
EXPECTED_CONTROL_METRICS = {
    "rmse_wh": 59.85291570400546,
    "mae_wh": 26.650501720144604,
    "r2": 0.5789433617557903,
}
PROMOTION_RMSE_TOLERANCE_WH = 0.10
PROMOTION_THRESHOLD_RMSE_WH = 59.75291570400546
EXPECTED_CONFIG_CHANGES = (
    "data.feature_count",
    "data.feature_variant_id",
    "lineage.feature_fingerprint",
    "lineage.scaler_bundle_checksum",
    "lineage.scaler_bundle_id",
    "model.input_size",
)


class E02PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E02PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E02PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e02_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_RELATIVE_PATH)


def _expected_challenger_config(root: Path) -> dict[str, Any]:
    e01 = _read_json(root / E01_RELATIVE_ROOT / "e01_config_snapshot.json")
    config = deepcopy(e01["v1_config"])
    features = get_feature_list(CHALLENGER_VARIANT)
    scaler_registry = _read_json(root / "artifacts/scaling/scaler_registry.json")
    scaler = scaler_registry["x_bundles"][CHALLENGER_VARIANT]
    config["data"]["feature_variant_id"] = CHALLENGER_VARIANT
    config["data"]["feature_count"] = len(features)
    config["model"]["input_size"] = len(features)
    config["lineage"]["feature_fingerprint"] = compute_feature_fingerprint(
        CHALLENGER_VARIANT, features
    )
    config["lineage"]["scaler_bundle_id"] = scaler["bundle_id"]
    config["lineage"]["scaler_bundle_checksum"] = scaler["artifact_sha256"]
    return config


def _validate_control_reuse(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    control = document.get("control", {})
    if control.get("reuse_policy") != "CONTROL_REUSED_FROM_E01":
        raise E02PreflightError("E02 control must be reused from accepted E01")
    e01_root = root / E01_RELATIVE_ROOT
    for name, expected in control.get("artifact_sha256", {}).items():
        path = e01_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise E02PreflightError(f"E01 control artifact changed: {name}")
    comparison = _read_json(e01_root / "e01_reproduction_comparison.json")
    observed = {
        "rmse_wh": comparison.get("observed_rmse_wh"),
        "mae_wh": comparison.get("observed_mae_wh"),
        "r2": comparison.get("observed_r2"),
    }
    if observed != EXPECTED_CONTROL_METRICS or control.get("pooled_metrics") != observed:
        raise E02PreflightError("E01 accepted control metrics mismatch")
    if comparison.get("target_population_match") is not True:
        raise E02PreflightError("E01 control target population was not canonical")
    if comparison.get("test_status") != "NOT_ACCESSED":
        raise E02PreflightError("E01 control does not satisfy the Test firewall")
    execution = _read_json(e01_root / "e01_execution_manifest.json")
    if execution.get("run_ids") != control.get("run_ids"):
        raise E02PreflightError("E01 control run ledger mismatch")
    run_ids = [
        run_id
        for stage in ("stage_a", "stage_b")
        for run_id in execution["run_ids"][stage].values()
    ]
    if len(run_ids) != 6 or len(set(run_ids)) != 6 or not all("_E01_" in value for value in run_ids):
        raise E02PreflightError("E01 control run ledger is incomplete")
    return {"status": "PASS", "run_count": 6, "metrics": observed}


def validate_e02_document(
    document: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise E02PreflightError("Wrong experiment ID")
    if document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E02PreflightError("Wrong V2 track")
    fs1 = get_feature_list(CHALLENGER_VARIANT)
    fs2 = get_feature_list(CONTROL_VARIANT)
    if len(fs1) != EXPECTED_FEATURE_COUNTS[CHALLENGER_VARIANT] or len(fs2) != EXPECTED_FEATURE_COUNTS[CONTROL_VARIANT]:
        raise E02PreflightError("Canonical feature count mismatch")
    if [feature for feature in fs2 if feature not in {"rv1", "rv2"}] != fs1:
        raise E02PreflightError("FS1_TF1 is not the ordered FS2_TF1 projection without rv1/rv2")
    if set(fs2) - set(fs1) != {"rv1", "rv2"} or set(fs1) - set(fs2):
        raise E02PreflightError("Unexpected FS1/FS2 feature delta")

    challenger = document.get("challenger", {})
    config = challenger.get("config")
    if not isinstance(config, dict):
        raise E02PreflightError("Missing challenger config")
    assert_no_test_access(split_id=str(config.get("data", {}).get("target_access_mode", "")))
    orchestration = document.get("v2_orchestration", {})
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    if challenger.get("feature_order") != fs1 or challenger.get("feature_count") != 31:
        raise E02PreflightError("Challenger feature projection mismatch")
    expected_config = _expected_challenger_config(root)
    if config != expected_config:
        raise E02PreflightError("E02 config differs beyond the canonical feature projection")
    e01_config = _read_json(root / E01_RELATIVE_ROOT / "e01_config_snapshot.json")["v1_config"]
    changed = assert_one_primary_change(EXPERIMENT_ID, e01_config, config)
    if changed != EXPECTED_CONFIG_CHANGES:
        raise E02PreflightError(f"Unexpected E02 changed fields: {changed}")
    fingerprint = compute_config_fingerprint(config)
    if challenger.get("config_fingerprint") != fingerprint:
        raise E02PreflightError("E02 challenger fingerprint mismatch")
    if challenger.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD":
        raise E02PreflightError("E02 must initialize FS1 from scratch")

    expected_orchestration = {
        "rolling_origin_protocol": "RO3_EXPANDING_PRETEST-v1",
        "folds": ["RO1", "RO2", "RO3"],
        "fold_count": 3,
        "fold_local_scaling": True,
        "control_retrained": False,
        "challenger_only_training": True,
        "training_authorized": False,
        "preflight_consumes_run_id": False,
        "registry_namespace": REGISTRY_NAMESPACE,
    }
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E02PreflightError(f"E02 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E02" not in value:
            raise E02PreflightError(f"E02-owned path required: {key}")
        assert_v2_artifact_path(value)
    selection = document.get("selection_policy", {})
    if (
        selection.get("primary_metric") != "pooled_rolling_origin_rmse_wh"
        or selection.get("promotion_rmse_tolerance_wh") != PROMOTION_RMSE_TOLERANCE_WH
        or selection.get("promotion_threshold_rmse_wh") != PROMOTION_THRESHOLD_RMSE_WH
        or selection.get("promotion_threshold_rmse_wh")
        != EXPECTED_CONTROL_METRICS["rmse_wh"] - PROMOTION_RMSE_TOLERANCE_WH
        or selection.get("promotion_rule")
        != "control_pooled_rmse_wh - challenger_pooled_rmse_wh >= promotion_rmse_tolerance_wh"
        or selection.get("below_tolerance_classification")
        != "NO_MEANINGFUL_IMPROVEMENT / TIE"
        or selection.get("threshold_kind")
        != "GOVERNANCE_THRESHOLD_NOT_STATISTICAL_CONFIDENCE_INTERVAL"
        or selection.get("human_locked") is not True
        or selection.get("weighted_composite") is not False
        or selection.get("test_metrics_allowed") is not False
    ):
        raise E02PreflightError("E02 selection policy is not safely locked")
    control_audit = _validate_control_reuse(root, document)
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID,
        "control_feature_set": CONTROL_VARIANT,
        "control_feature_count": 33,
        "challenger_feature_set": CHALLENGER_VARIANT,
        "challenger_feature_count": 31,
        "feature_set_delta": ["rv1", "rv2"],
        "changed_config_fields": list(changed),
        "one_primary_change": "PASS",
        "control_reuse": "CONTROL_REUSED_FROM_E01",
        "control_reuse_audit": control_audit,
        "challenger_config_fingerprint": fingerprint,
        "fresh_initialization": True,
        "training_candidates": [CHALLENGER_VARIANT],
        "registry_namespace": REGISTRY_NAMESPACE,
        "test_access": "NO",
        "training_executed": False,
        "inference_executed": False,
        "checkpoint_loaded": False,
        "promotion_rmse_tolerance_wh": PROMOTION_RMSE_TOLERANCE_WH,
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e02_document(load_e02_config(resolved), resolved)


def build_e02_candidate(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    config = deepcopy(document["challenger"]["config"])
    assert_no_test_access(split_id=config["data"]["target_access_mode"])
    return CandidateSpec(
        candidate_id=CANDIDATE_ID,
        model_family="TRANSFORMER_ENCODER",
        shortlist_position=0,
        config=config,
        config_fingerprint=document["challenger"]["config_fingerprint"],
        feature_variant_id=CHALLENGER_VARIANT,
        target_scaling_option="YS1",
        lookback_steps=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        source_phase="V2_E02_IMMUTABLE_SNAPSHOT",
        candidate_role="FEATURE_ABLATION_CHALLENGER",
    )


def _registry_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage = deepcopy(config["lineage"])
    return {
        "lineage": lineage,
        "feature_sets": {
            "variant_feature_counts": {CHALLENGER_VARIANT: 31},
            "variant_fingerprints": {
                CHALLENGER_VARIANT: lineage["feature_fingerprint"]
            },
        },
        "window_fingerprints": {},
    }


def build_e02_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.model_improvement_v2.pretest_adapter import (
        build_v2_pretest_dataset,
        load_phase44_fold_evidence,
    )
    from course_work.rolling_origin.real_run import RunContext

    validate_e02_document(document, root)
    candidate = build_e02_candidate(document)
    orchestration = document["v2_orchestration"]
    artifact_dir = root / "artifacts/model_improvement_v2/experiments/E02"
    registry_root = artifact_dir / "registry"
    run_root = artifact_dir / "runs"
    cache: dict[str, Any] = {}

    def dataset_factory(candidate_spec, _fold):
        if candidate_spec.feature_variant_id != CHALLENGER_VARIANT:
            raise E02PreflightError("E02 factory accepts only FS1_TF1")
        assert_no_test_access(split_id=candidate_spec.config["data"]["target_access_mode"])
        if "bundle" not in cache:
            cache["bundle"] = build_v2_pretest_dataset(
                root,
                tuple(document["challenger"]["feature_order"]),
                experiment_id=EXPERIMENT_ID,
                feature_variant_id=CHALLENGER_VARIANT,
            )
        dataset, _evidence, audit = cache["bundle"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E02 Test firewall failed")
        return dataset

    fold_evidence = load_phase44_fold_evidence(root)
    return RunContext(
        project_root=root,
        transformer_shortlist_path=root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=root / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=artifact_dir,
        registry_root=registry_root,
        run_root=run_root,
        scientific_max_epochs=50,
        scientific_patience=10,
        seed=SUPPORTED_SEED,
        device=str(candidate.config["runtime"]["device_type"]),
        is_rehearsal=False,
        rehearsal_synthetic=False,
        reuse_completed_runs=False,
        dataset_factory=dataset_factory,
        candidate_specs=(candidate,),
        registry_namespace=REGISTRY_NAMESPACE,
        seed_before_model_construction=True,
        validated_registry_lifecycle=True,
        execution_track=EXECUTION_TRACK,
        registry_upstream_context=_registry_upstream(candidate.config),
        robase_train_ids=fold_evidence.train_ids,
        robase_val_ids=fold_evidence.validation_ids,
        apply_fold_x_scaling=True,
    )


def _write_success_contracts(root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E02"
    observed = result.pooled_metrics_by_cid[CANDIDATE_ID]
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E02_EXECUTION-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "control_policy": "CONTROL_REUSED_FROM_E01",
        "control_run_ids": document["control"]["run_ids"],
        "challenger_run_ids": {
            "stage_a": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_a_run_ids.items()},
            "stage_b": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_b_run_ids.items()},
        },
        "test_status": "NOT_ACCESSED",
        "preflight": dict(preflight),
    }
    improvement_rmse_wh = (
        EXPECTED_CONTROL_METRICS["rmse_wh"] - float(observed["rmse_wh"])
    )
    promotion_status = (
        "PROMOTE_FS1_TF1"
        if improvement_rmse_wh >= PROMOTION_RMSE_TOLERANCE_WH
        else "NO_MEANINGFUL_IMPROVEMENT / TIE"
    )
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E02_FEATURE_ABLATION-v1",
        "experiment_id": EXPERIMENT_ID,
        "primary_change": "FEATURE_SET",
        "control_status": "CONTROL_REUSED_FROM_E01",
        "control_feature_set": CONTROL_VARIANT,
        "challenger_feature_set": CHALLENGER_VARIANT,
        "removed_features": ["rv1", "rv2"],
        "control_metrics": document["control"]["pooled_metrics"],
        "challenger_metrics": observed,
        "delta_rmse_wh": float(observed["rmse_wh"]) - EXPECTED_CONTROL_METRICS["rmse_wh"],
        "improvement_rmse_wh": improvement_rmse_wh,
        "target_population_policy": "IDENTICAL_PHASE44_RO1_RO2_RO3",
        "selection_metric": "pooled_rmse_wh",
        "promotion_rmse_tolerance_wh": PROMOTION_RMSE_TOLERANCE_WH,
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "promotion_status": promotion_status,
        "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(output / "e02_execution_manifest.json", canonical_json_bytes(manifest))
    atomic_write_bytes(output / "e02_feature_ablation_comparison.json", canonical_json_bytes(comparison))


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e02_config(resolved)
    preflight = validate_e02_document(document, resolved)
    context = build_e02_run_context(resolved, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary, file=sys.stderr)
        if result.exception:
            print(result.exception, file=sys.stderr)
        return result.exit_code
    _write_success_contracts(resolved, document, result, preflight)
    print(result.summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2 E02 feature-set ablation")
    parser.add_argument("--experiment", choices=[EXPERIMENT_ID], required=True)
    parser.add_argument("--mode", choices=["preflight", "official"], default="preflight")
    parser.add_argument("--seed", type=int, default=SUPPORTED_SEED)
    parser.add_argument("--authorize-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED:
        print("ERROR: E02 screening seed must be 42", file=sys.stderr)
        return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: E02 official mode requires --authorize-training.", file=sys.stderr)
        return 3
    if args.mode == "preflight" and args.authorize_training:
        print("ERROR: --authorize-training is valid only in official mode.", file=sys.stderr)
        return 2
    try:
        if args.mode == "official":
            return run_official()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (E02PreflightError, PermissionError, ValueError) as exc:
        print(f"E02 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
