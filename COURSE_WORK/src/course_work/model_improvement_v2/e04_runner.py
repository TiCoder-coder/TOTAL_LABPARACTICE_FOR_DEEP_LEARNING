"""E04 controlled residual LINEAR versus MLP prediction-head ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)

EXPERIMENT_ID = "E04"
SUPPORTED_SEED = 42
CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
CONTROL_HEAD = "LINEAR"
CHALLENGER_HEAD = "MLP"
PREDICTION_FORMULATION = "RESIDUAL_TO_PERSISTENCE"
FEATURE_VARIANT = "FS2_TF1"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E04"
REGISTRY_NAMESPACE = "V2_E04"
CONFIG_RELATIVE_PATH = Path("artifacts/model_improvement_v2/experiments/E04/e04_config_snapshot.json")
E03_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E03")
E01_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E01")
EXPECTED_LOCAL_METRICS = {
    "rmse_wh": 60.04654108949536,
    "mae_wh": 26.74870362368344,
    "r2": 0.5762147018706287,
}
EXPECTED_GLOBAL_METRICS = {
    "rmse_wh": 59.85291570400546,
    "mae_wh": 26.650501720144604,
    "r2": 0.5789433617557903,
}
GLOBAL_PROMOTION_THRESHOLD_RMSE_WH = 59.75291570400546
EXPECTED_CONFIG_CHANGES = ("model.prediction_head",)
EXPECTED_HEAD_SPEC = (
    "Linear(64,64)", "GELU", "Dropout(0.1)", "Linear(64,1)",
)


class E04PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E04PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E04PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e04_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_RELATIVE_PATH)


def _verify_artifacts(base: Path, expected: Mapping[str, Any], label: str) -> None:
    for name, checksum in expected.items():
        path = base / name
        if not path.is_file() or _sha256(path) != checksum:
            raise E04PreflightError(f"{label} immutable artifact changed: {name}")


def _validate_local_control(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    control = document.get("local_control", {})
    if control.get("reuse_policy") != "CONTROL_REUSED_FROM_E03":
        raise E04PreflightError("E04 local control must be reused from E03")
    if control.get("prediction_formulation") != PREDICTION_FORMULATION or control.get("prediction_head") != CONTROL_HEAD:
        raise E04PreflightError("E04 local control must be residual LINEAR")
    if control.get("test_status") != "NOT_ACCESSED":
        raise E04PreflightError("E03 local control violates Test firewall")
    e03_root = root / E03_RELATIVE_ROOT
    _verify_artifacts(e03_root, control.get("artifact_sha256", {}), "E03")
    comparison = _read_json(e03_root / "e03_prediction_ablation_comparison.json")
    if comparison.get("challenger_metrics") != EXPECTED_LOCAL_METRICS:
        raise E04PreflightError("E03 accepted residual LINEAR metrics mismatch")
    if comparison.get("promotion_status") != "NO_MEANINGFUL_IMPROVEMENT / TIE":
        raise E04PreflightError("E03 decision is not the accepted NOT PROMOTED decision")
    if comparison.get("test_status") != "NOT_ACCESSED" or control.get("pooled_metrics") != EXPECTED_LOCAL_METRICS:
        raise E04PreflightError("E03 local control evidence mismatch")
    execution = _read_json(e03_root / "e03_execution_manifest.json")
    if execution.get("challenger_run_ids") != control.get("run_ids"):
        raise E04PreflightError("E03 local-control run ledger mismatch")
    run_ids = [value for stage in control["run_ids"].values() for value in stage.values()]
    if len(run_ids) != 6 or len(set(run_ids)) != 6 or not all("_E03_" in value for value in run_ids):
        raise E04PreflightError("E03 local-control run ledger is incomplete")
    return {"status": "PASS", "run_count": 6, "metrics": EXPECTED_LOCAL_METRICS}


def _validate_global_baseline(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    baseline = document.get("global_baseline", {})
    if baseline.get("reuse_policy") != "GLOBAL_BASELINE_REUSED_FROM_E01":
        raise E04PreflightError("E04 global baseline must be reused from E01")
    if baseline.get("prediction_formulation") != "DIRECT" or baseline.get("test_status") != "NOT_ACCESSED":
        raise E04PreflightError("E01 global baseline contract mismatch")
    e01_root = root / E01_RELATIVE_ROOT
    _verify_artifacts(e01_root, baseline.get("artifact_sha256", {}), "E01")
    comparison = _read_json(e01_root / "e01_reproduction_comparison.json")
    observed = {
        "rmse_wh": comparison.get("observed_rmse_wh"),
        "mae_wh": comparison.get("observed_mae_wh"),
        "r2": comparison.get("observed_r2"),
    }
    if observed != EXPECTED_GLOBAL_METRICS or baseline.get("pooled_metrics") != observed:
        raise E04PreflightError("E01 accepted global metrics mismatch")
    if comparison.get("target_population_match") is not True or comparison.get("test_status") != "NOT_ACCESSED":
        raise E04PreflightError("E01 global baseline is not canonical pre-Test evidence")
    execution = _read_json(e01_root / "e01_execution_manifest.json")
    if execution.get("run_ids") != baseline.get("run_ids"):
        raise E04PreflightError("E01 global-baseline run ledger mismatch")
    return {"status": "PASS", "metrics": observed}


def validate_e04_document(document: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E04PreflightError("Wrong E04 identity/track")
    if document.get("objective") != "CONTROLLED_RESIDUAL_PREDICTION_HEAD_ABLATION":
        raise E04PreflightError("Wrong E04 objective")
    primary = document.get("primary_change", {})
    if primary != {"factor": "prediction_head", "control": CONTROL_HEAD, "challenger": CHALLENGER_HEAD}:
        raise E04PreflightError("E04 primary change must be LINEAR -> MLP prediction_head")

    challenger = document.get("challenger", {})
    config = challenger.get("config")
    if not isinstance(config, dict):
        raise E04PreflightError("Missing E04 challenger config")
    assert_no_test_access(split_id=str(config.get("data", {}).get("target_access_mode", "")))
    orchestration = document.get("v2_orchestration", {})
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    model = config.get("model", {})
    if model.get("prediction_formulation") != PREDICTION_FORMULATION or model.get("prediction_head") != CHALLENGER_HEAD:
        raise E04PreflightError("E04 must be RESIDUAL_TO_PERSISTENCE + MLP")
    if (model.get("d_model"), model.get("output_size"), model.get("dropout")) != (64, 1, 0.1):
        raise E04PreflightError("E04 MLP dimensions/dropout must be 64 -> 64 -> 1 with dropout 0.1")
    if tuple(challenger.get("head_spec", ())) != EXPECTED_HEAD_SPEC:
        raise E04PreflightError("E04 MLP head specification mismatch")
    data = config.get("data", {})
    if data.get("feature_variant_id") != FEATURE_VARIANT or data.get("feature_count") != 33 or model.get("input_size") != 33:
        raise E04PreflightError("E04 FS2_TF1 feature contract mismatch")
    from course_work.data.feature_sets import get_feature_list
    if tuple(challenger.get("feature_order", ())) != tuple(get_feature_list(FEATURE_VARIANT)):
        raise E04PreflightError("E04 feature order mismatch")

    e03_config = _read_json(root / E03_RELATIVE_ROOT / "e03_config_snapshot.json")["challenger"]["config"]
    changed = assert_one_primary_change(EXPERIMENT_ID, e03_config, config)
    if changed != EXPECTED_CONFIG_CHANGES:
        raise E04PreflightError(f"Unexpected E04 changed fields: {changed}")
    fingerprint = compute_config_fingerprint(config)
    if challenger.get("config_fingerprint") != fingerprint:
        raise E04PreflightError("E04 config fingerprint mismatch")
    if challenger.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD":
        raise E04PreflightError("E04 challenger must initialize from scratch")

    expected_orchestration = {
        "rolling_origin_protocol": "RO3_EXPANDING_PRETEST-v1",
        "folds": ["RO1", "RO2", "RO3"], "fold_count": 3,
        "fold_local_scaling": True, "local_control_retrained": False,
        "global_baseline_retrained": False, "challenger_only_training": True,
        "training_authorized": False, "test_access_authorized": False,
        "preflight_consumes_run_id": False, "registry_namespace": REGISTRY_NAMESPACE,
    }
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E04PreflightError(f"E04 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E04" not in value:
            raise E04PreflightError(f"E04-owned path required: {key}")
        assert_v2_artifact_path(value)

    selection = document.get("selection_policy", {})
    if (selection.get("primary_metric") != "pooled_rolling_origin_rmse_wh"
        or selection.get("global_promotion_threshold_rmse_wh") != GLOBAL_PROMOTION_THRESHOLD_RMSE_WH
        or selection.get("global_promotion_rule") != "e04_pooled_rmse_wh <= global_promotion_threshold_rmse_wh"
        or selection.get("weighted_composite") is not False
        or selection.get("test_metrics_allowed") is not False):
        raise E04PreflightError("E04 global promotion policy mismatch")

    local_audit = _validate_local_control(root, document)
    global_audit = _validate_global_baseline(root, document)
    return {
        "status": "PASS", "experiment_id": EXPERIMENT_ID,
        "candidate_id": CANDIDATE_ID, "prediction_formulation": PREDICTION_FORMULATION,
        "control_head": CONTROL_HEAD, "challenger_head": CHALLENGER_HEAD,
        "head_spec": list(EXPECTED_HEAD_SPEC), "changed_config_fields": list(changed),
        "one_primary_change": "PASS", "local_control_reuse": "CONTROL_REUSED_FROM_E03",
        "local_control_audit": local_audit, "global_baseline_audit": global_audit,
        "fresh_initialization": True, "training_candidates": ["RESIDUAL_TO_PERSISTENCE_MLP"],
        "registry_namespace": REGISTRY_NAMESPACE, "challenger_config_fingerprint": fingerprint,
        "global_promotion_threshold_rmse_wh": GLOBAL_PROMOTION_THRESHOLD_RMSE_WH,
        "test_access": "NO", "training_executed": False, "inference_executed": False,
        "checkpoint_loaded": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e04_document(load_e04_config(resolved), resolved)


def build_e04_candidate(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec
    config = deepcopy(document["challenger"]["config"])
    assert_no_test_access(split_id=config["data"]["target_access_mode"])
    return CandidateSpec(
        candidate_id=CANDIDATE_ID, model_family="TRANSFORMER_ENCODER",
        shortlist_position=0, config=config,
        config_fingerprint=document["challenger"]["config_fingerprint"],
        feature_variant_id=FEATURE_VARIANT, target_scaling_option="YS1",
        lookback_steps=72, boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        source_phase="V2_E04_IMMUTABLE_SNAPSHOT",
        candidate_role="RESIDUAL_MLP_HEAD_CHALLENGER",
    )


def _registry_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage = deepcopy(config["lineage"])
    return {"lineage": lineage, "feature_sets": {
        "variant_feature_counts": {FEATURE_VARIANT: 33},
        "variant_fingerprints": {FEATURE_VARIANT: lineage["feature_fingerprint"]},
    }, "window_fingerprints": {}}


def build_e04_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset, load_phase44_fold_evidence
    from course_work.rolling_origin.real_run import RunContext
    validate_e04_document(document, root)
    candidate = build_e04_candidate(document)
    artifact_dir = root / "artifacts/model_improvement_v2/experiments/E04"
    cache: dict[str, Any] = {}
    def dataset_factory(candidate_spec, _fold):
        if candidate_spec.feature_variant_id != FEATURE_VARIANT:
            raise E04PreflightError("E04 factory accepts only FS2_TF1")
        assert_no_test_access(split_id=candidate_spec.config["data"]["target_access_mode"])
        if "bundle" not in cache:
            cache["bundle"] = build_v2_pretest_dataset(
                root, tuple(document["challenger"]["feature_order"]),
                experiment_id=EXPERIMENT_ID, feature_variant_id=FEATURE_VARIANT,
            )
        dataset, _evidence, audit = cache["bundle"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E04 Test firewall failed")
        return dataset
    evidence = load_phase44_fold_evidence(root)
    return RunContext(
        project_root=root,
        transformer_shortlist_path=root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=root / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=artifact_dir, registry_root=artifact_dir / "registry", run_root=artifact_dir / "runs",
        scientific_max_epochs=50, scientific_patience=10, seed=SUPPORTED_SEED,
        device=str(candidate.config["runtime"]["device_type"]), is_rehearsal=False,
        rehearsal_synthetic=False, reuse_completed_runs=False,
        dataset_factory=dataset_factory, candidate_specs=(candidate,),
        registry_namespace=REGISTRY_NAMESPACE, seed_before_model_construction=True,
        validated_registry_lifecycle=True, execution_track=EXECUTION_TRACK,
        registry_upstream_context=_registry_upstream(candidate.config),
        robase_train_ids=evidence.train_ids, robase_val_ids=evidence.validation_ids,
        apply_fold_x_scaling=True,
    )


def _write_success_contracts(root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes
    output = root / "artifacts/model_improvement_v2/experiments/E04"
    observed = result.pooled_metrics_by_cid[CANDIDATE_ID]
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E04_EXECUTION-v1", "experiment_id": EXPERIMENT_ID,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "local_control_policy": "CONTROL_REUSED_FROM_E03",
        "global_baseline_policy": "GLOBAL_BASELINE_REUSED_FROM_E01",
        "local_control_run_ids": document["local_control"]["run_ids"],
        "global_baseline_run_ids": document["global_baseline"]["run_ids"],
        "challenger_run_ids": {
            "stage_a": {f"{cid}:{fold}": rid for (cid, fold), rid in result.stage_a_run_ids.items()},
            "stage_b": {f"{cid}:{fold}": rid for (cid, fold), rid in result.stage_b_run_ids.items()},
        }, "test_status": "NOT_ACCESSED", "preflight": dict(preflight),
    }
    local_delta = float(observed["rmse_wh"]) - EXPECTED_LOCAL_METRICS["rmse_wh"]
    global_delta = float(observed["rmse_wh"]) - EXPECTED_GLOBAL_METRICS["rmse_wh"]
    promoted = float(observed["rmse_wh"]) <= GLOBAL_PROMOTION_THRESHOLD_RMSE_WH
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E04_HEAD_ABLATION-v1", "experiment_id": EXPERIMENT_ID,
        "primary_change": "PREDICTION_HEAD", "prediction_formulation": PREDICTION_FORMULATION,
        "local_comparison": {
            "control": "E03_RESIDUAL_LINEAR", "challenger": "E04_RESIDUAL_MLP",
            "control_status": "CONTROL_REUSED_FROM_E03", "control_metrics": EXPECTED_LOCAL_METRICS,
            "challenger_metrics": observed, "delta_rmse_wh": local_delta,
        },
        "global_comparison": {
            "baseline": "E01_DIRECT_FS2_TF1", "challenger": "E04_RESIDUAL_MLP",
            "baseline_status": "GLOBAL_BASELINE_REUSED_FROM_E01", "baseline_metrics": EXPECTED_GLOBAL_METRICS,
            "challenger_metrics": observed, "delta_rmse_wh": global_delta,
            "promotion_threshold_rmse_wh": GLOBAL_PROMOTION_THRESHOLD_RMSE_WH,
            "promotion_rule": "e04_pooled_rmse_wh <= promotion_threshold_rmse_wh",
            "promotion_status": "PROMOTE_E04_GLOBAL" if promoted else "DO_NOT_PROMOTE_E04_GLOBAL",
        },
        "target_population_policy": "IDENTICAL_PHASE44_RO1_RO2_RO3",
        "selection_metric": "pooled_rmse_wh", "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(output / "e04_execution_manifest.json", canonical_json_bytes(manifest))
    atomic_write_bytes(output / "e04_head_ablation_comparison.json", canonical_json_bytes(comparison))


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e04_config(resolved)
    preflight = validate_e04_document(document, resolved)
    context = build_e04_run_context(resolved, document)
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
    parser = argparse.ArgumentParser(description="V2 E04 residual LINEAR vs MLP head ablation")
    parser.add_argument("--experiment", choices=[EXPERIMENT_ID], required=True)
    parser.add_argument("--mode", choices=["preflight", "official"], default="preflight")
    parser.add_argument("--seed", type=int, default=SUPPORTED_SEED)
    parser.add_argument("--authorize-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED:
        print(f"ERROR: E04 screening seed must be {SUPPORTED_SEED}", file=sys.stderr); return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: E04 official mode requires --authorize-training.", file=sys.stderr); return 3
    if args.mode == "preflight" and args.authorize_training:
        print("ERROR: --authorize-training is valid only in official mode.", file=sys.stderr); return 2
    try:
        if args.mode == "official":
            return run_official()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True)); return 0
    except (E04PreflightError, PermissionError, ValueError) as exc:
        print(f"E04 {args.mode.upper()} FAIL: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
