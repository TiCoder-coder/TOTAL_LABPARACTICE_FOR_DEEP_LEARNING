"""E01 orchestration for the isolated MODEL_IMPROVEMENT-v2 track.

Preflight is always non-training. Official execution is fully wired but refuses
unless the human supplies ``--authorize-training`` explicitly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from course_work.model_improvement_v2.contracts import (
    CANONICAL_DEVELOPMENT_BASELINE,
    TEST_ACCESS_AUTHORIZED,
    V2_ARTIFACT_ROOT,
    assert_no_test_access,
    assert_v2_artifact_path,
)


E01_CONFIG_RELATIVE_PATH = Path(
    "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json"
)
BASELINE_SNAPSHOT_RELATIVE_PATH = Path("artifacts/model_improvement_v2/baseline_snapshot.json")
SUPPORTED_EXPERIMENT = "E01"
SUPPORTED_SEED = 42
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E01"
REGISTRY_NAMESPACE = "V2_E01"
LOCKED_E01_COMPLETED_RUN_IDS = {
    "RO1_A": "RUN_V2_TR_E01_RO1_A_0001_3B5C1B75",
    "RO2_A": "RUN_V2_TR_E01_RO2_A_0002_1EA6841D",
    "RO3_A": "RUN_V2_TR_E01_RO3_A_0003_4C4B28D5",
    "RO1_B": "RUN_V2_TR_E01_RO1_B_0004_533B5F33",
    "RO2_B": "RUN_V2_TR_E01_RO2_B_0005_19A72D17",
    "RO3_B": "RUN_V2_TR_E01_RO3_B_0006_AA5D66A6",
}

REUSABLE_CORE_SYMBOLS: Mapping[str, tuple[str, ...]] = {
    "course_work.rolling_origin.real_run": (
        "RunContext",
        "run_real_pipeline",
    ),
    "course_work.model_improvement_v2.pretest_adapter": (
        "build_e01_pretest_dataset",
        "load_phase44_fold_evidence",
    ),
    "course_work.rolling_origin.folds": ("FoldId", "build_rolling_folds"),
    "course_work.rolling_origin.scaling": (
        "FoldLocalScalerBundle",
        "fit_fold_a_x_scaler",
        "fit_fold_a_y_scaler",
        "fit_fold_b_x_scaler",
        "fit_fold_b_y_scaler",
    ),
    "course_work.rolling_origin.refit_engine": ("RefitEngine",),
    "course_work.rolling_origin.pooling": ("compute_pooled_metrics",),
    "course_work.rolling_origin.artifacts": ("Phase44Artifacts", "write_all_artifacts"),
    "course_work.training.engine": ("TrainingEngine",),
    "course_work.experiments.registry": (
        "ExperimentRegistry",
        "compute_config_fingerprint",
    ),
}

E01_OUTPUT_CONTRACT = {
    "per_fold": (
        "stage_a_selected_epoch",
        "stage_a_validation_metric",
        "stage_a_checkpoint",
        "stage_b_refit_final_checkpoint",
        "stage_c_predictions",
        "fold_metrics",
        "scaler_population_audit",
    ),
    "aggregate": (
        "pooled_rmse_wh",
        "pooled_mae_wh",
        "pooled_r2",
        "worst_fold_rmse_wh",
        "fold_rmse_sd_wh",
        "execution_manifest",
        "reproduction_comparison",
    ),
}


class PreflightError(RuntimeError):
    """Raised when an E01 contract or reusable-core check fails."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Cannot load required JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Required JSON must contain an object: {path}")
    return payload


def load_e01_config_snapshot(root: Path) -> dict[str, Any]:
    return _load_json(root / E01_CONFIG_RELATIVE_PATH)


def load_v2_baseline_snapshot(root: Path) -> dict[str, Any]:
    return _load_json(root / BASELINE_SNAPSHOT_RELATIVE_PATH)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise PreflightError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _validate_test_firewall(document: Mapping[str, Any]) -> None:
    """Run before every future dataset or run construction boundary."""

    orchestration = document.get("v2_orchestration", {})
    config = document.get("v1_config", {})
    data = config.get("data", {}) if isinstance(config, Mapping) else {}
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    assert_no_test_access(split_id=str(data.get("target_access_mode", "")))
    _require_equal("global Test authorization", TEST_ACCESS_AUTHORIZED, False)


def _validate_core_availability() -> list[str]:
    checked: list[str] = []
    for module_name, symbol_names in REUSABLE_CORE_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol_name in symbol_names:
            if not hasattr(module, symbol_name):
                raise PreflightError(f"Reusable core symbol is missing: {module_name}.{symbol_name}")
            checked.append(f"{module_name}.{symbol_name}")
    return checked


def validate_e01_document(
    document: Mapping[str, Any], baseline_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate E01 without datasets, runs, models, checkpoints, or inference."""

    _validate_test_firewall(document)
    baseline = CANONICAL_DEVELOPMENT_BASELINE
    expected_baseline = baseline.to_dict()
    expected_baseline["folds"] = list(expected_baseline["folds"])

    _require_equal("track", document.get("track"), "MODEL_IMPROVEMENT-v2")
    _require_equal("experiment_id", document.get("experiment_id"), SUPPORTED_EXPERIMENT)
    _require_equal("candidate_id", document.get("candidate_id"), baseline.candidate_id)
    _require_equal("locked config fingerprint", document.get("locked_config_fingerprint"), baseline.config_fingerprint)
    _require_equal("baseline snapshot", baseline_snapshot.get("canonical_development_baseline"), expected_baseline)

    config = document.get("v1_config")
    if not isinstance(config, dict):
        raise PreflightError("v1_config must be an object")
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    orchestration = document.get("v2_orchestration", {})

    expected = {
        "feature set": (data.get("feature_variant_id"), baseline.feature_set),
        "feature count": (data.get("feature_count"), 33),
        "model input size": (model.get("input_size"), 33),
        "lookback": (data.get("lookback_steps"), baseline.lookback),
        "boundary protocol": (data.get("boundary_protocol"), baseline.boundary_protocol),
        "optimizer": (training.get("optimizer_name"), baseline.optimizer),
        "learning rate": (training.get("learning_rate"), baseline.learning_rate),
        "weight decay": (training.get("weight_decay"), 1e-3),
        "loss": (training.get("loss_name"), baseline.loss),
        "batch size": (training.get("batch_size"), 32),
        "max epochs": (training.get("max_epochs"), 50),
        "early stopping enabled": (training.get("early_stopping_enabled"), True),
        "early stopping metric": (training.get("early_stopping_metric"), "rmse_wh"),
        "early stopping mode": (training.get("early_stopping_mode"), "MIN"),
        "patience": (training.get("early_stopping_patience"), 10),
        "effective min delta": (training.get("early_stopping_min_delta", 0), 0),
        "gradient clipping": (training.get("gradient_clip_max_norm"), 1.0),
        "scheduler": (training.get("scheduler_name"), None),
        "rolling-origin protocol": (orchestration.get("rolling_origin_protocol"), baseline.rolling_origin_protocol),
        "fold IDs": (orchestration.get("folds"), list(baseline.folds)),
        "fold count": (orchestration.get("fold_count"), baseline.fold_count),
        "fold-local scaling": (orchestration.get("fold_local_scaling"), True),
        "snapshot training default": (orchestration.get("training_authorized"), False),
    }
    for label, (actual, wanted) in expected.items():
        _require_equal(label, actual, wanted)

    from course_work.data.feature_sets import EXPECTED_FEATURE_COUNTS, get_feature_list
    from course_work.experiments.registry import compute_config_fingerprint

    features = get_feature_list(baseline.feature_set)
    _require_equal("feature order", orchestration.get("feature_order"), features)
    _require_equal("registered feature count", EXPECTED_FEATURE_COUNTS[baseline.feature_set], 33)
    computed_fingerprint = compute_config_fingerprint(config)
    _require_equal("computed config fingerprint", computed_fingerprint, baseline.config_fingerprint)

    for field in ("artifact_root", "registry_root", "run_root"):
        path = orchestration.get(field)
        if not isinstance(path, str):
            raise PreflightError(f"{field} must be a V2-owned path")
        assert_v2_artifact_path(path)

    core_symbols = _validate_core_availability()
    return {
        "status": "PASS",
        "experiment_id": SUPPORTED_EXPERIMENT,
        "candidate_id": baseline.candidate_id,
        "config_fingerprint": computed_fingerprint,
        "feature_set": baseline.feature_set,
        "feature_count": len(features),
        "feature_order_verified": True,
        "lookback": baseline.lookback,
        "boundary_protocol": baseline.boundary_protocol,
        "rolling_origin_protocol": baseline.rolling_origin_protocol,
        "folds": list(baseline.folds),
        "fold_count": baseline.fold_count,
        "fold_local_scaling": True,
        "optimizer": baseline.optimizer,
        "learning_rate": baseline.learning_rate,
        "loss": baseline.loss,
        "initialization_policy": "SET_SEED_BEFORE_EVERY_MODEL_CONSTRUCTION",
        "reproduction_scope": "STRUCTURAL_CONFIG_REPRODUCTION",
        "v2_artifact_root": str(V2_ARTIFACT_ROOT),
        "registry_namespace": REGISTRY_NAMESPACE,
        "preflight_consumes_run_id": False,
        "test_access": "NO",
        "training_executed": False,
        "inference_executed": False,
        "checkpoint_loaded": False,
        "reusable_core_symbols_verified": core_symbols,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved_root = root if root is not None else project_root()
    return validate_e01_document(
        load_e01_config_snapshot(resolved_root),
        load_v2_baseline_snapshot(resolved_root),
    )


def build_e01_candidate(document: Mapping[str, Any]):
    """Construct the only learned candidate permitted to enter E01."""

    _validate_test_firewall(document)
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    baseline = CANONICAL_DEVELOPMENT_BASELINE
    if document.get("candidate_id") != baseline.candidate_id:
        raise PreflightError("E01 rejects C0, C1, LSTM, and every non-C2 candidate")
    config = deepcopy(document.get("v1_config"))
    if not isinstance(config, dict):
        raise PreflightError("E01 candidate config is missing")
    return CandidateSpec(
        candidate_id=baseline.candidate_id,
        model_family="TRANSFORMER_ENCODER",
        shortlist_position=0,
        config=config,
        config_fingerprint=baseline.config_fingerprint,
        feature_variant_id=baseline.feature_set,
        target_scaling_option="YS1",
        lookback_steps=baseline.lookback,
        boundary_protocol=baseline.boundary_protocol,
        source_phase="V2_E01_IMMUTABLE_SNAPSHOT",
        candidate_role="V1_EQUIVALENT_DEVELOPMENT_BASELINE",
    )


def _build_registry_upstream_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the V2 registry validation context from the fingerprinted snapshot."""

    lineage = deepcopy(config["lineage"])
    variant = str(config["data"]["feature_variant_id"])
    return {
        "lineage": lineage,
        "feature_sets": {
            "variant_feature_counts": {variant: int(config["data"]["feature_count"])},
            "variant_fingerprints": {variant: str(lineage["feature_fingerprint"])},
        },
        "window_fingerprints": {},
    }


def build_e01_run_context(root: Path, document: Mapping[str, Any]):
    """Build an official context only after the Test firewall has passed."""

    _validate_test_firewall(document)
    from course_work.rolling_origin.real_run import RunContext
    from course_work.model_improvement_v2.pretest_adapter import (
        build_e01_pretest_dataset,
        load_phase44_fold_evidence,
    )

    candidate = build_e01_candidate(document)
    artifact_dir = root / "artifacts/model_improvement_v2/experiments/E01"
    registry_root = root / "artifacts/model_improvement_v2/registry"
    run_root = root / "artifacts/model_improvement_v2/runs"
    for path in (artifact_dir, registry_root, run_root):
        assert_v2_artifact_path(f"COURSE_WORK/{path.relative_to(root).as_posix()}")

    adapter_cache = {}

    def _adapter_bundle():
        if "bundle" not in adapter_cache:
            adapter_cache["bundle"] = build_e01_pretest_dataset(
                root,
                tuple(document["v2_orchestration"]["feature_order"]),
            )
        return adapter_cache["bundle"]

    def dataset_factory(candidate_spec, _fold):
        if candidate_spec.candidate_id != CANONICAL_DEVELOPMENT_BASELINE.candidate_id:
            raise PreflightError("E01 dataset factory rejected a non-C2 candidate")
        _validate_test_firewall(document)
        dataset, _evidence, _audit = _adapter_bundle()
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
        device=str(document["v1_config"]["runtime"]["device_type"]),
        is_rehearsal=False,
        rehearsal_synthetic=False,
        reuse_completed_runs=False,
        dataset_factory=dataset_factory,
        candidate_specs=(candidate,),
        registry_namespace=REGISTRY_NAMESPACE,
        seed_before_model_construction=True,
        validated_registry_lifecycle=True,
        execution_track=EXECUTION_TRACK,
        registry_upstream_context=_build_registry_upstream_context(document["v1_config"]),
        robase_train_ids=fold_evidence.train_ids,
        robase_val_ids=fold_evidence.validation_ids,
        apply_fold_x_scaling=True,
    )


def _write_success_contracts(root: Path, result, preflight: Mapping[str, Any]) -> None:
    """Write E01 summaries only after successful scientific execution."""

    import csv
    import statistics
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output_dir = root / "artifacts/model_improvement_v2/experiments/E01"
    candidate_id = CANONICAL_DEVELOPMENT_BASELINE.candidate_id
    observed = result.pooled_metrics_by_cid[candidate_id]
    fold_metrics_path = output_dir / "rolling_origin_fold_metrics.csv"
    with fold_metrics_path.open(newline="", encoding="utf-8") as stream:
        fold_rows = [row for row in csv.DictReader(stream) if row["candidate_id"] == candidate_id]
    fold_rmse = [float(row["rmse_wh"]) for row in fold_rows]
    if len(fold_rmse) != 3:
        raise RuntimeError("E01 requires exactly three candidate fold metrics")

    canonical_fold_manifest = _load_json(
        root / "artifacts/rolling_origin/rolling_origin_fold_manifest.json"
    )
    canonical_outer_ids = {
        str(fold["fold_id"]): [str(value) for value in fold["outer_eval_ids"]]
        for fold in canonical_fold_manifest.get("folds", [])
    }
    observed_outer_ids: dict[str, list[str]] = {}
    for fold_id in CANONICAL_DEVELOPMENT_BASELINE.folds:
        prediction_path = (
            output_dir
            / "predictions"
            / f"outer_predictions_{candidate_id}_{fold_id}.csv"
        )
        with prediction_path.open(newline="", encoding="utf-8") as stream:
            observed_outer_ids[fold_id] = [str(row["target_id"]) for row in csv.DictReader(stream)]
    target_population_match = (
        set(canonical_outer_ids) == set(CANONICAL_DEVELOPMENT_BASELINE.folds)
        and all(
            observed_outer_ids[fold_id] == canonical_outer_ids[fold_id]
            for fold_id in CANONICAL_DEVELOPMENT_BASELINE.folds
        )
    )

    execution_manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E01_EXECUTION-v1",
        "experiment_id": "E01",
        "candidate_id": candidate_id,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "run_ids": {
            "stage_a": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_a_run_ids.items()},
            "stage_b": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_b_run_ids.items()},
        },
        "initialization_policy": "SET_SEED_BEFORE_EVERY_MODEL_CONSTRUCTION",
        "reproduction_scope": "STRUCTURAL_CONFIG_REPRODUCTION",
        "test_status": "NOT_ACCESSED",
        "output_contract": E01_OUTPUT_CONTRACT,
        "preflight": dict(preflight),
    }
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E01_REPRODUCTION_COMPARISON-v1",
        "experiment_id": "E01",
        "candidate_id": candidate_id,
        "baseline_rmse_wh": CANONICAL_DEVELOPMENT_BASELINE.baseline_pooled_rmse_wh,
        "observed_rmse_wh": float(observed["rmse_wh"]),
        "delta_rmse_wh": float(observed["rmse_wh"]) - CANONICAL_DEVELOPMENT_BASELINE.baseline_pooled_rmse_wh,
        "observed_mae_wh": float(observed["mae_wh"]),
        "observed_r2": float(observed["r2"]),
        "worst_fold_rmse_wh": max(fold_rmse),
        "fold_rmse_sd_wh": statistics.pstdev(fold_rmse),
        "target_population_match": target_population_match,
        "structural_config_match": preflight.get("status") == "PASS",
        "promotion_tolerance": "NOT_AUTHORIZED",
        "promotion_status": "HUMAN_REVIEW_REQUIRED",
        "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(output_dir / "e01_execution_manifest.json", canonical_json_bytes(execution_manifest))
    atomic_write_bytes(output_dir / "e01_reproduction_comparison.json", canonical_json_bytes(comparison))


def run_official_e01(root: Path | None = None) -> int:
    resolved_root = root if root is not None else project_root()
    document = load_e01_config_snapshot(resolved_root)
    preflight = validate_e01_document(document, load_v2_baseline_snapshot(resolved_root))
    context = build_e01_run_context(resolved_root, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary, file=sys.stderr)
        if result.exception:
            print(result.exception, file=sys.stderr)
        return result.exit_code
    _write_success_contracts(resolved_root, result, preflight)
    print(result.summary)
    return 0


def run_resume_stage_c_e01(root: Path | None = None) -> int:
    """Reuse the locked completed Stage A/B evidence and continue at Stage C."""

    resolved_root = root if root is not None else project_root()
    document = load_e01_config_snapshot(resolved_root)
    preflight = validate_e01_document(document, load_v2_baseline_snapshot(resolved_root))
    context = replace(
        build_e01_run_context(resolved_root, document),
        reuse_completed_runs=True,
        reuse_completed_run_ids=dict(LOCKED_E01_COMPLETED_RUN_IDS),
    )
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary, file=sys.stderr)
        if result.exception:
            print(result.exception, file=sys.stderr)
        return result.exit_code
    _write_success_contracts(resolved_root, result, preflight)
    print(result.summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MODEL_IMPROVEMENT-v2 E01 orchestrator")
    parser.add_argument("--experiment", required=True, choices=[SUPPORTED_EXPERIMENT])
    parser.add_argument(
        "--mode", choices=["preflight", "official", "resume-stage-c"], default="preflight"
    )
    parser.add_argument("--seed", type=int, default=SUPPORTED_SEED)
    parser.add_argument(
        "--authorize-training",
        action="store_true",
        help="Human-only explicit authorization for E01 official scientific training.",
    )
    parser.add_argument(
        "--authorize-stage-c-resume",
        action="store_true",
        help="Human-only authorization for checkpoint reuse and Stage-C inference; never trains.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED:
        print(f"ERROR: E01 screening seed must be {SUPPORTED_SEED}", file=sys.stderr)
        return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: --mode official requires explicit --authorize-training.", file=sys.stderr)
        return 3
    if args.mode == "resume-stage-c" and not args.authorize_stage_c_resume:
        print(
            "REFUSED: --mode resume-stage-c requires explicit --authorize-stage-c-resume.",
            file=sys.stderr,
        )
        return 3
    if args.authorize_training and args.mode != "official":
        print("ERROR: --authorize-training is valid only with --mode official.", file=sys.stderr)
        return 2
    if args.authorize_stage_c_resume and args.mode != "resume-stage-c":
        print(
            "ERROR: --authorize-stage-c-resume is valid only with --mode resume-stage-c.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.mode == "official":
            return run_official_e01()
        if args.mode == "resume-stage-c":
            return run_resume_stage_c_e01()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (PreflightError, PermissionError, ValueError) as exc:
        print(f"E01 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
