"""E16 three-layer Post-LN depth ablation; Human-gated training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)
from course_work.model_improvement_v2.e14_runner import bundle_config as e14_bundle_config

EXPERIMENT_ID = "E16"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E16"
REGISTRY_NAMESPACE = "V2_E16"
CONFIG_PATH = Path("artifacts/model_improvement_v2/experiments/E16/e16_config_snapshot.json")
E14_ROOT = Path("artifacts/model_improvement_v2/experiments/E14")
E15_ROOT = Path("artifacts/model_improvement_v2/experiments/E15")
CONTROL_ID = "TR_C2_ALT_LOOKBACK_E14_M1"
CANDIDATE_ID = "TR_C2_ALT_LOOKBACK_E16_DEPTH3_POSTLN"
CONTROL_METRICS = {
    "rmse_wh": 58.5534531302616,
    "mae_wh": 26.33779907425144,
    "r2": 0.5970279545320498,
}
CONTROL_FOLDS = [65.37273386173845, 50.772406234902675, 58.601149429283645]
PARAMETER_COUNT = 152193


class E16PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise E16PreflightError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E16PreflightError(f"Expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_e16_config(root: Path) -> dict[str, Any]:
    return _read(root / CONFIG_PATH)


def incumbent_config(root: Path) -> dict[str, Any]:
    return e14_bundle_config(root, "M1")


def depth_config(root: Path) -> dict[str, Any]:
    config = deepcopy(incumbent_config(root))
    config["model"]["num_layers"] = 3
    return config


def _control_audit(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    e15_decision_path = root / E15_ROOT / "e15_human_decision.json"
    e15_comparison_path = root / E15_ROOT / "e15_capacity_comparison.json"
    e15_config_path = root / E15_ROOT / "e15_config_snapshot.json"
    e15_decision = _read(e15_decision_path)
    if (
        e15_decision.get("decision") != "REJECT_E15"
        or e15_decision.get("accepted_incumbent") != CONTROL_ID
        or e15_decision.get("accepted_config_fingerprint")
        != compute_config_fingerprint(incumbent_config(root))
        or e15_decision.get("source_comparison_sha256") != _sha(e15_comparison_path)
        or e15_decision.get("source_config_snapshot_sha256") != _sha(e15_config_path)
        or e15_decision.get("decision_authority") != "HUMAN"
        or e15_decision.get("test_status") != "NOT_ACCESSED"
    ):
        raise E16PreflightError("E15 Human rejection lock mismatch")

    comparison = _read(e15_comparison_path)
    if any(
        item.get("promotion_evaluation", {}).get("eligible_for_human_promotion") is not False
        for item in comparison.get("challengers", [])
    ):
        raise E16PreflightError("E15 rejected challenger eligibility mismatch")

    base = incumbent_config(root)
    e14_decision = _read(root / E14_ROOT / "e14_human_decision.json")
    expected_metrics = {
        **CONTROL_METRICS,
        "worst_fold_rmse_wh": max(CONTROL_FOLDS),
        "fold_rmse_sd_wh": statistics.pstdev(CONTROL_FOLDS),
    }
    if (
        e14_decision.get("decision") != "PROMOTE_M1"
        or e14_decision.get("accepted_incumbent") != CONTROL_ID
        or e14_decision.get("accepted_config_fingerprint") != compute_config_fingerprint(base)
        or e14_decision.get("accepted_metrics") != expected_metrics
        or e14_decision.get("test_status") != "NOT_ACCESSED"
    ):
        raise E16PreflightError("E14-M1 incumbent lock mismatch")

    records = {
        record["run_id"]: record
        for record in map(
            json.loads,
            (root / E14_ROOT / "registry/experiment_registry.jsonl").read_text().splitlines(),
        )
    }
    for fold, run_id in e14_decision["stage_b_run_ids"].items():
        record = records.get(run_id)
        checkpoint = root / E14_ROOT / "runs" / run_id / "checkpoints/refit_final.pt"
        if (
            not record
            or record.get("status") != "COMPLETED"
            or record.get("candidate_id") != CONTROL_ID
            or record.get("sweep_stage") != f"{fold}_B"
            or not checkpoint.is_file()
            or _sha(checkpoint) != e14_decision["stage_b_checkpoint_sha256"][fold]
        ):
            raise E16PreflightError(f"E14-M1 checkpoint lineage mismatch: {fold}")

    control = document.get("accepted_incumbent", {})
    if (
        control.get("candidate_id") != CONTROL_ID
        or control.get("decision_source") != "E15_REJECT_E15"
        or control.get("config_fingerprint") != compute_config_fingerprint(base)
        or control.get("reuse_only") is not True
        or control.get("test_status") != "NOT_ACCESSED"
    ):
        raise E16PreflightError("E16 control snapshot mismatch")
    return {
        "status": "PASS",
        "decision": "REJECT_E15",
        "config_fingerprint": compute_config_fingerprint(base),
        "checkpoint_count": 3,
    }


def _parameter_count(config: Mapping[str, Any]) -> int:
    from course_work.rolling_origin.real_run import build_model_from_run_config

    model = build_model_from_run_config(config)
    return sum(parameter.numel() for parameter in model.parameters())


def validate_e16_document(
    document: Mapping[str, Any], root: Path, materialize_data: bool = True
) -> dict[str, Any]:
    if document.get("experiment_id") != "E16" or document.get("objective") != "POST_LN_DEPTH_EXPANSION":
        raise E16PreflightError("Wrong E16 identity")
    control = _control_audit(root, document)
    base = incumbent_config(root)
    config = depth_config(root)
    changed = set(assert_one_primary_change("E16", base, config))
    if changed != {"model.num_layers"}:
        raise E16PreflightError("E16 primary-change contract mismatch")

    challenger = document.get("challenger", {})
    expected_identity = (64, 4, 256, 3, "POST_LN")
    actual_identity = (
        challenger.get("d_model"),
        challenger.get("num_heads"),
        challenger.get("ffn_dim"),
        challenger.get("num_layers"),
        challenger.get("norm_order"),
    )
    if (
        challenger.get("candidate_id") != CANDIDATE_ID
        or actual_identity != expected_identity
        or config["model"].get("norm_first") is not False
        or challenger.get("config_fingerprint") != compute_config_fingerprint(config)
    ):
        raise E16PreflightError("E16 challenger identity/fingerprint mismatch")

    parameter_count = _parameter_count(config)
    if (
        parameter_count != PARAMETER_COUNT
        or challenger.get("parameter_count") != PARAMETER_COUNT
        or parameter_count >= 1_000_000
    ):
        raise E16PreflightError("E16 parameter-count gate failed")
    if (
        config["training"] != base["training"]
        or config["data"] != base["data"]
        or config["reproducibility"] != base["reproducibility"]
        or config["runtime"] != base["runtime"]
    ):
        raise E16PreflightError("Non-depth incumbent field changed")
    model_differences = {
        key
        for key in set(base["model"]) | set(config["model"])
        if base["model"].get(key) != config["model"].get(key)
    }
    if model_differences != {"num_layers"}:
        raise E16PreflightError("E16 changes a model field other than num_layers")
    if tuple(get_feature_list("FS2_TF1")) != tuple(get_feature_list(base["data"]["feature_variant_id"])):
        raise E16PreflightError("Feature order changed")
    assert_no_test_access(split_id=config["data"]["target_access_mode"])

    fixed = document.get("fixed_contract", {})
    required = {
        "feature_variant": "FS2_TF1",
        "feature_count": 33,
        "lookback": 72,
        "horizon": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "folds": ["RO1", "RO2", "RO3"],
        "fold_local_x_scaling": True,
        "fold_local_y_scaling": True,
        "prediction_formulation": "DIRECT",
        "prediction_head": "LINEAR",
        "residual_gate": False,
        "d_model": 64,
        "num_heads": 4,
        "ffn_dim": 256,
        "control_num_layers": 2,
        "challenger_num_layers": 3,
        "norm_order": "POST_LN",
        "dropout": 0.1,
        "optimizer": "AdamW",
        "learning_rate": 2e-4,
        "batch_size": 16,
        "weight_decay": 1e-3,
        "scheduler": "OFF",
        "loss_policy": "HYBRID_LEVEL_PLUS_DELTA",
        "lambda_delta": 0.1,
        "delta_beta_model_space": 1.0,
        "gradient_clip_max_norm": 1.0,
        "seed": 42,
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "parameter_count_limit_exclusive": 1_000_000,
    }
    if fixed != required:
        raise E16PreflightError("Fixed E16 scientific contract mismatch")

    expected_policy = {
        "rmse_min_improvement_wh": 0.1,
        "mae_max_degradation_wh": 0.25,
        "worst_fold_rmse_max_degradation_wh": 0.5,
        "fold_rmse_std_max_degradation_wh": 0.5,
        "automatic_promotion": False,
        "decision_authority": "HUMAN",
        "weighted_composite": False,
        "test_metrics_allowed": False,
    }
    if document.get("promotion_policy") != expected_policy:
        raise E16PreflightError("Promotion policy mismatch")
    orchestration = document.get("orchestration", {})
    if (
        orchestration.get("expected_new_training_runs") != 6
        or orchestration.get("control_retrained") is not False
        or orchestration.get("fresh_model_optimizer_each_run") is not True
        or orchestration.get("training_authorized") is not False
        or orchestration.get("test_access_authorized") is not False
    ):
        raise E16PreflightError("E16 orchestration mismatch")
    for key in ("artifact_root", "registry_root", "run_root"):
        assert_v2_artifact_path(orchestration[key])

    population = None
    if materialize_data:
        from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

        dataset, evidence, audit = build_v2_pretest_dataset(
            root,
            tuple(get_feature_list("FS2_TF1")),
            experiment_id="E16",
            feature_variant_id="FS2_TF1",
        )
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise E16PreflightError("Test firewall failed")
        population = {
            "status": "PASS",
            "population_fingerprint": base["lineage"]["population_fingerprint"],
            "target_count": len(dataset),
            "fold_fingerprints": {
                str(fold.fold_id): fold.fold_population_fingerprint for fold in evidence.folds
            },
        }
    return {
        "experiment": "E16",
        "status": "PASS",
        "accepted_incumbent": CONTROL_ID,
        "e15_human_decision": control["decision"],
        "control_reuse_status": control["status"],
        "challenger": CANDIDATE_ID,
        "primary_change": "model.num_layers:2->3",
        "architecture": {
            "d_model": 64,
            "num_heads": 4,
            "ffn_dim": 256,
            "num_layers": 3,
            "norm_order": "POST_LN",
        },
        "expected_new_training_runs": 6,
        "feature_variant": "FS2_TF1",
        "feature_count": 33,
        "candidate_config_fingerprint": compute_config_fingerprint(config),
        "parameter_count": parameter_count,
        "parameter_gate": "PASS",
        "population": population,
        "scaler_lineage": "FOLD_LOCAL_TRAIN_ONLY",
        "recovery_contract": "NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED",
        "test_rows_read": 0,
        "test_target_ids_seen": 0,
        "test_status": "NOT_ACCESSED",
        "training_executed": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return validate_e16_document(load_e16_config(root), root)


def build_candidates(root: Path):
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    config = depth_config(root)
    return (
        CandidateSpec(
            candidate_id=CANDIDATE_ID,
            model_family="TRANSFORMER_ENCODER",
            shortlist_position=1,
            config=config,
            config_fingerprint=compute_config_fingerprint(config),
            feature_variant_id="FS2_TF1",
            target_scaling_option="YS1",
            lookback_steps=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            source_phase="V2_E16_IMMUTABLE_SNAPSHOT",
            candidate_role="LOCKED_THREE_LAYER_POST_LN_CHALLENGER",
        ),
    )


def build_context(root: Path, document: Mapping[str, Any]):
    from course_work.model_improvement_v2.pretest_adapter import (
        build_v2_pretest_dataset,
        load_phase44_fold_evidence,
    )
    from course_work.rolling_origin.real_run import RunContext

    validate_e16_document(document, root, False)
    candidates = build_candidates(root)
    cache: dict[str, Any] = {}
    output = root / "artifacts/model_improvement_v2/experiments/E16"

    def factory(candidate, _fold):
        if candidate.candidate_id != CANDIDATE_ID:
            raise E16PreflightError("Unknown E16 candidate")
        if "data" not in cache:
            cache["data"] = build_v2_pretest_dataset(
                root,
                tuple(get_feature_list("FS2_TF1")),
                experiment_id="E16",
                feature_variant_id="FS2_TF1",
            )
        dataset, _evidence, audit = cache["data"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E16 Test firewall")
        return dataset

    evidence = load_phase44_fold_evidence(root)
    base = incumbent_config(root)
    upstream = {
        "lineage": deepcopy(base["lineage"]),
        "feature_sets": {
            "variant_feature_counts": {"FS2_TF1": 33},
            "variant_fingerprints": {"FS2_TF1": base["lineage"]["feature_fingerprint"]},
        },
        "window_fingerprints": {},
    }
    return RunContext(
        project_root=root,
        transformer_shortlist_path=root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=root / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=output,
        registry_root=output / "registry",
        run_root=output / "runs",
        scientific_max_epochs=50,
        scientific_patience=10,
        seed=42,
        device=str(base["runtime"]["device_type"]),
        dataset_factory=factory,
        candidate_specs=candidates,
        registry_namespace=REGISTRY_NAMESPACE,
        seed_before_model_construction=True,
        validated_registry_lifecycle=True,
        execution_track=EXECUTION_TRACK,
        registry_upstream_context=upstream,
        robase_train_ids=evidence.train_ids,
        robase_val_ids=evidence.validation_ids,
        apply_fold_x_scaling=True,
    )


def audit_completed_ledger(root: Path) -> dict[str, Any]:
    path = root / "artifacts/model_improvement_v2/experiments/E16/registry/experiment_registry.jsonl"
    if not path.is_file():
        raise E16PreflightError("E16 completed ledger unavailable")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {f"{CANDIDATE_ID}:RO{fold}_{stage}" for fold in (1, 2, 3) for stage in "AB"}
    locked: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for record in records:
        if record.get("status") != "COMPLETED":
            continue
        key = f"{record.get('candidate_id')}:{record.get('sweep_stage')}"
        if key not in expected or key in locked:
            raise E16PreflightError("E16 completed identity/duplicate mismatch")
        stage = record["sweep_stage"][-1]
        name = "best_checkpoint.pt" if stage == "A" else "refit_final.pt"
        checkpoint = root / "artifacts/model_improvement_v2/experiments/E16/runs" / record["run_id"] / "checkpoints" / name
        artifacts = [
            artifact
            for artifact in record.get("artifacts", [])
            if Path(artifact.get("artifact_path", "")).name == name
        ]
        if (
            not checkpoint.is_file()
            or len(artifacts) != 1
            or _sha(checkpoint) != artifacts[0].get("sha256")
            or record.get("config_fingerprint")
            != compute_config_fingerprint(record.get("config", {}))
            or record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST"
        ):
            raise E16PreflightError("E16 checkpoint/config/Test audit failed")
        locked[key] = record["run_id"]
        hashes[key] = _sha(checkpoint)
    if set(locked) != expected:
        raise E16PreflightError("E16 no-train resume requires exact six completed runs")
    return {
        "status": "PASS",
        "locked_run_ids": locked,
        "checkpoint_sha256": hashes,
        "training_reexecution_allowed": False,
        "test_status": "NOT_ACCESSED",
    }


def build_resume_context(root: Path, document: Mapping[str, Any]):
    context = build_context(root, document)
    audit = audit_completed_ledger(root)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["locked_run_ids"])
    return context, audit


def _promotion(metrics: Mapping[str, float], folds: list[float]) -> dict[str, Any]:
    gates = {
        "rmse_improvement": CONTROL_METRICS["rmse_wh"] - metrics["rmse_wh"] >= 0.1,
        "mae_guardrail": metrics["mae_wh"] - CONTROL_METRICS["mae_wh"] <= 0.25,
        "worst_fold_rmse_guardrail": max(folds) - max(CONTROL_FOLDS) <= 0.5,
        "fold_rmse_std_guardrail": statistics.pstdev(folds) - statistics.pstdev(CONTROL_FOLDS) <= 0.5,
    }
    return {
        "gates": gates,
        "eligible_for_human_promotion": all(gates.values()),
        "automatic_promotion": False,
        "human_decision_required": True,
    }


def _write_success(root: Path, result, preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E16"
    rows = list(csv.DictReader((output / "rolling_origin_results.csv").open()))
    folds = [float(row["rmse_wh"]) for row in rows if row["candidate_id"] == CANDIDATE_ID]
    metrics = result.pooled_metrics_by_cid[CANDIDATE_ID]
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E16_COMPARISON-v1",
        "control": {
            "source": "E14_M1_REUSED_READ_ONLY_AFTER_E15_REJECTION",
            "candidate_id": CONTROL_ID,
            "metrics": CONTROL_METRICS,
            "fold_rmse_wh": CONTROL_FOLDS,
        },
        "challenger": {
            "candidate_id": CANDIDATE_ID,
            "metrics": metrics,
            "fold_rmse_wh": folds,
            "parameter_count": PARAMETER_COUNT,
            "promotion_evaluation": _promotion(metrics, folds),
        },
        "decision": "HUMAN_REVIEW_REQUIRED",
        "test_status": "NOT_ACCESSED",
    }
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E16_EXECUTION-v1",
        "run_ids": {
            "stage_a": {f"{candidate}:{fold}": run_id for (candidate, fold), run_id in result.stage_a_run_ids.items()},
            "stage_b": {f"{candidate}:{fold}": run_id for (candidate, fold), run_id in result.stage_b_run_ids.items()},
        },
        "preflight": dict(preflight),
        "test_status": "NOT_ACCESSED",
    }
    ledger = [
        json.loads(line)
        for line in (output / "registry/experiment_registry.jsonl").read_text().splitlines()
        if line.strip()
    ]
    diagnostic_rows = []
    for record in ledger:
        if record.get("status") != "COMPLETED":
            continue
        started = record.get("started_at")
        completed = record.get("completed_at")
        seconds = (
            (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds()
            if started and completed
            else None
        )
        stage = record["sweep_stage"][-1]
        row = {
            "run_id": record["run_id"],
            "candidate_id": record["candidate_id"],
            "fold": record["sweep_stage"][:3],
            "stage": stage,
            "duration_seconds": seconds,
            "parameter_count": PARAMETER_COUNT,
            "parameter_memory_estimate_bytes_fp32": PARAMETER_COUNT * 4,
            "gradient_scope": "NOT_CAPTURED_BY_REFIT_ENGINE",
            "gradient_diagnostics": None,
        }
        if stage == "A":
            metrics_path = output / "runs" / record["run_id"] / "metrics/best_validation_metrics.json"
            metric_artifact = _read(metrics_path)
            row["gradient_scope"] = "CANONICAL_STAGE_A_AGGREGATE"
            row["gradient_diagnostics"] = metric_artifact.get("gradient_diagnostics")
        diagnostic_rows.append(row)
    report = {
        "schema": "MODEL_IMPROVEMENT_V2_E16_PARAMETER_RUNTIME_GRADIENT-v1",
        "runs": diagnostic_rows,
        "parameter_limit_exclusive": 1_000_000,
        "parameter_gate": "PASS",
        "gradient_reporting": "STAGE_A_AGGREGATE_STAGE_B_NOT_CAPTURED_NO_FABRICATION",
        "test_status": "NOT_ACCESSED",
    }
    for name, value in (
        ("e16_depth_comparison.json", comparison),
        ("e16_execution_manifest.json", manifest),
        ("e16_parameter_runtime_gradient_report.json", report),
    ):
        atomic_write_bytes(output / name, canonical_json_bytes(value))


def run_official(root: Path | None = None) -> int:
    root = root or project_root()
    document = load_e16_config(root)
    preflight = validate_e16_document(document, root)
    context = build_context(root, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code:
        print(result.summary, file=sys.stderr)
        return result.exit_code
    _write_success(root, result, preflight)
    print(result.summary)
    return 0


def run_resume_stage_c(root: Path | None = None) -> int:
    root = root or project_root()
    document = load_e16_config(root)
    preflight = validate_e16_document(document, root)
    context, _audit = build_resume_context(root, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code:
        print(result.summary, file=sys.stderr)
        return result.exit_code
    _write_success(root, result, preflight)
    print(result.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E16"], required=True)
    parser.add_argument("--mode", choices=["preflight", "official", "resume-stage-c"], default="preflight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: E16 official requires --authorize-training", file=sys.stderr)
        return 3
    if args.mode == "preflight" and args.authorize_training:
        return 2
    try:
        if args.mode == "official":
            return run_official()
        if args.mode == "resume-stage-c":
            if args.authorize_training:
                raise E16PreflightError("E16 resume-stage-c is no-train")
            return run_resume_stage_c()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (E16PreflightError, PermissionError, ValueError) as exc:
        print(f"E16 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
