"""E08 plain-SGD learning-rate audit with immutable E01 AdamW control."""
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

EXPERIMENT_ID = "E08"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E08"
REGISTRY_NAMESPACE = "V2_E08"
SUPPORTED_SEED = 42
BASE_CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
FEATURE_VARIANT = "FS2_TF1"
CONTROL_RMSE_WH = 59.85291570400546
CONTROL_METRICS = {
    "rmse_wh": CONTROL_RMSE_WH,
    "mae_wh": 26.650501720144604,
    "r2": 0.5789433617557903,
}
PROMOTION_THRESHOLD_RMSE_WH = 59.75291570400546
CHALLENGER_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
CANDIDATE_IDS = {
    1e-4: "TR_C2_ALT_LOOKBACK_SGD_LR_1E_MINUS_4",
    3e-4: "TR_C2_ALT_LOOKBACK_SGD_LR_3E_MINUS_4",
    1e-3: "TR_C2_ALT_LOOKBACK_SGD_LR_1E_MINUS_3",
    3e-3: "TR_C2_ALT_LOOKBACK_SGD_LR_3E_MINUS_3",
}
OPTIMIZER_CONFIG = {"momentum": 0.0, "nesterov": False}
CONFIG_RELATIVE_PATH = Path(
    "artifacts/model_improvement_v2/experiments/E08/e08_config_snapshot.json"
)
E01_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E01")


class E08PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E08PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E08PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e08_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_RELATIVE_PATH)


def _validate_control(
    root: Path, document: Mapping[str, Any], e01_config: Mapping[str, Any]
) -> dict[str, Any]:
    control = document.get("control", {})
    if control.get("source_experiment") != "E01":
        raise E08PreflightError("E08 control source must be E01")
    if control.get("reuse_policy") != "CONTROL_REUSED_FROM_E01_READ_ONLY":
        raise E08PreflightError("E08 control must be reuse-only")
    if control.get("training_required") is not False:
        raise E08PreflightError("E08 must not retrain the E01 control")
    if control.get("pooled_metrics") != CONTROL_METRICS:
        raise E08PreflightError("E01 control metrics mismatch")
    if control.get("test_status") != "NOT_ACCESSED":
        raise E08PreflightError("E01 control violates Test firewall")
    if control.get("config") != e01_config:
        raise E08PreflightError("E08 control config differs from E01")
    if control.get("config_fingerprint") != compute_config_fingerprint(e01_config):
        raise E08PreflightError("E08 control fingerprint mismatch")
    for name, expected_sha in control.get("artifact_sha256", {}).items():
        artifact = root / E01_RELATIVE_ROOT / name
        if not artifact.is_file() or _sha256(artifact) != expected_sha:
            raise E08PreflightError(f"E01 immutable control artifact changed: {name}")
    execution = _read_json(root / E01_RELATIVE_ROOT / "e01_execution_manifest.json")
    if execution.get("test_status") != "NOT_ACCESSED":
        raise E08PreflightError("E01 execution manifest violates Test firewall")
    if execution.get("run_ids") != control.get("run_ids"):
        raise E08PreflightError("E01 control run ledger mismatch")
    run_ids = [
        run_id
        for stage in control["run_ids"].values()
        for run_id in stage.values()
    ]
    if len(run_ids) != 6 or len(set(run_ids)) != 6 or not all("_E01_" in x for x in run_ids):
        raise E08PreflightError("E01 control requires six canonical run IDs")
    return {"status": "PASS", "run_count": 6, "metrics": CONTROL_METRICS}


def _validate_predecessor(root: Path, document: Mapping[str, Any]) -> None:
    predecessor = document.get("predecessor_decision", {})
    if (
        predecessor.get("experiment_id") != "E07"
        or predecessor.get("decision") != "RETAIN_E01_SCHEDULER_OFF"
        or predecessor.get("carry_forward_scheduler") is not False
        or predecessor.get("test_status") != "NOT_ACCESSED"
    ):
        raise E08PreflightError("E07 predecessor decision mismatch")
    path = root / "artifacts/model_improvement_v2/experiments/E07/e07_scheduler_ablation_comparison.json"
    if not path.is_file() or predecessor.get("artifact_sha256") != _sha256(path):
        raise E08PreflightError("E07 immutable comparison artifact changed")
    evidence = _read_json(path)
    if (
        evidence.get("decision", {}).get("action") != "RETAIN_E01_SCHEDULER_OFF"
        or evidence.get("test_status") != "NOT_ACCESSED"
    ):
        raise E08PreflightError("E07 accepted decision/Test status mismatch")


def _expected_candidate_config(
    e01_config: Mapping[str, Any], learning_rate: float
) -> dict[str, Any]:
    config = deepcopy(e01_config)
    training = config["training"]
    training["optimizer_name"] = "SGD"
    training["optimizer_config"] = deepcopy(OPTIMIZER_CONFIG)
    training["learning_rate"] = learning_rate
    training["weight_decay"] = 1e-3
    training["scheduler_name"] = None
    training["scheduler_config"] = None
    return config


def validate_e08_document(
    document: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise E08PreflightError("Wrong E08 experiment identity")
    if document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E08PreflightError("Wrong E08 track")
    if document.get("objective") != "PLAIN_SGD_OPTIMIZER_AND_LEARNING_RATE_AUDIT":
        raise E08PreflightError("Wrong E08 objective")
    primary = document.get("primary_change", {})
    if primary.get("factor") != "OPTIMIZER_FAMILY_WITH_PREDECLARED_LR":
        raise E08PreflightError("Wrong E08 primary factor")
    if primary.get("candidate_learning_rates") != list(CHALLENGER_LRS):
        raise E08PreflightError("Wrong E08 LR grid")

    e01_config = _read_json(
        root / E01_RELATIVE_ROOT / "e01_config_snapshot.json"
    )["v1_config"]
    control_audit = _validate_control(root, document, e01_config)
    _validate_predecessor(root, document)

    candidates = document.get("training_candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise E08PreflightError("E08 requires exactly four challengers")
    seen_ids: set[str] = set()
    seen_lrs: set[float] = set()
    seen_fingerprints: set[str] = set()
    changed_paths: dict[str, list[str]] = {}
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        lr = candidate.get("learning_rate")
        config = candidate.get("config")
        if lr not in CHALLENGER_LRS or cid != CANDIDATE_IDS[lr]:
            raise E08PreflightError("E08 challenger ID/LR mismatch")
        if not isinstance(config, dict) or config != _expected_candidate_config(e01_config, lr):
            raise E08PreflightError(f"E08 config mismatch for {cid}")
        assert_no_test_access(split_id=config["data"]["target_access_mode"])
        changed = assert_one_primary_change(EXPERIMENT_ID, e01_config, config)
        if not changed or any(
            not path.startswith((
                "training.optimizer_name",
                "training.optimizer_config",
                "training.learning_rate",
            ))
            for path in changed
        ):
            raise E08PreflightError(f"E08 out-of-factor config change for {cid}")
        training = config["training"]
        if (
            training.get("optimizer_name") != "SGD"
            or training.get("optimizer_config") != OPTIMIZER_CONFIG
            or training.get("weight_decay") != 1e-3
            or training.get("scheduler_name") is not None
            or training.get("scheduler_config") is not None
        ):
            raise E08PreflightError(f"E08 plain-SGD contract mismatch for {cid}")
        if config["model"].get("prediction_formulation", "DIRECT") != "DIRECT":
            raise E08PreflightError("E08 must use DIRECT prediction")
        fingerprint = compute_config_fingerprint(config)
        if candidate.get("config_fingerprint") != fingerprint:
            raise E08PreflightError(f"E08 fingerprint mismatch for {cid}")
        if candidate.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD":
            raise E08PreflightError(f"E08 initialization mismatch for {cid}")
        seen_ids.add(cid)
        seen_lrs.add(lr)
        seen_fingerprints.add(fingerprint)
        changed_paths[cid] = list(changed)
    if seen_ids != set(CANDIDATE_IDS.values()) or seen_lrs != set(CHALLENGER_LRS):
        raise E08PreflightError("E08 challenger matrix is incomplete or duplicated")
    if len(seen_fingerprints) != 4:
        raise E08PreflightError("E08 candidate fingerprints must be unique")

    from course_work.data.feature_sets import get_feature_list
    if tuple(document.get("feature_order", ())) != tuple(get_feature_list(FEATURE_VARIANT)):
        raise E08PreflightError("E08 FS2_TF1 feature order mismatch")
    fixed = document.get("fixed_contract", {})
    expected_fixed = {
        "prediction_formulation": "DIRECT",
        "feature_variant_id": "FS2_TF1",
        "feature_count": 33,
        "lookback_steps": 72,
        "horizon_steps": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "rolling_origin_protocol": "RO3_EXPANDING_PRETEST-v1",
        "folds": ["RO1", "RO2", "RO3"],
        "fold_local_x_scaling": True,
        "fold_local_y_scaling": True,
        "weight_decay": 1e-3,
        "loss_name": "MSE",
        "batch_size": 32,
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0,
        "gradient_clip_max_norm": 1.0,
        "seed": 42,
        "scheduler_name": None,
        "warmup": False,
    }
    if fixed != expected_fixed:
        raise E08PreflightError("E08 fixed scientific contract mismatch")

    orchestration = document.get("v2_orchestration", {})
    expected_orchestration = {
        "registry_namespace": REGISTRY_NAMESPACE,
        "execution_track": EXECUTION_TRACK,
        "shared_pretest_cache_key": "MODEL_IMPROVEMENT_V2_E08_SHARED_PRETEST",
        "training_candidate_count": 4,
        "control_retrained": False,
        "challengers_only_training": True,
        "expected_stage_a_runs": 12,
        "expected_stage_b_runs": 12,
        "expected_total_training_runs": 24,
        "preflight_consumes_run_id": False,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E08PreflightError(f"E08 orchestration mismatch: {key}")
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E08" not in value:
            raise E08PreflightError(f"E08-owned path required: {key}")
        assert_v2_artifact_path(value)

    recovery = document.get("recovery_policy", {})
    expected_recovery = {
        "mode": "NO_RETRAIN_STAGE_C_ONLY",
        "expected_completed_run_count": 24,
        "expected_stage_b_checkpoint_count": 12,
        "new_training_run_ids_allowed": False,
        "stage_a_reexecution_allowed": False,
        "stage_b_reexecution_allowed": False,
        "validate_registry_status": "COMPLETED",
        "validate_checkpoint_sha256": True,
        "test_access_authorized": False,
    }
    if recovery != expected_recovery:
        raise E08PreflightError("E08 recovery policy mismatch")

    selection = document.get("selection_policy", {})
    if selection != {
        "primary_metric": "pooled_rmse_wh",
        "secondary_metrics": ["pooled_mae_wh", "pooled_r2"],
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "promotion_rule": "challenger_pooled_rmse_wh <= 59.75291570400546",
        "fallback": "RETAIN_E01_ADAMW",
        "mae_can_promote": False,
        "test_metrics_allowed": False,
    }:
        raise E08PreflightError("E08 selection policy mismatch")

    from course_work.rolling_origin.real_run import (
        V2_DIRECT_METRIC_FLATTEN_TRACKS,
        V2_SHARED_PRETEST_CACHE_KEYS,
    )
    if V2_DIRECT_METRIC_FLATTEN_TRACKS != frozenset({
        "MODEL_IMPROVEMENT_V2_E06",
        "MODEL_IMPROVEMENT_V2_E07",
        "MODEL_IMPROVEMENT_V2_E08",
        "MODEL_IMPROVEMENT_V2_E09",
    }):
        raise E08PreflightError("E08 Stage-C flatten scope mismatch")
    if V2_SHARED_PRETEST_CACHE_KEYS.get(EXECUTION_TRACK) != orchestration["shared_pretest_cache_key"]:
        raise E08PreflightError("E08 shared pre-Test cache is not wired")

    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "control_reuse": "CONTROL_REUSED_FROM_E01_READ_ONLY",
        "control_audit": control_audit,
        "training_candidate_count": 4,
        "candidate_ids": [CANDIDATE_IDS[lr] for lr in CHALLENGER_LRS],
        "learning_rates": list(CHALLENGER_LRS),
        "optimizer": "SGD",
        "optimizer_config": OPTIMIZER_CONFIG,
        "scheduler": "OFF",
        "changed_config_paths": changed_paths,
        "one_primary_change": "PASS",
        "shared_pretest_cache": "PASS",
        "stage_c_flatten_protection": "PASS",
        "expected_stage_a_runs": 12,
        "expected_stage_b_runs": 12,
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "test_access": "NO",
        "training_executed": False,
        "inference_executed": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e08_document(load_e08_config(resolved), resolved)


def build_e08_candidates(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    specs = []
    for item in document["training_candidates"]:
        config = deepcopy(item["config"])
        assert_no_test_access(split_id=config["data"]["target_access_mode"])
        specs.append(CandidateSpec(
            candidate_id=item["candidate_id"],
            model_family="TRANSFORMER_ENCODER",
            shortlist_position=0,
            config=config,
            config_fingerprint=item["config_fingerprint"],
            feature_variant_id=FEATURE_VARIANT,
            target_scaling_option="YS1",
            lookback_steps=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            source_phase="V2_E08_IMMUTABLE_SNAPSHOT",
            candidate_role="PLAIN_SGD_LR_CHALLENGER",
        ))
    return tuple(specs)


def _registry_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage = deepcopy(config["lineage"])
    return {
        "lineage": lineage,
        "feature_sets": {
            "variant_feature_counts": {FEATURE_VARIANT: 33},
            "variant_fingerprints": {FEATURE_VARIANT: lineage["feature_fingerprint"]},
        },
        "window_fingerprints": {},
    }


def build_e08_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.model_improvement_v2.pretest_adapter import (
        build_v2_pretest_dataset,
        load_phase44_fold_evidence,
    )
    from course_work.rolling_origin.real_run import RunContext

    validate_e08_document(document, root)
    candidates = build_e08_candidates(document)
    artifact_dir = root / "artifacts/model_improvement_v2/experiments/E08"
    cache: dict[str, Any] = {}

    def dataset_factory(candidate_spec, _fold):
        if (
            candidate_spec.feature_variant_id != FEATURE_VARIANT
            or candidate_spec.candidate_id not in CANDIDATE_IDS.values()
        ):
            raise E08PreflightError("E08 dataset factory rejects unknown candidate")
        assert_no_test_access(split_id=candidate_spec.config["data"]["target_access_mode"])
        if "bundle" not in cache:
            cache["bundle"] = build_v2_pretest_dataset(
                root,
                tuple(document["feature_order"]),
                experiment_id=EXPERIMENT_ID,
                feature_variant_id=FEATURE_VARIANT,
            )
        dataset, _evidence, audit = cache["bundle"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E08 Test firewall failed")
        return dataset

    evidence = load_phase44_fold_evidence(root)
    return RunContext(
        project_root=root,
        transformer_shortlist_path=root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json",
        lstm_handoff_path=root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",
        phase_42_signoff_path=root / "artifacts/candidate_synthesis/phase_42_signoff.json",
        phase_43_signoff_path=root / "artifacts/lstm_tuning/phase_43_signoff.json",
        artifact_dir=artifact_dir,
        registry_root=artifact_dir / "registry",
        run_root=artifact_dir / "runs",
        scientific_max_epochs=50,
        scientific_patience=10,
        seed=SUPPORTED_SEED,
        device=str(candidates[0].config["runtime"]["device_type"]),
        is_rehearsal=False,
        rehearsal_synthetic=False,
        reuse_completed_runs=False,
        dataset_factory=dataset_factory,
        candidate_specs=candidates,
        registry_namespace=REGISTRY_NAMESPACE,
        seed_before_model_construction=True,
        validated_registry_lifecycle=True,
        execution_track=EXECUTION_TRACK,
        registry_upstream_context=_registry_upstream(candidates[0].config),
        robase_train_ids=evidence.train_ids,
        robase_val_ids=evidence.validation_ids,
        apply_fold_x_scaling=True,
        enable_stage_b_lr_replay=False,
    )


def _run_dir(root: Path, run_id: str) -> Path:
    return root / "artifacts/model_improvement_v2/experiments/E08/runs" / run_id


def _audit_resume_contract(root: Path) -> dict[str, Any]:
    registry_path = root / "artifacts/model_improvement_v2/experiments/E08/registry/experiment_registry.jsonl"
    if not registry_path.is_file():
        raise E08PreflightError("E08 registry missing for resume")
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 24 or len({row.get("run_id") for row in records}) != 24:
        raise E08PreflightError("E08 resume requires exactly 24 unique completed runs")
    ledger: dict[str, str] = {}
    epochs: dict[str, int] = {}
    checkpoint_sha256: dict[str, str] = {}
    for record in records:
        run_id = record.get("run_id")
        cid = record.get("candidate_id")
        stage = record.get("sweep_stage")
        if record.get("status") != "COMPLETED":
            raise E08PreflightError(f"E08 resume run is not COMPLETED: {run_id}")
        if cid not in CANDIDATE_IDS.values() or stage not in {
            "RO1_A", "RO2_A", "RO3_A", "RO1_B", "RO2_B", "RO3_B"
        }:
            raise E08PreflightError(f"E08 resume run identity mismatch: {run_id}")
        if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
            raise E08PreflightError(f"E08 resume Test-scoped run rejected: {run_id}")
        best_epoch = record.get("best_epoch")
        if not isinstance(best_epoch, int) or best_epoch < 1:
            raise E08PreflightError(f"E08 resume invalid best_epoch: {run_id}")
        key = f"{cid}:{stage}"
        if key in ledger:
            raise E08PreflightError(f"E08 duplicate resume key: {key}")
        ledger[key] = run_id
        epochs[key] = best_epoch
        checkpoint_name = "refit_final.pt" if stage.endswith("_B") else "best_checkpoint.pt"
        checkpoint = _run_dir(root, run_id) / "checkpoints" / checkpoint_name
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise E08PreflightError(f"E08 checkpoint missing/empty: {run_id}")
        actual_sha = _sha256(checkpoint)
        registered = [
            artifact for artifact in record.get("artifacts", [])
            if Path(artifact.get("artifact_path", "")).name == checkpoint_name
        ]
        if len(registered) != 1 or registered[0].get("sha256") != actual_sha:
            raise E08PreflightError(f"E08 checkpoint registration/SHA mismatch: {run_id}")
        if stage.endswith("_B"):
            checkpoint_sha256[key] = actual_sha
    expected_keys = {
        f"{cid}:RO{fold}_{stage}"
        for cid in CANDIDATE_IDS.values()
        for fold in (1, 2, 3)
        for stage in ("A", "B")
    }
    if set(ledger) != expected_keys or len(checkpoint_sha256) != 12:
        raise E08PreflightError("E08 resume ledger/checkpoint matrix mismatch")
    for cid in CANDIDATE_IDS.values():
        for fold in (1, 2, 3):
            if epochs[f"{cid}:RO{fold}_A"] != epochs[f"{cid}:RO{fold}_B"]:
                raise E08PreflightError(f"E08 Stage A/B epoch mismatch: {cid}/RO{fold}")
    return {
        "status": "PASS",
        "completed_run_count": 24,
        "stage_a_run_count": 12,
        "stage_b_run_count": 12,
        "stage_b_checkpoint_count": 12,
        "locked_run_ids": ledger,
        "locked_best_epochs": epochs,
        "stage_b_checkpoint_sha256": checkpoint_sha256,
        "new_training_run_ids_allowed": False,
        "training_reexecution_allowed": False,
        "test_access_authorized": False,
    }


def build_e08_resume_context(root: Path, document: Mapping[str, Any]):
    audit = _audit_resume_contract(root)
    context = build_e08_run_context(root, document)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["locked_run_ids"])
    return context, audit


def _write_success_contracts(
    root: Path,
    document: Mapping[str, Any],
    result,
    preflight: Mapping[str, Any],
    resume_audit: Mapping[str, Any] | None = None,
) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E08"
    control = document["control"]
    rows = [{
        "candidate_id": BASE_CANDIDATE_ID,
        "optimizer_name": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "momentum": None,
        "nesterov": None,
        "scheduler": "OFF",
        "source": "CONTROL_REUSED_FROM_E01_READ_ONLY",
        "pooled_metrics": CONTROL_METRICS,
    }]
    for lr in CHALLENGER_LRS:
        cid = CANDIDATE_IDS[lr]
        rows.append({
            "candidate_id": cid,
            "optimizer_name": "SGD",
            "learning_rate": lr,
            "weight_decay": 1e-3,
            "momentum": 0.0,
            "nesterov": False,
            "scheduler": "OFF",
            "source": "E08_TRAINED_CHALLENGER",
            "pooled_metrics": result.pooled_metrics_by_cid[cid],
        })
    ranked = sorted(rows, key=lambda row: float(row["pooled_metrics"]["rmse_wh"]))
    eligible = [
        row for row in rows
        if row["source"] == "E08_TRAINED_CHALLENGER"
        and float(row["pooled_metrics"]["rmse_wh"]) <= PROMOTION_THRESHOLD_RMSE_WH
    ]
    promoted = min(eligible, key=lambda row: float(row["pooled_metrics"]["rmse_wh"])) if eligible else None
    decision = {
        "action": "PROMOTE_PLAIN_SGD" if promoted else "RETAIN_E01_ADAMW",
        "selected_candidate_id": promoted["candidate_id"] if promoted else BASE_CANDIDATE_ID,
        "selected_optimizer": "SGD" if promoted else "AdamW",
        "selected_learning_rate": promoted["learning_rate"] if promoted else 3e-4,
        "best_observed_rmse_wh": float(ranked[0]["pooled_metrics"]["rmse_wh"]),
    }
    challenger_run_ids = {
        "stage_a": {
            f"{cid}:{fold}": run_id
            for (cid, fold), run_id in result.stage_a_run_ids.items()
        },
        "stage_b": {
            f"{cid}:{fold}": run_id
            for (cid, fold), run_id in result.stage_b_run_ids.items()
        },
    }
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E08_EXECUTION-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "control_policy": "CONTROL_REUSED_FROM_E01_READ_ONLY",
        "control_run_ids": control["run_ids"],
        "challenger_run_ids": challenger_run_ids,
        "selected_best_epochs": {
            f"{cid}:{fold}": epoch
            for (cid, fold), epoch in result.inner_best_epochs.items()
        },
        "optimizer_policy": "PLAIN_SGD_CONSTANT_LR_FRESH_STATE",
        "scheduler": "OFF",
        "resume_audit": dict(resume_audit) if resume_audit is not None else None,
        "test_status": "NOT_ACCESSED",
        "preflight": dict(preflight),
    }
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E08_SGD_LR_ABLATION-v1",
        "experiment_id": EXPERIMENT_ID,
        "primary_change": "OPTIMIZER_FAMILY_WITH_PREDECLARED_LR",
        "primary_metric": "pooled_rmse_wh",
        "results": rows,
        "ranking": [row["candidate_id"] for row in ranked],
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "promotion_rule": "challenger_pooled_rmse_wh <= 59.75291570400546",
        "decision": decision,
        "mae_can_promote": False,
        "target_population_policy": "IDENTICAL_PHASE44_RO1_RO2_RO3",
        "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(output / "e08_execution_manifest.json", canonical_json_bytes(manifest))
    atomic_write_bytes(output / "e08_sgd_lr_ablation_comparison.json", canonical_json_bytes(comparison))


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e08_config(resolved)
    preflight = validate_e08_document(document, resolved)
    context = build_e08_run_context(resolved, document)
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


def run_resume_stage_c(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e08_config(resolved)
    preflight = validate_e08_document(document, resolved)
    context, resume_audit = build_e08_resume_context(resolved, document)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result = run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary, file=sys.stderr)
        if result.exception:
            print(result.exception, file=sys.stderr)
        return result.exit_code
    _write_success_contracts(resolved, document, result, preflight, resume_audit)
    print(result.summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2 E08 plain-SGD optimizer audit")
    parser.add_argument("--experiment", choices=[EXPERIMENT_ID], required=True)
    parser.add_argument(
        "--mode", choices=["preflight", "official", "resume-stage-c"], default="preflight"
    )
    parser.add_argument("--seed", type=int, default=SUPPORTED_SEED)
    parser.add_argument("--authorize-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED:
        print(f"ERROR: E08 screening seed must be {SUPPORTED_SEED}", file=sys.stderr)
        return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: E08 official mode requires --authorize-training.", file=sys.stderr)
        return 3
    if args.mode != "official" and args.authorize_training:
        print("ERROR: --authorize-training is valid only in official mode.", file=sys.stderr)
        return 2
    try:
        if args.mode == "official":
            return run_official()
        if args.mode == "resume-stage-c":
            return run_resume_stage_c()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (E08PreflightError, PermissionError, ValueError) as exc:
        print(f"E08 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
