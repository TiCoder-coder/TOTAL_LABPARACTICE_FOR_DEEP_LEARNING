"""V2 E12 daily lag-144 feature ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.data.feature_sets import compute_feature_fingerprint, get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)
from course_work.model_improvement_v2.e12_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    LAG_DEFINITION,
    LAG_FEATURE,
    LAG_STEPS,
    build_e12_pretest_datasets,
    challenger_feature_order,
)

EXPERIMENT_ID = "E12"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E12"
REGISTRY_NAMESPACE = "V2_E12"
SEED = 42
CONTROL_ID = "TR_C2_ALT_LOOKBACK_E12_CONTROL"
CHALLENGER_ID = "TR_C2_ALT_LOOKBACK_E12_LAG144"
CONFIG_PATH = Path("artifacts/model_improvement_v2/experiments/E12/e12_config_snapshot.json")
E10_COMPARISON = Path("artifacts/model_improvement_v2/experiments/E10/e10_target_delta_ablation_comparison.json")
E11_COMPARISON = Path("artifacts/model_improvement_v2/experiments/E11/e11_rolling_target_ablation_comparison.json")
E11_MANIFEST = Path("artifacts/model_improvement_v2/experiments/E11/e11_execution_manifest.json")
E11_SIGNOFF = Path("artifacts/model_improvement_v2/experiments/E11/phase_44_signoff.json")
E11_REGISTRY = Path("artifacts/model_improvement_v2/experiments/E11/registry/experiment_registry.jsonl")
EXPECTED_INCUMBENT = {
    "rmse_wh": 59.85291570400546,
    "mae_wh": 26.650501720144604,
    "r2": 0.5789433617557903,
}
EXPECTED_E01_CONFIG_FINGERPRINT = "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24"
EXPECTED_COMMON_FINGERPRINT = "8c7441d7a2ad6f388ad9e392b5e234d24a2a1d39269ef340843881355208f036"
EXPECTED_FOLD_FINGERPRINTS = {
    "RO1": "d0bc40746e761f01af29906f3626227a2af56e927d026aaaa7881a35e1565903",
    "RO2": "b2b4256a70bcbb05714b8e73668afd61491055884211f68c4a8850febb7d8051",
    "RO3": EXPECTED_COMMON_FINGERPRINT,
}
EXPECTED_ORIGINAL_TARGET_COUNT = 16630
EXPECTED_TARGET_COUNT = 16558
EXPECTED_TRAIN_TARGET_COUNT = 13598
EXPECTED_LOST_TARGET_COUNT = 72
EXPECTED_LOST_TARGET_FINGERPRINT = "d653d9cccd3c3fc7f2dd2c17609606ec233d58345c82ca0d859072f14a9de3e8"
EXPECTED_FIRST_TARGET_ID = "TGT_00000216"
WINDOW_CONTRACT_FINGERPRINT = "0c04c73e51fe13df8509b273fd2a030060304f4225f0d32690cd386ef051537e"
SCALER_CONTRACT_CHECKSUM = "6fae08714dba3a3c6b6939b3ba0a7b2a354225bc336690300d299efbc65ee6b8"
CONTROL_CONFIG_FINGERPRINT = "1c9b5b51be1ee37b9d08f256802d6d46fe6bbca4b00b48f31bb6f1047fb9b17f"
CHALLENGER_CONFIG_FINGERPRINT = "768718d43d8060ddec366eb42c0ea38c6e313ab7de6201bf8f00e1d4e48c25ab"


class E12PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E12PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E12PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e12_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_PATH)


def _audit_predecessor_decisions(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    decisions = document.get("predecessor_human_decisions", {})
    requirements = {
        "E10": (E10_COMPARISON, "carry_forward_delta_features"),
        "E11": (E11_COMPARISON, "carry_forward_rolling_features"),
    }
    audited: dict[str, Any] = {}
    for experiment_id, (relative, carry_key) in requirements.items():
        decision = decisions.get(experiment_id, {})
        path = root / relative
        if (
            decision.get("decision") != "REJECTED"
            or decision.get("decision_authority") != "HUMAN"
            or decision.get("accepted_incumbent_after_decision") != "E01_DIRECT_FS2_TF1"
            or decision.get(carry_key) is not False
            or decision.get("comparison_sha256") != _sha256(path)
        ):
            raise E12PreflightError(f"{experiment_id} Human rejection lock mismatch")
        evidence = _read_json(path)
        if (
            evidence.get("test_status") != "NOT_ACCESSED"
            or evidence.get("promotion_evaluation", {}).get("eligible_for_human_promotion") is not False
        ):
            raise E12PreflightError(f"{experiment_id} rejection evidence mismatch")
        audited[experiment_id] = {
            "status": "PASS",
            "decision": "REJECTED",
            "comparison_sha256": _sha256(path),
        }

    manifest = _read_json(root / E11_MANIFEST)
    signoff = _read_json(root / E11_SIGNOFF)
    if (
        manifest.get("status") != "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED"
        or manifest.get("test_status") != "NOT_ACCESSED"
        or signoff.get("overall_status") not in {"PASS", "PASS_WITH_WARNING"}
        or signoff.get("test_status") != "NOT_ACCESSED"
    ):
        raise E12PreflightError("E11 recovery/signoff is not complete")
    canonical_ids = tuple(manifest.get("run_ids", {}).get("stage_a", {}).values()) + tuple(
        manifest.get("run_ids", {}).get("stage_b", {}).values()
    )
    if len(canonical_ids) != 12 or len(set(canonical_ids)) != 12:
        raise E12PreflightError("E11 recovery does not contain 12 unique canonical runs")
    records = [
        json.loads(line)
        for line in (root / E11_REGISTRY).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record.get("run_id"): record for record in records}
    for run_id in canonical_ids:
        record = by_id.get(run_id, {})
        if record.get("status") != "COMPLETED":
            raise E12PreflightError(f"E11 canonical run is not COMPLETED: {run_id}")
        if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
            raise E12PreflightError(f"E11 canonical run is Test-scoped: {run_id}")
        checkpoints = [
            artifact
            for artifact in record.get("artifacts", [])
            if artifact.get("artifact_type") == "BEST_CHECKPOINT"
        ]
        if len(checkpoints) != 1:
            raise E12PreflightError(f"E11 canonical checkpoint registration mismatch: {run_id}")
        checkpoint = root / checkpoints[0]["artifact_path"]
        if not checkpoint.is_file() or _sha256(checkpoint) != checkpoints[0].get("sha256"):
            raise E12PreflightError(f"E11 canonical checkpoint checksum mismatch: {run_id}")
    preserved = {
        "RUN_V2_TR_E11_RO1_A_0001_11B8DF31": "FAILED",
        "RUN_V2_TR_E11_RO1_B_0009_41FB2473": "RUNNING",
    }
    if any(by_id.get(run_id, {}).get("status") != status for run_id, status in preserved.items()):
        raise E12PreflightError("E11 failed/interrupted provenance evidence was not preserved")
    audited["E11_RECOVERY"] = {
        "status": "PASS",
        "canonical_completed_runs": 12,
        "stage_a_reused": 6,
        "stage_b_reused": 1,
        "stage_b_trained_during_recovery": 5,
        "failed_interrupted_evidence_preserved": True,
    }
    return audited


def _expected_configs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _read_json(
        root / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json"
    )["v1_config"]
    if compute_config_fingerprint(source) != EXPECTED_E01_CONFIG_FINGERPRINT:
        raise E12PreflightError("E01 source config fingerprint changed")
    control = deepcopy(source)
    control["data"]["train_sample_count"] = EXPECTED_TRAIN_TARGET_COUNT
    control["lineage"]["population_version"] = "MODEL_IMPROVEMENT_V2_E12_COMMON-v1"
    control["lineage"]["population_fingerprint"] = EXPECTED_COMMON_FINGERPRINT
    control["lineage"]["window_version"] = "MODEL_IMPROVEMENT_V2_E12_L72_WB0-v1"
    control["lineage"]["window_fingerprint"] = WINDOW_CONTRACT_FINGERPRINT
    challenger = deepcopy(control)
    order = challenger_feature_order()
    challenger["data"]["feature_variant_id"] = CHALLENGER_VARIANT
    challenger["data"]["feature_count"] = len(order)
    challenger["model"]["input_size"] = len(order)
    challenger["lineage"]["feature_fingerprint"] = compute_feature_fingerprint(
        CHALLENGER_VARIANT, list(order)
    )
    challenger["lineage"]["scaler_bundle_id"] = "FOLD_LOCAL__E12__FS2_TF1_LAG144"
    challenger["lineage"]["scaler_bundle_checksum"] = SCALER_CONTRACT_CHECKSUM
    return control, challenger


def validate_e12_document(
    document: Mapping[str, Any], root: Path, *, materialize_data: bool = True
) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E12PreflightError("Wrong E12 identity/track")
    incumbent = document.get("accepted_incumbent", {})
    if (
        incumbent.get("source_experiment") != "E01"
        or incumbent.get("candidate_id") != "TR_C2_ALT_LOOKBACK"
        or incumbent.get("pooled_metrics") != EXPECTED_INCUMBENT
        or incumbent.get("test_status") != "NOT_ACCESSED"
    ):
        raise E12PreflightError("Accepted incumbent is not locked E01")
    predecessor = _audit_predecessor_decisions(root, document)
    primary = document.get("primary_change", {})
    if primary != {
        "factor": "DAILY_LAG_144_FEATURE",
        "feature_name": LAG_FEATURE,
        "lag_steps": LAG_STEPS,
        "definition": LAG_DEFINITION,
        "past_target_only": True,
        "off_by_one_alignment": "CONTEXT_ROW_s_READS_TARGET_y_s_minus_144",
        "imputation": "NONE",
    }:
        raise E12PreflightError("E12 lag-144 causal contract mismatch")

    expected_control, expected_challenger = _expected_configs(root)
    control = document.get("control", {})
    challenger = document.get("challenger", {})
    expected_items = (
        (control, CONTROL_ID, CONTROL_VARIANT, 33, CONTROL_CONFIG_FINGERPRINT, expected_control),
        (challenger, CHALLENGER_ID, CHALLENGER_VARIANT, 34, CHALLENGER_CONFIG_FINGERPRINT, expected_challenger),
    )
    for item, candidate_id, variant, count, fingerprint, config in expected_items:
        assert_no_test_access(split_id=config["data"]["target_access_mode"])
        if (
            item.get("candidate_id") != candidate_id
            or item.get("feature_variant_id") != variant
            or item.get("feature_count") != count
            or item.get("config_fingerprint") != fingerprint
            or compute_config_fingerprint(config) != fingerprint
            or item.get("training_required") is not True
            or item.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
        ):
            raise E12PreflightError(f"E12 {variant} candidate/config/init mismatch")
    if control.get("reuse_previous_metrics") is not False:
        raise E12PreflightError("E12 control must retrain on the lag-144 common population")
    rejected = {
        "appliances_delta_1",
        "appliances_abs_delta_1",
        "appliances_delta_2",
        "appliances_roll_mean_3",
        "appliances_roll_mean_6",
        "appliances_roll_mean_12",
        "appliances_roll_std_6",
        "appliances_roll_std_12",
        "appliances_roll_max_12",
        "appliances_roll_min_12",
    }
    if rejected.intersection(challenger_feature_order()):
        raise E12PreflightError("Rejected E10/E11 feature block leaked into E12")
    changed = assert_one_primary_change(EXPERIMENT_ID, expected_control, expected_challenger)
    stripped: list[dict[str, Any]] = []
    for original in (expected_control, expected_challenger):
        config = deepcopy(original)
        for group, keys in (
            ("data", ("feature_variant_id", "feature_count")),
            ("model", ("input_size",)),
            ("lineage", ("feature_fingerprint", "scaler_bundle_id", "scaler_bundle_checksum")),
        ):
            for key in keys:
                config[group].pop(key, None)
        stripped.append(config)
    if stripped[0] != stripped[1]:
        raise E12PreflightError("E12 changes model/training outside the lag-144 feature")
    e01 = _read_json(
        root / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json"
    )["v1_config"]
    if expected_control["model"] != e01["model"] or expected_control["training"] != e01["training"]:
        raise E12PreflightError("E12 control is not E01-equivalent in model/training")

    selection = document.get("selection_policy", {})
    expected_selection = {
        "primary_metric": "pooled_rmse_wh",
        "rmse_min_improvement_wh": 0.10,
        "mae_max_degradation_wh": 0.25,
        "worst_fold_rmse_max_degradation_wh": 0.50,
        "fold_rmse_std_max_degradation_wh": 0.50,
        "all_guardrails_required": True,
        "weighted_composite": False,
        "test_metrics_allowed": False,
        "automatic_promotion": False,
        "decision_authority": "HUMAN",
    }
    if selection != expected_selection:
        raise E12PreflightError("E12 Wave2 promotion guardrails are not locked")
    orchestration = document.get("v2_orchestration", {})
    expected_orchestration = {
        "registry_namespace": REGISTRY_NAMESPACE,
        "execution_track": EXECUTION_TRACK,
        "training_candidate_count": 2,
        "expected_stage_a_runs": 6,
        "expected_stage_b_runs": 6,
        "expected_total_training_runs": 12,
        "preflight_consumes_run_id": False,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E12PreflightError(f"E12 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E12" not in value:
            raise E12PreflightError(f"E12 path mismatch: {key}")
        assert_v2_artifact_path(value)

    population = None
    if materialize_data:
        datasets, evidence, audit = build_e12_pretest_datasets(root)
        control_ids = datasets[CONTROL_VARIANT].window_records["target_id"].tolist()
        challenger_ids = datasets[CHALLENGER_VARIANT].window_records["target_id"].tolist()
        actual_folds = {
            str(fold.fold_id): fold.fold_population_fingerprint for fold in evidence.folds
        }
        if (
            control_ids != challenger_ids
            or audit.common_population_fingerprint != EXPECTED_COMMON_FINGERPRINT
            or actual_folds != EXPECTED_FOLD_FINGERPRINTS
            or audit.common_target_count != EXPECTED_TARGET_COUNT
            or audit.lost_target_count != EXPECTED_LOST_TARGET_COUNT
            or audit.lost_target_fingerprint != EXPECTED_LOST_TARGET_FINGERPRINT
            or audit.first_eligible_target_id != EXPECTED_FIRST_TARGET_ID
            or audit.control.test_rows_read
            or audit.challenger.test_rows_read
            or audit.control.test_target_ids_seen
            or audit.challenger.test_target_ids_seen
            or not audit.no_future_target_access
            or not audit.off_by_one_alignment_verified
            or any(value != "PASS" for value in evidence.fingerprint_status.values())
        ):
            raise E12PreflightError("E12 common population/lag alignment/Test firewall failed")
        expected_population = document.get("common_population", {})
        if (
            expected_population.get("expected_target_count") != audit.common_target_count
            or expected_population.get("expected_population_fingerprint") != audit.common_population_fingerprint
            or expected_population.get("lost_target_count") != audit.lost_target_count
            or expected_population.get("lost_target_fingerprint") != audit.lost_target_fingerprint
        ):
            raise E12PreflightError("E12 config population/lost-target lock mismatch")
        population = {
            "status": "PASS",
            "target_count": audit.common_target_count,
            "population_fingerprint": audit.common_population_fingerprint,
            "fold_fingerprints": actual_folds,
            "lost_target_count": audit.lost_target_count,
            "first_eligible_target_id": audit.first_eligible_target_id,
            "ordered_target_ids_identical": True,
            "test_rows_read": 0,
            "test_target_ids_seen": 0,
        }
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "accepted_incumbent": "E01_DIRECT_FS2_TF1",
        "predecessor_audit": predecessor,
        "one_primary_change": "PASS",
        "changed_paths": list(changed),
        "population": population,
        "training_candidates": [CONTROL_ID, CHALLENGER_ID],
        "expected_total_training_runs": 12,
        "fold_local_x_scaling": True,
        "fold_local_y_scaling": True,
        "test_access": "NO",
        "training_executed": False,
        "inference_executed": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e12_document(load_e12_config(resolved), resolved)


def build_e12_candidates(document: Mapping[str, Any], root: Path):
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    control, challenger = _expected_configs(root)
    items = (
        (document["control"], control),
        (document["challenger"], challenger),
    )
    return tuple(
        CandidateSpec(
            candidate_id=item["candidate_id"],
            model_family="TRANSFORMER_ENCODER",
            shortlist_position=0,
            config=deepcopy(config),
            config_fingerprint=item["config_fingerprint"],
            feature_variant_id=item["feature_variant_id"],
            target_scaling_option="YS1",
            lookback_steps=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            source_phase="V2_E12_IMMUTABLE_SNAPSHOT",
            candidate_role=item["role"],
        )
        for item, config in items
    )


def build_e12_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.rolling_origin.real_run import RunContext

    validate_e12_document(document, root, materialize_data=False)
    candidates = build_e12_candidates(document, root)
    datasets, evidence, audit = build_e12_pretest_datasets(root)

    def dataset_factory(candidate, _fold):
        if candidate.candidate_id not in {CONTROL_ID, CHALLENGER_ID}:
            raise E12PreflightError("E12 dataset factory rejects candidate")
        if audit.control.test_rows_read or audit.challenger.test_rows_read:
            raise PermissionError("E12 Test firewall failed")
        return datasets[candidate.feature_variant_id]

    output = root / "artifacts/model_improvement_v2/experiments/E12"
    feature_sets = {
        CONTROL_VARIANT: {
            "feature_count": 33,
            "feature_fingerprint": candidates[0].config["lineage"]["feature_fingerprint"],
        },
        CHALLENGER_VARIANT: {
            "feature_count": 34,
            "feature_fingerprint": candidates[1].config["lineage"]["feature_fingerprint"],
        },
    }
    upstream = {
        "lineage": {},
        "feature_sets": {
            "variant_feature_counts": {key: value["feature_count"] for key, value in feature_sets.items()},
            "variant_fingerprints": {key: value["feature_fingerprint"] for key, value in feature_sets.items()},
        },
        "window_fingerprints": {"E12_COMMON_L72_WB0": WINDOW_CONTRACT_FINGERPRINT},
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
        seed=SEED,
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
        registry_upstream_context=upstream,
        robase_train_ids=evidence.train_ids,
        robase_val_ids=evidence.validation_ids,
        apply_fold_x_scaling=True,
    )


def _audit_resume_contract(root: Path) -> dict[str, Any]:
    """Lock only completed, checksum-verified E12 Stage A/B runs."""

    registry_path = root / "artifacts/model_improvement_v2/experiments/E12/registry/experiment_registry.jsonl"
    if not registry_path.is_file():
        raise E12PreflightError("E12 registry missing for partial recovery")
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_candidates = {
        CONTROL_ID: (CONTROL_VARIANT, 33),
        CHALLENGER_ID: (CHALLENGER_VARIANT, 34),
    }
    control_config, challenger_config = _expected_configs(root)
    expected_base_configs = {
        CONTROL_ID: control_config,
        CHALLENGER_ID: challenger_config,
    }
    valid_stages = {f"RO{fold}_{stage}" for fold in (1, 2, 3) for stage in ("A", "B")}
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    preserved_noncompleted: list[str] = []
    for record in records:
        run_id = str(record.get("run_id", ""))
        candidate = record.get("candidate_id")
        stage = record.get("sweep_stage")
        if candidate not in expected_candidates or stage not in valid_stages:
            raise E12PreflightError(f"E12 recovery record mismatch: {run_id}")
        config = record.get("config", {})
        if (
            record.get("test_access_authorized") is not False
            or config.get("data", {}).get("target_access_mode") == "TEST"
        ):
            raise E12PreflightError(f"E12 recovery Test-scoped record: {run_id}")
        if record.get("status") != "COMPLETED":
            preserved_noncompleted.append(run_id)
            continue
        by_key.setdefault((candidate, stage), []).append(record)

    ledger: dict[str, str] = {}
    checkpoint_sha: dict[str, str] = {}
    best_epochs: dict[str, int] = {}
    run_root = root / "artifacts/model_improvement_v2/experiments/E12/runs"
    for (candidate, stage), completed in by_key.items():
        if len(completed) != 1:
            raise E12PreflightError(
                f"E12 recovery has duplicate COMPLETED runs for {(candidate, stage)}"
            )
        record = completed[0]
        run_id = record["run_id"]
        variant, count = expected_candidates[candidate]
        config = record.get("config", {})
        data = config.get("data", {})
        model = config.get("model", {})
        lineage = config.get("lineage", {})
        expected_stage = stage[-1]
        expected_fold = stage[:3]
        if (
            data.get("feature_variant_id") != variant
            or data.get("feature_count") != count
            or model.get("input_size") != count
            or data.get("target_access_mode") != "VALIDATION"
            or lineage.get("rolling_origin_candidate_id") != candidate
            or lineage.get("rolling_origin_fold_id") != expected_fold
            or lineage.get("rolling_origin_stage") != expected_stage
            or lineage.get("population_fingerprint") != EXPECTED_COMMON_FINGERPRINT
        ):
            raise E12PreflightError(f"E12 completed-run config mismatch: {run_id}")
        best_epoch = record.get("best_epoch")
        if not isinstance(best_epoch, int) or best_epoch <= 0:
            raise E12PreflightError(f"E12 completed-run best epoch invalid: {run_id}")
        expected_static = deepcopy(expected_base_configs[candidate])
        if expected_stage == "B":
            expected_static["training"]["max_epochs"] = best_epoch
            expected_static["training"]["early_stopping_enabled"] = False
            expected_static["training"]["early_stopping_patience"] = 0
        actual_static = deepcopy(config)
        rolling_only_fields = (
            "rolling_origin_candidate_id",
            "rolling_origin_fold_id",
            "rolling_origin_stage",
            "rolling_origin_metric_population_fingerprint",
            "rolling_origin_metric_population_count",
        )
        for field in rolling_only_fields:
            expected_static["lineage"].pop(field, None)
            actual_static["lineage"].pop(field, None)
        if actual_static != expected_static:
            raise E12PreflightError(f"E12 completed-run scientific config changed: {run_id}")
        if record.get("config_fingerprint") != compute_config_fingerprint(config):
            raise E12PreflightError(f"E12 completed-run config fingerprint mismatch: {run_id}")
        checkpoint_name = "best_checkpoint.pt" if expected_stage == "A" else "refit_final.pt"
        checkpoint = run_root / run_id / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise E12PreflightError(f"E12 completed-run checkpoint missing: {run_id}")
        actual_sha = _sha256(checkpoint)
        registered = [
            artifact for artifact in record.get("artifacts", [])
            if Path(str(artifact.get("artifact_path", ""))).name == checkpoint_name
        ]
        if len(registered) != 1 or registered[0].get("sha256") != actual_sha:
            raise E12PreflightError(f"E12 completed-run checkpoint SHA mismatch: {run_id}")
        if expected_stage == "A":
            required = (
                run_root / run_id / "training_history.csv",
                run_root / run_id / "metrics" / "best_validation_metrics.json",
            )
            if not all(path.is_file() for path in required):
                raise E12PreflightError(f"E12 Stage-A evidence incomplete: {run_id}")
        key = f"{candidate}:{stage}"
        ledger[key] = run_id
        checkpoint_sha[key] = actual_sha
        best_epochs[key] = best_epoch

    expected_keys = {
        f"{candidate}:RO{fold}_{stage}"
        for candidate in expected_candidates
        for fold in (1, 2, 3)
        for stage in ("A", "B")
    }
    if not ledger:
        raise E12PreflightError("E12 partial recovery found no reusable COMPLETED runs")
    for key in set(ledger):
        if key.endswith("_B"):
            stage_a_key = key[:-1] + "A"
            if stage_a_key not in ledger or best_epochs[stage_a_key] != best_epochs[key]:
                raise E12PreflightError(f"E12 Stage-B parent/epoch mismatch: {key}")
    missing = sorted(expected_keys - set(ledger))
    return {
        "status": "PASS",
        "locked_run_ids": ledger,
        "checkpoint_sha256": checkpoint_sha,
        "best_epochs": best_epochs,
        "missing_run_keys": missing,
        "reused_completed_count": len(ledger),
        "training_run_count_required": len(missing),
        "noncompleted_evidence_preserved": sorted(preserved_noncompleted),
        "test_status": "NOT_ACCESSED",
    }


def build_e12_resume_context(root: Path, document: Mapping[str, Any]):
    context = build_e12_run_context(root, document)
    audit = _audit_resume_contract(root)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["locked_run_ids"])
    context.allow_partial_stage_recovery = bool(audit["missing_run_keys"])
    return context, audit


def evaluate_promotion(
    control: Mapping[str, float],
    challenger: Mapping[str, float],
    control_fold_rmse: list[float],
    challenger_fold_rmse: list[float],
) -> dict[str, Any]:
    gates = {
        "rmse_improvement": float(control["rmse_wh"]) - float(challenger["rmse_wh"]) >= 0.10,
        "mae_guardrail": float(challenger["mae_wh"]) - float(control["mae_wh"]) <= 0.25,
        "worst_fold_rmse_guardrail": max(challenger_fold_rmse) - max(control_fold_rmse) <= 0.50,
        "fold_rmse_std_guardrail": statistics.pstdev(challenger_fold_rmse) - statistics.pstdev(control_fold_rmse) <= 0.50,
    }
    return {
        "gates": gates,
        "eligible_for_human_promotion": all(gates.values()),
        "automatic_promotion": False,
        "human_decision_required": True,
    }


def _write_success(
    root: Path,
    document: Mapping[str, Any],
    result,
    preflight: Mapping[str, Any],
    recovery_audit: Mapping[str, Any] | None = None,
) -> None:
    from course_work.rolling_origin.real_run import fit_fold_local_scaler
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E12"
    rows = list(csv.DictReader((output / "rolling_origin_results.csv").open(encoding="utf-8")))
    folds = {
        candidate_id: [float(row["rmse_wh"]) for row in rows if row["candidate_id"] == candidate_id]
        for candidate_id in (CONTROL_ID, CHALLENGER_ID)
    }
    metrics = result.pooled_metrics_by_cid
    promotion = evaluate_promotion(
        metrics[CONTROL_ID], metrics[CHALLENGER_ID], folds[CONTROL_ID], folds[CHALLENGER_ID]
    )
    datasets, evidence, causal = build_e12_pretest_datasets(root)
    context = build_e12_run_context(root, document)
    candidates = {candidate.candidate_id: candidate for candidate in context.candidate_specs}
    scaler_rows = []
    for fold in evidence.folds:
        for candidate_id in (CONTROL_ID, CHALLENGER_ID):
            candidate = candidates[candidate_id]
            dataset = datasets[candidate.feature_variant_id]
            for stage in ("A", "B"):
                bundle, audit = fit_fold_local_scaler(
                    candidate=candidate,
                    fold=fold,
                    fold_dataset=dataset,
                    fit_stage=stage,
                    ctx=context,
                )
                scaler_rows.append({
                    **audit.as_row(),
                    "feature_order": list(dataset.feature_order),
                    "feature_count": len(dataset.feature_order),
                    "bundle_checksum_verified": bundle.checksum() == audit.bundle_checksum,
                })
    documents = {
        "e12_execution_manifest.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_EXECUTION-v1",
            "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
            "accepted_incumbent_at_entry": "E01_DIRECT_FS2_TF1",
            "predecessor_human_decisions": {"E10": "REJECTED", "E11": "REJECTED"},
            "run_ids": {
                "stage_a": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_a_run_ids.items()},
                "stage_b": {f"{cid}:{fold}": run_id for (cid, fold), run_id in result.stage_b_run_ids.items()},
            },
            "selected_best_epochs": {f"{cid}:{fold}": epoch for (cid, fold), epoch in result.inner_best_epochs.items()},
            "preflight": dict(preflight),
            "recovery": dict(recovery_audit) if recovery_audit is not None else None,
            "test_status": "NOT_ACCESSED",
        },
        "e12_lag144_ablation_comparison.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_LAG144_ABLATION-v1",
            "experiment_id": "E12",
            "primary_change": "DAILY_LAG_144_FEATURE",
            "control_metrics": metrics[CONTROL_ID],
            "challenger_metrics": metrics[CHALLENGER_ID],
            "control_fold_rmse_wh": folds[CONTROL_ID],
            "challenger_fold_rmse_wh": folds[CHALLENGER_ID],
            "control_worst_fold_rmse_wh": max(folds[CONTROL_ID]),
            "challenger_worst_fold_rmse_wh": max(folds[CHALLENGER_ID]),
            "control_fold_rmse_std_wh": statistics.pstdev(folds[CONTROL_ID]),
            "challenger_fold_rmse_std_wh": statistics.pstdev(folds[CHALLENGER_ID]),
            "delta_rmse_challenger_minus_control": float(metrics[CHALLENGER_ID]["rmse_wh"]) - float(metrics[CONTROL_ID]["rmse_wh"]),
            "promotion_evaluation": promotion,
            "decision": "HUMAN_REVIEW_REQUIRED",
            "test_status": "NOT_ACCESSED",
        },
        "e12_feature_contract.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_FEATURE_CONTRACT-v1",
            "base_variant": CONTROL_VARIANT,
            "challenger_variant": CHALLENGER_VARIANT,
            "base_feature_order": list(get_feature_list(CONTROL_VARIANT)),
            "appended_feature_order": [LAG_FEATURE],
            "challenger_feature_order": list(challenger_feature_order()),
            "feature_fingerprint": document["challenger"]["feature_fingerprint"],
            "definition": LAG_DEFINITION,
            "lag_steps": LAG_STEPS,
            "imputation": "NONE",
            "status": "PASS",
        },
        "e12_causal_audit.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_CAUSAL_AUDIT-v1",
            **causal.to_dict(),
            "status": "PASS",
        },
        "e12_lost_target_accounting.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_LOST_TARGET_ACCOUNTING-v1",
            "original_phase44_target_count": causal.original_phase44_target_count,
            "common_target_count": causal.common_target_count,
            "lost_target_count": causal.lost_target_count,
            "lost_target_ids": list(causal.lost_target_ids),
            "lost_target_fingerprint": causal.lost_target_fingerprint,
            "first_eligible_target_id": causal.first_eligible_target_id,
            "reason": "L72 first context row requires y[s-144]",
            "status": "PASS",
        },
        "e12_common_population_audit.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_POPULATION_AUDIT-v1",
            "common_target_count": causal.common_target_count,
            "common_population_fingerprint": causal.common_population_fingerprint,
            "control_challenger_ordered_ids_identical": True,
            "folds": [fold.as_dict() for fold in evidence.folds],
            "test_rows_read": 0,
            "test_target_ids_seen": 0,
            "status": "PASS",
        },
        "e12_scaler_lineage_audit.json": {
            "schema": "MODEL_IMPROVEMENT_V2_E12_SCALER_AUDIT-v1",
            "fold_local_x_scaling": True,
            "fold_local_y_scaling": True,
            "audit_rows": scaler_rows,
            "all_passed": all(
                row["status"] == "PASS"
                and row["bundle_checksum_verified"]
                and row["test_rows_used"] == 0
                and row["outer_eval_rows_used"] == 0
                for row in scaler_rows
            ),
            "test_status": "NOT_ACCESSED",
        },
    }
    if recovery_audit is not None:
        documents["e12_partial_recovery_contract.json"] = {
            "schema": "MODEL_IMPROVEMENT_V2_E12_RECOVERY-v1",
            **dict(recovery_audit),
        }
    for name, payload in documents.items():
        atomic_write_bytes(output / name, canonical_json_bytes(payload))


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e12_config(resolved)
    preflight = validate_e12_document(document, resolved)
    context = build_e12_run_context(resolved, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code:
        print(result.summary, file=sys.stderr)
        return result.exit_code
    _write_success(resolved, document, result, preflight)
    print(result.summary)
    return 0


def run_resume_partial(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e12_config(resolved)
    preflight = validate_e12_document(document, resolved)
    context, audit = build_e12_resume_context(resolved, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code:
        print(result.summary, file=sys.stderr)
        return result.exit_code
    _write_success(resolved, document, result, preflight, audit)
    print(result.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2 E12 daily lag-144 feature ablation")
    parser.add_argument("--experiment", choices=["E12"], required=True)
    parser.add_argument(
        "--mode", choices=["preflight", "official", "resume-partial"], default="preflight"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        print("ERROR: E12 screening seed must be 42", file=sys.stderr)
        return 2
    if args.mode == "official" and not args.authorize_training:
        print("REFUSED: E12 official mode requires --authorize-training.", file=sys.stderr)
        return 3
    if args.mode == "resume-partial" and not args.authorize_training:
        print(
            "REFUSED: E12 resume-partial may train missing Stage A/B runs and "
            "requires --authorize-training.",
            file=sys.stderr,
        )
        return 3
    if args.mode == "preflight" and args.authorize_training:
        print("ERROR: --authorize-training is valid only in official mode.", file=sys.stderr)
        return 2
    try:
        if args.mode == "official":
            return run_official()
        if args.mode == "resume-partial":
            return run_resume_partial()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (E12PreflightError, PermissionError, ValueError) as exc:
        print(f"E12 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
