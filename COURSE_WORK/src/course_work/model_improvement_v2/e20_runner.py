"""E20 matched three-seed finalist confirmation; Human-gated training.

Seed 42 is immutable cross-experiment evidence.  This runner can create new
scientific runs only for seeds 123 and 2026, in separate E20-owned ledgers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)
from course_work.model_improvement_v2.e14_runner import bundle_config as e14_bundle_config
from course_work.model_improvement_v2.runner import load_e01_config_snapshot

EXPERIMENT_ID = "E20"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E20"
REGISTRY_NAMESPACE = "V2_E20"
CONTROL_ID = "TR_C2_ALT_LOOKBACK"
FINALIST_ID = "TR_C2_ALT_LOOKBACK_E14_M1"
ALL_SEEDS = (42, 123, 2026)
NEW_SEEDS = (123, 2026)
FOLDS = ("RO1", "RO2", "RO3")
CONFIG_PATH = Path("artifacts/model_improvement_v2/experiments/E20/e20_config_snapshot.json")
E20_ROOT = Path("artifacts/model_improvement_v2/experiments/E20")
PLAN_PATH = Path("docs/plan/plan_before_process/model_improvement_v2_e20_pre_process_plan.md")
POPULATION_FINGERPRINT = "a40ded8802e90008535d268720bad9e9dcca5eee1436ddb359deea3ad39a1987"
BASE_FINGERPRINTS = {
    "CONTROL": "585c5e79e6a1c8c49efdd2f7767e6d3d4c628835acfda360a2fa66a2ef5c1e24",
    "FINALIST": "ff99dba3b57d38308f412a76c936b3a1f2a99ab8835e465417090ebfda49bc77",
}
SEEDED_FINGERPRINTS = {
    123: {
        "CONTROL": "bbd544add153421ffc45e13c6b5d25cdf250dfeb0ed836ea4b086c344dea35fe",
        "FINALIST": "0b4f1b61cb8400324df51e97e05050ec42b9f74a79a0aa650538c23a6ad6d775",
    },
    2026: {
        "CONTROL": "f0c3a5d53fa4124435901ae7cd073d964b8ab25dd3443aa0bce744ac1c325cd4",
        "FINALIST": "aa3d6a5366877f02ce0130020d4ec0bd39ed92677dad4561a7b502252c6e3369",
    },
}
SEED42_RUNS = {
    "CONTROL": {
        "RO1_A": "RUN_V2_TR_E01_RO1_A_0001_3B5C1B75",
        "RO2_A": "RUN_V2_TR_E01_RO2_A_0002_1EA6841D",
        "RO3_A": "RUN_V2_TR_E01_RO3_A_0003_4C4B28D5",
        "RO1_B": "RUN_V2_TR_E01_RO1_B_0004_533B5F33",
        "RO2_B": "RUN_V2_TR_E01_RO2_B_0005_19A72D17",
        "RO3_B": "RUN_V2_TR_E01_RO3_B_0006_AA5D66A6",
    },
    "FINALIST": {
        "RO1_A": "RUN_V2_TR_E14_RO1_A_0001_B5DD2C2E",
        "RO2_A": "RUN_V2_TR_E14_RO2_A_0006_EC2A3FCE",
        "RO3_A": "RUN_V2_TR_E14_RO3_A_0011_DFA1252E",
        "RO1_B": "RUN_V2_TR_E14_RO1_B_0016_668187E3",
        "RO2_B": "RUN_V2_TR_E14_RO2_B_0021_AD87B689",
        "RO3_B": "RUN_V2_TR_E14_RO3_B_0026_ABCFB6CD",
    },
}


class E20PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise E20PreflightError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E20PreflightError(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise E20PreflightError(f"Missing registry: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise E20PreflightError(f"Malformed registry: {path}")
    return {str(row["run_id"]): row for row in rows}


def load_e20_config(root: Path) -> dict[str, Any]:
    return _read(root / CONFIG_PATH)


def base_config(root: Path, role: str) -> dict[str, Any]:
    if role == "CONTROL":
        config = deepcopy(load_e01_config_snapshot(root)["v1_config"])
    elif role == "FINALIST":
        config = deepcopy(e14_bundle_config(root, "M1"))
    else:
        raise E20PreflightError(f"Unsupported E20 role: {role}")
    if compute_config_fingerprint(config) != BASE_FINGERPRINTS[role]:
        raise E20PreflightError(f"{role} base config fingerprint drift")
    return config


def seeded_config(root: Path, role: str, seed: int) -> dict[str, Any]:
    if seed not in ALL_SEEDS:
        raise E20PreflightError(f"Unregistered E20 seed: {seed}")
    config = base_config(root, role)
    for key in ("seed", "global_seed", "dataloader_seed"):
        config["reproducibility"][key] = seed
    if seed != 42:
        changed = set(assert_one_primary_change("E20", base_config(root, role), config))
        if changed != {
            "reproducibility.seed",
            "reproducibility.global_seed",
            "reproducibility.dataloader_seed",
        }:
            raise E20PreflightError(f"{role}/seed-{seed} changes more than seed policy")
        if compute_config_fingerprint(config) != SEEDED_FINGERPRINTS[seed][role]:
            raise E20PreflightError(f"{role}/seed-{seed} fingerprint drift")
    return config


def expected_new_run_identities() -> tuple[str, ...]:
    return tuple(
        f"seed_{seed}:{candidate_id}:RO{fold}_{stage}"
        for seed in NEW_SEEDS
        for candidate_id in (CONTROL_ID, FINALIST_ID)
        for fold in (1, 2, 3)
        for stage in "AB"
    )


def _decision_audit(root: Path) -> dict[str, Any]:
    e14 = _read(root / "artifacts/model_improvement_v2/experiments/E14/e14_human_decision.json")
    e15 = _read(root / "artifacts/model_improvement_v2/experiments/E15/e15_human_decision.json")
    e16 = _read(root / "artifacts/model_improvement_v2/experiments/E16/e16_human_decision.json")
    e18 = _read(root / "artifacts/model_improvement_v2/experiments/E18/e18_human_decision.json")
    if (
        e14.get("decision") != "PROMOTE_M1"
        or e14.get("accepted_incumbent") != FINALIST_ID
        or e14.get("accepted_config_fingerprint") != BASE_FINGERPRINTS["FINALIST"]
        or e15.get("decision") != "REJECT_E15"
        or e15.get("accepted_incumbent") != FINALIST_ID
        or e16.get("decision") != "REJECT_E16"
        or e16.get("accepted_incumbent") != FINALIST_ID
        or e16.get("e17_eligibility") != "DEFERRED"
        or e18.get("decision") != "ACCEPT_FINETUNING_ELIGIBILITY"
        or e18.get("active_incumbent") != FINALIST_ID
        or any(doc.get("test_status") != "NOT_ACCESSED" for doc in (e14, e15, e16, e18))
    ):
        raise E20PreflightError("Human finalist decision chain mismatch")
    return {"status": "PASS", "finalist": FINALIST_ID, "e19_status": "NOT_TRIGGERED"}


def _checkpoint_path(run_root: Path, record: Mapping[str, Any]) -> Path:
    name = "best_checkpoint.pt" if str(record["sweep_stage"]).endswith("_A") else "refit_final.pt"
    return run_root / str(record["run_id"]) / "checkpoints" / name


def _audit_source_group(
    root: Path,
    *,
    role: str,
    candidate_id: str,
    registry_path: Path,
    run_root: Path,
    prediction_root: Path,
    expected_outer_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    registry = _records(registry_path)
    source = SEED42_RUNS[role]
    checkpoint_hashes: dict[str, str] = {}
    for identity, run_id in source.items():
        record = registry.get(run_id)
        if record is None:
            raise E20PreflightError(f"Missing seed-42 source run: {run_id}")
        config = record.get("config", {})
        checkpoint = _checkpoint_path(run_root, record)
        config_path = run_root / run_id / "config.json"
        checkpoint_artifacts = [
            item for item in record.get("artifacts", [])
            if Path(str(item.get("artifact_path", ""))).name == checkpoint.name
        ]
        config_artifacts = [
            item for item in record.get("artifacts", [])
            if Path(str(item.get("artifact_path", ""))).name == "config.json"
        ]
        expected_stage = identity[-1]
        expected_fold = identity[:3]
        expected_config = base_config(root, role)
        expected_config["lineage"].update(
            rolling_origin_fold_id=expected_fold,
            rolling_origin_stage=expected_stage,
            rolling_origin_candidate_id=candidate_id,
        )
        if expected_stage == "A":
            expected_config["lineage"].update(
                rolling_origin_metric_population_fingerprint=config.get("lineage", {}).get(
                    "rolling_origin_metric_population_fingerprint"
                ),
                rolling_origin_metric_population_count=config.get("lineage", {}).get(
                    "rolling_origin_metric_population_count"
                ),
            )
        else:
            expected_config["training"].update(
                max_epochs=record.get("best_epoch"),
                early_stopping_enabled=False,
                early_stopping_patience=0,
            )
        if (
            record.get("status") != "COMPLETED"
            or record.get("candidate_id") != candidate_id
            or record.get("sweep_stage") != identity
            or config.get("reproducibility", {}).get("seed") != 42
            or config.get("reproducibility", {}).get("global_seed") != 42
            or config.get("reproducibility", {}).get("dataloader_seed") != 42
            or config.get("data", {}).get("feature_variant_id") != "FS2_TF1"
            or config.get("data", {}).get("feature_count") != 33
            or config.get("data", {}).get("target_access_mode") != "VALIDATION"
            or config.get("lineage", {}).get("population_fingerprint") != POPULATION_FINGERPRINT
            or config.get("lineage", {}).get("rolling_origin_fold_id") != expected_fold
            or config.get("lineage", {}).get("rolling_origin_stage") != expected_stage
            or record.get("config_fingerprint") != compute_config_fingerprint(config)
            or config != expected_config
            or not checkpoint.is_file()
            or len(checkpoint_artifacts) != 1
            or checkpoint_artifacts[0].get("sha256") != _sha(checkpoint)
            or not config_path.is_file()
            or len(config_artifacts) != 1
            or config_artifacts[0].get("sha256") != _sha(config_path)
        ):
            raise E20PreflightError(f"Seed-42 source lineage mismatch: {run_id}")
        checkpoint_hashes[identity] = _sha(checkpoint)
    for fold in FOLDS:
        path = prediction_root / f"outer_predictions_{candidate_id}_{fold}.csv"
        if not path.is_file():
            raise E20PreflightError(f"Missing seed-42 prediction evidence: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            observed = [str(row["target_id"]) for row in csv.DictReader(stream)]
        if observed != list(expected_outer_ids[fold]):
            raise E20PreflightError(f"Seed-42 ordered target population mismatch: {role}/{fold}")
    return {
        "status": "PASS",
        "run_count": len(source),
        "run_ids": dict(source),
        "checkpoint_sha256": checkpoint_hashes,
    }


def audit_seed42_reuse(root: Path) -> dict[str, Any]:
    from course_work.model_improvement_v2.pretest_adapter import load_phase44_fold_evidence

    evidence = load_phase44_fold_evidence(root)
    outer = {str(fold.fold_id): tuple(fold.outer_eval_ids) for fold in evidence.folds}
    control = _audit_source_group(
        root,
        role="CONTROL",
        candidate_id=CONTROL_ID,
        registry_path=root / "artifacts/model_improvement_v2/registry/experiment_registry.jsonl",
        run_root=root / "artifacts/model_improvement_v2/runs",
        prediction_root=root / "artifacts/model_improvement_v2/experiments/E01/predictions",
        expected_outer_ids=outer,
    )
    finalist = _audit_source_group(
        root,
        role="FINALIST",
        candidate_id=FINALIST_ID,
        registry_path=root / "artifacts/model_improvement_v2/experiments/E14/registry/experiment_registry.jsonl",
        run_root=root / "artifacts/model_improvement_v2/experiments/E14/runs",
        prediction_root=root / "artifacts/model_improvement_v2/experiments/E14/predictions",
        expected_outer_ids=outer,
    )
    e18 = _read(root / "artifacts/model_improvement_v2/experiments/E18/e18_human_decision.json")
    for fold in FOLDS:
        source = e18["source_checkpoints"][fold]
        if (
            source["run_id"] != SEED42_RUNS["CONTROL"][f"{fold}_B"]
            or source["sha256"] != control["checkpoint_sha256"][f"{fold}_B"]
        ):
            raise E20PreflightError(f"E18/E01 checkpoint lock mismatch: {fold}")
    e14 = _read(root / "artifacts/model_improvement_v2/experiments/E14/e14_human_decision.json")
    for fold in FOLDS:
        if (
            e14["stage_b_run_ids"][fold] != SEED42_RUNS["FINALIST"][f"{fold}_B"]
            or e14["stage_b_checkpoint_sha256"][fold]
            != finalist["checkpoint_sha256"][f"{fold}_B"]
        ):
            raise E20PreflightError(f"E14 finalist checkpoint lock mismatch: {fold}")
    return {
        "status": "PASS",
        "seed": 42,
        "reused_runs": 12,
        "retrain_allowed": False,
        "control": control,
        "finalist": finalist,
        "population_fingerprint": POPULATION_FINGERPRINT,
    }


def validate_e20_document(
    document: Mapping[str, Any], root: Path, *, materialize_data: bool = True
) -> dict[str, Any]:
    if (
        document.get("schema") != "MODEL_IMPROVEMENT_V2_E20_CONFIG-v1"
        or document.get("experiment_id") != EXPERIMENT_ID
        or document.get("objective") != "MATCHED_THREE_SEED_FINALIST_CONFIRMATION"
    ):
        raise E20PreflightError("Wrong E20 config identity")
    decision = _decision_audit(root)
    if _sha(root / PLAN_PATH) != document.get("source_plan_sha256"):
        raise E20PreflightError("E20 source plan checksum mismatch")
    if document.get("finalist_selection") != {
        "status": "PASS",
        "finalist": FINALIST_ID,
        "e19_status": "NOT_TRIGGERED",
    }:
        raise E20PreflightError("E20 finalist/E19 lock mismatch")
    protocol = document.get("protocol", {})
    if protocol != {
        "all_seeds": [42, 123, 2026],
        "new_seeds": [123, 2026],
        "folds": ["RO1", "RO2", "RO3"],
        "stages": ["A", "B"],
        "fold_local_x_scaling": True,
        "fold_local_y_scaling": True,
        "matched_data_order_policy": True,
        "best_seed_selection_forbidden": True,
        "hyperparameter_retuning_forbidden": True,
    }:
        raise E20PreflightError("E20 seed/fold protocol mismatch")
    matrix = document.get("locked_configurations", [])
    if {row.get("role") for row in matrix} != {"CONTROL", "FINALIST"} or len(matrix) != 2:
        raise E20PreflightError("E20 requires exactly control and finalist")
    computed: dict[str, dict[str, str]] = {}
    expected_ids = {"CONTROL": CONTROL_ID, "FINALIST": FINALIST_ID}
    for row in matrix:
        role = row["role"]
        if (
            row.get("candidate_id") != expected_ids[role]
            or row.get("base_config_fingerprint") != BASE_FINGERPRINTS[role]
            or row.get("seed_config_fingerprints")
            != {str(seed): SEEDED_FINGERPRINTS[seed][role] for seed in NEW_SEEDS}
        ):
            raise E20PreflightError(f"E20 {role} identity/fingerprint mismatch")
        computed[role] = {
            str(seed): compute_config_fingerprint(seeded_config(root, role, seed))
            for seed in NEW_SEEDS
        }
    orchestration = document.get("orchestration", {})
    if (
        orchestration.get("registry_namespace") != REGISTRY_NAMESPACE
        or orchestration.get("execution_track") != EXECUTION_TRACK
        or orchestration.get("expected_total_evidence") != 36
        or orchestration.get("seed42_reused_runs") != 12
        or orchestration.get("expected_new_training_runs") != 24
        or orchestration.get("seed42_retrain") is not False
        or orchestration.get("fresh_model_optimizer_each_new_run") is not True
        or orchestration.get("training_authorized") is not False
        or orchestration.get("test_access_authorized") is not False
    ):
        raise E20PreflightError("E20 orchestration mismatch")
    for value in orchestration.get("seed_roots", {}).values():
        for path in value.values():
            assert_v2_artifact_path(path)
    if document.get("promotion_policy") != {
        "best_seed_selection": False,
        "hyperparameter_retuning": False,
        "paired_seed_comparison": True,
        "weighted_composite": False,
        "automatic_promotion": False,
        "decision_authority": "HUMAN",
        "test_metrics_allowed": False,
    }:
        raise E20PreflightError("E20 promotion/governance policy mismatch")
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    assert_no_test_access(split_id=base_config(root, "CONTROL")["data"]["target_access_mode"])
    seed42 = audit_seed42_reuse(root)
    population = None
    if materialize_data:
        from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset

        dataset, evidence, audit = build_v2_pretest_dataset(
            root,
            tuple(get_feature_list("FS2_TF1")),
            experiment_id="E20",
            feature_variant_id="FS2_TF1",
        )
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise E20PreflightError("E20 Test firewall failed")
        population = {
            "status": "PASS",
            "population_fingerprint": POPULATION_FINGERPRINT,
            "target_count": len(dataset),
            "fold_fingerprints": {
                str(fold.fold_id): fold.fold_population_fingerprint for fold in evidence.folds
            },
        }
    return {
        "experiment": "E20",
        "status": "PASS",
        "accepted_incumbent": FINALIST_ID,
        "control": CONTROL_ID,
        "e19_status": decision["e19_status"],
        "all_seeds": list(ALL_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "expected_total_evidence": 36,
        "expected_new_training_runs": 24,
        "expected_new_run_identities": list(expected_new_run_identities()),
        "seed42_reuse": seed42,
        "candidate_config_fingerprints": computed,
        "population": population,
        "feature_variant": "FS2_TF1",
        "feature_count": 33,
        "scaler_lineage": "FOLD_LOCAL_TRAIN_ONLY",
        "fresh_model_optimizer_each_new_run": True,
        "test_rows_read": 0,
        "test_target_ids_seen": 0,
        "test_status": "NOT_ACCESSED",
        "training_executed": False,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return validate_e20_document(load_e20_config(root), root)


def build_candidates(root: Path, seed: int):
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    roles = (("CONTROL", CONTROL_ID, "MATCHED_E01_CONTROL"), ("FINALIST", FINALIST_ID, "WAVE2_FINALIST"))
    return tuple(
        CandidateSpec(
            candidate_id=candidate_id,
            model_family="TRANSFORMER_ENCODER",
            shortlist_position=index,
            config=seeded_config(root, role, seed),
            config_fingerprint=SEEDED_FINGERPRINTS[seed][role],
            feature_variant_id="FS2_TF1",
            target_scaling_option="YS1",
            lookback_steps=72,
            boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            source_phase="V2_E20_IMMUTABLE_SNAPSHOT",
            candidate_role=candidate_role,
        )
        for index, (role, candidate_id, candidate_role) in enumerate(roles, 1)
    )


def build_context(root: Path, document: Mapping[str, Any], seed: int):
    from course_work.model_improvement_v2.pretest_adapter import (
        build_v2_pretest_dataset,
        load_phase44_fold_evidence,
    )
    from course_work.rolling_origin.real_run import RunContext

    if seed not in NEW_SEEDS:
        raise E20PreflightError("E20 may create runs only for seeds 123 and 2026")
    validate_e20_document(document, root, materialize_data=False)
    candidates = build_candidates(root, seed)
    cache: dict[str, Any] = {}
    output = root / E20_ROOT / f"seed_{seed}"

    def factory(candidate, _fold):
        if candidate.candidate_id not in {CONTROL_ID, FINALIST_ID}:
            raise E20PreflightError("Unknown E20 candidate")
        if "data" not in cache:
            cache["data"] = build_v2_pretest_dataset(
                root,
                tuple(get_feature_list("FS2_TF1")),
                experiment_id="E20",
                feature_variant_id="FS2_TF1",
            )
        dataset, _evidence, audit = cache["data"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E20 Test firewall")
        return dataset

    evidence = load_phase44_fold_evidence(root)
    lineage = deepcopy(base_config(root, "CONTROL")["lineage"])
    upstream = {
        "lineage": lineage,
        "feature_sets": {
            "variant_feature_counts": {"FS2_TF1": 33},
            "variant_fingerprints": {"FS2_TF1": lineage["feature_fingerprint"]},
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
        seed=seed,
        device=str(base_config(root, "CONTROL")["runtime"]["device_type"]),
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


def audit_completed_seed_ledger(root: Path, seed: int, *, require_all: bool) -> dict[str, Any]:
    output = root / E20_ROOT / f"seed_{seed}"
    path = output / "registry/experiment_registry.jsonl"
    expected = {
        f"{candidate_id}:RO{fold}_{stage}"
        for candidate_id in (CONTROL_ID, FINALIST_ID)
        for fold in (1, 2, 3)
        for stage in "AB"
    }
    if not path.is_file():
        if require_all:
            raise E20PreflightError(f"Seed-{seed} completed ledger unavailable")
        return {"status": "PASS", "completed": {}, "missing": sorted(expected)}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    completed: dict[str, str] = {}
    for record in rows:
        if record.get("status") != "COMPLETED":
            continue
        key = f"{record.get('candidate_id')}:{record.get('sweep_stage')}"
        if key not in expected or key in completed:
            raise E20PreflightError(f"Seed-{seed} duplicate/unexpected completed run: {key}")
        role = "CONTROL" if record.get("candidate_id") == CONTROL_ID else "FINALIST"
        config = record.get("config", {})
        checkpoint = _checkpoint_path(output / "runs", record)
        config_path = output / "runs" / str(record["run_id"]) / "config.json"
        registered = [
            item for item in record.get("artifacts", [])
            if Path(str(item.get("artifact_path", ""))).name == checkpoint.name
        ]
        registered_config = [
            item for item in record.get("artifacts", [])
            if Path(str(item.get("artifact_path", ""))).name == "config.json"
        ]
        fold, stage = str(record["sweep_stage"]).split("_")
        expected_config = seeded_config(root, role, seed)
        expected_config["lineage"].update(
            rolling_origin_fold_id=fold,
            rolling_origin_stage=stage,
            rolling_origin_candidate_id=record["candidate_id"],
        )
        if stage == "A":
            expected_config["lineage"].update(
                rolling_origin_metric_population_fingerprint=config.get("lineage", {}).get(
                    "rolling_origin_metric_population_fingerprint"
                ),
                rolling_origin_metric_population_count=config.get("lineage", {}).get(
                    "rolling_origin_metric_population_count"
                ),
            )
        else:
            expected_config["training"].update(
                max_epochs=record.get("best_epoch"),
                early_stopping_enabled=False,
                early_stopping_patience=0,
            )
        if (
            config.get("reproducibility", {}).get("seed") != seed
            or config.get("data", {}).get("target_access_mode") != "VALIDATION"
            or compute_config_fingerprint(config) != record.get("config_fingerprint")
            or config != expected_config
            or not checkpoint.is_file()
            or len(registered) != 1
            or registered[0].get("sha256") != _sha(checkpoint)
            or not config_path.is_file()
            or len(registered_config) != 1
            or registered_config[0].get("sha256") != _sha(config_path)
            or role not in {"CONTROL", "FINALIST"}
        ):
            raise E20PreflightError(f"Seed-{seed} completed run audit failed: {key}")
        completed[key] = str(record["run_id"])
    missing = expected - set(completed)
    for key in set(completed):
        if key.endswith("_B"):
            parent_key = key[:-1] + "A"
            if parent_key not in completed:
                raise E20PreflightError(f"Seed-{seed} Stage B has no completed Stage-A parent: {key}")
            stage_b = next(row for row in rows if row.get("run_id") == completed[key])
            stage_a = next(row for row in rows if row.get("run_id") == completed[parent_key])
            if stage_b.get("best_epoch") != stage_a.get("best_epoch"):
                raise E20PreflightError(f"Seed-{seed} Stage-B exact-epoch lineage mismatch: {key}")
    if require_all and missing:
        raise E20PreflightError(f"Seed-{seed} canonical runs missing: {sorted(missing)}")
    return {"status": "PASS", "completed": completed, "missing": sorted(missing)}


def build_recovery_context(root: Path, document: Mapping[str, Any], seed: int, *, finalization_only: bool):
    context = build_context(root, document, seed)
    audit = audit_completed_seed_ledger(root, seed, require_all=finalization_only)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["completed"])
    context.allow_partial_stage_recovery = not finalization_only
    return context, audit


def _read_pooled(path: Path, candidate_id: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next((row for row in csv.DictReader(stream) if row["candidate_id"] == candidate_id), None)
    if row is None:
        raise E20PreflightError(f"Missing pooled metrics for {candidate_id}: {path}")
    return {
        "rmse_wh": float(row["pooled_rmse_wh"]),
        "mae_wh": float(row["pooled_mae_wh"]),
        "r2": float(row["pooled_r2"]),
    }


def _write_final_outputs(root: Path, results: Mapping[int, Any], preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    root_out = root / E20_ROOT
    metrics: dict[str, dict[str, dict[str, float]]] = {
        "42": {
            "CONTROL": _read_pooled(root / "artifacts/model_improvement_v2/experiments/E01/rolling_origin_pooled_metrics.csv", CONTROL_ID),
            "FINALIST": _read_pooled(root / "artifacts/model_improvement_v2/experiments/E14/rolling_origin_pooled_metrics.csv", FINALIST_ID),
        }
    }
    manifests: dict[str, Any] = {"42": {"reuse": deepcopy(SEED42_RUNS)}}
    for seed, result in results.items():
        metrics[str(seed)] = {
            "CONTROL": dict(result.pooled_metrics_by_cid[CONTROL_ID]),
            "FINALIST": dict(result.pooled_metrics_by_cid[FINALIST_ID]),
        }
        manifests[str(seed)] = {
            "stage_a": {f"{cid}:{fold}": rid for (cid, fold), rid in result.stage_a_run_ids.items()},
            "stage_b": {f"{cid}:{fold}": rid for (cid, fold), rid in result.stage_b_run_ids.items()},
        }
    summary: dict[str, Any] = {}
    for role in ("CONTROL", "FINALIST"):
        values = [metrics[str(seed)][role]["rmse_wh"] for seed in ALL_SEEDS]
        summary[role] = {"mean_rmse_wh": statistics.mean(values), "sd_rmse_wh": statistics.pstdev(values)}
    paired = {
        str(seed): metrics[str(seed)]["FINALIST"]["rmse_wh"] - metrics[str(seed)]["CONTROL"]["rmse_wh"]
        for seed in ALL_SEEDS
    }
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E20_MATCHED_SEED_COMPARISON-v1",
        "metrics_by_seed": metrics,
        "across_seed_summary": summary,
        "paired_finalist_minus_control_rmse_wh": paired,
        "best_seed_selection": "FORBIDDEN",
        "hyperparameter_retuning": "FORBIDDEN",
        "decision": "HUMAN_REVIEW_REQUIRED",
        "test_status": "NOT_ACCESSED",
    }
    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E20_EXECUTION-v1",
        "expected_total_evidence": 36,
        "seed42_reused_runs": 12,
        "new_runs": 24,
        "run_ids": manifests,
        "preflight": dict(preflight),
        "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(root_out / "e20_matched_seed_comparison.json", canonical_json_bytes(comparison))
    atomic_write_bytes(root_out / "e20_execution_manifest.json", canonical_json_bytes(manifest))


def _run(mode: str, root: Path | None = None) -> int:
    root = root or project_root()
    document = load_e20_config(root)
    preflight = validate_e20_document(document, root)
    from course_work.rolling_origin.real_run import run_real_pipeline

    if mode == "official":
        for seed in NEW_SEEDS:
            ledger = root / E20_ROOT / f"seed_{seed}/registry/experiment_registry.jsonl"
            if ledger.is_file() and ledger.read_text().strip():
                raise E20PreflightError(
                    f"Seed-{seed} ledger already exists; use resume-partial to avoid duplicates"
                )
    results: dict[int, Any] = {}
    for seed in NEW_SEEDS:
        if mode == "official":
            context = build_context(root, document, seed)
        else:
            context, _audit = build_recovery_context(
                root, document, seed, finalization_only=(mode == "finalize-only")
            )
        result = run_real_pipeline(context)
        if result.exit_code:
            print(result.summary, file=sys.stderr)
            return result.exit_code
        results[seed] = result
    _write_final_outputs(root, results, preflight)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MODEL_IMPROVEMENT-v2 E20 orchestrator")
    parser.add_argument("--experiment", required=True, choices=["E20"])
    parser.add_argument("--mode", choices=["preflight", "official", "resume-partial", "finalize-only"], default="preflight")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(NEW_SEEDS))
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--authorize-stage-c-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if tuple(args.seeds) != NEW_SEEDS:
        print("ERROR: E20 new-run seeds must be exactly 123 2026; seed 42 is reuse-only", file=sys.stderr)
        return 2
    if args.mode in {"official", "resume-partial"} and not args.authorize_training:
        print(f"REFUSED: E20 {args.mode} requires --authorize-training", file=sys.stderr)
        return 3
    if args.mode == "finalize-only" and not args.authorize_stage_c_resume:
        print("REFUSED: E20 finalize-only requires --authorize-stage-c-resume", file=sys.stderr)
        return 3
    if args.authorize_training and args.mode not in {"official", "resume-partial"}:
        return 2
    if args.authorize_stage_c_resume and args.mode != "finalize-only":
        return 2
    try:
        if args.mode == "preflight":
            print(json.dumps(run_preflight(), indent=2, sort_keys=True))
            return 0
        return _run(args.mode)
    except (E20PreflightError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"E20 {args.mode.upper()} FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
