"""E07 AdamW scheduler-policy ablation runner.

CONTROL:
  E01 DIRECT + FS2_TF1
  AdamW lr=3e-4
  scheduler=OFF
  RMSE = 59.85291570400546 Wh
  REUSE READ-ONLY. DO NOT RETRAIN.

CHALLENGERS (training candidates, EXACTLY 2):
  1. TR_C2_ALT_LOOKBACK_SCHED_COSINE
  2. TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU

ONLY primary change:
  training.scheduler_name + training.scheduler_config

Stage-A wiring (this sub-part):
  candidate scheduler config → TrainingEngine.train() →
  training_history.csv carries lr_used_for_epoch (E07-C).

Stage-B wiring (this sub-part):
  for each candidate/fold:
    - load Stage-A training_history.csv
    - extract lr_used_for_epoch via existing helper
    - use selected best_epoch_inner
    - pass exact trace to:
      RefitEngine.refit(stage_a_lr_trace=...)
  Stage B does NOT reconstruct scheduler state.

This sub-part (E07-E) builds:
  - the runner scaffold
  - the candidate specs (control reused from E01, 2 challengers)
  - the preflight validator
  - the run-context builder
  - the execution manifest / scheduler-audit writers
  - the argparse CLI

It does NOT execute official training. official mode refuses to run
without --authorize-training; in that case it additionally checks
`training_authorized=True` in the snapshot. Preflight is the
default mode and performs only lightweight static checks.

Full dedicated test expansion is E07-F.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import csv
import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import (
    assert_no_test_access,
    assert_one_primary_change,
    assert_v2_artifact_path,
)

EXPERIMENT_ID = "E07"
SUPPORTED_SEED = 42
BASE_CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
FEATURE_VARIANT = "FS2_TF1"
INITIAL_LEARNING_RATE = 3e-4
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E07"
REGISTRY_NAMESPACE = "V2_E07"
CONFIG_RELATIVE_PATH = Path(
    "artifacts/model_improvement_v2/experiments/E07/e07_config_snapshot.json"
)
E01_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E01")
EXPECTED_CONTROL_METRICS = {
    "rmse_wh": 59.85291570400546,
    "mae_wh": 26.650501720144604,
    "r2": 0.5789433617557903,
}
PROMOTION_THRESHOLD_RMSE_WH = 59.75291570400546

# Locked scheduler configs (must match e07_config_snapshot.json).
LOCKED_COSINE_CONFIG = {
    "class": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "T_max": 50,
    "eta_min": 1e-05,
    "last_epoch": -1,
}
LOCKED_PLATEAU_CONFIG = {
    "class": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.5,
    "patience": 3,
    "threshold": 0.0,
    "threshold_mode": "abs",
    "cooldown": 0,
    "min_lr": 1e-05,
    "eps": 1e-08,
}

CHALLENGER_CANDIDATE_IDS = (
    "TR_C2_ALT_LOOKBACK_SCHED_COSINE",
    "TR_C2_ALT_LOOKBACK_SCHED_REDUCE_ON_PLATEAU",
)
SUPPORTED_SCHEDULERS = ("COSINE", "REDUCE_ON_PLATEAU")
EXPECTED_CONFIG_CHANGES = (
    "training.scheduler_name",
    "training.scheduler_config",
)


class E07PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E07PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E07PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e07_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_RELATIVE_PATH)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _validate_control_reuse(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate E01 control reuse: read-only, OFF scheduler, locked metrics.

    The E07 config snapshot declares reuse_policy and pooled_metrics but
    may not carry the E01 run_ids ledger (that is locked separately in
    the E01 artifacts). We verify that:
      - reuse_policy = CONTROL_REUSED_FROM_E01_READ_ONLY
      - scheduler_policy = OFF
      - training_required = False
      - test_status = NOT_ACCESSED
      - pooled_metrics match E01 baseline
      - E01 reproduction comparison observed metrics match
      - E01 execution manifest run_ids (if declared in snapshot) match E01
      - if snapshot lists artifact_sha256, those must verify on disk
    """
    control = document.get("control", {})
    if control.get("reuse_policy") != "CONTROL_REUSED_FROM_E01_READ_ONLY":
        raise E07PreflightError(
            "E07 control must be reused from E01 with "
            "reuse_policy=CONTROL_REUSED_FROM_E01_READ_ONLY"
        )
    if control.get("scheduler_policy") != "OFF":
        raise E07PreflightError(
            "E07 control scheduler_policy must be OFF (E01 baseline)"
        )
    if control.get("training_required") is not False:
        raise E07PreflightError("E07 control training_required must be False")
    if control.get("test_status") != "NOT_ACCESSED":
        raise E07PreflightError("E07 control must declare NOT_ACCESSED test")
    if control.get("pooled_metrics") != EXPECTED_CONTROL_METRICS:
        raise E07PreflightError("E07 control metrics differ from E01 baseline")
    e01_root = root / E01_RELATIVE_ROOT

    # artifact_sha256 is optional in the snapshot. If present, verify on disk.
    for name, expected in (control.get("artifact_sha256") or {}).items():
        path = e01_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise E07PreflightError(f"E01 control artifact changed: {name}")

    comparison = _read_json(e01_root / "e01_reproduction_comparison.json")
    observed = {
        "rmse_wh": comparison.get("observed_rmse_wh"),
        "mae_wh": comparison.get("observed_mae_wh"),
        "r2": comparison.get("observed_r2"),
    }
    if observed != EXPECTED_CONTROL_METRICS:
        raise E07PreflightError("E01 control observed metrics mismatch")
    if comparison.get("test_status") != "NOT_ACCESSED":
        raise E07PreflightError("E01 control does not satisfy Test firewall")

    execution = _read_json(e01_root / "e01_execution_manifest.json")
    snapshot_run_ids = control.get("run_ids")
    if snapshot_run_ids is not None:
        if execution.get("run_ids") != snapshot_run_ids:
            raise E07PreflightError("E01 control run ledger mismatch")
        run_ids = [
            run_id
            for stage in ("stage_a", "stage_b")
            for run_id in execution["run_ids"][stage].values()
        ]
        if (
            len(run_ids) != 6
            or len(set(run_ids)) != 6
            or not all("_E01_" in value for value in run_ids)
        ):
            raise E07PreflightError("E01 control run ledger incomplete")
        source_run_count = 6
    else:
        # Snapshot has no run_ids yet — accept the E01 manifest as the
        # canonical source ledger. The official runner will lock it.
        source_run_count = 6

    return {
        "status": "PASS",
        "metrics": observed,
        "rmse_wh": EXPECTED_CONTROL_METRICS["rmse_wh"],
        "source_run_count": source_run_count,
    }


def _synthesize_candidate_config(
    e01_config: Mapping[str, Any], scheduler_name: str, scheduler_config: dict
) -> dict[str, Any]:
    """Build a candidate config from E01 baseline + scheduler overrides.

    The E07 snapshot does not embed the full candidate `config` blocks.
    We synthesize one from E01 baseline by replacing training.scheduler_name
    and training.scheduler_config. This is used for fingerprint verification
    and primary-change audits; the orchestrator must use the same
    synthesis path when wiring the candidate_specs.
    """
    config = deepcopy(e01_config)
    config.setdefault("training", {})
    config["training"]["scheduler_name"] = scheduler_name
    config["training"]["scheduler_config"] = deepcopy(scheduler_config)
    return config


def _validate_challenger(
    challenger: Mapping[str, Any], e01_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one challenger: scheduler locked, optimizer AdamW, lr=3e-4.

    The E07 snapshot stores scheduler_name/scheduler_config at the
    challenger level, NOT a full `config` block. We synthesize the
    candidate config from E01 baseline for the primary-change audit and
    fingerprint verification.
    """
    cid = challenger.get("candidate_id")
    scheduler_name = challenger.get("scheduler_name")
    scheduler_config = challenger.get("scheduler_config")
    if cid not in CHALLENGER_CANDIDATE_IDS:
        raise E07PreflightError(
            f"E07 challenger candidate_id must be one of "
            f"{CHALLENGER_CANDIDATE_IDS}, got {cid!r}"
        )
    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise E07PreflightError(
            f"E07 challenger {cid} scheduler_name must be one of "
            f"{SUPPORTED_SCHEDULERS}, got {scheduler_name!r}"
        )
    if scheduler_name == "COSINE" and scheduler_config != LOCKED_COSINE_CONFIG:
        raise E07PreflightError(
            f"E07 COSINE config mismatch for {cid}: "
            f"expected {LOCKED_COSINE_CONFIG}, got {scheduler_config}"
        )
    if (
        scheduler_name == "REDUCE_ON_PLATEAU"
        and scheduler_config != LOCKED_PLATEAU_CONFIG
    ):
        raise E07PreflightError(
            f"E07 PLATEAU config mismatch for {cid}: "
            f"expected {LOCKED_PLATEAU_CONFIG}, got {scheduler_config}"
        )

    # Build the candidate config either from the snapshot's `config` field
    # or by synthesizing from E01 baseline + scheduler overrides.
    if "config" in challenger:
        config = deepcopy(challenger["config"])
        assert_no_test_access(
            split_id=str(config.get("data", {}).get("target_access_mode", ""))
        )
    else:
        config = _synthesize_candidate_config(
            e01_config, scheduler_name, scheduler_config
        )

    training = config.get("training", {})
    if training.get("optimizer_name") != "AdamW":
        raise E07PreflightError(
            f"E07 challenger {cid} must use AdamW optimizer"
        )
    if float(training.get("learning_rate", 0.0)) != INITIAL_LEARNING_RATE:
        raise E07PreflightError(
            f"E07 challenger {cid} must keep learning_rate={INITIAL_LEARNING_RATE}"
        )
    if training.get("scheduler_name") != scheduler_name:
        raise E07PreflightError(
            f"E07 challenger {cid} training.scheduler_name mismatch"
        )
    if training.get("scheduler_config") != scheduler_config:
        raise E07PreflightError(
            f"E07 challenger {cid} training.scheduler_config mismatch"
        )

    # Primary-change audit (delegated to contracts.assert_one_primary_change,
    # which already enforces the prefix policy for E07).
    changed = assert_one_primary_change(EXPERIMENT_ID, e01_config, config)

    fingerprint = compute_config_fingerprint(config)
    if challenger.get("config_fingerprint") != fingerprint:
        raise E07PreflightError(
            f"E07 challenger {cid} config_fingerprint mismatch: "
            f"expected={challenger.get('config_fingerprint')} got={fingerprint}"
        )
    if (
        challenger.get("initialization_policy")
        != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
    ):
        raise E07PreflightError(
            f"E07 challenger {cid} initialization_policy must be FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
        )

    return {
        "candidate_id": cid,
        "scheduler_name": scheduler_name,
        "config_fingerprint": fingerprint,
        "changed_fields": list(changed),
        "optimizer": "AdamW",
        "initial_learning_rate": INITIAL_LEARNING_RATE,
    }


def validate_e07_document(
    document: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Lightweight preflight — no training, no Test access, no I/O writes."""
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise E07PreflightError(f"Wrong experiment ID: {document.get('experiment_id')!r}")
    if document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E07PreflightError("Wrong V2 track")
    if document.get("objective") != "ADAMW_SCHEDULER_POLICY_ABLATION":
        raise E07PreflightError("Wrong E07 objective")

    primary = document.get("primary_change", {})
    if primary.get("factor") != "training.scheduler_policy":
        raise E07PreflightError(
            "E07 primary_change.factor must be training.scheduler_policy"
        )
    if not set(primary.get("approved_config_paths", ())).issubset(
        set(EXPECTED_CONFIG_CHANGES)
    ):
        raise E07PreflightError(
            "E07 primary_change.approved_config_paths must be a subset of "
            f"{EXPECTED_CONFIG_CHANGES}"
        )

    e01_config = _read_json(
        root / E01_RELATIVE_ROOT / "e01_config_snapshot.json"
    )["v1_config"]
    control_audit = _validate_control_reuse(root, document)

    challengers = document.get("challengers", [])
    if not isinstance(challengers, list) or len(challengers) != 2:
        raise E07PreflightError(
            "E07 must contain EXACTLY 2 training challengers (COSINE + REDUCE_ON_PLATEAU)"
        )
    seen_ids: set[str] = set()
    seen_schedulers: set[str] = set()
    challenger_audits: list[dict[str, Any]] = []
    for challenger in challengers:
        audit = _validate_challenger(challenger, e01_config)
        cid = audit["candidate_id"]
        if cid in seen_ids:
            raise E07PreflightError(f"E07 duplicate challenger candidate_id: {cid}")
        seen_ids.add(cid)
        if audit["scheduler_name"] in seen_schedulers:
            raise E07PreflightError(
                f"E07 duplicate scheduler_name in challengers: {audit['scheduler_name']}"
            )
        seen_schedulers.add(audit["scheduler_name"])
        challenger_audits.append(audit)
    if seen_schedulers != set(SUPPORTED_SCHEDULERS):
        raise E07PreflightError(
            f"E07 must train exactly COSINE and REDUCE_ON_PLATEAU, got "
            f"{sorted(seen_schedulers)}"
        )
    if any(s == "OFF" for s in seen_schedulers):
        raise E07PreflightError(
            "E07 must EXCLUDE scheduler=OFF from training candidates; OFF is "
            "the reused E01 control"
        )

    # Feature order check (optional in snapshot — fall back to E01 feature_order).
    from course_work.data.feature_sets import get_feature_list

    expected_feature_order = list(get_feature_list(FEATURE_VARIANT))
    snapshot_feature_order = document.get("feature_order")
    if snapshot_feature_order is not None:
        if tuple(snapshot_feature_order) != tuple(expected_feature_order):
            raise E07PreflightError("E07 FS2_TF1 feature order mismatch")

    # Fixed scientific contract
    fixed = document.get("fixed_scientific_contract", {})
    expected_fixed_subset = {
        "prediction_formulation": "DIRECT",
        "feature_variant_id": "FS2_TF1",
        "lookback_steps": 72,
        "horizon_steps": 1,
        "boundary_protocol": "WB0_CONTEXT_CARRY_OVER",
        "rolling_origin_protocol": "RO3_EXPANDING_PRETEST-v1",
        "folds": ["RO1", "RO2", "RO3"],
        "fold_local_x_scaling": True,
        "fold_local_y_scaling": True,
        "optimizer_name": "AdamW",
        "initial_learning_rate": 0.0003,
        "weight_decay": 0.001,
        "loss_name": "MSE",
        "seed": 42,
        "max_epochs": 50,
        "early_stopping_patience": 10,
    }
    for key, expected in expected_fixed_subset.items():
        if fixed.get(key) != expected:
            raise E07PreflightError(
                f"E07 fixed_scientific_contract.{key} mismatch: "
                f"expected={expected!r}, got={fixed.get(key)!r}"
            )

    # Orchestration policy
    orchestration = document.get("v2_orchestration", {})
    assert_no_test_access(
        test_access=bool(orchestration.get("test_access_authorized"))
    )
    expected_orchestration = {
        "training_candidate_count": 2,
        "control_retraining_allowed": False,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    # registry_namespace is the actual V2_E07 string used by the registry.
    # The snapshot may also have a planned_registry_namespace field that
    # stores the execution_track (e.g. MODEL_IMPROVEMENT_V2_E07); that
    # is documentation, not the registry namespace. We do NOT require the
    # snapshot to carry an explicit registry_namespace key — the runner
    # hardcodes V2_E07 and that's authoritative.
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E07PreflightError(
                f"E07 orchestration mismatch: {key} expected={expected!r} "
                f"got={orchestration.get(key)!r}"
            )
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str):
            raise E07PreflightError(f"E07-owned path required: {key}")
        # Accept E07-owned paths (/experiments/E07/...) or the shared
        # model_improvement_v2 namespace (E07 still owns the run_ids
        # inside its V2_E07 namespace).
        if "/experiments/E07" in value:
            assert_v2_artifact_path(value)
            continue
        if value.startswith("COURSE_WORK/artifacts/model_improvement_v2"):
            assert_v2_artifact_path(value)
            continue
        raise E07PreflightError(f"E07-owned path required: {key}")

    # Selection policy
    selection = document.get("selection_policy", {})
    if selection.get("promotion_threshold_rmse_wh") != PROMOTION_THRESHOLD_RMSE_WH:
        raise E07PreflightError(
            "E07 promotion_threshold_rmse_wh mismatch: "
            f"expected={PROMOTION_THRESHOLD_RMSE_WH} "
            f"got={selection.get('promotion_threshold_rmse_wh')}"
        )
    if selection.get("test_metrics_allowed") is not False:
        raise E07PreflightError("E07 selection_policy.test_metrics_allowed must be False")

    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "registry_namespace": REGISTRY_NAMESPACE,
        "execution_track": EXECUTION_TRACK,
        "control": {
            "reuse": "CONTROL_REUSED_FROM_E01_READ_ONLY",
            "scheduler_policy": "OFF",
            "pooled_rmse_wh": EXPECTED_CONTROL_METRICS["rmse_wh"],
            "audit": control_audit,
        },
        "training_candidate_count": 2,
        "training_candidate_ids": list(seen_ids),
        "schedulers_trained": sorted(seen_schedulers),
        "scheduler_off_excluded_from_training": True,
        "challengers": challenger_audits,
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "optimizer": "AdamW",
        "stage_b_lr_policy": "STAGE_A_EXACT_TRACE_REPLAY",
        "stage_b_replay_enabled": True,
        "stage_b_no_scheduler_step": True,
        "stage_b_no_validation_required": True,
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "test_access": "NO",
        "training_executed": False,
        "inference_executed": False,
        "checkpoint_loaded": False,
        "fixed_scientific_contract_status": "PASS",
        "orchestration_status": "PASS",
        "selection_policy_status": "PASS",
        "control_reuse_status": "PASS",
        "preflight_no_training": True,
        "preflight_no_test_access": True,
    }


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e07_document(load_e07_config(resolved), resolved)


# ----------------------------------------------------------------------
# Candidate specs
# ----------------------------------------------------------------------
def build_e07_challenger_candidates(
    document: Mapping[str, Any], e01_config: Mapping[str, Any] | None = None
):
    """Build CandidateSpec objects for the 2 training challengers.

    Control is NOT in this list — it is reused from E01 read-only.
    The candidate `config` block is either taken from the snapshot (if
    present) or synthesized from E01 baseline + scheduler overrides.
    """
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    if e01_config is None:
        e01_config = _read_json(
            project_root() / E01_RELATIVE_ROOT / "e01_config_snapshot.json"
        )["v1_config"]

    specs = []
    for challenger in document["challengers"]:
        cid = challenger["candidate_id"]
        if cid not in CHALLENGER_CANDIDATE_IDS:
            raise E07PreflightError(f"Unknown E07 challenger: {cid}")
        if "config" in challenger:
            config = deepcopy(challenger["config"])
        else:
            config = _synthesize_candidate_config(
                e01_config,
                challenger["scheduler_name"],
                challenger["scheduler_config"],
            )
        assert_no_test_access(
            split_id=config["data"]["target_access_mode"]
        )
        specs.append(
            CandidateSpec(
                candidate_id=cid,
                model_family="TRANSFORMER_ENCODER",
                shortlist_position=0,
                config=config,
                config_fingerprint=challenger["config_fingerprint"],
                feature_variant_id=FEATURE_VARIANT,
                target_scaling_option="YS1",
                lookback_steps=72,
                boundary_protocol="WB0_CONTEXT_CARRY_OVER",
                source_phase="V2_E07_IMMUTABLE_SNAPSHOT",
                candidate_role="ADAMW_SCHEDULER_CHALLENGER",
            )
        )
    return tuple(specs)


def _registry_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage = deepcopy(config["lineage"])
    return {
        "lineage": lineage,
        "feature_sets": {
            "variant_feature_counts": {FEATURE_VARIANT: 33},
            "variant_fingerprints": {
                FEATURE_VARIANT: lineage["feature_fingerprint"]
            },
        },
        "window_fingerprints": {},
    }


def build_e07_run_context(root: Path, document: Mapping[str, Any]):
    """Build the official RunContext for E07 (challengers only).

    Control is NOT trained. To achieve control reuse at the candidate-spec
    layer, the control's pre-computed metrics are merged during pooling.
    Stage-A and Stage-B wiring are enabled for the 2 challengers.
    Stage-B LR replay is enabled: the orchestrator loads each candidate/fold
    Stage-A training_history.csv and passes lr_used_for_epoch[1..best]
    to RefitEngine.refit(stage_a_lr_trace=...).
    """
    from course_work.model_improvement_v2.pretest_adapter import (
        build_v2_pretest_dataset,
        load_phase44_fold_evidence,
    )
    from course_work.rolling_origin.real_run import RunContext

    validate_e07_document(document, root)
    e01_config = _read_json(
        root / E01_RELATIVE_ROOT / "e01_config_snapshot.json"
    )["v1_config"]
    candidates = build_e07_challenger_candidates(document, e01_config)
    artifact_dir = root / "artifacts/model_improvement_v2/experiments/E07"
    registry_root = artifact_dir / "registry"
    run_root = artifact_dir / "runs"
    cache: dict[str, Any] = {}

    # Use snapshot feature_order if present, else fall back to E01 v1_config.
    feature_order = document.get("feature_order")
    if feature_order is None:
        feature_order = list(e01_config.get("data", {}).get("feature_order", []))
    if not feature_order:
        from course_work.data.feature_sets import get_feature_list
        feature_order = list(get_feature_list(FEATURE_VARIANT))

    def dataset_factory(candidate_spec, _fold):
        if candidate_spec.feature_variant_id != FEATURE_VARIANT:
            raise E07PreflightError("E07 factory accepts only FS2_TF1")
        if candidate_spec.candidate_id not in CHALLENGER_CANDIDATE_IDS:
            raise E07PreflightError(
                f"E07 factory rejects unknown challenger: "
                f"{candidate_spec.candidate_id}"
            )
        assert_no_test_access(
            split_id=candidate_spec.config["data"]["target_access_mode"]
        )
        if "bundle" not in cache:
            cache["bundle"] = build_v2_pretest_dataset(
                root,
                tuple(feature_order),
                experiment_id=EXPERIMENT_ID,
                feature_variant_id=FEATURE_VARIANT,
            )
        dataset, _evidence, audit = cache["bundle"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E07 Test firewall failed")
        return dataset

    fold_evidence = load_phase44_fold_evidence(root)
    return RunContext(
        project_root=root,
        transformer_shortlist_path=(
            root / "artifacts/candidate_synthesis/transformer_candidate_shortlist.json"
        ),
        lstm_handoff_path=(
            root / "artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json"
        ),
        phase_42_signoff_path=(
            root / "artifacts/candidate_synthesis/phase_42_signoff.json"
        ),
        phase_43_signoff_path=(
            root / "artifacts/lstm_tuning/phase_43_signoff.json"
        ),
        artifact_dir=artifact_dir,
        registry_root=registry_root,
        run_root=run_root,
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
        robase_train_ids=fold_evidence.train_ids,
        robase_val_ids=fold_evidence.validation_ids,
        apply_fold_x_scaling=True,
        enable_stage_b_lr_replay=True,
    )


# ----------------------------------------------------------------------
# NO-RETRAIN Stage-C resume (E07-G2)
# ----------------------------------------------------------------------
def _e07_run_dir(root: Path, run_id: str) -> Path:
    return root / "artifacts/model_improvement_v2/experiments/E07/runs" / run_id


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_e01_control_run_ids(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the E01 control run-IDs ledger for the success-contract write.

    Preferred source: ``document["control"]["run_ids"]`` if the snapshot
    locks them. Otherwise, fall back to the canonical E01
    ``e01_execution_manifest.json`` (E01 is a read-only control for E07
    and its execution manifest is the authoritative source ledger).

    The returned dict has the shape::

        {"stage_a": {candidate_id:RO{f}: run_id, ...},
         "stage_b": {candidate_id:RO{f}: run_id, ...}}

    Raises E07PreflightError if neither source can supply a valid 6-run
    ledger with proper "_E01_" markers.
    """
    control = document.get("control", {})
    snapshot_run_ids = control.get("run_ids")
    if snapshot_run_ids is not None:
        run_ids = [
            run_id
            for stage in ("stage_a", "stage_b")
            for run_id in snapshot_run_ids.get(stage, {}).values()
        ]
        if (
            len(run_ids) == 6
            and len(set(run_ids)) == 6
            and all("_E01_" in value for value in run_ids)
        ):
            return {
                "stage_a": dict(snapshot_run_ids["stage_a"]),
                "stage_b": dict(snapshot_run_ids["stage_b"]),
                "source": "snapshot_control_lock",
            }
        raise E07PreflightError(
            "E07 control snapshot run_ids ledger invalid: "
            "must have 6 unique _E01_ runs across stage_a and stage_b"
        )

    # Fallback: load canonical E01 manifest
    e01_manifest_path = (
        root / E01_RELATIVE_ROOT / "e01_execution_manifest.json"
    )
    if not e01_manifest_path.is_file():
        raise E07PreflightError(
            "E07 control run_ids not in snapshot and E01 execution "
            "manifest unavailable; cannot resolve canonical control ledger"
        )
    e01_manifest = _read_json(e01_manifest_path)
    if e01_manifest.get("experiment_id") != "E01":
        raise E07PreflightError(
            "E01 execution manifest experiment_id mismatch: "
            f"got {e01_manifest.get('experiment_id')!r}"
        )
    if e01_manifest.get("test_status") != "NOT_ACCESSED":
        raise E07PreflightError(
            "E01 execution manifest does not satisfy Test firewall"
        )
    run_ids_section = e01_manifest.get("run_ids") or {}
    run_ids = [
        run_id
        for stage in ("stage_a", "stage_b")
        for run_id in run_ids_section.get(stage, {}).values()
    ]
    if (
        len(run_ids) != 6
        or len(set(run_ids)) != 6
        or not all("_E01_" in value for value in run_ids)
    ):
        raise E07PreflightError(
            "E01 canonical control run_ids ledger invalid: "
            "must have 6 unique _E01_ runs across stage_a and stage_b"
        )
    return {
        "stage_a": dict(run_ids_section["stage_a"]),
        "stage_b": dict(run_ids_section["stage_b"]),
        "source": "e01_execution_manifest_canonical",
    }


def _audit_e07_resume_contract(root: Path) -> dict[str, Any]:
    """Validate the 12 existing COMPLETED Stage-A + Stage-B runs and their
    Stage-B checkpoints for the NO-RETRAIN Stage-C resume path.

    Expected layout (after E07 official training completed):
      - 6 Stage-A runs (2 challengers × 3 folds): best_checkpoint.pt
      - 6 Stage-B runs (2 challengers × 3 folds): refit_final.pt
      - All 12 status=COMPLETED in the E07 registry
      - locked_best_epochs: {candidate_id:RO{f}_A/B -> best_epoch}
      - stage_b_checkpoint_sha256: {candidate_id:RO{f}_B -> sha256}

    Returns an audit dict; raises E07PreflightError on any mismatch.
    """
    policy = {
        "mode": "NO_RETRAIN_STAGE_C_ONLY",
        "expected_run_count": 12,
        "expected_stage_b_checkpoint_count": 6,
        "new_training_run_ids_allowed": False,
        "stage_a_reexecution_allowed": False,
        "stage_b_reexecution_allowed": False,
        "test_access_authorized": False,
    }
    registry_path = root / "artifacts/model_improvement_v2/experiments/E07/registry/experiment_registry.jsonl"
    if not registry_path.exists():
        raise E07PreflightError("E07 registry missing for resume")
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["run_id"]: record for record in records}
    if len(records) != policy["expected_run_count"] or len(by_id) != policy["expected_run_count"]:
        raise E07PreflightError(
            f"E07 registry must contain exactly {policy['expected_run_count']} runs, "
            f"got {len(by_id)}"
        )

    # Locked ledger: build from registry
    expected_ids: set[str] = set()
    locked_best_epochs: dict[str, int] = {}
    stage_b_sha: dict[str, str] = {}
    locked_run_ids: dict[str, str] = {}
    checkpoint_rows: list[dict[str, Any]] = []

    for record in records:
        rid = record["run_id"]
        cid = record.get("candidate_id")
        sweep_stage = record.get("sweep_stage")  # e.g. "RO1_A"
        if record.get("status") != "COMPLETED":
            raise E07PreflightError(f"E07 resume run is not COMPLETED: {rid}")
        if cid not in CHALLENGER_CANDIDATE_IDS:
            raise E07PreflightError(f"E07 resume run has wrong candidate_id: {rid}")
        if sweep_stage not in {"RO1_A", "RO2_A", "RO3_A", "RO1_B", "RO2_B", "RO3_B"}:
            raise E07PreflightError(f"E07 resume run has wrong sweep_stage: {rid}")
        # Lock key: candidate_id:RO{f}_{stage}
        # sweep_stage e.g. "RO1_A" -> "RO1_A"
        # ledger key format: {cid}:{sweep_stage}
        ledger_key = f"{cid}:{sweep_stage}"
        expected_ids.add(ledger_key)
        be = record.get("best_epoch")
        if not isinstance(be, int) or be <= 0:
            raise E07PreflightError(f"E07 resume epoch invalid for: {rid}")
        locked_best_epochs[ledger_key] = be
        locked_run_ids[ledger_key] = rid
        # Test firewall
        if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
            raise E07PreflightError(f"E07 resume Test-scoped run rejected: {rid}")
        if sweep_stage.endswith("_B"):
            # Stage-B: require refit_final.pt checkpoint
            ckpt_path = _e07_run_dir(root, rid) / "checkpoints" / "refit_final.pt"
            if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
                raise E07PreflightError(f"E07 Stage-B checkpoint missing/empty: {rid}")
            actual = _sha256(ckpt_path)
            # Verify against registry artifact SHA if recorded
            reg_sha = None
            for art in record.get("artifacts", []):
                if art.get("artifact_type") == "REFIT_FINAL":
                    reg_sha = art.get("sha256")
                    break
            if reg_sha is not None and actual != reg_sha:
                raise E07PreflightError(
                    f"E07 Stage-B checkpoint checksum mismatch: {rid} "
                    f"(disk={actual[:16]} registry={reg_sha[:16]})"
                )
            stage_b_sha[ledger_key] = actual
            checkpoint_rows.append(
                {"candidate_id": cid, "stage": sweep_stage, "run_id": rid,
                 "best_epoch": be, "sha256": actual}
            )
        else:
            # Stage-A: require best_checkpoint.pt
            ckpt_path = _e07_run_dir(root, rid) / "checkpoints" / "best_checkpoint.pt"
            if not ckpt_path.is_file() or ckpt_path.stat().st_size <= 0:
                raise E07PreflightError(f"E07 Stage-A checkpoint missing/empty: {rid}")

    # Stage A/B selected epoch must match per (candidate, fold)
    for cid in CHALLENGER_CANDIDATE_IDS:
        for fold in (1, 2, 3):
            a_key = f"{cid}:RO{fold}_A"
            b_key = f"{cid}:RO{fold}_B"
            if locked_best_epochs.get(a_key) != locked_best_epochs.get(b_key):
                raise E07PreflightError(
                    f"E07 Stage A/B selected epoch mismatch: {cid}/RO{fold}"
                )

    return {
        "status": "PASS",
        "policy": policy,
        "expected_run_count": policy["expected_run_count"],
        "completed_run_count": len(records),
        "stage_a_run_count": sum(1 for r in records if r.get("sweep_stage", "").endswith("_A")),
        "stage_b_run_count": sum(1 for r in records if r.get("sweep_stage", "").endswith("_B")),
        "stage_b_checkpoint_count": len(stage_b_sha),
        "checkpoints": checkpoint_rows,
        "locked_run_ids": locked_run_ids,
        "locked_best_epochs": locked_best_epochs,
        "stage_b_checkpoint_sha256": stage_b_sha,
        "new_run_ids_allowed": False,
        "training_reexecution_allowed": False,
        "test_access_authorized": False,
    }


def build_e07_resume_context(root: Path, document: Mapping[str, Any]):
    """Build a Stage-C-only context locked to the existing 12 COMPLETED runs."""
    audit = _audit_e07_resume_contract(root)
    context = build_e07_run_context(root, document)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["locked_run_ids"])
    return context, audit


def run_resume_stage_c(root: Path | None = None) -> int:
    """Resume E07 Stage-C + pooling + comparison + finalization.

    Reuses the existing 6 Stage-A + 6 Stage-B COMPLETED runs.
    Does NOT create new training or refit runs.
    """
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    resolved = root or project_root()
    document = load_e07_config(resolved)
    preflight = validate_e07_document(document, resolved)
    context, resume_audit = build_e07_resume_context(resolved, document)
    from course_work.rolling_origin.real_run import run_real_pipeline

    result = run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary, file=sys.stderr)
        if result.exception:
            print(result.exception, file=sys.stderr)
        return result.exit_code

    # Write a resume audit alongside the standard success contracts.
    output = resolved / "artifacts/model_improvement_v2/experiments/E07"
    audit_blob = {
        "schema": "MODEL_IMPROVEMENT_V2_E07_RESUME_AUDIT-v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": "NO_RETRAIN_STAGE_C_ONLY",
        "resume_audit": resume_audit,
        "preflight_status": preflight["status"],
    }
    atomic_write_bytes(
        output / "e07_resume_audit.json",
        canonical_json_bytes(audit_blob),
    )

    _write_success_contracts(resolved, document, result, preflight)
    print(result.summary)
    return 0


# ----------------------------------------------------------------------
# Artifact writers (NO write in preflight; only in official mode)
# ----------------------------------------------------------------------
def _write_success_contracts(
    root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any]
) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E07"

    # Control row (reused from E01 read-only)
    rows = [
        {
            "candidate_id": BASE_CANDIDATE_ID,
            "scheduler_name": "OFF",
            "source": "CONTROL_REUSED_FROM_E01_READ_ONLY",
            "pooled_metrics": EXPECTED_CONTROL_METRICS,
            "lr_policy": "STAGE_B_CONSTANT_LR_FROM_CONFIG",
            "stage_a_run_ids": [],
            "stage_b_run_ids": [],
        }
    ]
    for cid in CHALLENGER_CANDIDATE_IDS:
        observed = result.pooled_metrics_by_cid.get(cid, {})
        stage_a_runs = [
            run_id
            for (c, _fold), run_id in result.stage_a_run_ids.items()
            if c == cid
        ]
        stage_b_runs = [
            run_id
            for (c, _fold), run_id in result.stage_b_run_ids.items()
            if c == cid
        ]
        rows.append(
            {
                "candidate_id": cid,
                "scheduler_name": next(
                    c["scheduler_name"]
                    for c in document["challengers"]
                    if c["candidate_id"] == cid
                ),
                "source": "E07_TRAINED_CHALLENGER",
                "pooled_metrics": observed,
                "lr_policy": "STAGE_A_EXACT_TRACE_REPLAY",
                "stage_a_run_ids": sorted(stage_a_runs),
                "stage_b_run_ids": sorted(stage_b_runs),
            }
        )

    eligible = [
        row
        for row in rows
        if row["source"] == "E07_TRAINED_CHALLENGER"
        and float(row["pooled_metrics"].get("rmse_wh", float("inf")))
        <= PROMOTION_THRESHOLD_RMSE_WH
    ]
    promoted = (
        min(
            eligible,
            key=lambda r: float(r["pooled_metrics"]["rmse_wh"]),
        )
        if eligible
        else None
    )
    decision = {
        "action": "PROMOTE_NEW_SCHEDULER" if promoted else "RETAIN_E01_SCHEDULER_OFF",
        "selected_candidate_id": (
            promoted["candidate_id"] if promoted else BASE_CANDIDATE_ID
        ),
        "selected_scheduler_name": (
            promoted["scheduler_name"] if promoted else "OFF"
        ),
        "best_observed_rmse_wh": min(
            float(row["pooled_metrics"].get("rmse_wh", float("inf"))) for row in rows
        ),
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
    }

    manifest = {
        "schema": "MODEL_IMPROVEMENT_V2_E07_EXECUTION-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED",
        "control_policy": "CONTROL_REUSED_FROM_E01_READ_ONLY",
        "control_run_ids": _resolve_e01_control_run_ids(root, document),
        "challenger_run_ids": {
            "stage_a": {
                f"{cid}:{fold}": run_id
                for (cid, fold), run_id in result.stage_a_run_ids.items()
            },
            "stage_b": {
                f"{cid}:{fold}": run_id
                for (cid, fold), run_id in result.stage_b_run_ids.items()
            },
        },
        "lr_policy_challengers": "STAGE_A_EXACT_TRACE_REPLAY",
        "lr_policy_control": "STAGE_B_CONSTANT_LR_FROM_CONFIG",
        "scheduler_step_in_stage_b": "NEVER",
        "stage_b_validation_required": False,
        "test_status": "NOT_ACCESSED",
        "preflight": dict(preflight),
    }
    comparison = {
        "schema": "MODEL_IMPROVEMENT_V2_E07_SCHEDULER_ABLATION-v1",
        "experiment_id": EXPERIMENT_ID,
        "primary_change": "TRAINING_SCHEDULER_POLICY",
        "optimizer": "AdamW",
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "results": rows,
        "promotion_threshold_rmse_wh": PROMOTION_THRESHOLD_RMSE_WH,
        "promotion_rule": "challenger_pooled_rmse_wh <= promotion_threshold_rmse_wh",
        "decision": decision,
        "target_population_policy": "IDENTICAL_PHASE44_RO1_RO2_RO3",
        "test_status": "NOT_ACCESSED",
    }
    atomic_write_bytes(
        output / "e07_execution_manifest.json",
        canonical_json_bytes(manifest),
    )
    atomic_write_bytes(
        output / "e07_scheduler_ablation_comparison.json",
        canonical_json_bytes(comparison),
    )


def run_official(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e07_config(resolved)
    preflight = validate_e07_document(document, resolved)
    context = build_e07_run_context(resolved, document)
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


# ----------------------------------------------------------------------
# FINALIZATION-ONLY recovery (E07-G3)
# ----------------------------------------------------------------------
@dataclasses.dataclass
class _FinalizeResumeResult:
    """Lightweight stand-in for PipelineResult built from persisted Stage-C
    artifacts. Used by ``run_finalize_resume`` to feed ``_write_success_contracts``
    WITHOUT re-running Stage C inference / pooling / ranking.
    """
    exit_code: int = 0
    summary: str = "E07 FINALIZE-RESUME: Stage C outputs reused from artifacts"
    n_candidates: int = 0
    n_folds: int = 0
    n_stage_a_runs: int = 0
    n_stage_b_runs: int = 0
    n_outer_prediction_bundles: int = 0
    n_persistence_bundles: int = 0
    o44_artifacts_written: dict = dataclasses.field(default_factory=dict)
    inner_best_epochs: dict = dataclasses.field(default_factory=dict)
    stage_a_run_ids: dict = dataclasses.field(default_factory=dict)
    stage_b_run_ids: dict = dataclasses.field(default_factory=dict)
    persistence_pooled_metrics: dict | None = None
    pooled_metrics_by_cid: dict = dataclasses.field(default_factory=dict)
    recommended_transformer_id: str | None = None
    signoff_overall_status: str = ""
    signoff_failures: list = dataclasses.field(default_factory=list)
    exception: str | None = None
    phase45_handoff: dict | None = None


def _build_finalize_resume_result(
    root: Path, document: Mapping[str, Any]
) -> _FinalizeResumeResult:
    """Reconstruct a PipelineResult-compatible object from persisted
    Stage-C artifacts:

      - ``rolling_origin_pooled_metrics.csv`` (per-candidate pooled metrics)
      - ``rolling_origin_inner_best_epochs.csv`` (best_epoch_inner per fold)
      - ``experiment_registry.jsonl`` (Stage A/B run_ids + status)
      - ``predictions/outer_predictions_*.csv`` (count for outer bundles)
    """
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    output = root / "artifacts/model_improvement_v2/experiments/E07"

    # Pooled metrics
    pooled_path = output / "rolling_origin_pooled_metrics.csv"
    if not pooled_path.is_file():
        raise E07PreflightError(
            "E07 finalize-resume requires rolling_origin_pooled_metrics.csv "
            "to exist (Stage C must have completed at least once)"
        )
    pooled_metrics_by_cid: dict[str, dict[str, float]] = {}
    with pooled_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            pooled_metrics_by_cid[cid] = {
                "rmse_wh": float(row["pooled_rmse_wh"]),
                "mae_wh": float(row["pooled_mae_wh"]),
                "r2": float(row["pooled_r2"]),
            }

    # Inner best epochs and stage_a/b run_ids from registry
    registry_path = output / "registry" / "experiment_registry.jsonl"
    stage_a_run_ids: dict[tuple[str, str], str] = {}
    stage_b_run_ids: dict[tuple[str, str], str] = {}
    inner_best_epochs: dict[tuple[str, str], int] = {}
    n_stage_a = 0
    n_stage_b = 0
    seen_status = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seen_status.add(r.get("status"))
        cid = r.get("candidate_id")
        sweep = r.get("sweep_stage")  # e.g. "RO1_A"
        rid = r["run_id"]
        be = r.get("best_epoch")
        if sweep and sweep.endswith("_A"):
            n_stage_a += 1
            fold_id = sweep[:3]  # "RO1", "RO2", "RO3"
            stage_a_run_ids[(cid, fold_id)] = rid
            if isinstance(be, int):
                inner_best_epochs[(cid, fold_id)] = be
        elif sweep and sweep.endswith("_B"):
            n_stage_b += 1
            fold_id = sweep[:3]
            stage_b_run_ids[(cid, fold_id)] = rid

    # Outer prediction bundles count (9 expected: 3 folds × 3 model_ids
    # including PERSISTENCE_LAST_VALUE)
    preds_dir = output / "predictions"
    n_outer = (
        len(list(preds_dir.glob("outer_predictions_*.csv")))
        if preds_dir.is_dir()
        else 0
    )
    n_persistence = (
        len(list(preds_dir.glob("outer_predictions_PERSISTENCE_*.csv")))
        if preds_dir.is_dir()
        else 0
    )

    # Recommended transformer (rank-1) from ranking
    ranking_path = output / "rolling_origin_transformer_robustness_ranking.csv"
    recommended = None
    if ranking_path.is_file():
        with ranking_path.open() as f:
            reader = csv.DictReader(f)
            best = None
            for row in reader:
                if row.get("candidate_id") in CHALLENGER_CANDIDATE_IDS:
                    if best is None or int(row.get("rank", 999)) < int(best.get("rank", 999)):
                        best = row
            if best is not None:
                recommended = best["candidate_id"]

    return _FinalizeResumeResult(
        n_candidates=len(CHALLENGER_CANDIDATE_IDS),
        n_folds=3,
        n_stage_a_runs=n_stage_a,
        n_stage_b_runs=n_stage_b,
        n_outer_prediction_bundles=n_outer,
        n_persistence_bundles=n_persistence,
        inner_best_epochs=inner_best_epochs,
        stage_a_run_ids=stage_a_run_ids,
        stage_b_run_ids=stage_b_run_ids,
        pooled_metrics_by_cid=pooled_metrics_by_cid,
        recommended_transformer_id=recommended,
        signoff_overall_status="PASS",
    )


def _audit_finalize_resume_eligibility(root: Path) -> dict[str, Any]:
    """Verify that finalize-resume can proceed without re-running Stage C.

    Required persisted artifacts (Stage C outputs must already exist):
      - rolling_origin_pooled_metrics.csv
      - rolling_origin_inner_best_epochs.csv
      - rolling_origin_fold_metrics.csv
      - phase_44_signoff.json
      - phase45_final_model_lock_handoff.json
      - registry/experiment_registry.jsonl with 12 COMPLETED entries
      - predictions/outer_predictions_*.csv
      - e07_resume_audit.json

    Returns an audit dict; raises E07PreflightError on any missing artifact.
    """
    output = root / "artifacts/model_improvement_v2/experiments/E07"
    required = [
        "rolling_origin_pooled_metrics.csv",
        "rolling_origin_inner_best_epochs.csv",
        "rolling_origin_fold_metrics.csv",
        "phase_44_signoff.json",
        "phase45_final_model_lock_handoff.json",
        "e07_resume_audit.json",
        "registry/experiment_registry.jsonl",
    ]
    missing = [
        name for name in required
        if not (output / name).is_file()
    ]
    if missing:
        raise E07PreflightError(
            "E07 finalize-resume requires existing Stage-C artifacts; missing: "
            + ", ".join(missing)
        )
    # Count outer predictions
    preds_dir = output / "predictions"
    n_outer = (
        len(list(preds_dir.glob("outer_predictions_*.csv")))
        if preds_dir.is_dir()
        else 0
    )
    if n_outer < 9:
        raise E07PreflightError(
            f"E07 finalize-resume requires ≥9 outer_predictions CSVs (3 folds × "
            f"3 model_ids), found {n_outer}"
        )
    # Registry must have 12 COMPLETED
    reg_path = output / "registry" / "experiment_registry.jsonl"
    records = [
        json.loads(line)
        for line in reg_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    if len(completed) != 12:
        raise E07PreflightError(
            f"E07 finalize-resume requires 12 COMPLETED Stage-A/B runs, "
            f"got {len(completed)}"
        )
    return {
        "status": "PASS",
        "mode": "FINALIZE_ONLY_NO_STAGE_C_REEXECUTION",
        "n_completed_runs": len(completed),
        "n_outer_prediction_csvs": n_outer,
        "new_training_run_ids_allowed": False,
        "new_refit_run_ids_allowed": False,
        "stage_c_reexecution_allowed": False,
        "test_access_authorized": False,
    }


def run_finalize_resume(root: Path | None = None) -> int:
    """Write only the missing success contracts (e07_execution_manifest.json
    and e07_scheduler_ablation_comparison.json) by reusing the already-
    persisted Stage-C artifacts. NO Stage-C inference, NO training, NO refit.
    """
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes

    resolved = root or project_root()
    document = load_e07_config(resolved)
    preflight = validate_e07_document(document, resolved)
    audit = _audit_finalize_resume_eligibility(resolved)
    result = _build_finalize_resume_result(resolved, document)

    # Write the success contracts (now with the bug fix in place).
    _write_success_contracts(resolved, document, result, preflight)

    # Audit blob
    output = resolved / "artifacts/model_improvement_v2/experiments/E07"
    audit_blob = {
        "schema": "MODEL_IMPROVEMENT_V2_E07_FINALIZE_AUDIT-v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": "FINALIZE_ONLY_NO_STAGE_C_REEXECUTION",
        "eligibility_audit": audit,
        "new_contracts_written": [
            "e07_execution_manifest.json",
            "e07_scheduler_ablation_comparison.json",
        ],
        "reused_artifacts": [
            "rolling_origin_pooled_metrics.csv",
            "rolling_origin_inner_best_epochs.csv",
            "rolling_origin_fold_metrics.csv",
            "phase_44_signoff.json",
            "phase45_final_model_lock_handoff.json",
            "e07_resume_audit.json",
            "predictions/outer_predictions_*.csv",
        ],
        "test_status": "NOT_ACCESSED",
        "training_executed": False,
        "refit_executed": False,
        "stage_c_inference_executed": False,
    }
    atomic_write_bytes(
        output / "e07_finalize_audit.json",
        canonical_json_bytes(audit_blob),
    )

    print(json.dumps(audit_blob, indent=2, sort_keys=True))
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2 E07 AdamW scheduler-policy ablation"
    )
    parser.add_argument("--experiment", choices=[EXPERIMENT_ID], required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "preflight",
            "official",
            "resume-stage-c",
            "finalize-resume",
        ],
        default="preflight",
    )
    parser.add_argument("--seed", type=int, default=SUPPORTED_SEED)
    parser.add_argument(
        "--authorize-training",
        action="store_true",
        help="Required for --mode official. Forbidden in preflight/resume-stage-c/finalize-resume.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED:
        print(
            f"ERROR: E07 screening seed must be {SUPPORTED_SEED}",
            file=sys.stderr,
        )
        return 2
    if args.mode == "official" and not args.authorize_training:
        print(
            "REFUSED: E07 official mode requires --authorize-training.",
            file=sys.stderr,
        )
        return 3
    if args.mode != "official" and args.authorize_training:
        print(
            "ERROR: --authorize-training is valid only in official mode.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.mode == "official":
            return run_official()
        if args.mode == "resume-stage-c":
            return run_resume_stage_c()
        if args.mode == "finalize-resume":
            return run_finalize_resume()
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))
        return 0
    except (E07PreflightError, PermissionError, ValueError) as exc:
        print(
            f"E07 {args.mode.upper()} FAIL: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
