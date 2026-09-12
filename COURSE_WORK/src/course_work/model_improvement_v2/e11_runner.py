"""V2 E11 past-only rolling-target feature ablation."""
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
from course_work.model_improvement_v2.e11_pretest import (
    CHALLENGER_VARIANT,
    CONTROL_VARIANT,
    ROLLING_DEFINITIONS,
    ROLLING_FEATURES,
    build_e11_pretest_datasets,
    challenger_feature_order,
)

EXPERIMENT_ID = "E11"
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E11"
REGISTRY_NAMESPACE = "V2_E11"
SEED = 42
CONTROL_ID = "TR_C2_ALT_LOOKBACK_E11_CONTROL"
CHALLENGER_ID = "TR_C2_ALT_LOOKBACK_E11_ROLL7"
CONFIG_PATH = Path("artifacts/model_improvement_v2/experiments/E11/e11_config_snapshot.json")
E10_COMPARISON = Path("artifacts/model_improvement_v2/experiments/E10/e10_target_delta_ablation_comparison.json")
EXPECTED_INCUMBENT = {"rmse_wh": 59.85291570400546, "mae_wh": 26.650501720144604, "r2": 0.5789433617557903}
EXPECTED_COMMON_FINGERPRINT = "a4b7c9479867884cd9e8e2f0a50f634735a31e4adb6f6fb9f90d944c34de4731"
SCALER_CONTRACT_CHECKSUM = "f319cc52689dbab56efc4711ea11b8485f7c4b533952b663f7748c455111e629"


class E11PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E11PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E11PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e11_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_PATH)


def _audit_e10_decision(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    decision = document.get("predecessor_human_decision", {})
    path = root / E10_COMPARISON
    if (
        decision.get("experiment_id") != "E10"
        or decision.get("decision") != "REJECTED"
        or decision.get("accepted_incumbent_after_decision") != "E01_DIRECT_FS2_TF1"
        or decision.get("carry_forward_delta_features") is not False
        or decision.get("decision_authority") != "HUMAN"
        or decision.get("scientific_evidence_mutation_allowed") is not False
    ):
        raise E11PreflightError("E10 Human rejection/feature carry-forward lock mismatch")
    if not path.is_file() or decision.get("comparison_sha256") != _sha256(path):
        raise E11PreflightError("Frozen E10 comparison artifact checksum mismatch")
    evidence = _read_json(path)
    gates = evidence.get("promotion_evaluation", {}).get("gates", {})
    if (
        evidence.get("test_status") != "NOT_ACCESSED"
        or evidence.get("promotion_evaluation", {}).get("eligible_for_human_promotion") is not False
        or gates.get("fold_rmse_std_guardrail") is not False
    ):
        raise E11PreflightError("E10 rejection evidence/Test status mismatch")
    gaps = decision.get("reporting_gaps_recorded", [])
    if not isinstance(gaps, list) or len(gaps) < 4:
        raise E11PreflightError("E10 reporting gaps must remain documented")
    return {"status": "PASS", "human_decision": "E10_REJECTED", "accepted_incumbent": "E01", "reporting_gaps_preserved": True, "e10_artifact_sha256": _sha256(path)}


def _expected_configs(root: Path) -> tuple[dict, dict]:
    base = deepcopy(_read_json(root / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json")["v1_config"])
    challenger = deepcopy(base)
    order = challenger_feature_order()
    challenger["data"]["feature_variant_id"] = CHALLENGER_VARIANT
    challenger["data"]["feature_count"] = len(order)
    challenger["model"]["input_size"] = len(order)
    challenger["lineage"]["feature_fingerprint"] = compute_feature_fingerprint(CHALLENGER_VARIANT, list(order))
    challenger["lineage"]["scaler_bundle_id"] = "FOLD_LOCAL__E11__FS2_TF1_ROLL7"
    challenger["lineage"]["scaler_bundle_checksum"] = SCALER_CONTRACT_CHECKSUM
    return base, challenger


def validate_e11_document(document: Mapping[str, Any], root: Path, *, materialize_data: bool = True) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E11PreflightError("Wrong E11 identity/track")
    incumbent = document.get("accepted_incumbent", {})
    if incumbent.get("source_experiment") != "E01" or incumbent.get("pooled_metrics") != EXPECTED_INCUMBENT or incumbent.get("test_status") != "NOT_ACCESSED":
        raise E11PreflightError("Accepted incumbent is not locked E01")
    predecessor = _audit_e10_decision(root, document)
    primary = document.get("primary_change", {})
    if (
        primary.get("factor") != "PAST_ONLY_ROLLING_TARGET_FEATURE_BLOCK"
        or primary.get("rolling_features") != list(ROLLING_FEATURES)
        or primary.get("definitions") != ROLLING_DEFINITIONS
        or primary.get("trailing_includes_current_observation") is not True
        or primary.get("centered") is not False
        or primary.get("ddof") != 0
        or primary.get("partial_windows") is not False
        or primary.get("imputation") != "NONE"
        or primary.get("past_target_only") is not True
    ):
        raise E11PreflightError("E11 rolling-feature causal contract mismatch")

    control, challenger = document.get("control", {}), document.get("challenger", {})
    expected_control, expected_challenger = _expected_configs(root)
    if control.get("candidate_id") != CONTROL_ID or challenger.get("candidate_id") != CHALLENGER_ID:
        raise E11PreflightError("E11 candidate identity mismatch")
    if control.get("config") != expected_control or control.get("training_required") is not True or control.get("reuse_e01_metrics") is not False:
        raise E11PreflightError("E11 control must retrain E01-equivalent config")
    if challenger.get("config") != expected_challenger or challenger.get("training_required") is not True:
        raise E11PreflightError("E11 challenger config mismatch")
    for item, variant, order in (
        (control, CONTROL_VARIANT, tuple(get_feature_list(CONTROL_VARIANT))),
        (challenger, CHALLENGER_VARIANT, challenger_feature_order()),
    ):
        config = item["config"]
        assert_no_test_access(split_id=config["data"]["target_access_mode"])
        if (
            tuple(item.get("feature_order", ())) != order
            or item.get("feature_count") != len(order)
            or item.get("config_fingerprint") != compute_config_fingerprint(config)
            or item.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD"
        ):
            raise E11PreflightError(f"E11 {variant} feature/config/init mismatch")
    if any(name in challenger["feature_order"] for name in ("appliances_delta_1", "appliances_abs_delta_1", "appliances_delta_2")):
        raise E11PreflightError("Rejected E10 delta block leaked into E11")
    changed = assert_one_primary_change(EXPERIMENT_ID, control["config"], challenger["config"])
    stripped = []
    for original in (control["config"], challenger["config"]):
        cfg = deepcopy(original)
        for group, keys in (("data", ("feature_variant_id", "feature_count")), ("model", ("input_size",)), ("lineage", ("feature_fingerprint", "scaler_bundle_id", "scaler_bundle_checksum"))):
            for key in keys:
                cfg[group].pop(key, None)
        stripped.append(cfg)
    if stripped[0] != stripped[1]:
        raise E11PreflightError("E11 changes model/training outside rolling-feature block")

    selection = document.get("selection_policy", {})
    expected_selection = {"primary_metric":"pooled_rmse_wh","rmse_min_improvement_wh":0.10,"mae_max_degradation_wh":0.25,"worst_fold_rmse_max_degradation_wh":0.50,"fold_rmse_std_max_degradation_wh":0.50,"all_guardrails_required":True,"weighted_composite":False,"test_metrics_allowed":False,"automatic_promotion":False,"decision_authority":"HUMAN"}
    if selection != expected_selection:
        raise E11PreflightError("E11 Wave2 promotion guardrails are not locked")
    orchestration = document.get("v2_orchestration", {})
    expected_orchestration = {"registry_namespace":REGISTRY_NAMESPACE,"execution_track":EXECUTION_TRACK,"training_candidate_count":2,"expected_stage_a_runs":6,"expected_stage_b_runs":6,"expected_total_training_runs":12,"preflight_consumes_run_id":False,"training_authorized":False,"test_access_authorized":False}
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E11PreflightError(f"E11 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E11" not in value:
            raise E11PreflightError(f"E11 path mismatch: {key}")
        assert_v2_artifact_path(value)

    population = None
    if materialize_data:
        datasets, evidence, audit = build_e11_pretest_datasets(root)
        control_ids = datasets[CONTROL_VARIANT].window_records["target_id"].tolist()
        challenger_ids = datasets[CHALLENGER_VARIANT].window_records["target_id"].tolist()
        if (
            control_ids != challenger_ids
            or audit.common_population_fingerprint != EXPECTED_COMMON_FINGERPRINT
            or audit.control.test_rows_read
            or audit.challenger.test_rows_read
            or any(value != "PASS" for value in evidence.fingerprint_status.values())
        ):
            raise E11PreflightError("E11 common population/fingerprint/Test firewall failed")
        expected_pop = document.get("common_population", {})
        if expected_pop.get("expected_target_count") != len(control_ids) or expected_pop.get("expected_population_fingerprint") != audit.common_population_fingerprint:
            raise E11PreflightError("E11 config population lock mismatch")
        population = {"status":"PASS","target_count":len(control_ids),"population_fingerprint":audit.common_population_fingerprint,"fold_fingerprints":{str(f.fold_id):f.fold_population_fingerprint for f in evidence.folds},"test_rows_read":0,"test_target_ids_seen":0}
    return {"status":"PASS","experiment_id":"E11","accepted_incumbent":"E01_DIRECT_FS2_TF1","predecessor_audit":predecessor,"one_primary_change":"PASS","changed_paths":list(changed),"training_candidates":[CONTROL_ID,CHALLENGER_ID],"expected_total_training_runs":12,"population":population,"fold_local_x_scaling":True,"fold_local_y_scaling":True,"test_access":"NO","training_executed":False,"inference_executed":False}


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e11_document(load_e11_config(resolved), resolved)


def build_e11_candidates(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec
    return tuple(
        CandidateSpec(
            candidate_id=item["candidate_id"], model_family="TRANSFORMER_ENCODER",
            shortlist_position=0, config=deepcopy(item["config"]),
            config_fingerprint=item["config_fingerprint"],
            feature_variant_id=item["feature_variant_id"], target_scaling_option="YS1",
            lookback_steps=72, boundary_protocol="WB0_CONTEXT_CARRY_OVER",
            source_phase="V2_E11_IMMUTABLE_SNAPSHOT", candidate_role=item["role"],
        )
        for item in (document["control"], document["challenger"])
    )


def build_e11_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.rolling_origin.real_run import RunContext
    validate_e11_document(document, root, materialize_data=False)
    candidates = build_e11_candidates(document)
    cache: dict[str, Any] = {}

    def dataset_factory(candidate, _fold):
        if candidate.candidate_id not in {CONTROL_ID, CHALLENGER_ID}:
            raise E11PreflightError("E11 dataset factory rejects candidate")
        if "bundle" not in cache:
            cache["bundle"] = build_e11_pretest_datasets(root)
        datasets, _evidence, audit = cache["bundle"]
        if audit.control.test_rows_read or audit.challenger.test_rows_read:
            raise PermissionError("E11 Test firewall failed")
        return datasets[candidate.feature_variant_id]

    _, evidence, _ = build_e11_pretest_datasets(root)
    output = root / "artifacts/model_improvement_v2/experiments/E11"
    variants = {item["feature_variant_id"]: item for item in (document["control"], document["challenger"])}
    upstream = {"lineage":{},"feature_sets":{"variant_feature_counts":{key:item["feature_count"] for key,item in variants.items()},"variant_fingerprints":{key:item["config"]["lineage"]["feature_fingerprint"] for key,item in variants.items()}},"window_fingerprints":{}}
    return RunContext(project_root=root,transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json",lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json",phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json",artifact_dir=output,registry_root=output/"registry",run_root=output/"runs",scientific_max_epochs=50,scientific_patience=10,seed=SEED,device=str(candidates[0].config["runtime"]["device_type"]),is_rehearsal=False,rehearsal_synthetic=False,reuse_completed_runs=False,dataset_factory=dataset_factory,candidate_specs=candidates,registry_namespace=REGISTRY_NAMESPACE,seed_before_model_construction=True,validated_registry_lifecycle=True,execution_track=EXECUTION_TRACK,registry_upstream_context=upstream,robase_train_ids=evidence.train_ids,robase_val_ids=evidence.validation_ids,apply_fold_x_scaling=True)


def evaluate_promotion(control: Mapping[str, float], challenger: Mapping[str, float], control_fold_rmse: list[float], challenger_fold_rmse: list[float]) -> dict[str, Any]:
    gates = {
        "rmse_improvement": float(control["rmse_wh"]) - float(challenger["rmse_wh"]) >= 0.10,
        "mae_guardrail": float(challenger["mae_wh"]) - float(control["mae_wh"]) <= 0.25,
        "worst_fold_rmse_guardrail": max(challenger_fold_rmse) - max(control_fold_rmse) <= 0.50,
        "fold_rmse_std_guardrail": statistics.pstdev(challenger_fold_rmse) - statistics.pstdev(control_fold_rmse) <= 0.50,
    }
    return {"gates":gates,"eligible_for_human_promotion":all(gates.values()),"automatic_promotion":False,"human_decision_required":True}


def _audit_resume_contract(root: Path) -> dict[str, Any]:
    """Audit the registry for E11 partial-resume.

    Strict invariants:
      - Registry file exists and is readable.
      - All 6 Stage A keys (2 candidates x 3 folds) are COMPLETED in the
        registry (no Stage-A retraining is permitted under any resume mode).
      - Any registry record claiming `target_access_mode == "TEST"` is
        rejected (Test firewall).
      - For each candidate/fold, at most ONE registry record exists
        (no duplicate run IDs per candidate-stage key).

    Flexible invariants (partial-resume aware):
      - Stage B keys may be COMPLETED (reused) OR missing / RUNNING /
        FAILED (will be freshly trained by the orchestrator under
        `allow_partial_stage_b_training=True`).
      - The failed predecessor and interrupted ROLL7 RO1_B record are
        tolerated: their Stage-B slot is reported as missing and a
        fresh Stage-B refit is scheduled.

    Returns a dict containing:
      - `status`: "PASS"
      - `reuse_run_ids`: {candidate:stage -> run_id} for COMPLETED Stage-A
        and COMPLETED Stage-B records (read-only ledger).
      - `missing_stage_b_keys`: ordered list of (candidate_id, "RO{f}_B")
        tuples for Stage-B pairs that must be trained fresh.
      - `trained_partial_resume_allowed`: True (governance flag).
      - `stage_b_checkpoint_sha256`: per Stage-B COMPLETED key.
      - `new_training_run_ids_allowed`: True (Stage-B only).
      - `training_reexecution_allowed`: True (Stage-B only).
      - `test_access_authorized`: False.
    """
    registry_path = root / "artifacts/model_improvement_v2/experiments/E11/registry/experiment_registry.jsonl"
    if not registry_path.is_file():
        raise E11PreflightError("E11 registry missing for no-train resume")
    records = [json.loads(line) for line in registry_path.read_text().splitlines() if line.strip()]
    expected_ids = {CONTROL_ID, CHALLENGER_ID}
    valid_stages = {f"RO{fold}_{part}" for fold in (1,2,3) for part in ("A","B")}
    ledger: dict[str, str] = {}
    checkpoint_sha: dict[str, str] = {}
    seen_keys: set[tuple[str, str]] = set()
    # Two-pass: first collect all COMPLETED records by key, then fail
    # only if a key has zero COMPLETED records. This handles the
    # CONTROL RO1_A case where a FAILED predecessor (run 0001) and a
    # COMPLETED replacement (run 0002) share the same candidate-stage key.
    by_key: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        run_id = record.get("run_id")
        candidate = record.get("candidate_id")
        stage = record.get("sweep_stage")
        status = record.get("status")
        if candidate not in expected_ids or stage not in valid_stages:
            raise E11PreflightError(f"E11 resume record mismatch: {run_id}")
        if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
            raise E11PreflightError(f"E11 resume Test-scoped record: {run_id}")
        key = (candidate, stage)
        by_key.setdefault(key, []).append(record)
    for key, key_records in by_key.items():
        candidate, stage = key
        if len(key_records) > 1:
            # Multiple records for the same candidate-stage key (e.g.,
            # FAILED predecessor + COMPLETED replacement). Tolerated only
            # when exactly one record is COMPLETED and others are
            # RUNNING/FAILED.
            completed = [r for r in key_records if r.get("status") == "COMPLETED"]
            if len(completed) != 1:
                raise E11PreflightError(
                    f"E11 resume duplicate records for {key}: "
                    f"{[r['run_id'] for r in key_records]} (need exactly 1 COMPLETED)"
                )
            chosen = completed[0]
        else:
            chosen = key_records[0]
        seen_keys.add(key)
        run_id = chosen["run_id"]
        if stage.endswith("_A"):
            if chosen.get("status") != "COMPLETED":
                raise E11PreflightError(
                    f"E11 resume requires COMPLETED Stage A: {run_id} (status={chosen.get('status')})"
                )
            ledger[f"{candidate}:{stage}"] = run_id
        elif stage.endswith("_B"):
            if chosen.get("status") == "COMPLETED":
                checkpoint = root / "artifacts/model_improvement_v2/experiments/E11/runs" / run_id / "checkpoints/refit_final.pt"
                if not checkpoint.is_file():
                    raise E11PreflightError(f"E11 Stage-B checkpoint missing: {run_id}")
                actual = _sha256(checkpoint)
                registered = [
                    a for a in chosen.get("artifacts", [])
                    if Path(a.get("artifact_path", "")).name == "refit_final.pt"
                ]
                if len(registered) != 1 or registered[0].get("sha256") != actual:
                    raise E11PreflightError(f"E11 checkpoint SHA mismatch: {run_id}")
                ledger[f"{candidate}:{stage}"] = run_id
                checkpoint_sha[f"{candidate}:{stage}"] = actual
            # else: RUNNING/FAILED/missing -> treated as missing, will be retrained.
    # All 6 Stage A keys must be present in ledger.
    stage_a_keys = {f"{cid}:RO{fold}_A" for cid in expected_ids for fold in (1,2,3)}
    missing_stage_a = stage_a_keys - set(ledger.keys())
    if missing_stage_a:
        raise E11PreflightError(
            f"E11 resume requires COMPLETED Stage A for all 6 pairs; missing: {sorted(missing_stage_a)}"
        )
    # Identify missing Stage-B keys (must be trained fresh under partial resume).
    stage_b_keys = {f"{cid}:RO{fold}_B" for cid in expected_ids for fold in (1,2,3)}
    present_stage_b = set(ledger.keys()) & stage_b_keys
    missing_stage_b_keys = sorted(stage_b_keys - present_stage_b)
    partial_training_required = bool(missing_stage_b_keys)
    # Also: any registry record that claims RUNNING/FAILED Stage-B but
    # has no COMPLETED replacement is reported in `missing_stage_b_keys`.
    return {
        "status": "PASS",
        "locked_run_ids": ledger,
        "stage_b_checkpoint_sha256": checkpoint_sha,
        "missing_stage_b_keys": missing_stage_b_keys,
        "trained_partial_resume_allowed": partial_training_required,
        "new_training_run_ids_allowed": partial_training_required,
        "training_reexecution_allowed": partial_training_required,
        "test_access_authorized": False,
    }


def build_e11_resume_context(root: Path, document: Mapping[str, Any]):
    """Build a RunContext for E11 partial-resume.

    Reuses 6 Stage-A COMPLETED runs (no Stage-A retrain) + any COMPLETED
    Stage-B refits. Schedules fresh Stage-B refits for the missing
    Stage-B keys (governance: 4 missing keys + 1 interrupted replacement
    = 5 fresh Stage-B refits in the current E11 state).
    """
    audit = _audit_resume_contract(root)
    context = build_e11_run_context(root, document)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(audit["locked_run_ids"])
    context.allow_partial_stage_b_training = bool(audit["missing_stage_b_keys"])
    return context, audit


def _write_success(root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any], resume_audit: Mapping[str, Any] | None = None) -> None:
    from course_work.rolling_origin.real_run import fit_fold_local_scaler
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes
    output=root/"artifacts/model_improvement_v2/experiments/E11"
    rows=list(csv.DictReader((output/"rolling_origin_results.csv").open(encoding="utf-8")))
    folds={cid:[float(row["rmse_wh"]) for row in rows if row["candidate_id"]==cid] for cid in (CONTROL_ID,CHALLENGER_ID)}
    metrics=result.pooled_metrics_by_cid
    promotion=evaluate_promotion(metrics[CONTROL_ID],metrics[CHALLENGER_ID],folds[CONTROL_ID],folds[CHALLENGER_ID])
    datasets,evidence,causal=build_e11_pretest_datasets(root)
    context=build_e11_run_context(root,document); candidates={c.candidate_id:c for c in context.candidate_specs}
    scaler_rows=[]
    for fold in evidence.folds:
        for candidate_id in (CONTROL_ID,CHALLENGER_ID):
            candidate=candidates[candidate_id]; dataset=datasets[candidate.feature_variant_id]
            for stage in ("A","B"):
                bundle,audit=fit_fold_local_scaler(candidate=candidate,fold=fold,fold_dataset=dataset,fit_stage=stage,ctx=context)
                scaler_rows.append({**audit.as_row(),"feature_order":list(dataset.feature_order),"feature_count":len(dataset.feature_order),"bundle_checksum_verified":bundle.checksum()==audit.bundle_checksum})
    population={"schema":"MODEL_IMPROVEMENT_V2_E11_POPULATION_AUDIT-v1","common_target_count":causal.common_target_count,"common_population_fingerprint":causal.common_population_fingerprint,"control_challenger_ordered_ids_identical":True,"folds":[fold.as_dict() for fold in evidence.folds],"test_rows_read":0,"test_target_ids_seen":0,"status":"PASS"}
    feature_contract={"schema":"MODEL_IMPROVEMENT_V2_E11_FEATURE_CONTRACT-v1","base_variant":"FS2_TF1","challenger_variant":CHALLENGER_VARIANT,"base_feature_order":list(get_feature_list(CONTROL_VARIANT)),"appended_feature_order":list(ROLLING_FEATURES),"challenger_feature_order":list(challenger_feature_order()),"feature_fingerprint":document["challenger"]["config"]["lineage"]["feature_fingerprint"],"definitions":dict(ROLLING_DEFINITIONS),"ddof":0,"centered":False,"partial_windows":False,"imputation":"NONE","status":"PASS"}
    comparison={"schema":"MODEL_IMPROVEMENT_V2_E11_ROLLING_TARGET_ABLATION-v1","experiment_id":"E11","primary_change":"PAST_ONLY_ROLLING_TARGET_FEATURE_BLOCK","control_metrics":metrics[CONTROL_ID],"challenger_metrics":metrics[CHALLENGER_ID],"control_fold_rmse_wh":folds[CONTROL_ID],"challenger_fold_rmse_wh":folds[CHALLENGER_ID],"control_worst_fold_rmse_wh":max(folds[CONTROL_ID]),"challenger_worst_fold_rmse_wh":max(folds[CHALLENGER_ID]),"control_fold_rmse_std_wh":statistics.pstdev(folds[CONTROL_ID]),"challenger_fold_rmse_std_wh":statistics.pstdev(folds[CHALLENGER_ID]),"delta_rmse_challenger_minus_control":float(metrics[CHALLENGER_ID]["rmse_wh"])-float(metrics[CONTROL_ID]["rmse_wh"]),"promotion_evaluation":promotion,"decision":"HUMAN_REVIEW_REQUIRED","test_status":"NOT_ACCESSED"}
    run_ids={"stage_a":{f"{cid}:{fold}":run_id for (cid,fold),run_id in result.stage_a_run_ids.items()},"stage_b":{f"{cid}:{fold}":run_id for (cid,fold),run_id in result.stage_b_run_ids.items()}}
    recovery=resume_audit or _audit_resume_contract(root)
    manifest={"schema":"MODEL_IMPROVEMENT_V2_E11_EXECUTION-v1","status":"SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED","accepted_incumbent_at_entry":"E01_DIRECT_FS2_TF1","e10_human_decision":"REJECTED","run_ids":run_ids,"selected_best_epochs":{f"{cid}:{fold}":epoch for (cid,fold),epoch in result.inner_best_epochs.items()},"required_e11_audits":{"feature_contract":"e11_feature_contract.json","causal_audit":"e11_causal_audit.json","population_audit":"e11_common_population_audit.json","scaler_audit":"e11_scaler_lineage_audit.json","recovery_contract":"e11_no_retrain_recovery_contract.json"},"preflight":dict(preflight),"test_status":"NOT_ACCESSED"}
    documents={"e11_execution_manifest.json":manifest,"e11_rolling_target_ablation_comparison.json":comparison,"e11_feature_contract.json":feature_contract,"e11_causal_audit.json":{"schema":"MODEL_IMPROVEMENT_V2_E11_CAUSAL_AUDIT-v1",**causal.to_dict(),"status":"PASS"},"e11_common_population_audit.json":population,"e11_scaler_lineage_audit.json":{"schema":"MODEL_IMPROVEMENT_V2_E11_SCALER_AUDIT-v1","fold_local_x_scaling":True,"fold_local_y_scaling":True,"audit_rows":scaler_rows,"all_passed":all(row["status"]=="PASS" and row["bundle_checksum_verified"] and row["test_rows_used"]==0 and row["outer_eval_rows_used"]==0 for row in scaler_rows),"test_status":"NOT_ACCESSED"},"e11_no_retrain_recovery_contract.json":{"schema":"MODEL_IMPROVEMENT_V2_E11_RECOVERY-v1",**dict(recovery)}}
    for name,payload in documents.items(): atomic_write_bytes(output/name,canonical_json_bytes(payload))


def run_official(root: Path | None = None) -> int:
    resolved=root or project_root(); document=load_e11_config(resolved); preflight=validate_e11_document(document,resolved); context=build_e11_run_context(resolved,document)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(context)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(resolved,document,result,preflight); print(result.summary); return 0


def run_resume_stage_c(root: Path | None = None) -> int:
    resolved=root or project_root(); document=load_e11_config(resolved); preflight=validate_e11_document(document,resolved); context,audit=build_e11_resume_context(resolved,document)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(context)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(resolved,document,result,preflight,audit); print(result.summary); return 0


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="V2 E11 past-only rolling-target audit"); parser.add_argument("--experiment",choices=["E11"],required=True); parser.add_argument("--mode",choices=["preflight","official","resume-stage-c"],default="preflight"); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--authorize-training",action="store_true"); args=parser.parse_args(argv)
    if args.seed!=42: print("ERROR: E11 screening seed must be 42",file=sys.stderr); return 2
    if args.mode=="official" and not args.authorize_training: print("REFUSED: E11 official mode requires --authorize-training.",file=sys.stderr); return 3
    # E11 partial-resume trains new Stage-B refits; require Human authorization.
    if args.mode=="resume-stage-c" and not args.authorize_training: print("REFUSED: E11 resume-stage-c trains partial Stage-B refits and requires --authorize-training.",file=sys.stderr); return 3
    if args.mode=="preflight" and args.authorize_training: print("ERROR: --authorize-training is valid only in official or resume-stage-c mode.",file=sys.stderr); return 2
    try:
        if args.mode=="official": return run_official()
        if args.mode=="resume-stage-c": return run_resume_stage_c()
        print(json.dumps(run_preflight(),indent=2,sort_keys=True)); return 0
    except (E11PreflightError,PermissionError,ValueError) as exc: print(f"E11 {args.mode.upper()} FAIL: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
