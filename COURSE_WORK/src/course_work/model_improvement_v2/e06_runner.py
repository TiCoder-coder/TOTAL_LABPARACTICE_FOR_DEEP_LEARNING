"""E06 constant AdamW learning-rate audit with immutable E01 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import assert_no_test_access, assert_one_primary_change, assert_v2_artifact_path

EXPERIMENT_ID = "E06"
SUPPORTED_SEED = 42
BASE_CANDIDATE_ID = "TR_C2_ALT_LOOKBACK"
FEATURE_VARIANT = "FS2_TF1"
CONTROL_LR = 3e-4
CHALLENGER_LRS = (5e-5, 1e-4, 2e-4)
ALL_LRS = (*CHALLENGER_LRS, CONTROL_LR)
CANDIDATE_IDS = {
    5e-5: "TR_C2_ALT_LOOKBACK_LR_5E5",
    1e-4: "TR_C2_ALT_LOOKBACK_LR_1E4",
    2e-4: "TR_C2_ALT_LOOKBACK_LR_2E4",
}
EXPECTED_CONFIG_CHANGES = ("training.learning_rate",)
EXECUTION_TRACK = "MODEL_IMPROVEMENT_V2_E06"
REGISTRY_NAMESPACE = "V2_E06"
CONFIG_RELATIVE_PATH = Path("artifacts/model_improvement_v2/experiments/E06/e06_config_snapshot.json")
E01_RELATIVE_ROOT = Path("artifacts/model_improvement_v2/experiments/E01")
EXPECTED_CONTROL_METRICS = {"rmse_wh": 59.85291570400546, "mae_wh": 26.650501720144604, "r2": 0.5789433617557903}
PROMOTION_THRESHOLD_RMSE_WH = 59.75291570400546


class E06PreflightError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E06PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise E06PreflightError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e06_config(root: Path) -> dict[str, Any]:
    return _read_json(root / CONFIG_RELATIVE_PATH)


def _verify_artifacts(base: Path, expected: Mapping[str, Any], label: str) -> None:
    for name, checksum in expected.items():
        path = base / name
        if not path.is_file() or _sha256(path) != checksum:
            raise E06PreflightError(f"{label} immutable artifact changed: {name}")


def _validate_control(root: Path, document: Mapping[str, Any], e01_config: Mapping[str, Any]) -> dict[str, Any]:
    control = document.get("control", {})
    if control.get("reuse_policy") != "CONTROL_REUSED_FROM_E01" or control.get("learning_rate") != CONTROL_LR:
        raise E06PreflightError("E06 must reuse E01 lr=3e-4 control")
    if control.get("test_status") != "NOT_ACCESSED" or control.get("pooled_metrics") != EXPECTED_CONTROL_METRICS:
        raise E06PreflightError("E01 control metrics/Test status mismatch")
    if assert_one_primary_change("E01", e01_config, control.get("config", {})) != ():
        raise E06PreflightError("E06 control config differs from E01")
    if control.get("config_fingerprint") != compute_config_fingerprint(e01_config):
        raise E06PreflightError("E06 control fingerprint mismatch")
    base = root / E01_RELATIVE_ROOT
    _verify_artifacts(base, control.get("artifact_sha256", {}), "E01")
    comparison = _read_json(base / "e01_reproduction_comparison.json")
    observed = {"rmse_wh": comparison.get("observed_rmse_wh"), "mae_wh": comparison.get("observed_mae_wh"), "r2": comparison.get("observed_r2")}
    if observed != EXPECTED_CONTROL_METRICS or comparison.get("target_population_match") is not True or comparison.get("test_status") != "NOT_ACCESSED":
        raise E06PreflightError("E01 accepted control evidence mismatch")
    execution = _read_json(base / "e01_execution_manifest.json")
    if execution.get("run_ids") != control.get("run_ids"):
        raise E06PreflightError("E01 control run ledger mismatch")
    run_ids = [rid for stage in control["run_ids"].values() for rid in stage.values()]
    if len(run_ids) != 6 or len(set(run_ids)) != 6 or not all("_E01_" in rid for rid in run_ids):
        raise E06PreflightError("E01 control run ledger incomplete")
    return {"status": "PASS", "learning_rate": CONTROL_LR, "run_count": 6, "metrics": observed}


def _validate_rejected_residuals(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "E03": ("e03_prediction_ablation_comparison.json", 60.04654108949536, "NO_MEANINGFUL_IMPROVEMENT / TIE"),
        "E04": ("e04_head_ablation_comparison.json", 60.66433497520889, "DO_NOT_PROMOTE_E04_GLOBAL"),
        "E05": ("e05_gate_ablation_comparison.json", 60.681364484407354, "DO_NOT_PROMOTE_E05_GLOBAL"),
    }
    evidence = document.get("rejected_residual_evidence", [])
    if len(evidence) != 3:
        raise E06PreflightError("E06 requires locked E03/E04/E05 rejection evidence")
    for row in evidence:
        exp = row.get("experiment_id")
        if exp not in expected:
            raise E06PreflightError("Unexpected residual predecessor")
        name, rmse, status = expected[exp]
        path = root / f"artifacts/model_improvement_v2/experiments/{exp}" / name
        if row.get("comparison_artifact") != name or row.get("artifact_sha256") != _sha256(path):
            raise E06PreflightError(f"{exp} rejection artifact mismatch")
        if row.get("pooled_rmse_wh") != rmse or row.get("promotion_status") != status:
            raise E06PreflightError(f"{exp} rejection decision mismatch")
        if row.get("test_status") != "NOT_ACCESSED" or row.get("policy") != "NOT_CARRIED_FORWARD":
            raise E06PreflightError(f"{exp} cannot be carried into E06")
    return {"status": "PASS", "returned_to_e01_direct": True}



def _audit_resume_contract(root: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    resume = document.get("resume_stage_c", {})
    expected_policy = {
        "mode": "NO_RETRAIN_STAGE_C_ONLY",
        "expected_run_count": 18,
        "expected_stage_b_checkpoint_count": 9,
        "new_training_run_ids_allowed": False,
        "stage_a_reexecution_allowed": False,
        "stage_b_reexecution_allowed": False,
        "test_access_authorized": False,
    }
    for key, expected in expected_policy.items():
        if resume.get(key) != expected:
            raise E06PreflightError(f"E06 resume policy mismatch: {key}")
    ledger = resume.get("locked_run_ids", {})
    locked_epochs = resume.get("locked_best_epochs", {})
    checkpoint_hashes = resume.get("stage_b_checkpoint_sha256", {})
    if len(ledger) != 18 or len(locked_epochs) != 18 or len(checkpoint_hashes) != 9:
        raise E06PreflightError("E06 resume ledger must lock 18 runs and 9 Stage-B checkpoints")
    registry_path = root / "artifacts/model_improvement_v2/experiments/E06/registry/experiment_registry.jsonl"
    records = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["run_id"]: record for record in records}
    if len(records) != 18 or len(by_id) != 18:
        raise E06PreflightError("E06 registry must contain exactly the original 18 runs")
    checkpoint_rows = []
    for key, run_id in ledger.items():
        candidate_id, sweep_stage = key.rsplit(":", 1)
        record = by_id.get(run_id)
        if record is None or record.get("status") != "COMPLETED":
            raise E06PreflightError(f"E06 resume run is not COMPLETED: {run_id}")
        if record.get("candidate_id") != candidate_id or record.get("sweep_stage") != sweep_stage:
            raise E06PreflightError(f"E06 resume run identity mismatch: {run_id}")
        if record.get("best_epoch") != locked_epochs.get(key):
            raise E06PreflightError(f"E06 resume epoch mismatch: {run_id}")
        if record.get("config", {}).get("data", {}).get("target_access_mode") == "TEST":
            raise E06PreflightError(f"E06 resume Test-scoped run rejected: {run_id}")
        if sweep_stage.endswith("_B"):
            artifacts = [
                artifact for artifact in record.get("artifacts", [])
                if artifact.get("artifact_type") == "BEST_CHECKPOINT"
                and Path(artifact.get("artifact_path", "")).name == "refit_final.pt"
            ]
            if len(artifacts) != 1:
                raise E06PreflightError(f"E06 Stage-B checkpoint registration mismatch: {run_id}")
            checkpoint = root / "artifacts/model_improvement_v2/experiments/E06/runs" / run_id / "checkpoints/refit_final.pt"
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise E06PreflightError(f"E06 Stage-B checkpoint missing/empty: {run_id}")
            actual = _sha256(checkpoint)
            if actual != checkpoint_hashes.get(key) or actual != artifacts[0].get("sha256"):
                raise E06PreflightError(f"E06 Stage-B checkpoint checksum mismatch: {run_id}")
            checkpoint_rows.append({"candidate_id": candidate_id, "stage": sweep_stage, "run_id": run_id, "best_epoch": record["best_epoch"], "sha256": actual})
    for candidate_id in CANDIDATE_IDS.values():
        for fold in (1, 2, 3):
            a_key = f"{candidate_id}:RO{fold}_A"
            b_key = f"{candidate_id}:RO{fold}_B"
            if locked_epochs[a_key] != locked_epochs[b_key]:
                raise E06PreflightError(f"E06 Stage A/B selected epoch mismatch: {candidate_id}/RO{fold}")
    return {"status": "PASS", "completed_run_count": 18, "stage_a_run_count": 9, "stage_b_run_count": 9, "checkpoint_count": 9, "new_run_ids_allowed": False, "training_reexecution_allowed": False, "checkpoints": checkpoint_rows}

def validate_e06_document(document: Mapping[str, Any], root: Path) -> dict[str, Any]:
    if document.get("experiment_id") != EXPERIMENT_ID or document.get("track") != "MODEL_IMPROVEMENT-v2":
        raise E06PreflightError("Wrong E06 identity/track")
    if document.get("objective") != "CONSTANT_ADAMW_LEARNING_RATE_AUDIT":
        raise E06PreflightError("Wrong E06 objective")
    primary = document.get("primary_change", {})
    if primary != {"factor": "training.learning_rate", "candidate_values": list(ALL_LRS)}:
        raise E06PreflightError("E06 LR candidate set mismatch")
    e01_config = _read_json(root / E01_RELATIVE_ROOT / "e01_config_snapshot.json")["v1_config"]
    control_audit = _validate_control(root, document, e01_config)

    candidates = document.get("training_candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise E06PreflightError("E06 must contain exactly three training candidates")
    seen_lrs: set[float] = set(); seen_ids: set[str] = set(); fingerprints: set[str] = set()
    for candidate in candidates:
        lr = candidate.get("learning_rate"); cid = candidate.get("candidate_id"); config = candidate.get("config")
        if lr not in CHALLENGER_LRS or cid != CANDIDATE_IDS[lr] or not isinstance(config, dict):
            raise E06PreflightError("E06 challenger identity/LR mismatch")
        assert_no_test_access(split_id=str(config.get("data", {}).get("target_access_mode", "")))
        changed = assert_one_primary_change(EXPERIMENT_ID, e01_config, config)
        if changed != EXPECTED_CONFIG_CHANGES or config["training"].get("learning_rate") != lr:
            raise E06PreflightError(f"E06 {cid} changes more than learning_rate")
        if config["training"].get("optimizer_name") != "AdamW" or config["training"].get("scheduler_name") is not None or config["training"].get("scheduler_config") is not None:
            raise E06PreflightError("E06 requires constant-LR AdamW with scheduler OFF")
        if config["model"].get("prediction_formulation", "DIRECT") != "DIRECT":
            raise E06PreflightError("E06 must return to DIRECT E01 configuration")
        fingerprint = compute_config_fingerprint(config)
        if candidate.get("config_fingerprint") != fingerprint or candidate.get("initialization_policy") != "FRESH_FROM_SEED_42_NO_CHECKPOINT_LOAD":
            raise E06PreflightError("E06 challenger fingerprint/initialization mismatch")
        seen_lrs.add(lr); seen_ids.add(cid); fingerprints.add(fingerprint)
    if seen_lrs != set(CHALLENGER_LRS) or seen_ids != set(CANDIDATE_IDS.values()) or len(fingerprints) != 3:
        raise E06PreflightError("E06 challenger set is not unique and complete")

    from course_work.data.feature_sets import get_feature_list
    if tuple(document.get("feature_order", ())) != tuple(get_feature_list(FEATURE_VARIANT)):
        raise E06PreflightError("E06 FS2_TF1 feature order mismatch")
    fixed = document.get("fixed_contract", {})
    expected_fixed = {"prediction_formulation":"DIRECT","feature_variant_id":"FS2_TF1","feature_count":33,"lookback_steps":72,"horizon_steps":1,"boundary_protocol":"WB0_CONTEXT_CARRY_OVER","rolling_origin_protocol":"RO3_EXPANDING_PRETEST-v1","folds":["RO1","RO2","RO3"],"fold_local_x_scaling":True,"fold_local_y_scaling":True,"optimizer_name":"AdamW","weight_decay":1e-3,"loss_name":"MSE","seed":42,"scheduler_name":None,"warmup":False}
    if fixed != expected_fixed:
        raise E06PreflightError("E06 fixed scientific contract mismatch")

    orchestration = document.get("v2_orchestration", {})
    assert_no_test_access(test_access=bool(orchestration.get("test_access_authorized")))
    expected_orchestration = {"registry_namespace":REGISTRY_NAMESPACE,"training_candidate_count":3,"control_retrained":False,"challengers_only_training":True,"fresh_initialization_per_candidate":True,"same_initialization_seed_across_lr":True,"preflight_consumes_run_id":False,"training_authorized":False,"test_access_authorized":False}
    for key, expected in expected_orchestration.items():
        if orchestration.get(key) != expected:
            raise E06PreflightError(f"E06 orchestration mismatch: {key}")
    for key in ("artifact_root", "registry_root", "run_root"):
        value = orchestration.get(key)
        if not isinstance(value, str) or "/experiments/E06" not in value:
            raise E06PreflightError(f"E06-owned path required: {key}")
        assert_v2_artifact_path(value)
    selection = document.get("selection_policy", {})
    if selection != {"primary_metric":"pooled_rolling_origin_rmse_wh","all_lr_values":list(ALL_LRS),"global_baseline_rmse_wh":EXPECTED_CONTROL_METRICS["rmse_wh"],"promotion_threshold_rmse_wh":PROMOTION_THRESHOLD_RMSE_WH,"promotion_rule":"candidate_pooled_rmse_wh <= promotion_threshold_rmse_wh","fallback":"RETAIN_E01_LR_3E4","weighted_composite":False,"test_metrics_allowed":False}:
        raise E06PreflightError("E06 selection policy mismatch")
    rejected_audit = _validate_rejected_residuals(root, document)
    resume_audit = _audit_resume_contract(root, document)
    return {"status":"PASS","experiment_id":EXPERIMENT_ID,"control_learning_rate":CONTROL_LR,"control_reuse":"CONTROL_REUSED_FROM_E01","control_audit":control_audit,"training_candidate_count":3,"training_candidates":list(CHALLENGER_LRS),"candidate_ids":[CANDIDATE_IDS[lr] for lr in CHALLENGER_LRS],"changed_config_fields":list(EXPECTED_CONFIG_CHANGES),"one_primary_change":"PASS","fresh_initialization_per_candidate":True,"same_initialization_seed_across_lr":True,"scheduler":"OFF","rejected_residual_audit":rejected_audit,"resume_audit":resume_audit,"promotion_threshold_rmse_wh":PROMOTION_THRESHOLD_RMSE_WH,"registry_namespace":REGISTRY_NAMESPACE,"test_access":"NO","training_executed":False,"inference_executed":False,"checkpoint_loaded":False}


def run_preflight(root: Path | None = None) -> dict[str, Any]:
    resolved = root or project_root()
    return validate_e06_document(load_e06_config(resolved), resolved)


def build_e06_candidates(document: Mapping[str, Any]):
    from course_work.rolling_origin.candidate_loader import CandidateSpec
    specs=[]
    for item in document["training_candidates"]:
        config=deepcopy(item["config"]); assert_no_test_access(split_id=config["data"]["target_access_mode"])
        specs.append(CandidateSpec(candidate_id=item["candidate_id"],model_family="TRANSFORMER_ENCODER",shortlist_position=0,config=config,config_fingerprint=item["config_fingerprint"],feature_variant_id=FEATURE_VARIANT,target_scaling_option="YS1",lookback_steps=72,boundary_protocol="WB0_CONTEXT_CARRY_OVER",source_phase="V2_E06_IMMUTABLE_SNAPSHOT",candidate_role="ADAMW_CONSTANT_LR_CHALLENGER"))
    return tuple(specs)


def _registry_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage=deepcopy(config["lineage"])
    return {"lineage":lineage,"feature_sets":{"variant_feature_counts":{FEATURE_VARIANT:33},"variant_fingerprints":{FEATURE_VARIANT:lineage["feature_fingerprint"]}},"window_fingerprints":{}}


def build_e06_run_context(root: Path, document: Mapping[str, Any]):
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset, load_phase44_fold_evidence
    from course_work.rolling_origin.real_run import RunContext
    validate_e06_document(document, root); candidates=build_e06_candidates(document)
    artifact_dir=root/"artifacts/model_improvement_v2/experiments/E06"; cache={}
    def dataset_factory(candidate_spec, _fold):
        if candidate_spec.feature_variant_id != FEATURE_VARIANT or candidate_spec.candidate_id not in CANDIDATE_IDS.values():
            raise E06PreflightError("E06 dataset factory rejects unknown candidate")
        assert_no_test_access(split_id=candidate_spec.config["data"]["target_access_mode"])
        if "bundle" not in cache:
            cache["bundle"]=build_v2_pretest_dataset(root,tuple(document["feature_order"]),experiment_id=EXPERIMENT_ID,feature_variant_id=FEATURE_VARIANT)
        dataset,_evidence,audit=cache["bundle"]
        if audit.test_rows_read or audit.test_target_ids_seen:
            raise PermissionError("E06 Test firewall failed")
        return dataset
    evidence=load_phase44_fold_evidence(root)
    return RunContext(project_root=root,transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json",lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json",phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json",artifact_dir=artifact_dir,registry_root=artifact_dir/"registry",run_root=artifact_dir/"runs",scientific_max_epochs=50,scientific_patience=10,seed=SUPPORTED_SEED,device=str(candidates[0].config["runtime"]["device_type"]),is_rehearsal=False,rehearsal_synthetic=False,reuse_completed_runs=False,dataset_factory=dataset_factory,candidate_specs=candidates,registry_namespace=REGISTRY_NAMESPACE,seed_before_model_construction=True,validated_registry_lifecycle=True,execution_track=EXECUTION_TRACK,registry_upstream_context=_registry_upstream(candidates[0].config),robase_train_ids=evidence.train_ids,robase_val_ids=evidence.validation_ids,apply_fold_x_scaling=True)


def _write_success_contracts(root: Path, document: Mapping[str, Any], result, preflight: Mapping[str, Any]) -> None:
    from course_work.utils.artifacts import atomic_write_bytes, canonical_json_bytes
    output=root/"artifacts/model_improvement_v2/experiments/E06"
    rows=[{"candidate_id":BASE_CANDIDATE_ID,"learning_rate":CONTROL_LR,"source":"CONTROL_REUSED_FROM_E01","pooled_metrics":EXPECTED_CONTROL_METRICS}]
    for lr in CHALLENGER_LRS:
        cid=CANDIDATE_IDS[lr]; rows.append({"candidate_id":cid,"learning_rate":lr,"source":"E06_TRAINED_CHALLENGER","pooled_metrics":result.pooled_metrics_by_cid[cid]})
    rows.sort(key=lambda row: row["learning_rate"])
    best=min(rows,key=lambda row:float(row["pooled_metrics"]["rmse_wh"]))
    eligible=[row for row in rows if row["source"] == "E06_TRAINED_CHALLENGER" and float(row["pooled_metrics"]["rmse_wh"]) <= PROMOTION_THRESHOLD_RMSE_WH]
    promoted=min(eligible,key=lambda row:float(row["pooled_metrics"]["rmse_wh"])) if eligible else None
    decision={"action":"PROMOTE_NEW_LR" if promoted else "RETAIN_E01_LR_3E4","selected_learning_rate":promoted["learning_rate"] if promoted else CONTROL_LR,"selected_candidate_id":promoted["candidate_id"] if promoted else BASE_CANDIDATE_ID,"best_observed_learning_rate":best["learning_rate"],"best_observed_rmse_wh":best["pooled_metrics"]["rmse_wh"]}
    manifest={"schema":"MODEL_IMPROVEMENT_V2_E06_EXECUTION-v1","experiment_id":EXPERIMENT_ID,"status":"SCIENTIFIC_EXECUTION_COMPLETE_HUMAN_REVIEW_REQUIRED","control_policy":"CONTROL_REUSED_FROM_E01","control_run_ids":document["control"]["run_ids"],"challenger_run_ids":{"stage_a":{f"{cid}:{fold}":rid for (cid,fold),rid in result.stage_a_run_ids.items()},"stage_b":{f"{cid}:{fold}":rid for (cid,fold),rid in result.stage_b_run_ids.items()}},"test_status":"NOT_ACCESSED","preflight":dict(preflight)}
    comparison={"schema":"MODEL_IMPROVEMENT_V2_E06_ADAMW_LR_AUDIT-v1","experiment_id":EXPERIMENT_ID,"primary_change":"LEARNING_RATE","optimizer":"AdamW","scheduler":"OFF","results":rows,"promotion_threshold_rmse_wh":PROMOTION_THRESHOLD_RMSE_WH,"promotion_rule":"candidate_pooled_rmse_wh <= promotion_threshold_rmse_wh","decision":decision,"target_population_policy":"IDENTICAL_PHASE44_RO1_RO2_RO3","test_status":"NOT_ACCESSED"}
    atomic_write_bytes(output/"e06_execution_manifest.json",canonical_json_bytes(manifest)); atomic_write_bytes(output/"e06_lr_audit_comparison.json",canonical_json_bytes(comparison))



def build_e06_resume_context(root: Path, document: Mapping[str, Any]):
    """Build a Stage-C-only context locked to the original 18 completed runs."""
    _audit_resume_contract(root, document)
    context = build_e06_run_context(root, document)
    context.reuse_completed_runs = True
    context.reuse_completed_run_ids = dict(document["resume_stage_c"]["locked_run_ids"])
    return context


def run_resume_stage_c(root: Path | None = None) -> int:
    resolved = root or project_root()
    document = load_e06_config(resolved)
    preflight = validate_e06_document(document, resolved)
    context = build_e06_resume_context(resolved, document)
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

def run_official(root: Path | None = None) -> int:
    resolved=root or project_root(); document=load_e06_config(resolved); preflight=validate_e06_document(document,resolved); context=build_e06_run_context(resolved,document)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(context)
    if result.exit_code != 0:
        print(result.summary,file=sys.stderr)
        if result.exception: print(result.exception,file=sys.stderr)
        return result.exit_code
    _write_success_contracts(resolved,document,result,preflight); print(result.summary); return 0


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="V2 E06 constant AdamW learning-rate audit")
    parser.add_argument("--experiment",choices=[EXPERIMENT_ID],required=True); parser.add_argument("--mode",choices=["preflight","official","resume-stage-c"],default="preflight"); parser.add_argument("--seed",type=int,default=SUPPORTED_SEED); parser.add_argument("--authorize-training",action="store_true"); return parser


def main(argv: list[str] | None = None) -> int:
    args=build_parser().parse_args(argv)
    if args.seed != SUPPORTED_SEED: print(f"ERROR: E06 screening seed must be {SUPPORTED_SEED}",file=sys.stderr); return 2
    if args.mode == "official" and not args.authorize_training: print("REFUSED: E06 official mode requires --authorize-training.",file=sys.stderr); return 3
    if args.mode != "official" and args.authorize_training: print("ERROR: --authorize-training is valid only in official mode.",file=sys.stderr); return 2
    try:
        if args.mode == "official": return run_official()
        if args.mode == "resume-stage-c": return run_resume_stage_c()
        print(json.dumps(run_preflight(),indent=2,sort_keys=True)); return 0
    except (E06PreflightError,PermissionError,ValueError) as exc: print(f"E06 {args.mode.upper()} FAIL: {exc}",file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
