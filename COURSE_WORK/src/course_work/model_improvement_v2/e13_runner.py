"""E13 hybrid level-plus-delta loss audit; preparation is training-free."""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import assert_no_test_access, assert_one_primary_change, assert_v2_artifact_path
from course_work.model_improvement_v2.hybrid_loss import ALLOWED_LAMBDAS, BETA_MODEL_SPACE, LOSS_POLICY
EXPERIMENT_ID="E13"; EXECUTION_TRACK="MODEL_IMPROVEMENT_V2_E13"; REGISTRY_NAMESPACE="V2_E13"; FEATURE_VARIANT="FS2_TF1"; SEED=42
CONFIG_PATH=Path("artifacts/model_improvement_v2/experiments/E13/e13_config_snapshot.json")
E01_ROOT=Path("artifacts/model_improvement_v2/experiments/E01"); E12_ROOT=Path("artifacts/model_improvement_v2/experiments/E12")
CONTROL_METRICS={"rmse_wh":59.85291570400546,"mae_wh":26.650501720144604,"r2":0.5789433617557903}
CONTROL_FOLDS=[69.01338295186963,50.59540710120016,58.51676739199306]
CANDIDATE_IDS={0.10:"TR_C2_ALT_LOOKBACK_E13_LAMBDA_010",0.25:"TR_C2_ALT_LOOKBACK_E13_LAMBDA_025",0.50:"TR_C2_ALT_LOOKBACK_E13_LAMBDA_050"}
class E13PreflightError(RuntimeError): pass
def project_root()->Path: return Path(__file__).resolve().parents[3]
def _read(path:Path)->dict[str,Any]:
    try: value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: raise E13PreflightError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value,dict): raise E13PreflightError(f"Expected JSON object: {path}")
    return value
def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""): h.update(block)
    return h.hexdigest()
def load_e13_config(root:Path)->dict[str,Any]: return _read(root/CONFIG_PATH)
def _e01_config(root:Path)->dict[str,Any]: return _read(root/E01_ROOT/"e01_config_snapshot.json")["v1_config"]
def candidate_config(root:Path,value:float)->dict[str,Any]:
    cfg=deepcopy(_e01_config(root)); cfg["training"].update({"loss_policy":LOSS_POLICY,"lambda_delta":value,"delta_beta_model_space":BETA_MODEL_SPACE}); return cfg
def _control_audit(root:Path,document:Mapping[str,Any])->dict[str,Any]:
    control=document["control_reuse"]
    hashes={"e01_config_snapshot.json":"7e3b064e00a3e1148053d798228ce232fc3faa7464a803b697ab4915b0a5a101","e01_execution_manifest.json":"377b50b2230a024c93650dc7f31cb6f06f9a0d55c83869ffc55bddf061c2553a","e01_reproduction_comparison.json":"a73e94c21acd18195a4350c6ae14d65f84927688bacaeacbc50292b0f07be5e5"}
    if control!={"source":"E01","candidate_id":"E01_DIRECT_FS2_TF1","reuse_only":True,"metrics":CONTROL_METRICS,"test_status":"NOT_ACCESSED","artifact_sha256":hashes}: raise E13PreflightError("E13 E01 control lock mismatch")
    if any(_sha(root/E01_ROOT/name)!=digest for name,digest in hashes.items()): raise E13PreflightError("E01 immutable control artifact changed")
    comp=_read(root/E01_ROOT/"e01_reproduction_comparison.json"); manifest=_read(root/E01_ROOT/"e01_execution_manifest.json")
    if comp.get("target_population_match") is not True or comp.get("test_status")!="NOT_ACCESSED" or manifest.get("test_status")!="NOT_ACCESSED": raise E13PreflightError("E01 control population/Test evidence failed")
    if len([v for group in manifest["run_ids"].values() for v in group.values()])!=6: raise E13PreflightError("E01 run ledger incomplete")
    return {"status":"PASS","config_fingerprint":compute_config_fingerprint(_e01_config(root)),"population_fingerprint":_e01_config(root)["lineage"]["population_fingerprint"],"run_count":6}
def _e12_rejection_audit(root:Path,document:Mapping[str,Any])->dict[str,Any]:
    lock=document["e12_human_decision"]
    if lock!={"decision":"REJECTED","challenger":"TR_C2_ALT_LOOKBACK_E12_LAG144","carry_forward_lag144":False,"incumbent_retained":"E01_DIRECT_FS2_TF1","test_status":"NOT_ACCESSED"}: raise E13PreflightError("E12 Human rejection lock mismatch")
    comp=_read(root/E12_ROOT/"e12_lag144_ablation_comparison.json")
    if comp.get("promotion_evaluation",{}).get("eligible_for_human_promotion") is not False or comp.get("test_status")!="NOT_ACCESSED" or any(comp.get("promotion_evaluation",{}).get("gates",{}).values()): raise E13PreflightError("E12 rejection evidence mismatch")
    return {"status":"PASS","decision":"REJECTED","lag144_carried_forward":False,"comparison_sha256":_sha(root/E12_ROOT/"e12_lag144_ablation_comparison.json")}
def validate_e13_document(document:Mapping[str,Any],root:Path,materialize_data:bool=True)->dict[str,Any]:
    if document.get("experiment_id")!="E13" or document.get("objective")!="HYBRID_LEVEL_PLUS_DELTA_LOSS": raise E13PreflightError("Wrong E13 identity/objective")
    control=_control_audit(root,document); e12=_e12_rejection_audit(root,document); base=_e01_config(root)
    if document.get("challenger_lambdas")!=list(ALLOWED_LAMBDAS): raise E13PreflightError("E13 lambda matrix mismatch")
    if document.get("loss_contract")!={"level":"MSE(y_pred_model,y_next_model)","delta_true_model":"(y_next_raw-y_context_raw)/y_scaler_scale","delta_pred_model":"y_pred_model-y_context_model","delta":"SmoothL1(delta_pred_model,delta_true_model,beta=1.0)","total":"level_mse+lambda_delta*delta_smooth_l1","space":"FOLD_LOCAL_Y_STANDARDIZED"}: raise E13PreflightError("E13 loss algebra lock mismatch")
    fingerprints={}
    for value in ALLOWED_LAMBDAS:
        cfg=candidate_config(root,value); assert_no_test_access(split_id=cfg["data"]["target_access_mode"])
        changed=assert_one_primary_change("E13",base,cfg)
        if changed!=("training.delta_beta_model_space","training.lambda_delta","training.loss_policy"): raise E13PreflightError("E13 changes more than loss policy")
        if cfg["data"]["feature_variant_id"]!="FS2_TF1" or cfg["data"]["feature_count"]!=33 or tuple(get_feature_list("FS2_TF1"))!=tuple(document["feature_order"]): raise E13PreflightError("E13 feature order/count changed")
        fingerprints[str(value)]=compute_config_fingerprint(cfg)
    if document.get("promotion_policy")!={"rmse_min_improvement_wh":0.10,"mae_max_degradation_wh":0.25,"worst_fold_rmse_max_degradation_wh":0.50,"fold_rmse_std_max_degradation_wh":0.50,"lower_mae_alone_can_promote":False,"automatic_promotion":False,"decision_authority":"HUMAN","test_metrics_allowed":False}: raise E13PreflightError("E13 promotion locks unresolved")
    orch=document.get("orchestration",{})
    if orch.get("expected_new_training_runs")!=18 or orch.get("control_retrained") is not False or orch.get("training_authorized") is not False or orch.get("test_access_authorized") is not False: raise E13PreflightError("E13 orchestration mismatch")
    for key in ("artifact_root","registry_root","run_root"): assert_v2_artifact_path(orch[key])
    population=None
    if materialize_data:
        from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
        dataset,evidence,audit=build_v2_pretest_dataset(root,tuple(document["feature_order"]),experiment_id="E13",feature_variant_id="FS2_TF1")
        if audit.test_rows_read or audit.test_target_ids_seen or tuple(evidence.train_ids+evidence.validation_ids)!=tuple(dataset.window_records["target_id"]): raise E13PreflightError("E13 population/Test firewall failed")
        population={"status":"PASS","population_fingerprint":base["lineage"]["population_fingerprint"],"target_count":len(dataset),"fold_fingerprints":{str(f.fold_id):f.fold_population_fingerprint for f in evidence.folds},"test_rows_read":0,"test_target_ids_seen":0}
    return {"experiment":"E13","status":"PASS","accepted_incumbent":"E01_DIRECT_FS2_TF1","feature_variant":"FS2_TF1","feature_count":33,"challenger_lambdas":list(ALLOWED_LAMBDAS),"expected_new_training_runs":18,"control_reuse_status":control["status"],"e12_decision_audit":e12,"candidate_config_fingerprints":fingerprints,"population":population,"algebra_scaling_audit":"PASS","test_rows_read":0,"test_target_ids_seen":0,"test_status":"NOT_ACCESSED","training_executed":False}
def run_preflight(root:Path|None=None)->dict[str,Any]:
    root=root or project_root(); return validate_e13_document(load_e13_config(root),root)
def build_candidates(root:Path):
    from course_work.rolling_origin.candidate_loader import CandidateSpec
    return tuple(CandidateSpec(candidate_id=CANDIDATE_IDS[v],model_family="TRANSFORMER_ENCODER",shortlist_position=0,config=candidate_config(root,v),config_fingerprint=compute_config_fingerprint(candidate_config(root,v)),feature_variant_id="FS2_TF1",target_scaling_option="YS1",lookback_steps=72,boundary_protocol="WB0_CONTEXT_CARRY_OVER",source_phase="V2_E13_IMMUTABLE_SNAPSHOT",candidate_role="HYBRID_LOSS_CHALLENGER") for v in ALLOWED_LAMBDAS)
def build_context(root:Path,document:Mapping[str,Any]):
    from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset,load_phase44_fold_evidence
    from course_work.rolling_origin.real_run import RunContext
    validate_e13_document(document,root,materialize_data=False); candidates=build_candidates(root); cache={}
    def factory(candidate,_fold):
        if candidate.candidate_id not in CANDIDATE_IDS.values(): raise E13PreflightError("E13 rejects candidate")
        if "bundle" not in cache: cache["bundle"]=build_v2_pretest_dataset(root,tuple(document["feature_order"]),experiment_id="E13",feature_variant_id="FS2_TF1")
        ds,_e,a=cache["bundle"]
        if a.test_rows_read or a.test_target_ids_seen: raise PermissionError("E13 Test firewall failed")
        return ds
    evidence=load_phase44_fold_evidence(root); base=candidates[0].config; output=root/"artifacts/model_improvement_v2/experiments/E13"
    upstream={"lineage":deepcopy(base["lineage"]),"feature_sets":{"variant_feature_counts":{"FS2_TF1":33},"variant_fingerprints":{"FS2_TF1":base["lineage"]["feature_fingerprint"]}},"window_fingerprints":{}}
    return RunContext(project_root=root,transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json",lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json",phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json",artifact_dir=output,registry_root=output/"registry",run_root=output/"runs",scientific_max_epochs=50,scientific_patience=10,seed=42,device=str(base["runtime"]["device_type"]),dataset_factory=factory,candidate_specs=candidates,registry_namespace="V2_E13",seed_before_model_construction=True,validated_registry_lifecycle=True,execution_track=EXECUTION_TRACK,registry_upstream_context=upstream,robase_train_ids=evidence.train_ids,robase_val_ids=evidence.validation_ids,apply_fold_x_scaling=True)
def audit_completed_run_ledger(root:Path)->dict[str,Any]:
    path=root/"artifacts/model_improvement_v2/experiments/E13/registry/experiment_registry.jsonl"
    if not path.is_file(): raise E13PreflightError("E13 registry missing for recovery")
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]; expected={f"{cid}:RO{fold}_{stage}" for cid in CANDIDATE_IDS.values() for fold in (1,2,3) for stage in ("A","B")}; stage_a_expected={key for key in expected if key.endswith("_A")}; ledger={}; hashes={}; noncompleted=[]
    for record in records:
        if record.get("status")!="COMPLETED":
            noncompleted.append({"run_id":record.get("run_id"),"status":record.get("status"),"candidate_id":record.get("candidate_id"),"sweep_stage":record.get("sweep_stage")})
            continue
        key=f"{record.get('candidate_id')}:{record.get('sweep_stage')}"
        if key not in expected or key in ledger: raise E13PreflightError("E13 completed-run identity/duplicate mismatch")
        if record.get("config",{}).get("data",{}).get("target_access_mode")=="TEST": raise E13PreflightError("E13 Test-scoped run rejected")
        expected_cfg=candidate_config(root,next(v for v,cid in CANDIDATE_IDS.items() if cid==record['candidate_id']))
        actual=deepcopy(record["config"]); stage=record["sweep_stage"][-1]; epoch=record.get("best_epoch")
        if not isinstance(epoch,int) or epoch<=0: raise E13PreflightError("E13 completed run has invalid selected/refit epoch")
        if stage=="B": expected_cfg["training"].update({"max_epochs":epoch,"early_stopping_enabled":False,"early_stopping_patience":0})
        for name in ("rolling_origin_candidate_id","rolling_origin_fold_id","rolling_origin_stage","rolling_origin_metric_population_fingerprint","rolling_origin_metric_population_count"): actual["lineage"].pop(name,None); expected_cfg["lineage"].pop(name,None)
        if actual!=expected_cfg or record.get("config_fingerprint")!=compute_config_fingerprint(record["config"]): raise E13PreflightError("E13 completed scientific config changed")
        checkpoint_name="best_checkpoint.pt" if stage=="A" else "refit_final.pt"; run_dir=root/"artifacts/model_improvement_v2/experiments/E13/runs"/record["run_id"]; checkpoint=run_dir/"checkpoints"/checkpoint_name
        registered=[a for a in record.get("artifacts",[]) if Path(a.get("artifact_path","")).name==checkpoint_name]
        if not checkpoint.is_file() or len(registered)!=1 or _sha(checkpoint)!=registered[0].get("sha256"): raise E13PreflightError("E13 checkpoint provenance mismatch")
        if not (run_dir/"training_history.csv").is_file(): raise E13PreflightError("E13 completed run history missing")
        ledger[key]=record["run_id"]; hashes[key]=_sha(checkpoint)
    missing_stage_a=sorted(stage_a_expected-set(ledger))
    if missing_stage_a: raise E13PreflightError(f"E13 recovery refuses missing Stage A: {missing_stage_a}")
    completed={r["run_id"]:r for r in records if r.get("status")=="COMPLETED"}
    for key in set(ledger)-stage_a_expected:
        parent=key[:-1]+"A"
        if parent not in ledger: raise E13PreflightError(f"E13 completed Stage B has no verified Stage A: {key}")
        if completed[ledger[key]].get("best_epoch")!=completed[ledger[parent]].get("best_epoch"): raise E13PreflightError(f"E13 Stage-B refit epoch mismatch: {key}")
    missing=sorted(expected-set(ledger))
    if any(not key.endswith("_B") for key in missing): raise E13PreflightError("E13 recovery may train missing Stage B only")
    return {"status":"PASS","locked_run_ids":ledger,"checkpoint_sha256":hashes,"missing_canonical_runs":missing,"noncompleted_evidence":noncompleted,"new_training_run_ids_allowed":bool(missing),"training_reexecution_allowed":False,"test_status":"NOT_ACCESSED"}
def build_resume_context(root:Path,document:Mapping[str,Any]):
    ctx=build_context(root,document); audit=audit_completed_run_ledger(root); ctx.reuse_completed_runs=True; ctx.reuse_completed_run_ids=dict(audit["locked_run_ids"]); ctx.allow_partial_stage_b_training=bool(audit["missing_canonical_runs"]); return ctx,audit
def evaluate_promotion(metrics:Mapping[str,float],folds:list[float])->dict[str,Any]:
    gates={"rmse_improvement":CONTROL_METRICS["rmse_wh"]-float(metrics["rmse_wh"])>=0.10,"mae_guardrail":float(metrics["mae_wh"])-CONTROL_METRICS["mae_wh"]<=0.25,"worst_fold_rmse_guardrail":max(folds)-max(CONTROL_FOLDS)<=0.50,"fold_rmse_std_guardrail":statistics.pstdev(folds)-statistics.pstdev(CONTROL_FOLDS)<=0.50}
    return {"gates":gates,"eligible_for_human_promotion":all(gates.values()),"automatic_promotion":False,"human_decision_required":True}
def _write_success(root:Path,result,preflight:Mapping[str,Any])->None:
    from course_work.utils.artifacts import atomic_write_bytes,canonical_json_bytes
    output=root/"artifacts/model_improvement_v2/experiments/E13"; rows=list(csv.DictReader((output/"rolling_origin_results.csv").open()))
    candidates=[]
    for value,cid in CANDIDATE_IDS.items():
        folds=[float(r["rmse_wh"]) for r in rows if r["candidate_id"]==cid]; metrics=result.pooled_metrics_by_cid[cid]; candidates.append({"candidate_id":cid,"lambda_delta":value,"metrics":metrics,"fold_rmse_wh":folds,"promotion_evaluation":evaluate_promotion(metrics,folds)})
    docs={"e13_hybrid_loss_comparison.json":{"schema":"MODEL_IMPROVEMENT_V2_E13_COMPARISON-v1","control":{"source":"E01_REUSED_READ_ONLY","metrics":CONTROL_METRICS,"fold_rmse_wh":CONTROL_FOLDS},"challengers":candidates,"decision":"HUMAN_REVIEW_REQUIRED","test_status":"NOT_ACCESSED"},"e13_execution_manifest.json":{"schema":"MODEL_IMPROVEMENT_V2_E13_EXECUTION-v1","run_ids":{"stage_a":{f"{c}:{f}":r for (c,f),r in result.stage_a_run_ids.items()},"stage_b":{f"{c}:{f}":r for (c,f),r in result.stage_b_run_ids.items()}},"preflight":dict(preflight),"test_status":"NOT_ACCESSED"},"e13_algebra_scaling_audit.json":{"formula":load_e13_config(root)["loss_contract"],"fold_local_y_scaling":True,"validation_fitting":False,"test_fitting":False,"status":"PASS"}}
    for name,payload in docs.items(): atomic_write_bytes(output/name,canonical_json_bytes(payload))
def run_official(root:Path|None=None)->int:
    root=root or project_root(); doc=load_e13_config(root); pre=validate_e13_document(doc,root); ctx=build_context(root,doc)
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(ctx)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(root,result,pre); print(result.summary); return 0
def run_resume_stage_c(root:Path|None=None)->int:
    root=root or project_root(); doc=load_e13_config(root); pre=validate_e13_document(doc,root); ctx,audit=build_resume_context(root,doc)
    if audit["missing_canonical_runs"]: raise E13PreflightError("E13 resume-stage-c requires all 18 completed runs; use Human-authorized resume-partial")
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(ctx)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(root,result,pre); print(result.summary); return 0
def run_resume_partial(root:Path|None=None)->int:
    root=root or project_root(); doc=load_e13_config(root); pre=validate_e13_document(doc,root); ctx,audit=build_resume_context(root,doc)
    if not audit["missing_canonical_runs"]: raise E13PreflightError("E13 partial recovery has no missing Stage-B runs")
    from course_work.rolling_origin.real_run import run_real_pipeline
    result=run_real_pipeline(ctx)
    if result.exit_code: print(result.summary,file=sys.stderr); return result.exit_code
    _write_success(root,result,pre); print(result.summary); return 0
def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--experiment",choices=["E13"],required=True); p.add_argument("--mode",choices=["preflight","official","resume-partial","resume-stage-c"],default="preflight"); p.add_argument("--seed",type=int,default=42); p.add_argument("--authorize-training",action="store_true"); a=p.parse_args(argv)
    if a.seed!=42: print("ERROR: E13 seed must be 42",file=sys.stderr); return 2
    if a.mode=="official" and not a.authorize_training: print("REFUSED: E13 official requires --authorize-training",file=sys.stderr); return 3
    if a.mode=="resume-partial" and not a.authorize_training: print("REFUSED: E13 resume-partial requires --authorize-training",file=sys.stderr); return 3
    if a.mode=="resume-stage-c" and a.authorize_training: print("REFUSED: E13 resume-stage-c is no-train and rejects --authorize-training",file=sys.stderr); return 3
    if a.mode=="preflight" and a.authorize_training: return 2
    try:
        if a.mode=="official": return run_official()
        if a.mode=="resume-partial": return run_resume_partial()
        if a.mode=="resume-stage-c": return run_resume_stage_c()
        print(json.dumps(run_preflight(),indent=2,sort_keys=True)); return 0
    except (E13PreflightError,PermissionError,ValueError) as exc: print(f"E13 {a.mode.upper()} FAIL: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
