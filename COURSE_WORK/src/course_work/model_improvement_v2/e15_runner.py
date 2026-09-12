"""E15 moderate proportional width/FFN search; Human-gated training."""
from __future__ import annotations

import argparse,csv,hashlib,json,statistics,sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any,Mapping

from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import assert_no_test_access,assert_one_primary_change,assert_v2_artifact_path
from course_work.model_improvement_v2.e14_runner import bundle_config as e14_bundle_config

EXPERIMENT_ID="E15";EXECUTION_TRACK="MODEL_IMPROVEMENT_V2_E15";REGISTRY_NAMESPACE="V2_E15"
CONFIG_PATH=Path("artifacts/model_improvement_v2/experiments/E15/e15_config_snapshot.json");E14_ROOT=Path("artifacts/model_improvement_v2/experiments/E14")
CONTROL_ID="TR_C2_ALT_LOOKBACK_E14_M1";CONTROL_METRICS={"rmse_wh":58.5534531302616,"mae_wh":26.33779907425144,"r2":0.5970279545320498}
CONTROL_FOLDS=[65.37273386173845,50.772406234902675,58.601149429283645]
WIDTH_BUNDLES={"W96":(96,384,227041),"W128":(128,512,401025)}
CANDIDATE_IDS={"W96":"TR_C2_ALT_LOOKBACK_E15_W96_FFN384","W128":"TR_C2_ALT_LOOKBACK_E15_W128_FFN512"}

class E15PreflightError(RuntimeError):pass
def project_root():return Path(__file__).resolve().parents[3]
def _read(path):
 try:value=json.loads(Path(path).read_text())
 except Exception as exc:raise E15PreflightError(f"Cannot read {path}: {exc}") from exc
 if not isinstance(value,dict):raise E15PreflightError(f"Expected object: {path}")
 return value
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_e15_config(root):return _read(root/CONFIG_PATH)
def incumbent_config(root):return e14_bundle_config(root,"M1")
def width_config(root,key):
 config=deepcopy(incumbent_config(root));width,ffn,_=WIDTH_BUNDLES[key];config["model"].update(d_model=width,ffn_dim=ffn);return config

def _control_audit(root,document):
 decision=_read(root/E14_ROOT/"e14_human_decision.json");comparison=root/E14_ROOT/"e14_bundle_comparison.json";base=incumbent_config(root)
 expected={"rmse_wh":CONTROL_METRICS["rmse_wh"],"mae_wh":CONTROL_METRICS["mae_wh"],"r2":CONTROL_METRICS["r2"],"worst_fold_rmse_wh":max(CONTROL_FOLDS),"fold_rmse_sd_wh":statistics.pstdev(CONTROL_FOLDS)}
 if decision.get("decision")!="PROMOTE_M1" or decision.get("accepted_incumbent")!=CONTROL_ID or decision.get("accepted_config_fingerprint")!=compute_config_fingerprint(base) or decision.get("accepted_metrics")!=expected or decision.get("source_comparison_sha256")!=_sha(comparison) or decision.get("test_status")!="NOT_ACCESSED":raise E15PreflightError("E14 Human promotion lock mismatch")
 control=document.get("accepted_incumbent",{})
 if control.get("candidate_id")!=CONTROL_ID or control.get("config_fingerprint")!=compute_config_fingerprint(base) or control.get("reuse_only") is not True or control.get("test_status")!="NOT_ACCESSED":raise E15PreflightError("E15 control snapshot mismatch")
 records={r["run_id"]:r for r in map(json.loads,(root/E14_ROOT/"registry/experiment_registry.jsonl").read_text().splitlines())}
 for fold,run_id in decision["stage_b_run_ids"].items():
  record=records.get(run_id);checkpoint=root/E14_ROOT/"runs"/run_id/"checkpoints/refit_final.pt"
  if not record or record.get("status")!="COMPLETED" or record.get("candidate_id")!=CONTROL_ID or record.get("sweep_stage")!=f"{fold}_B" or not checkpoint.is_file() or _sha(checkpoint)!=decision["stage_b_checkpoint_sha256"][fold]:raise E15PreflightError(f"E14 M1 checkpoint lineage mismatch: {fold}")
 return {"status":"PASS","config_fingerprint":compute_config_fingerprint(base),"checkpoint_count":3}

def _parameter_count(config):
 from course_work.rolling_origin.real_run import build_model_from_run_config
 model=build_model_from_run_config(config);return sum(parameter.numel() for parameter in model.parameters())

def validate_e15_document(document:Mapping[str,Any],root:Path,materialize_data=True):
 if document.get("experiment_id")!="E15" or document.get("objective")!="MODERATE_PROPORTIONAL_WIDTH_FFN_CAPACITY_SEARCH":raise E15PreflightError("Wrong E15 identity")
 control=_control_audit(root,document);base=incumbent_config(root);rows=document.get("challengers",[])
 if len(rows)!=2 or {row.get("bundle_id") for row in rows}!=set(WIDTH_BUNDLES):raise E15PreflightError("Width bundle matrix mismatch")
 fingerprints={};parameters={}
 for row in rows:
  key=row["bundle_id"];width,ffn,count=WIDTH_BUNDLES[key];config=width_config(root,key);changed=set(assert_one_primary_change("E15",base,config))
  if changed!={"model.d_model","model.ffn_dim"} or (row.get("d_model"),row.get("num_heads"),row.get("ffn_dim"),row.get("num_layers"))!=(width,4,ffn,2) or row.get("candidate_id")!=CANDIDATE_IDS[key] or row.get("config_fingerprint")!=compute_config_fingerprint(config):raise E15PreflightError(f"{key} width/config fingerprint mismatch")
  actual_count=_parameter_count(config)
  if actual_count!=count or row.get("parameter_count")!=count or actual_count>=1_000_000:raise E15PreflightError(f"{key} parameter-count gate failed")
  if config["training"]!=base["training"] or config["data"]!=base["data"] or config["reproducibility"]!=base["reproducibility"] or config["runtime"]!=base["runtime"]:raise E15PreflightError("Non-capacity incumbent field changed")
  assert_no_test_access(split_id=config["data"]["target_access_mode"]);fingerprints[key]=compute_config_fingerprint(config);parameters[key]=actual_count
 if tuple(get_feature_list("FS2_TF1"))!=tuple(get_feature_list(base["data"]["feature_variant_id"])):raise E15PreflightError("Feature order changed")
 fixed=document.get("fixed_contract",{});required={"feature_variant":"FS2_TF1","feature_count":33,"lookback":72,"horizon":1,"boundary_protocol":"WB0_CONTEXT_CARRY_OVER","folds":["RO1","RO2","RO3"],"fold_local_x_scaling":True,"fold_local_y_scaling":True,"prediction_formulation":"DIRECT","prediction_head":"LINEAR","residual_gate":False,"num_heads":4,"num_layers":2,"dropout":.1,"optimizer":"AdamW","learning_rate":2e-4,"batch_size":16,"weight_decay":1e-3,"scheduler":"OFF","loss_policy":"HYBRID_LEVEL_PLUS_DELTA","lambda_delta":.1,"delta_beta_model_space":1.,"gradient_clip_max_norm":1.,"seed":42,"max_epochs":50,"early_stopping_patience":10,"parameter_count_limit_exclusive":1_000_000}
 if fixed!=required:raise E15PreflightError("Fixed E15 scientific contract mismatch")
 policy=document.get("promotion_policy",{});expected_policy={"rmse_min_improvement_wh":.1,"mae_max_degradation_wh":.25,"worst_fold_rmse_max_degradation_wh":.5,"fold_rmse_std_max_degradation_wh":.5,"automatic_promotion":False,"decision_authority":"HUMAN","weighted_composite":False,"test_metrics_allowed":False}
 if policy!=expected_policy:raise E15PreflightError("Promotion policy mismatch")
 orchestration=document.get("orchestration",{})
 if orchestration.get("expected_new_training_runs")!=12 or orchestration.get("control_retrained") is not False or orchestration.get("training_authorized") is not False or orchestration.get("test_access_authorized") is not False:raise E15PreflightError("Orchestration mismatch")
 for key in ("artifact_root","registry_root","run_root"):assert_v2_artifact_path(orchestration[key])
 population=None
 if materialize_data:
  from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
  dataset,evidence,audit=build_v2_pretest_dataset(root,tuple(get_feature_list("FS2_TF1")),experiment_id="E15",feature_variant_id="FS2_TF1")
  if audit.test_rows_read or audit.test_target_ids_seen:raise E15PreflightError("Test firewall failed")
  population={"status":"PASS","population_fingerprint":base["lineage"]["population_fingerprint"],"target_count":len(dataset),"fold_fingerprints":{str(f.fold_id):f.fold_population_fingerprint for f in evidence.folds}}
 return {"experiment":"E15","status":"PASS","accepted_incumbent":CONTROL_ID,"control_reuse_status":control["status"],"challengers":list(WIDTH_BUNDLES),"expected_new_training_runs":12,"feature_variant":"FS2_TF1","feature_count":33,"candidate_config_fingerprints":fingerprints,"parameter_counts":parameters,"parameter_gate":"PASS","parameter_memory_estimate_bytes_fp32":{key:value*4 for key,value in parameters.items()},"population":population,"scaler_lineage":"FOLD_LOCAL_TRAIN_ONLY","recovery_contract":"NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED","test_rows_read":0,"test_target_ids_seen":0,"test_status":"NOT_ACCESSED","training_executed":False}

def run_preflight(root=None):root=root or project_root();return validate_e15_document(load_e15_config(root),root)
def build_candidates(root):
 from course_work.rolling_origin.candidate_loader import CandidateSpec
 return tuple(CandidateSpec(candidate_id=CANDIDATE_IDS[key],model_family="TRANSFORMER_ENCODER",shortlist_position=index,config=width_config(root,key),config_fingerprint=compute_config_fingerprint(width_config(root,key)),feature_variant_id="FS2_TF1",target_scaling_option="YS1",lookback_steps=72,boundary_protocol="WB0_CONTEXT_CARRY_OVER",source_phase="V2_E15_IMMUTABLE_SNAPSHOT",candidate_role="LOCKED_PROPORTIONAL_WIDTH_BUNDLE") for index,key in enumerate(WIDTH_BUNDLES,1))
def build_context(root,document):
 from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset,load_phase44_fold_evidence
 from course_work.rolling_origin.real_run import RunContext
 validate_e15_document(document,root,False);candidates=build_candidates(root);cache={};output=root/"artifacts/model_improvement_v2/experiments/E15"
 def factory(candidate,_fold):
  if candidate.candidate_id not in CANDIDATE_IDS.values():raise E15PreflightError("Unknown E15 candidate")
  if "data" not in cache:cache["data"]=build_v2_pretest_dataset(root,tuple(get_feature_list("FS2_TF1")),experiment_id="E15",feature_variant_id="FS2_TF1")
  dataset,_e,audit=cache["data"]
  if audit.test_rows_read or audit.test_target_ids_seen:raise PermissionError("E15 Test firewall")
  return dataset
 evidence=load_phase44_fold_evidence(root);base=incumbent_config(root);upstream={"lineage":deepcopy(base["lineage"]),"feature_sets":{"variant_feature_counts":{"FS2_TF1":33},"variant_fingerprints":{"FS2_TF1":base["lineage"]["feature_fingerprint"]}},"window_fingerprints":{}}
 return RunContext(project_root=root,transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json",lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json",phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json",artifact_dir=output,registry_root=output/"registry",run_root=output/"runs",scientific_max_epochs=50,scientific_patience=10,seed=42,device=str(base["runtime"]["device_type"]),dataset_factory=factory,candidate_specs=candidates,registry_namespace=REGISTRY_NAMESPACE,seed_before_model_construction=True,validated_registry_lifecycle=True,execution_track=EXECUTION_TRACK,registry_upstream_context=upstream,robase_train_ids=evidence.train_ids,robase_val_ids=evidence.validation_ids,apply_fold_x_scaling=True)

def audit_completed_ledger(root):
 path=root/"artifacts/model_improvement_v2/experiments/E15/registry/experiment_registry.jsonl"
 if not path.is_file():raise E15PreflightError("E15 completed ledger unavailable")
 records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()];expected={f"{cid}:RO{fold}_{stage}" for cid in CANDIDATE_IDS.values() for fold in (1,2,3) for stage in "AB"};locked={};hashes={}
 for record in records:
  if record.get("status")!="COMPLETED":continue
  key=f"{record.get('candidate_id')}:{record.get('sweep_stage')}"
  if key not in expected or key in locked:raise E15PreflightError("E15 completed identity/duplicate mismatch")
  stage=record["sweep_stage"][-1];name="best_checkpoint.pt" if stage=="A" else "refit_final.pt";checkpoint=root/"artifacts/model_improvement_v2/experiments/E15/runs"/record["run_id"]/"checkpoints"/name;artifacts=[a for a in record.get("artifacts",[]) if Path(a.get("artifact_path","")).name==name]
  if not checkpoint.is_file() or len(artifacts)!=1 or _sha(checkpoint)!=artifacts[0].get("sha256") or record.get("config_fingerprint")!=compute_config_fingerprint(record.get("config",{})) or record.get("config",{}).get("data",{}).get("target_access_mode")=="TEST":raise E15PreflightError("E15 checkpoint/config/Test audit failed")
  locked[key]=record["run_id"];hashes[key]=_sha(checkpoint)
 if set(locked)!=expected:raise E15PreflightError("E15 no-train resume requires exact 12 completed runs")
 return {"status":"PASS","locked_run_ids":locked,"checkpoint_sha256":hashes,"training_reexecution_allowed":False,"test_status":"NOT_ACCESSED"}
def build_resume_context(root,document):
 context=build_context(root,document);audit=audit_completed_ledger(root);context.reuse_completed_runs=True;context.reuse_completed_run_ids=dict(audit["locked_run_ids"]);return context,audit
def _promotion(metrics,folds):
 gates={"rmse_improvement":CONTROL_METRICS["rmse_wh"]-metrics["rmse_wh"]>=.1,"mae_guardrail":metrics["mae_wh"]-CONTROL_METRICS["mae_wh"]<=.25,"worst_fold_rmse_guardrail":max(folds)-max(CONTROL_FOLDS)<=.5,"fold_rmse_std_guardrail":statistics.pstdev(folds)-statistics.pstdev(CONTROL_FOLDS)<=.5};return {"gates":gates,"eligible_for_human_promotion":all(gates.values()),"automatic_promotion":False,"human_decision_required":True}
def _write_success(root,result,preflight):
 from course_work.utils.artifacts import atomic_write_bytes,canonical_json_bytes
 output=root/"artifacts/model_improvement_v2/experiments/E15";rows=list(csv.DictReader((output/"rolling_origin_results.csv").open()));challengers=[]
 for key,candidate_id in CANDIDATE_IDS.items():
  folds=[float(row["rmse_wh"]) for row in rows if row["candidate_id"]==candidate_id];metrics=result.pooled_metrics_by_cid[candidate_id];challengers.append({"bundle_id":key,"candidate_id":candidate_id,"metrics":metrics,"fold_rmse_wh":folds,"parameter_count":WIDTH_BUNDLES[key][2],"promotion_evaluation":_promotion(metrics,folds)})
 comparison={"schema":"MODEL_IMPROVEMENT_V2_E15_COMPARISON-v1","control":{"source":"E14_M1_REUSED_READ_ONLY","candidate_id":CONTROL_ID,"metrics":CONTROL_METRICS,"fold_rmse_wh":CONTROL_FOLDS},"challengers":challengers,"decision":"HUMAN_REVIEW_REQUIRED","test_status":"NOT_ACCESSED"}
 manifest={"schema":"MODEL_IMPROVEMENT_V2_E15_EXECUTION-v1","run_ids":{"stage_a":{f"{c}:{f}":r for (c,f),r in result.stage_a_run_ids.items()},"stage_b":{f"{c}:{f}":r for (c,f),r in result.stage_b_run_ids.items()}},"preflight":dict(preflight),"test_status":"NOT_ACCESSED"}
 ledger=[json.loads(line) for line in (output/"registry/experiment_registry.jsonl").read_text().splitlines() if line.strip()];runtime=[]
 for record in ledger:
  if record.get("status")!="COMPLETED":continue
  started=record.get("started_at");completed=record.get("completed_at");seconds=(datetime.fromisoformat(completed)-datetime.fromisoformat(started)).total_seconds() if started and completed else None;key=next(key for key,value in CANDIDATE_IDS.items() if value==record["candidate_id"])
  runtime.append({"run_id":record["run_id"],"candidate_id":record["candidate_id"],"fold":record["sweep_stage"][:3],"stage":record["sweep_stage"][-1],"duration_seconds":seconds,"parameter_count":WIDTH_BUNDLES[key][2],"parameter_memory_estimate_bytes_fp32":WIDTH_BUNDLES[key][2]*4,"peak_runtime_memory":"NOT_CAPTURED_BY_CANONICAL_ENGINE"})
 report={"schema":"MODEL_IMPROVEMENT_V2_E15_PARAMETER_RUNTIME-v1","runs":runtime,"parameter_limit_exclusive":1_000_000,"parameter_gate":"PASS","memory_reporting":"STATIC_PARAMETER_BYTES_ONLY_NO_FABRICATED_PEAK_MEMORY","test_status":"NOT_ACCESSED"}
 for name,value in (("e15_capacity_comparison.json",comparison),("e15_execution_manifest.json",manifest),("e15_parameter_runtime_report.json",report)):atomic_write_bytes(output/name,canonical_json_bytes(value))
def run_official(root=None):
 root=root or project_root();document=load_e15_config(root);preflight=validate_e15_document(document,root);context=build_context(root,document);from course_work.rolling_origin.real_run import run_real_pipeline;result=run_real_pipeline(context)
 if result.exit_code:print(result.summary,file=sys.stderr);return result.exit_code
 _write_success(root,result,preflight);print(result.summary);return 0
def run_resume_stage_c(root=None):
 root=root or project_root();document=load_e15_config(root);preflight=validate_e15_document(document,root);context,_=build_resume_context(root,document);from course_work.rolling_origin.real_run import run_real_pipeline;result=run_real_pipeline(context)
 if result.exit_code:print(result.summary,file=sys.stderr);return result.exit_code
 _write_success(root,result,preflight);print(result.summary);return 0
def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument("--experiment",choices=["E15"],required=True);parser.add_argument("--mode",choices=["preflight","official","resume-stage-c"],default="preflight");parser.add_argument("--seed",type=int,default=42);parser.add_argument("--authorize-training",action="store_true");args=parser.parse_args(argv)
 if args.seed!=42:return 2
 if args.mode=="official" and not args.authorize_training:print("REFUSED: E15 official requires --authorize-training",file=sys.stderr);return 3
 if args.mode=="preflight" and args.authorize_training:return 2
 try:
  if args.mode=="official":return run_official()
  if args.mode=="resume-stage-c":
   if args.authorize_training:raise E15PreflightError("E15 resume-stage-c is no-train")
   return run_resume_stage_c()
  print(json.dumps(run_preflight(),indent=2,sort_keys=True));return 0
 except (E15PreflightError,PermissionError,ValueError) as exc:print(f"E15 {args.mode.upper()} FAIL: {exc}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
