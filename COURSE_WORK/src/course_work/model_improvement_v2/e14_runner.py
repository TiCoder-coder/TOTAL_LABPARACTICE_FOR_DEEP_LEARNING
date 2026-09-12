"""E14 locked multi-factor bundle experiment; Human-gated training."""
from __future__ import annotations
import argparse,csv,hashlib,json,statistics,sys
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Any,Mapping
from course_work.data.feature_sets import get_feature_list
from course_work.experiments.registry import compute_config_fingerprint
from course_work.model_improvement_v2.contracts import assert_no_test_access,assert_one_primary_change,assert_v2_artifact_path
from course_work.model_improvement_v2.e13_runner import candidate_config as e13_config
EXPERIMENT_ID="E14";EXECUTION_TRACK="MODEL_IMPROVEMENT_V2_E14";REGISTRY_NAMESPACE="V2_E14";CONFIG_PATH=Path("artifacts/model_improvement_v2/experiments/E14/e14_config_snapshot.json");E13_ROOT=Path("artifacts/model_improvement_v2/experiments/E13")
CONTROL_ID="TR_C2_ALT_LOOKBACK_E13_LAMBDA_010";CONTROL_METRICS={"rmse_wh":59.60512359272541,"mae_wh":26.612618306519018,"r2":0.5824225085537367}
CONTROL_FOLDS=[69.03091214638307,50.53768355426747,57.78210074486984]
BUNDLES={"M1":(2e-4,16,.10,1.),"M2":(3e-4,16,.05,1.),"M3":(5e-4,32,.05,1.),"M4":(5e-4,64,.10,1.),"M5":(3e-4,32,.20,.5)};CANDIDATE_IDS={k:f"TR_C2_ALT_LOOKBACK_E14_{k}" for k in BUNDLES}
class E14PreflightError(RuntimeError):pass
def project_root():return Path(__file__).resolve().parents[3]
def _read(p):
 try:o=json.loads(Path(p).read_text())
 except Exception as e:raise E14PreflightError(f"Cannot read {p}: {e}") from e
 if not isinstance(o,dict):raise E14PreflightError(f"Expected object: {p}")
 return o
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_e14_config(root):return _read(root/CONFIG_PATH)
def incumbent_config(root):return e13_config(root,.10)
def bundle_config(root,key):
 c=deepcopy(incumbent_config(root));lr,batch,drop,clip=BUNDLES[key];c["training"].update(learning_rate=lr,batch_size=batch,gradient_clip_max_norm=clip);c["model"]["dropout"]=drop;return c
def _control_audit(root,doc):
 d=_read(root/E13_ROOT/"e13_human_decision.json");comparison=root/E13_ROOT/"e13_hybrid_loss_comparison.json";base=incumbent_config(root)
 if d.get("decision")!="PROMOTE" or d.get("accepted_incumbent")!=CONTROL_ID or d.get("lambda_delta")!=.1 or d.get("accepted_metrics")!=CONTROL_METRICS or d.get("test_status")!="NOT_ACCESSED" or d.get("source_comparison_sha256")!=_sha(comparison):raise E14PreflightError("E13 Human promotion lock mismatch")
 c=doc.get("accepted_incumbent",{})
 if c.get("candidate_id")!=CONTROL_ID or c.get("metrics")!=CONTROL_METRICS or c.get("reuse_only") is not True or c.get("config_fingerprint")!=compute_config_fingerprint(base):raise E14PreflightError("E14 control mismatch")
 records={r["run_id"]:r for r in map(json.loads,(root/E13_ROOT/"registry/experiment_registry.jsonl").read_text().splitlines())}
 for fold,rid in d["stage_b_run_ids"].items():
  r=records.get(rid);p=root/E13_ROOT/"runs"/rid/"checkpoints/refit_final.pt"
  if not r or r.get("status")!="COMPLETED" or r.get("candidate_id")!=CONTROL_ID or r.get("sweep_stage")!=f"{fold}_B" or not p.is_file() or _sha(p)!=d["stage_b_checkpoint_sha256"][fold]:raise E14PreflightError(f"E13 checkpoint lineage mismatch: {fold}")
 return {"status":"PASS","candidate_id":CONTROL_ID,"config_fingerprint":compute_config_fingerprint(base),"checkpoint_count":3}
def validate_e14_document(doc:Mapping[str,Any],root:Path,materialize_data=True):
 if doc.get("experiment_id")!="E14" or doc.get("objective")!="BOUNDED_STAGED_MULTI_FACTOR_TRAINING_SEARCH":raise E14PreflightError("Wrong E14 identity")
 control=_control_audit(root,doc);rows=doc.get("bundles",[]);base=incumbent_config(root)
 if len(rows)!=5 or {x.get("bundle_id") for x in rows}!=set(BUNDLES):raise E14PreflightError("Bundle matrix mismatch")
 fps={}
 for row in rows:
  k=row["bundle_id"];values=BUNDLES[k];cfg=bundle_config(root,k);changed=set(assert_one_primary_change("E14",base,cfg));expected={p for p,v in zip(("training.learning_rate","training.batch_size","model.dropout","training.gradient_clip_max_norm"),values) if v!={"training.learning_rate":base["training"]["learning_rate"],"training.batch_size":base["training"]["batch_size"],"model.dropout":base["model"]["dropout"],"training.gradient_clip_max_norm":base["training"]["gradient_clip_max_norm"]}[p]}
  if changed!=expected or tuple(row[x] for x in ("learning_rate","batch_size","dropout","gradient_clip_max_norm"))!=values or row.get("candidate_id")!=CANDIDATE_IDS[k] or row.get("config_fingerprint")!=compute_config_fingerprint(cfg):raise E14PreflightError(f"{k} bundle/fingerprint mismatch")
  if cfg["training"].get("loss_policy")!="HYBRID_LEVEL_PLUS_DELTA" or cfg["training"].get("lambda_delta")!=.1 or cfg["training"].get("optimizer_name")!="AdamW" or cfg["training"].get("weight_decay")!=1e-3 or cfg["training"].get("scheduler_name") is not None or cfg["data"].get("feature_count")!=33:raise E14PreflightError("Incumbent fields changed")
  assert_no_test_access(split_id=cfg["data"]["target_access_mode"]);fps[k]=compute_config_fingerprint(cfg)
 if tuple(get_feature_list("FS2_TF1"))!=tuple(get_feature_list(base["data"]["feature_variant_id"])):raise E14PreflightError("Feature order changed")
 orch=doc["orchestration"]
 if orch.get("expected_new_training_runs")!=30 or orch.get("control_retrained") is not False or orch.get("training_authorized") is not False or orch.get("test_access_authorized") is not False:raise E14PreflightError("Orchestration mismatch")
 fixed=doc.get("fixed_contract",{})
 expected_fixed={"feature_variant":"FS2_TF1","feature_count":33,"lookback":72,"horizon":1,"boundary_protocol":"WB0_CONTEXT_CARRY_OVER","folds":["RO1","RO2","RO3"],"fold_local_x_scaling":True,"fold_local_y_scaling":True,"optimizer":"AdamW","weight_decay":1e-3,"scheduler":"OFF","loss_policy":"HYBRID_LEVEL_PLUS_DELTA","lambda_delta":.1,"delta_beta_model_space":1.,"seed":42,"max_epochs":50,"early_stopping_patience":10}
 if fixed!=expected_fixed:raise E14PreflightError("Fixed scientific contract mismatch")
 promotion=doc.get("promotion_policy",{})
 expected_promotion={"rmse_min_improvement_wh":.1,"mae_max_degradation_wh":.25,"worst_fold_rmse_max_degradation_wh":.5,"fold_rmse_std_max_degradation_wh":.5,"automatic_promotion":False,"decision_authority":"HUMAN","weighted_composite":False,"test_metrics_allowed":False}
 if promotion!=expected_promotion:raise E14PreflightError("Promotion policy mismatch")
 for key in ("artifact_root","registry_root","run_root"):assert_v2_artifact_path(orch[key])
 pop=None
 if materialize_data:
  from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
  ds,e,a=build_v2_pretest_dataset(root,tuple(get_feature_list("FS2_TF1")),experiment_id="E14",feature_variant_id="FS2_TF1")
  if a.test_rows_read or a.test_target_ids_seen:raise E14PreflightError("Test firewall failed")
  pop={"status":"PASS","population_fingerprint":base["lineage"]["population_fingerprint"],"target_count":len(ds),"fold_fingerprints":{str(f.fold_id):f.fold_population_fingerprint for f in e.folds}}
 return {"experiment":"E14","status":"PASS","accepted_incumbent":CONTROL_ID,"control_reuse_status":control["status"],"bundle_ids":list(BUNDLES),"expected_new_training_runs":30,"feature_variant":"FS2_TF1","feature_count":33,"candidate_config_fingerprints":fps,"population":pop,"scaler_lineage":"FOLD_LOCAL_TRAIN_ONLY","equal_budget_audit":{"status":"PASS","basis":"IDENTICAL_MAX_EPOCH_AND_EARLY_STOPPING_BUDGET","max_epochs":50,"patience":10,"batch_size_is_registered_bundle_factor":True},"promotion_policy_status":"PASS","expected_outputs":["rolling_origin_results.csv","rolling_origin_inner_best_epochs.csv","rolling_origin_fold_metrics.csv","rolling_origin_pooled_metrics.csv","e14_bundle_comparison.json","e14_execution_manifest.json","e14_equal_budget_audit.json","e14_parameter_runtime_report.json","e14_lr_gradient_trace_audit.json","e14_ranking_guardrails.json"],"recovery_contract":"NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED","test_rows_read":0,"test_target_ids_seen":0,"test_status":"NOT_ACCESSED","training_executed":False}
def run_preflight(root=None):root=root or project_root();return validate_e14_document(load_e14_config(root),root)
def build_candidates(root):
 from course_work.rolling_origin.candidate_loader import CandidateSpec
 return tuple(CandidateSpec(candidate_id=CANDIDATE_IDS[k],model_family="TRANSFORMER_ENCODER",shortlist_position=i,config=bundle_config(root,k),config_fingerprint=compute_config_fingerprint(bundle_config(root,k)),feature_variant_id="FS2_TF1",target_scaling_option="YS1",lookback_steps=72,boundary_protocol="WB0_CONTEXT_CARRY_OVER",source_phase="V2_E14_IMMUTABLE_SNAPSHOT",candidate_role="LOCKED_MULTI_FACTOR_BUNDLE") for i,k in enumerate(BUNDLES,1))
def build_context(root,doc):
 from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset,load_phase44_fold_evidence
 from course_work.rolling_origin.real_run import RunContext
 validate_e14_document(doc,root,False);candidates=build_candidates(root);cache={};out=root/"artifacts/model_improvement_v2/experiments/E14"
 def factory(c,_f):
  if c.candidate_id not in CANDIDATE_IDS.values():raise E14PreflightError("Unknown E14 candidate")
  if "data" not in cache:cache["data"]=build_v2_pretest_dataset(root,tuple(get_feature_list("FS2_TF1")),experiment_id="E14",feature_variant_id="FS2_TF1")
  ds,_e,a=cache["data"]
  if a.test_rows_read or a.test_target_ids_seen:raise PermissionError("E14 Test firewall")
  return ds
 e=load_phase44_fold_evidence(root);base=incumbent_config(root);up={"lineage":deepcopy(base["lineage"]),"feature_sets":{"variant_feature_counts":{"FS2_TF1":33},"variant_fingerprints":{"FS2_TF1":base["lineage"]["feature_fingerprint"]}},"window_fingerprints":{}}
 return RunContext(project_root=root,transformer_shortlist_path=root/"artifacts/candidate_synthesis/transformer_candidate_shortlist.json",lstm_handoff_path=root/"artifacts/lstm_tuning/phase44_rolling_origin_lstm_handoff.json",phase_42_signoff_path=root/"artifacts/candidate_synthesis/phase_42_signoff.json",phase_43_signoff_path=root/"artifacts/lstm_tuning/phase_43_signoff.json",artifact_dir=out,registry_root=out/"registry",run_root=out/"runs",scientific_max_epochs=50,scientific_patience=10,seed=42,device=str(base["runtime"]["device_type"]),dataset_factory=factory,candidate_specs=candidates,registry_namespace=REGISTRY_NAMESPACE,seed_before_model_construction=True,validated_registry_lifecycle=True,execution_track=EXECUTION_TRACK,registry_upstream_context=up,robase_train_ids=e.train_ids,robase_val_ids=e.validation_ids,apply_fold_x_scaling=True)
def audit_completed_ledger(root):
 p=root/"artifacts/model_improvement_v2/experiments/E14/registry/experiment_registry.jsonl"
 if not p.is_file():raise E14PreflightError("E14 completed ledger unavailable")
 records=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];expected={f"{cid}:RO{f}_{s}" for cid in CANDIDATE_IDS.values() for f in (1,2,3) for s in ('A','B')};locked={};hashes={}
 for r in records:
  if r.get('status')!='COMPLETED':continue
  key=f"{r.get('candidate_id')}:{r.get('sweep_stage')}"
  if key not in expected or key in locked:raise E14PreflightError("E14 completed identity/duplicate mismatch")
  stage=r['sweep_stage'][-1];name='best_checkpoint.pt' if stage=='A' else 'refit_final.pt';ck=root/"artifacts/model_improvement_v2/experiments/E14/runs"/r['run_id']/"checkpoints"/name;arts=[a for a in r.get('artifacts',[]) if Path(a.get('artifact_path','')).name==name]
  bundle=next(k for k,cid in CANDIDATE_IDS.items() if cid==r.get('candidate_id'));cfg=bundle_config(root,bundle);actual=r.get('config',{});expected=deepcopy(cfg)
  if stage=='A':
   expected['lineage']['rolling_origin_stage']='A';expected['lineage']['rolling_origin_metric_population_fingerprint']=actual.get('lineage',{}).get('rolling_origin_metric_population_fingerprint');expected['lineage']['rolling_origin_metric_population_count']=actual.get('lineage',{}).get('rolling_origin_metric_population_count')
   if not r.get('best_epoch') or not expected['lineage']['rolling_origin_metric_population_fingerprint'] or not expected['lineage']['rolling_origin_metric_population_count']:raise E14PreflightError("E14 Stage-A selection/population evidence missing")
  else:
   akey=f"{r.get('candidate_id')}:{r['sweep_stage'][:-1]}A";arun=next((x for x in records if x.get('status')=='COMPLETED' and f"{x.get('candidate_id')}:{x.get('sweep_stage')}"==akey),None)
   if not arun or r.get('best_epoch')!=arun.get('best_epoch'):raise E14PreflightError("E14 Stage-B refit epoch lineage mismatch")
   expected['lineage']['rolling_origin_stage']='B';expected['training'].update(max_epochs=arun['best_epoch'],early_stopping_enabled=False,early_stopping_patience=0)
  if not ck.is_file() or len(arts)!=1 or _sha(ck)!=arts[0].get('sha256') or r.get('config_fingerprint')!=compute_config_fingerprint(actual) or actual!=expected or actual.get('data',{}).get('target_access_mode')=='TEST':raise E14PreflightError("E14 completed checkpoint/config/Test audit failed")
  locked[key]=r['run_id'];hashes[key]=_sha(ck)
 if set(locked)!=expected:raise E14PreflightError("E14 no-train resume requires exact 30 completed runs")
 return {'status':'PASS','locked_run_ids':locked,'checkpoint_sha256':hashes,'training_reexecution_allowed':False,'test_status':'NOT_ACCESSED'}
def build_resume_context(root,doc):
 ctx=build_context(root,doc);audit=audit_completed_ledger(root);ctx.reuse_completed_runs=True;ctx.reuse_completed_run_ids=dict(audit['locked_run_ids']);return ctx,audit
def _promotion(metrics,folds):
 gates={'rmse_improvement':CONTROL_METRICS['rmse_wh']-metrics['rmse_wh']>=.1,'mae_guardrail':metrics['mae_wh']-CONTROL_METRICS['mae_wh']<=.25,'worst_fold_rmse_guardrail':max(folds)-max(CONTROL_FOLDS)<=.5,'fold_rmse_std_guardrail':statistics.pstdev(folds)-statistics.pstdev(CONTROL_FOLDS)<=.5};return {'gates':gates,'eligible_for_human_promotion':all(gates.values()),'automatic_promotion':False,'human_decision_required':True}
def _write_success(root,result,pre):
 from course_work.utils.artifacts import atomic_write_bytes,canonical_json_bytes
 out=root/"artifacts/model_improvement_v2/experiments/E14";rows=list(csv.DictReader((out/'rolling_origin_results.csv').open()));chall=[]
 for key,cid in CANDIDATE_IDS.items():
  folds=[float(r['rmse_wh']) for r in rows if r['candidate_id']==cid];m=result.pooled_metrics_by_cid[cid];chall.append({'bundle_id':key,'candidate_id':cid,'metrics':m,'fold_rmse_wh':folds,'promotion_evaluation':_promotion(m,folds)})
 comparison={'schema':'MODEL_IMPROVEMENT_V2_E14_COMPARISON-v1','control':{'source':'E13_REUSED_READ_ONLY','candidate_id':CONTROL_ID,'metrics':CONTROL_METRICS,'fold_rmse_wh':CONTROL_FOLDS},'challengers':chall,'decision':'HUMAN_REVIEW_REQUIRED','test_status':'NOT_ACCESSED'}
 manifest={'schema':'MODEL_IMPROVEMENT_V2_E14_EXECUTION-v1','run_ids':{'stage_a':{f'{c}:{f}':r for (c,f),r in result.stage_a_run_ids.items()},'stage_b':{f'{c}:{f}':r for (c,f),r in result.stage_b_run_ids.items()}},'preflight':dict(pre),'test_status':'NOT_ACCESSED'}
 ranking=sorted(chall,key=lambda x:x['metrics']['rmse_wh']);ranking_doc={'schema':'MODEL_IMPROVEMENT_V2_E14_RANKING-v1','ranking':[{'rank':i,'bundle_id':x['bundle_id'],'candidate_id':x['candidate_id'],'metrics':x['metrics'],'promotion_evaluation':x['promotion_evaluation']} for i,x in enumerate(ranking,1)],'selection_rule':'BEST_ELIGIBLE_POOLED_RMSE_AFTER_GUARDRAILS','automatic_promotion':False,'human_decision_required':True,'test_status':'NOT_ACCESSED'}
 budget={'schema':'MODEL_IMPROVEMENT_V2_E14_EQUAL_BUDGET-v1','status':'PASS','basis':'IDENTICAL_MAX_EPOCH_AND_EARLY_STOPPING_BUDGET','max_epochs':50,'early_stopping_patience':10,'stage_b_exact_selected_epoch':True,'batch_size_is_registered_bundle_factor':True,'equal_optimizer_step_count_claimed':False,'test_status':'NOT_ACCESSED'}
 ledger=[json.loads(x) for x in (out/'registry/experiment_registry.jsonl').read_text().splitlines() if x.strip()];runtime=[];traces=[]
 for r in ledger:
  if r.get('status')!='COMPLETED' or r.get('candidate_id') not in CANDIDATE_IDS.values():continue
  started=r.get('started_at');completed=r.get('completed_at');seconds=(datetime.fromisoformat(completed)-datetime.fromisoformat(started)).total_seconds() if started and completed else None;run_dir=out/'runs'/r['run_id'];history=list(csv.DictReader((run_dir/'training_history.csv').open()));stage=r['sweep_stage'][-1];params=None;grad=None
  if stage=='A':
   metrics_doc=_read(run_dir/'metrics/best_validation_metrics.json');grad=metrics_doc.get('gradient_diagnostics')
  else:
   params=_read(run_dir/'refit_status.json').get('trainable_parameters')
  runtime.append({'run_id':r['run_id'],'candidate_id':r['candidate_id'],'fold':r['sweep_stage'][:-2],'stage':stage,'duration_seconds':seconds,'trainable_parameters':params,'epochs_completed':len(history)})
  traces.append({'run_id':r['run_id'],'candidate_id':r['candidate_id'],'fold':r['sweep_stage'][:-2],'stage':stage,'lr_used_for_epoch':[float(x['lr_used_for_epoch']) for x in history],'gradient_diagnostics':grad,'gradient_scope':'STAGE_A_AGGREGATE' if stage=='A' else 'NOT_EMITTED_BY_CANONICAL_REFIT_ENGINE'})
 runtime_doc={'schema':'MODEL_IMPROVEMENT_V2_E14_PARAMETER_RUNTIME-v1','runs':runtime,'test_status':'NOT_ACCESSED'};trace_doc={'schema':'MODEL_IMPROVEMENT_V2_E14_LR_GRADIENT_TRACE-v1','runs':traces,'lr_trace_status':'PASS','gradient_diagnostic_scope':'CANONICAL_STAGE_A_AGGREGATE','test_status':'NOT_ACCESSED'}
 for name,obj in (('e14_bundle_comparison.json',comparison),('e14_execution_manifest.json',manifest),('e14_equal_budget_audit.json',budget),('e14_parameter_runtime_report.json',runtime_doc),('e14_lr_gradient_trace_audit.json',trace_doc),('e14_ranking_guardrails.json',ranking_doc)):atomic_write_bytes(out/name,canonical_json_bytes(obj))
def run_official(root=None):
 root=root or project_root();doc=load_e14_config(root);pre=validate_e14_document(doc,root);ctx=build_context(root,doc);from course_work.rolling_origin.real_run import run_real_pipeline;result=run_real_pipeline(ctx)
 if result.exit_code:print(result.summary,file=sys.stderr);return result.exit_code
 _write_success(root,result,pre);print(result.summary);return 0
def run_resume_stage_c(root=None):
 root=root or project_root();doc=load_e14_config(root);pre=validate_e14_document(doc,root);ctx,_=build_resume_context(root,doc);from course_work.rolling_origin.real_run import run_real_pipeline;result=run_real_pipeline(ctx)
 if result.exit_code:print(result.summary,file=sys.stderr);return result.exit_code
 _write_success(root,result,pre);print(result.summary);return 0
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--experiment",choices=["E14"],required=True);p.add_argument("--mode",choices=["preflight","official","resume-stage-c"],default="preflight");p.add_argument("--seed",type=int,default=42);p.add_argument("--authorize-training",action="store_true");a=p.parse_args(argv)
 if a.seed!=42:return 2
 if a.mode=="official" and not a.authorize_training:print("REFUSED: E14 official requires --authorize-training",file=sys.stderr);return 3
 if a.mode=="preflight" and a.authorize_training:return 2
 try:
  if a.mode=="official":return run_official()
  if a.mode=="resume-stage-c":
   if a.authorize_training:raise E14PreflightError("E14 resume-stage-c is no-train")
   return run_resume_stage_c()
  print(json.dumps(run_preflight(),indent=2,sort_keys=True));return 0
 except (E14PreflightError,PermissionError,ValueError) as e:print(f"E14 {a.mode.upper()} FAIL: {e}",file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
