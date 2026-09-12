import pytest
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e14_runner import BUNDLES,CANDIDATE_IDS,build_candidates,build_context,bundle_config,incumbent_config,load_e14_config,main,project_root,run_preflight
from course_work.experiments.registry import ExperimentRegistry,validate_run_config
from course_work.rolling_origin.real_run import assert_context_invariants
def test_human_promoted_e13_is_control():
 p=run_preflight(project_root());assert p['accepted_incumbent']=='TR_C2_ALT_LOOKBACK_E13_LAMBDA_010';assert p['control_reuse_status']=='PASS';assert p['test_status']=='NOT_ACCESSED'
def test_exact_locked_bundle_matrix_and_primary_factor():
 root=project_root();base=incumbent_config(root)
 for key,values in BUNDLES.items():
  cfg=bundle_config(root,key);assert assert_one_primary_change('E14',base,cfg);assert (cfg['training']['learning_rate'],cfg['training']['batch_size'],cfg['model']['dropout'],cfg['training']['gradient_clip_max_norm'])==values;assert cfg['training']['loss_policy']=='HYBRID_LEVEL_PLUS_DELTA';assert cfg['training']['lambda_delta']==.1
def test_feature_population_scaler_and_test_firewall():
 p=run_preflight(project_root());assert p['feature_variant']=='FS2_TF1';assert p['feature_count']==33;assert p['population']['target_count']==16630;assert p['scaler_lineage']=='FOLD_LOCAL_TRAIN_ONLY';assert p['test_rows_read']==p['test_target_ids_seen']==0
def test_exact_five_candidates_and_context():
 root=project_root();c=build_candidates(root);assert len(c)==5;assert {x.candidate_id for x in c}==set(CANDIDATE_IDS.values());ctx=build_context(root,load_e14_config(root));assert_context_invariants(ctx);assert ctx.registry_namespace=='V2_E14'
def test_out_of_bundle_change_rejected():
 root=project_root();base=incumbent_config(root);cfg=bundle_config(root,'M1');cfg['model']['d_model']=96
 with pytest.raises(ValueError):assert_one_primary_change('E14',base,cfg)
def test_official_requires_human_authorization():assert main(['--experiment','E14','--mode','official','--seed','42'])==3
def test_preflight_scope_and_recovery_lock():
 p=run_preflight(project_root());assert p['expected_new_training_runs']==30;assert p['training_executed'] is False;assert p['recovery_contract']=='NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED';assert p['equal_budget_audit']['status']=='PASS';assert p['promotion_policy_status']=='PASS';assert {'e14_bundle_comparison.json','e14_parameter_runtime_report.json','e14_lr_gradient_trace_audit.json','e14_ranking_guardrails.json'}<=set(p['expected_outputs'])
def test_e14_registry_accepts_only_preregistered_batch_sizes(tmp_path):
 root=project_root();ctx=build_context(root,load_e14_config(root));cfg16=bundle_config(root,'M1');cfg64=bundle_config(root,'M4')
 registry=ExperimentRegistry(project_root=root,registry_root=tmp_path/'registry',run_root=tmp_path/'runs',run_id_namespace='V2_E14',upstream_context_override=ctx.registry_upstream_context)
 assert validate_run_config(cfg16,registry.upstream_context,registry.run_id_namespace)['training']['batch_size']==16
 assert validate_run_config(cfg64,registry.upstream_context,registry.run_id_namespace)['training']['batch_size']==64
 registered=registry.register_run(cfg16,experiment_family='ROLLING_ORIGIN',execution_type='ROBUSTNESS',candidate_id=CANDIDATE_IDS['M1'],sweep_stage='RO1_A',rerun_reason='PHASE44_CORRECTIVE_RERUN')
 assert registered['run_id'].startswith('RUN_V2_TR_E14_RO1_A_')
 with pytest.raises(ValueError,match='Invalid batch size'):validate_run_config(cfg16,registry.upstream_context,'V2_E13')
 invalid=bundle_config(root,'M1');invalid['training']['batch_size']=8
 with pytest.raises(ValueError,match='Invalid batch size'):validate_run_config(invalid,registry.upstream_context,registry.run_id_namespace)
