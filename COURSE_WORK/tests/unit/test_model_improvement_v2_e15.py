import pytest
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e15_runner import CANDIDATE_IDS,WIDTH_BUNDLES,build_candidates,build_context,incumbent_config,load_e15_config,main,project_root,run_preflight,width_config
from course_work.experiments.registry import ExperimentRegistry
from course_work.rolling_origin.real_run import assert_context_invariants

def test_e14_m1_human_decision_is_exact_control():
 preflight=run_preflight(project_root());assert preflight["accepted_incumbent"]=="TR_C2_ALT_LOOKBACK_E14_M1";assert preflight["control_reuse_status"]=="PASS"
def test_exact_width_bundles_and_only_primary_factor():
 root=project_root();base=incumbent_config(root)
 for key,(width,ffn,count) in WIDTH_BUNDLES.items():
  config=width_config(root,key);assert set(assert_one_primary_change("E15",base,config))=={"model.d_model","model.ffn_dim"};assert (config["model"]["d_model"],config["model"]["num_heads"],config["model"]["ffn_dim"],config["model"]["num_layers"])==(width,4,ffn,2);assert config["training"]==base["training"];assert count<1_000_000
def test_parameter_population_scaler_and_test_firewall():
 preflight=run_preflight(project_root());assert preflight["parameter_counts"]=={"W96":227041,"W128":401025};assert preflight["parameter_gate"]=="PASS";assert preflight["feature_count"]==33;assert preflight["population"]["target_count"]==16630;assert preflight["scaler_lineage"]=="FOLD_LOCAL_TRAIN_ONLY";assert preflight["test_rows_read"]==preflight["test_target_ids_seen"]==0
def test_exact_two_candidates_and_context():
 root=project_root();candidates=build_candidates(root);assert len(candidates)==2;assert {c.candidate_id for c in candidates}==set(CANDIDATE_IDS.values());context=build_context(root,load_e15_config(root));assert_context_invariants(context);assert context.registry_namespace=="V2_E15"
def test_non_capacity_change_is_rejected():
 root=project_root();base=incumbent_config(root);config=width_config(root,"W96");config["training"]["learning_rate"]=3e-4
 with pytest.raises(ValueError):assert_one_primary_change("E15",base,config)
def test_official_requires_separate_human_authorization():assert main(["--experiment","E15","--mode","official","--seed","42"])==3
def test_preflight_scope_and_no_training():
 preflight=run_preflight(project_root());assert preflight["expected_new_training_runs"]==12;assert preflight["training_executed"] is False;assert preflight["recovery_contract"]=="NO_RETRAIN_EXACT_COMPLETED_LEDGER_REQUIRED";assert preflight["test_status"]=="NOT_ACCESSED"
def test_e15_registry_namespace_and_inherited_batch16(tmp_path):
 root=project_root();context=build_context(root,load_e15_config(root));config=width_config(root,"W96");registry=ExperimentRegistry(project_root=root,registry_root=tmp_path/"registry",run_root=tmp_path/"runs",run_id_namespace="V2_E15",upstream_context_override=context.registry_upstream_context);record=registry.register_run(config,experiment_family="ROLLING_ORIGIN",execution_type="ROBUSTNESS",candidate_id=CANDIDATE_IDS["W96"],sweep_stage="RO1_A",rerun_reason="PHASE44_CORRECTIVE_RERUN");assert record["run_id"].startswith("RUN_V2_TR_E15_RO1_A_");assert record["config"]["training"]["batch_size"]==16
