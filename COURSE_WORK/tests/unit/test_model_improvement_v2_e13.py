from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from course_work.experiments.registry import ExperimentRegistry
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e13_runner import ALLOWED_LAMBDAS, CANDIDATE_IDS, E13PreflightError, audit_completed_run_ledger, build_candidates, build_context, candidate_config, load_e13_config, main, project_root, run_preflight, run_resume_stage_c
from course_work.model_improvement_v2.hybrid_loss import compute_hybrid_loss
from course_work.training.engine import HISTORY_COLUMNS, history_columns_for_training
from course_work.rolling_origin.real_run import assert_context_invariants
from course_work.rolling_origin.scaling import FoldLocalScalerBundle
def _bundle(mean=100.0,scale=20.0): return {"scaler":SimpleNamespace(mean_=np.array([mean]),scale_=np.array([scale]))}
def _training(value=0.25): return {"loss_name":"MSE","loss_policy":"HYBRID_LEVEL_PLUS_DELTA","lambda_delta":value,"delta_beta_model_space":1.0}
def test_exact_hybrid_algebra_and_units():
    pred=torch.tensor([[1.5],[0.0]]); nxt=torch.tensor([[2.0],[1.0]]); context_raw=torch.tensor([[120.0],[80.0]])
    c=compute_hybrid_loss(pred,nxt,context_raw,_bundle(),_training())
    context=(context_raw-100.0)/20.0; true_delta=nxt-context; pred_delta=pred-context
    assert torch.allclose(c.level_mse,torch.nn.functional.mse_loss(pred,nxt))
    assert torch.allclose(c.delta_smooth_l1,torch.nn.functional.smooth_l1_loss(pred_delta,true_delta,beta=1.0))
    assert torch.allclose(c.total_loss,c.level_mse+0.25*c.delta_smooth_l1)
def test_delta_true_raw_identity_in_standardized_space():
    y_next_raw=torch.tensor([[140.0]]); context_raw=torch.tensor([[110.0]]); y_next=(y_next_raw-100.0)/20.0; pred=torch.tensor([[1.0]])
    c=compute_hybrid_loss(pred,y_next,context_raw,_bundle(),_training(.10))
    expected=torch.nn.functional.smooth_l1_loss(pred-(context_raw-100.0)/20.0,(y_next_raw-context_raw)/20.0,beta=1.0)
    assert torch.allclose(c.delta_smooth_l1,expected)
def test_canonical_fold_scaler_dict_preserves_e13_y_lineage():
    bundle=FoldLocalScalerBundle(bundle_id='E13_TEST',fit_stage='B',fold_id='RO1',candidate_id=CANDIDATE_IDS[.10],target_scaling_option='YS1',feature_variant_id='FS2_TF1',lookback_steps=72,boundary_protocol='WB0_CONTEXT_CARRY_OVER',revin_enabled=False,fit_population_fingerprint='outer-train-fingerprint',fit_target_ids=(1,2,3),fit_raw_row_count=3,y_mean=100.0,y_std=20.0)
    checksum=bundle.checksum(); serialized=bundle.as_dict()
    assert serialized['checksum']==checksum
    assert serialized['y_mean']==100.0 and serialized['y_std']==20.0
    assert float(serialized['scaler'].mean_[0])==100.0
    assert float(serialized['scaler'].scale_[0])==20.0
    components=compute_hybrid_loss(torch.tensor([[1.5]]),torch.tensor([[2.0]]),torch.tensor([[120.0]]),serialized,_training(.10))
    assert torch.isfinite(components.total_loss)
def test_invalid_lambda_beta_and_missing_context_rejected():
    with pytest.raises(ValueError): compute_hybrid_loss(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1),_bundle(),{**_training(),"lambda_delta":.2})
    with pytest.raises(ValueError): compute_hybrid_loss(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1),_bundle(),{**_training(),"delta_beta_model_space":.5})
def test_only_loss_policy_changes_and_no_lag144():
    root=project_root(); base=__import__('json').loads((root/'artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json').read_text())['v1_config']
    for value in ALLOWED_LAMBDAS:
        cfg=candidate_config(root,value); assert assert_one_primary_change('E13',base,cfg)==('training.delta_beta_model_space','training.lambda_delta','training.loss_policy'); assert cfg['data']['feature_variant_id']=='FS2_TF1'; assert cfg['data']['feature_count']==33; assert 'LAG144' not in str(cfg)
def test_preflight_population_firewall_and_control_reuse():
    p=run_preflight(project_root()); assert p['status']=='PASS'; assert p['control_reuse_status']=='PASS'; assert p['expected_new_training_runs']==18; assert p['test_rows_read']==p['test_target_ids_seen']==0; assert p['training_executed'] is False
def test_three_candidates_and_context_invariants():
    root=project_root(); candidates=build_candidates(root); assert [c.config['training']['lambda_delta'] for c in candidates]==list(ALLOWED_LAMBDAS); ctx=build_context(root,load_e13_config(root)); assert_context_invariants(ctx); assert ctx.reuse_completed_runs is False
def test_e13_component_history_is_scoped_and_complete():
    hybrid=history_columns_for_training(_training()); ordinary=history_columns_for_training({"loss_name":"MSE"})
    assert ordinary==HISTORY_COLUMNS
    assert hybrid[-4:]==["level_mse","delta_smooth_l1","weighted_delta_loss","total_loss"]
def test_e13_registry_accepts_locked_hybrid_config(tmp_path):
    root=project_root(); ctx=build_context(root,load_e13_config(root)); cfg=candidate_config(root,.10); cfg["lineage"].update({"rolling_origin_candidate_id":"TR_C2_ALT_LOOKBACK_E13_LAMBDA_010","rolling_origin_fold_id":"RO1","rolling_origin_stage":"A"})
    registry=ExperimentRegistry(project_root=root,registry_root=tmp_path/"registry",run_root=tmp_path/"runs",run_id_namespace="V2_E13",upstream_context_override=ctx.registry_upstream_context)
    record=registry.register_run(cfg,experiment_family="ROLLING_ORIGIN",execution_type="ROBUSTNESS",candidate_id="TR_C2_ALT_LOOKBACK_E13_LAMBDA_010",sweep_stage="RO1_A")
    assert record["run_id"].startswith("RUN_V2_TR_E13_RO1_A_")
def test_official_refuses_without_human_authorization():
    assert main(['--experiment','E13','--mode','official','--seed','42'])==3
    assert main(['--experiment','E13','--mode','resume-partial','--seed','42'])==3
    assert main(['--experiment','E13','--mode','resume-stage-c','--seed','42','--authorize-training'])==3
def test_e13_current_recovery_ledger_is_stage_b_only_and_preserves_failure():
    audit=audit_completed_run_ledger(project_root())
    assert 9<=len(audit['locked_run_ids'])<=18
    assert sum(key.endswith('_A') for key in audit['locked_run_ids'])==9
    assert len(audit['missing_canonical_runs'])==18-len(audit['locked_run_ids'])
    assert all(key.endswith('_B') for key in audit['missing_canonical_runs'])
    assert any(item['status']=='FAILED' for item in audit['noncompleted_evidence'])
def test_e13_partial_invariant_requires_all_stage_a():
    root=project_root(); ctx=build_context(root,load_e13_config(root))
    stage_a={f"{cid}:RO{fold}_A":f"A-{fold}" for cid in CANDIDATE_IDS.values() for fold in (1,2,3)}
    ctx.reuse_completed_runs=True; ctx.allow_partial_stage_b_training=True; ctx.reuse_completed_run_ids=stage_a
    assert_context_invariants(ctx)
    broken=deepcopy(ctx); broken.reuse_completed_run_ids=dict(stage_a); broken.reuse_completed_run_ids.pop(next(iter(broken.reuse_completed_run_ids)))
    with pytest.raises(RuntimeError,match='all nine COMPLETED Stage A'): assert_context_invariants(broken)
def test_e13_complete_ledger_is_no_train_resume():
    root=project_root(); ctx=build_context(root,load_e13_config(root)); ctx.reuse_completed_runs=True
    ctx.reuse_completed_run_ids={f"{cid}:RO{fold}_{stage}":f"{stage}-{fold}" for cid in CANDIDATE_IDS.values() for fold in (1,2,3) for stage in ('A','B')}
    assert_context_invariants(ctx)
def test_resume_stage_c_refuses_partial_ledger_before_pipeline(monkeypatch,tmp_path):
    import course_work.model_improvement_v2.e13_runner as runner
    monkeypatch.setattr(runner,'load_e13_config',lambda root:{})
    monkeypatch.setattr(runner,'validate_e13_document',lambda document,root:{})
    monkeypatch.setattr(runner,'build_resume_context',lambda root,document:(object(),{'missing_canonical_runs':['candidate:RO1_B']}))
    with pytest.raises(E13PreflightError,match='all 18 completed runs'):
        run_resume_stage_c(tmp_path)
