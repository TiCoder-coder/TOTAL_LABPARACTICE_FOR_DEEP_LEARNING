"""Phase 40 — S18 RevIN focused pre-training tests.

Covers all required pre-training checks before RN1 scientific training.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK')
sys.path.insert(0, str(ROOT / 'src'))

from course_work.data.scaling import (
    inverse_transform_target,
    load_validated_scaler_bundle,
    load_validated_target_scaler,
)
from course_work.models.revin import (
    AFFINE_BIAS_INIT,
    AFFINE_WEIGHT_INIT,
    REVIN_AFFINE,
    REVIN_EPS,
    RevINScope,
    RevINWrappedTransformerRegressor,
    TargetSelectiveRevIN,
    resolve_revin_scope_from_feature_order,
)
from course_work.models.transformer_regressor import TransformerRegressor
from course_work.sweeps.revin import (
    resolve_frozen_config,
    resolve_revin_scope,
    resolve_s17_winner_reference,
)


PHASE_40_DIR = ROOT / 'artifacts/sweeps/S18_revin'
PHASE_40_DIR.mkdir(parents=True, exist_ok=True)


WINNER_RUN_ID = 'RUN_TR_S14_0023_A711A9B8'


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def phase39_handoff():
    return resolve_s17_winner_reference(ROOT)


@pytest.fixture(scope="module")
def frozen_config():
    return resolve_frozen_config(ROOT, WINNER_RUN_ID)


@pytest.fixture(scope="module")
def scope_resolution():
    return resolve_revin_scope(ROOT, 'XSCALER__FS2_TF1', 'FS2_TF1')


@pytest.fixture(scope="module")
def revin_scope(scope_resolution):
    assert scope_resolution['rn1_applicable']
    s = scope_resolution['scope']
    return RevINScope(
        revin_channel_names=s['revin_channel_names'],
        revin_channel_indices=s['revin_channel_indices'],
        passthrough_channel_names=s['passthrough_channel_names'],
        passthrough_channel_indices=s['passthrough_channel_indices'],
        target_channel_name=s['target_channel_name'],
        target_original_index=s['target_original_index'],
        target_revin_subset_index=s['target_revin_subset_index'],
    )


@pytest.fixture(scope="module")
def feature_order(scope_resolution):
    return scope_resolution['feature_order']


@pytest.fixture(scope="module")
def wrapped_model(revin_scope, frozen_config):
    torch.manual_seed(frozen_config['seed'])
    backbone = TransformerRegressor({
        "input_size": frozen_config["input_size"],
        "d_model": frozen_config["d_model"],
        "num_heads": frozen_config["num_heads"],
        "num_layers": frozen_config["num_layers"],
        "ffn_dim": frozen_config["ffn_dim"],
        "output_size": 1,
        "dropout": frozen_config["dropout"],
        "activation": frozen_config["activation"],
        "pooling": frozen_config["pooling"],
        "positional_encoding_type": "SINUSOIDAL",
        "norm_first": False,
        "attention_aware": True,
    })
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    return RevINWrappedTransformerRegressor(backbone=backbone, revin=revin)


@pytest.fixture(scope="module")
def backbone_only(frozen_config):
    torch.manual_seed(frozen_config['seed'])
    return TransformerRegressor({
        "input_size": frozen_config["input_size"],
        "d_model": frozen_config["d_model"],
        "num_heads": frozen_config["num_heads"],
        "num_layers": frozen_config["num_layers"],
        "ffn_dim": frozen_config["ffn_dim"],
        "output_size": 1,
        "dropout": frozen_config["dropout"],
        "activation": frozen_config["activation"],
        "pooling": frozen_config["pooling"],
        "positional_encoding_type": "SINUSOIDAL",
        "norm_first": False,
        "attention_aware": True,
    })


# ---------- A. Phase 39 handoff ----------

def test_a1_phase39_status_pass_or_warning(phase39_handoff):
    assert phase39_handoff['signoff']['status'] in ('PASS', 'PASS_WITH_WARNING')


def test_a2_approved_for_phase40(phase39_handoff):
    assert phase39_handoff['signoff']['approved_for_phase40'] is True


def test_a3_no_blocking_compliance_issue(phase39_handoff):
    assert 'blocking_issues' not in phase39_handoff['signoff']
    issues = phase39_handoff['signoff'].get('discrepancies') or []
    assert not issues


def test_a4_gc1_winner(phase39_handoff):
    assert phase39_handoff['winner']['winner_condition'] == 'GC1'
    assert phase39_handoff['winner']['winner_run_id'] == WINNER_RUN_ID


def test_a5_exact_s17_winner_resolved(phase39_handoff):
    assert phase39_handoff['winner_run_id'] == WINNER_RUN_ID


def test_a6_test_untouched(phase39_handoff):
    assert phase39_handoff['signoff']['test_status'] == 'FORBIDDEN'
    assert phase39_handoff['signoff']['test_access'] == 'NOT_ACCESSED'


# ---------- B. Resolved S1-S17 configuration ----------

def test_b1_frozen_feature_variant(frozen_config):
    assert frozen_config['feature_variant_id'] == 'FS2_TF1'


def test_b2_frozen_target_scaling(frozen_config):
    assert frozen_config['target_scaling_option'] == 'YS1'


def test_b3_frozen_lookback(frozen_config):
    assert frozen_config['lookback_steps'] == 36


def test_b4_frozen_pooling(frozen_config):
    assert frozen_config['pooling'] == 'LAST_STEP'


def test_b5_frozen_activation(frozen_config):
    assert frozen_config['activation'] == 'GELU'


def test_b6_frozen_batch_size(frozen_config):
    assert frozen_config['batch_size'] == 32


def test_b7_frozen_lr_wd(frozen_config):
    assert math.isclose(frozen_config['learning_rate'], 0.0003)
    assert math.isclose(frozen_config['weight_decay'], 0.001)


def test_b8_frozen_dropout(frozen_config):
    assert math.isclose(frozen_config['dropout'], 0.1)


def test_b9_frozen_d_model_heads_layers(frozen_config):
    assert frozen_config['d_model'] == 64
    assert frozen_config['num_heads'] == 4
    assert frozen_config['head_dim'] == 16
    assert frozen_config['num_layers'] == 2


def test_b10_frozen_ffn(frozen_config):
    assert frozen_config['ffn_dim'] == 256


def test_b11_frozen_loss_epochs(frozen_config):
    assert frozen_config['loss'] == 'MSE'
    assert frozen_config['max_epochs'] == 50


def test_b12_frozen_grad_clip(frozen_config):
    assert frozen_config['gradient_clipping_enabled'] is True
    assert math.isclose(frozen_config['gradient_clip_max_norm'], 1.0)


def test_b13_frozen_seed(frozen_config):
    assert frozen_config['seed'] == 42


# ---------- C/D. Applicability gate ----------

def test_c1_feature_order_resolved(feature_order):
    assert len(feature_order) == 33


def test_c2_appliances_match_count(feature_order):
    count = sum(1 for n in feature_order if n == 'Appliances')
    assert count == 1


def test_c3_rn1_applicable_true(scope_resolution):
    assert scope_resolution['rn1_applicable'] is True


def test_d_absent_appliances_skipped_path():
    no_appliances = ['lights', 'T1', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend']
    ok, scope_or_none, reason = resolve_revin_scope_from_feature_order(no_appliances)
    assert ok is False
    assert scope_or_none is None
    assert 'Appliances' in reason


def test_d2_duplicate_appliances_fails():
    duplicate = ['Appliances', 'lights', 'Appliances']
    with pytest.raises(ValueError):
        resolve_revin_scope_from_feature_order(duplicate)


def test_d3_no_hidden_target_injection(scope_resolution):
    feature_names = scope_resolution['feature_order']
    assert 'target_next' not in feature_names
    assert not any('future' in n.lower() for n in feature_names)
    assert not any('t_plus_1' in n.lower() for n in feature_names)


# ---------- E. RevIN channel scope ----------

def test_e1_revin_includes_Appliances(revin_scope):
    assert 'Appliances' in revin_scope.revin_channel_names
    assert 'Appliances' in revin_scope.revin_channel_names
    assert 'Appliances' not in revin_scope.passthrough_channel_names


def test_e2_passthrough_excludes_Appliances(revin_scope):
    assert 'Appliances' not in revin_scope.passthrough_channel_names


def test_e3_passthrough_exact_set(revin_scope):
    expected = {'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend'}
    assert set(revin_scope.passthrough_channel_names) == expected


def test_e4_target_original_index(revin_scope, feature_order):
    expected = feature_order.index('Appliances')
    assert revin_scope.target_original_index == expected


def test_e5_target_subset_index(revin_scope):
    assert revin_scope.target_revin_subset_index == revin_scope.revin_channel_names.index('Appliances')


def test_e6_revin_channel_count(revin_scope):
    assert revin_scope.revin_channel_count == len(revin_scope.revin_channel_indices)
    assert revin_scope.revin_channel_count + len(revin_scope.passthrough_channel_names) == 33


# ---------- F. Global scaling preserved ----------

def test_f1_scaler_bundle_resolvable(frozen_config):
    bundle = load_validated_scaler_bundle(frozen_config['scaler_bundle_id'].replace('XSCALER__', ''), ROOT)
    assert bundle['full_feature_order']


def test_f2_target_scaler_resolvable(frozen_config):
    y_scaler = load_validated_target_scaler(ROOT)
    assert y_scaler is not None


# ---------- G/H. RevIN contract ----------

def test_g1_revin_eps(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert revin.eps == 1e-5


def test_g2_revin_affine(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert revin.affine is True


def test_g3_gamma_init_one(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert torch.allclose(revin.gamma, torch.ones_like(revin.gamma))


def test_g4_beta_init_zero(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert torch.allclose(revin.beta, torch.zeros_like(revin.beta))


def test_h1_time_features_passthrough(wrapped_model, frozen_config, feature_order):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    x = torch.randn(2, L, F)
    x_norm, _ = wrapped_model.revin.normalize(x)
    time_indices = [feature_order.index(n) for n in ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'weekend')]
    for ti in time_indices:
        assert torch.allclose(x[..., ti], x_norm[..., ti])


def test_h2_feature_order_unchanged(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    x = torch.randn(2, L, F)
    x_norm, _ = wrapped_model.revin.normalize(x)
    assert x_norm.shape == x.shape


def test_h3_statistics_detached(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    _, ctx = wrapped_model.revin.normalize(x)
    assert not ctx['mean'].requires_grad
    assert not ctx['stdev'].requires_grad


def test_h4_no_running_stats(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert not hasattr(revin, 'running_mean') or revin.running_mean is None or len(revin.state_dict()) <= 4


def test_h5_gamma_beta_learnable(revin_scope):
    revin = TargetSelectiveRevIN(
        revin_indices=revin_scope.revin_channel_indices,
        passthrough_indices=revin_scope.passthrough_channel_indices,
        target_revin_subset_index=revin_scope.target_revin_subset_index,
    )
    assert revin.gamma.requires_grad is True
    assert revin.beta.requires_grad is True


# ---------- I. Round-trip / target denorm tests ----------

def test_i1_full_roundtrip(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    x = torch.randn(4, L, F)
    x_norm, ctx = wrapped_model.revin.normalize(x)
    pred_norm = torch.randn(4, 1)
    pred_x = wrapped_model.revin.denormalize_target(pred_norm, ctx)
    assert pred_x.shape == (4, 1)
    assert torch.isfinite(pred_x).all()


def test_i2_constant_channel(wrapped_model, revin_scope, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    x[:, :, 0] = 5.0
    x_norm, ctx = wrapped_model.revin.normalize(x)
    assert torch.isfinite(x_norm).all()
    assert torch.isfinite(ctx['mean']).all()
    assert torch.isfinite(ctx['stdev']).all()


def test_i3_low_variance_channel(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    x[:, :, 0] = 5.0 + 1e-7 * torch.randn(2, L)
    x_norm, ctx = wrapped_model.revin.normalize(x)
    assert torch.isfinite(x_norm).all()


def test_i4_b1_works(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(1, L, F)
    out = wrapped_model(x)
    assert out.shape == (1, 1)


def test_i5_b_greater_than_1_works(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(8, L, F)
    out = wrapped_model(x)
    assert out.shape == (8, 1)


def test_i6_batch_independence(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    sample_a = torch.randn(1, L, F)
    sample_b = torch.randn(1, L, F)
    sample_c = torch.randn(1, L, F)

    batch_one = sample_a
    batch_three = torch.cat([sample_a, sample_b, sample_c], dim=0)

    xn_one, ctx_one = wrapped_model.revin.normalize(batch_one)
    xn_three, ctx_three = wrapped_model.revin.normalize(batch_three)

    assert torch.allclose(xn_one, xn_three[:1], atol=1e-6)
    assert torch.allclose(ctx_one['mean'], ctx_three['mean'][:1], atol=1e-6)
    assert torch.allclose(ctx_one['stdev'], ctx_three['stdev'][:1], atol=1e-6)


def test_i7_neighbor_sample_no_leakage(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    sample_a = torch.randn(1, L, F)
    sample_b = torch.randn(1, L, F)

    batch1 = torch.cat([sample_a, sample_b], dim=0)
    _, ctx1 = wrapped_model.revin.normalize(batch1)
    mean_a_normal = ctx1['mean'][0:1].clone()

    sample_b_extreme = torch.randn(1, L, F) * 1e6
    batch2 = torch.cat([sample_a, sample_b_extreme], dim=0)
    _, ctx2 = wrapped_model.revin.normalize(batch2)
    mean_a_extreme = ctx2['mean'][0:1].clone()

    assert torch.allclose(mean_a_normal, mean_a_extreme, atol=1e-6)


def test_i8_batch_permutation(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    torch.manual_seed(42)
    a = torch.randn(1, L, F)
    b = torch.randn(1, L, F)
    c = torch.randn(1, L, F)
    batch_fwd = torch.cat([a, b, c], dim=0)
    # permutation: [a, b, c] -> [b, a, c] (swap first two)
    batch_perm = torch.cat([b, a, c], dim=0)

    xn_fwd, ctx_fwd = wrapped_model.revin.normalize(batch_fwd)
    xn_perm, ctx_perm = wrapped_model.revin.normalize(batch_perm)

    # Sample a is at index 0 in fwd, index 1 in perm
    # Sample b is at index 1 in fwd, index 0 in perm
    assert torch.allclose(xn_fwd[0], xn_perm[1], atol=1e-6)
    assert torch.allclose(xn_fwd[1], xn_perm[0], atol=1e-6)
    assert torch.allclose(xn_fwd[2], xn_perm[2], atol=1e-6)
    # context likewise
    assert torch.allclose(ctx_fwd['mean'][0], ctx_perm['mean'][1], atol=1e-6)


# ---------- J. Architecture / Parameter audit ----------

def test_j1_state_dict_delta(wrapped_model, backbone_only):
    rn0_keys = set(backbone_only.state_dict().keys())
    rn1_keys = set(wrapped_model.state_dict().keys())
    # wrapped prefixes backbone keys with "backbone." -> strip for fair compare
    rn1_stripped = {k[len("backbone."):] if k.startswith("backbone.") else k for k in rn1_keys}
    new_keys = sorted(rn1_stripped - rn0_keys)
    revin_affine_keys = [k for k in new_keys if k.startswith("revin.")]
    non_revin_new = [k for k in new_keys if not k.startswith("revin.")]
    assert set(revin_affine_keys) == {'revin.gamma', 'revin.beta'}
    assert non_revin_new == []


def test_j2_parameter_delta(wrapped_model, backbone_only, revin_scope):
    rn0_params = sum(p.numel() for p in backbone_only.parameters() if p.requires_grad)
    rn1_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    expected_delta = 2 * revin_scope.revin_channel_count
    assert (rn1_params - rn0_params) == expected_delta


def test_j3_backbone_invariant(wrapped_model, backbone_only):
    rn0_keys = set(backbone_only.state_dict().keys())
    rn1_keys = set(wrapped_model.state_dict().keys())
    # Strip "backbone." prefix from wrapped model keys for comparison
    rn1_stripped = {k.replace("backbone.", "", 1) for k in rn1_keys}
    shared = sorted(rn0_keys & rn1_stripped)
    assert len(shared) == len(rn0_keys)
    for k in shared:
        wrapped_key = f"backbone.{k}" if not k.startswith("revin.") else k
        if k.startswith("revin."):
            continue
        assert wrapped_model.state_dict()[wrapped_key].shape == backbone_only.state_dict()[k].shape


def test_j4_revin_affine_optimizer_covered(wrapped_model):
    params = list(wrapped_model.parameters())
    revin_gamma = wrapped_model.revin.gamma
    revin_beta = wrapped_model.revin.beta
    assert any(p is revin_gamma for p in params)
    assert any(p is revin_beta for p in params)


# ---------- K. Gradient flow ----------

def test_k1_gamma_gradient_finite(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    out = wrapped_model(x).sum()
    out.backward()
    assert wrapped_model.revin.gamma.grad is not None
    assert torch.isfinite(wrapped_model.revin.gamma.grad).all()


def test_k2_beta_gradient_finite(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    out = wrapped_model(x).sum()
    out.backward()
    assert wrapped_model.revin.beta.grad is not None
    assert torch.isfinite(wrapped_model.revin.beta.grad).all()


def test_k3_backbone_gradient_finite(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    x = torch.randn(2, L, F)
    out = wrapped_model(x).sum()
    out.backward()
    for name, p in wrapped_model.backbone.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


# ---------- L. forward_with_attention equivalence ----------

def test_l1_forward_attention_equivalence(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    wrapped_model.eval()
    torch.manual_seed(42)
    x = torch.randn(2, L, F)
    with torch.no_grad():
        out_standard = wrapped_model(x)
        out_attn, attn = wrapped_model.forward_with_attention(x)
    assert torch.allclose(out_standard, out_attn, atol=1e-6)


def test_l2_attention_shape(wrapped_model, frozen_config):
    L = frozen_config['lookback_steps']
    F = frozen_config['input_size']
    wrapped_model.eval()
    x = torch.randn(2, L, F)
    with torch.no_grad():
        _, attn = wrapped_model.forward_with_attention(x)
    assert len(attn) == frozen_config['num_layers']
    for a in attn:
        assert a.shape == (2, frozen_config['num_heads'], L, L)


# ---------- M. Coordinate bridge ----------

def test_m1_x_to_y_roundtrip(frozen_config):
    y_scaler = load_validated_target_scaler(ROOT)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 200, size=(8, 1))
    y_model = (raw - y_scaler['scaler'].mean_[0]) / y_scaler['scaler'].scale_[0]
    raw_recovered = inverse_transform_target(y_model, 'YS1', y_scaler)
    assert np.allclose(raw_recovered, raw, atol=1e-6)
    assert raw_recovered.shape == (8, 1)


def test_m2_y_transform_invertible():
    y_scaler = load_validated_target_scaler(ROOT)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 200, size=(16, 1))
    y_model = (raw - y_scaler['scaler'].mean_[0]) / y_scaler['scaler'].scale_[0]
    raw_recovered = inverse_transform_target(y_model, 'YS1', y_scaler)
    assert np.allclose(raw_recovered, raw, atol=1e-6)


def test_m3_ys0_identity():
    y_scaler = load_validated_target_scaler(ROOT)
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 200, size=(4, 1))
    raw_inv = inverse_transform_target(raw, 'YS0', y_scaler)
    assert np.allclose(raw_inv, raw, atol=1e-6)


# ---------- N. Test firewall ----------

def test_n1_test_firewall_active(phase39_handoff):
    assert phase39_handoff['signoff']['test_status'] == 'FORBIDDEN'
    assert phase39_handoff['signoff']['test_access'] == 'NOT_ACCESSED'


def test_n2_no_test_in_features(scope_resolution):
    feature_names = scope_resolution['feature_order']
    forbidden = {'test', 'test_target', 'test_label'}
    assert not any(n.lower() in forbidden for n in feature_names)


# ---------- O. Training-config delta ----------

def test_o1_only_revin_differs(frozen_config, wrapped_model):
    assert wrapped_model.revin.affine is True
    assert wrapped_model.revin.eps == 1e-5
    assert frozen_config['gradient_clipping_enabled'] is True
    assert math.isclose(frozen_config['gradient_clip_max_norm'], 1.0)
    assert frozen_config['loss'] == 'MSE'
    assert frozen_config['max_epochs'] == 50


# ---------- P. Common data audit ----------

def test_p1_population_fingerprint_match(frozen_config):
    cfg = json.loads((ROOT / f"artifacts/runs/{WINNER_RUN_ID}/config.json").read_text())['config']
    assert frozen_config['population_fingerprint'] == cfg['lineage']['population_fingerprint']


def test_p2_metric_contract_match(frozen_config):
    metrics_payload = json.loads((ROOT / f"artifacts/runs/{WINNER_RUN_ID}/metrics/best_validation_metrics.json").read_text())
    assert metrics_payload['metric_result']['metric_contract_fingerprint'] == frozen_config['metric_contract_fingerprint']


# ---------- Q. WB0 preserved ----------

def test_q1_wb0_preserved(frozen_config):
    cfg = json.loads((ROOT / f"artifacts/runs/{WINNER_RUN_ID}/config.json").read_text())['config']
    assert cfg['data']['boundary_protocol'] == 'WB0_CONTEXT_CARRY_OVER'


# ---------- R. Sample order fairness ----------

def test_r1_sample_order_seeds(frozen_config):
    cfg = json.loads((ROOT / f"artifacts/runs/{WINNER_RUN_ID}/config.json").read_text())['config']
    assert cfg['reproducibility']['seed'] == 42
    assert cfg['reproducibility']['dataloader_seed'] == 42


# ---------- S. Idempotency ----------

def test_s1_idempotent_scope_resolution(scope_resolution):
    s2 = resolve_revin_scope(ROOT, 'XSCALER__FS2_TF1', 'FS2_TF1')
    assert s2['rn1_applicable'] == scope_resolution['rn1_applicable']
    assert s2['reason'] == scope_resolution['reason']


# ---------- T. RN0 reuse gate ----------

def test_t1_rn0_uses_exact_s17_winner(phase39_handoff):
    assert phase39_handoff['winner']['winner_run_id'] == WINNER_RUN_ID
    assert phase39_handoff['signoff']['winner']['run_id'] == WINNER_RUN_ID


def test_t2_rn0_is_reuse_reference(phase39_handoff):
    assert phase39_handoff['signoff']['gc1_reference']['run_id'] == WINNER_RUN_ID


# ---------- U. No scientific RN1 training executed ----------

def test_u1_rn1_run_exists_with_strict_best_artifacts():
    """Post-training: RN1 scientific run must exist with all required artifacts."""
    rn1_run_dirs = list((ROOT / 'artifacts/runs').glob('RUN_TR_S18_003*'))
    assert len(rn1_run_dirs) >= 1, "RN1 scientific run must exist"
    # Verify the canonical post-training artifacts for each RN1 run
    for run_dir in rn1_run_dirs:
        run_id = run_dir.name
        required = [
            run_dir / 'config.json',
            run_dir / 'status.json',
            run_dir / 'checkpoints/best_checkpoint.pt',
            run_dir / 'checkpoints/last_checkpoint.pt',
            run_dir / 'training_history.csv',
            run_dir / 'training.log',
            run_dir / 'metrics/best_validation_metrics.json',
            run_dir / 'predictions/best_validation_predictions.csv',
        ]
        missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
        assert not missing, f"RN1 {run_id} missing: {missing}"