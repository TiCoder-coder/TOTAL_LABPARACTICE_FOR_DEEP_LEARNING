#!/usr/bin/env python3
"""Strict BEST verification for Phase 39 GC0.

Following canonical pattern from Phase 36/37/38 verification functions.
"""

import math
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK/src')))

from course_work.data.datasets import build_train_validation_loaders
from course_work.data.scaling import inverse_transform_target, load_validated_target_scaler
from course_work.evaluation.metrics import EvaluationMode, compute_regression_metrics
from course_work.models.transformer_regressor import TransformerRegressor
from course_work.utils.artifacts import read_json
from course_work.utils.reproducibility import set_seed

ROOT = Path('/Users/vientu/Deep Learning/TOTAL_LABPARACTICE_FOR_DEEP_LEARNING/COURSE_WORK')
RUN_ID = 'RUN_TR_S17_0029_082F7FF5'
SEED = 42

def main():
    config = read_json(ROOT / f'artifacts/runs/{RUN_ID}/config.json')['config']
    metrics = read_json(ROOT / f'artifacts/runs/{RUN_ID}/metrics/best_validation_metrics.json')
    predictions = pd.read_csv(ROOT / f'artifacts/runs/{RUN_ID}/predictions/best_validation_predictions.csv')
    sample_ids = predictions['sample_idx'].astype(int).tolist()

    # Build model config from saved config
    model_config = {
        'input_size': int(config['model']['input_size']),
        'd_model': int(config['model']['d_model']),
        'num_heads': int(config['model']['num_heads']),
        'num_layers': int(config['model']['num_layers']),
        'ffn_dim': int(config['model']['ffn_dim']),
        'output_size': int(config['model']['output_size']),
        'dropout': float(config['model']['dropout']),
        'activation': str(config['model']['activation']),
        'pooling': str(config['model']['pooling']),
        'positional_encoding_type': str(config['model']['positional_encoding_type']),
        'norm_first': bool(config['model'].get('norm_first', False)),
        'attention_aware': bool(config['model'].get('attention_aware', True)),
    }

    # Strict load
    checkpoint = torch.load(ROOT / f'artifacts/runs/{RUN_ID}/checkpoints/best_checkpoint.pt',
                            map_location='cpu', weights_only=False)
    set_seed(SEED)
    model = TransformerRegressor(model_config)
    incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, \
        f'incompatible keys: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}'
    model.eval()

    # Load validation data
    loaders = build_train_validation_loaders(
        project_root=ROOT,
        variant_id=config['data']['feature_variant_id'],
        lookback=int(config['data']['lookback_steps']),
        target_option=config['data']['target_scaling_option'],
        batch_size=int(config['training']['batch_size']),
        seed=int(config['reproducibility']['seed']),
        num_workers=0,
        device_type='cpu',
    )
    validation_loader = loaders['VALIDATION'][0]
    target_scaler = load_validated_target_scaler(ROOT)

    # Inference
    recomputed_sample_ids: list[int] = []
    recomputed_y_true: list[float] = []
    recomputed_y_pred: list[float] = []
    with torch.no_grad():
        for batch in validation_loader:
            model_predictions = model(batch['x'])
            if model_predictions.shape != batch['y_model'].shape:
                raise RuntimeError("prediction shape mismatch")
            prediction_wh = inverse_transform_target(
                model_predictions.cpu().numpy().reshape(-1, 1), 'YS1', target_scaler
            ).reshape(-1)
            recomputed_sample_ids.extend(batch['sample_idx'].cpu().numpy().tolist())
            recomputed_y_true.extend(batch['y_raw_wh'].cpu().numpy().reshape(-1).tolist())
            recomputed_y_pred.extend(prediction_wh.tolist())

    recomputed = compute_regression_metrics(
        np.asarray(recomputed_y_true, dtype=np.float64),
        np.asarray(recomputed_y_pred, dtype=np.float64),
        np.asarray(recomputed_sample_ids, dtype=np.int64),
        'VALIDATION',
        EvaluationMode.VALIDATION.value,
        config['lineage']['population_fingerprint'],
        RUN_ID,
        config['model'].get('model_name', config['model']['model_family']),
        lookback_steps=int(config['data']['lookback_steps']),
        horizon_steps=int(config['data']['horizon_steps']),
        target_scaling_option=config['data']['target_scaling_option'],
        project_root=ROOT,
    )

    stored_rmse = metrics['metric_result']['rmse_wh']
    stored_mae = metrics['metric_result']['mae_wh']
    stored_r2 = metrics['metric_result']['r2']

    print(f'Stored RMSE: {stored_rmse}')
    print(f'Recomputed RMSE: {recomputed.rmse_wh}')
    print(f'RMSE delta: {abs(recomputed.rmse_wh - stored_rmse)}')
    print(f'Stored MAE: {stored_mae}')
    print(f'Recomputed MAE: {recomputed.mae_wh}')
    print(f'MAE delta: {abs(recomputed.mae_wh - stored_mae)}')
    print(f'Stored R2: {stored_r2}')
    print(f'Recomputed R2: {recomputed.r2}')
    print(f'R2 delta: {abs(recomputed.r2 - stored_r2)}')

    tolerance = 1e-6
    rmse_ok = math.isclose(recomputed.rmse_wh, stored_rmse, rel_tol=tolerance, abs_tol=tolerance)
    mae_ok = math.isclose(recomputed.mae_wh, stored_mae, rel_tol=tolerance, abs_tol=tolerance)
    r2_ok = math.isclose(recomputed.r2, stored_r2, rel_tol=tolerance, abs_tol=tolerance)
    n_ok = recomputed.n_samples == int(metrics['metric_result']['n_samples'])
    pop_ok = recomputed.population_fingerprint == metrics['metric_result']['population_fingerprint']
    sample_ok = np.array_equal(np.asarray(recomputed_sample_ids, dtype=np.int64),
                               np.asarray(sample_ids, dtype=np.int64))

    status = 'PASS' if (rmse_ok and mae_ok and r2_ok and n_ok and pop_ok and sample_ok) else 'FAIL'
    print(f'\nSTRICT BEST: {status}')
    print(f'Tolerance: {tolerance}')
    return status, {
        'stored_rmse': stored_rmse,
        'recomputed_rmse': recomputed.rmse_wh,
        'rmse_delta': abs(recomputed.rmse_wh - stored_rmse),
        'stored_mae': stored_mae,
        'recomputed_mae': recomputed.mae_wh,
        'mae_delta': abs(recomputed.mae_wh - stored_mae),
        'stored_r2': stored_r2,
        'recomputed_r2': recomputed.r2,
        'r2_delta': abs(recomputed.r2 - stored_r2),
        'n_samples': recomputed.n_samples,
        'population_fingerprint': recomputed.population_fingerprint,
        'tolerance': tolerance,
    }


if __name__ == "__main__":
    main()