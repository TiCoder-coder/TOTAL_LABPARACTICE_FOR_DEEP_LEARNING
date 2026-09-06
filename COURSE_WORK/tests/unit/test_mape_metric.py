import math

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_percentage_error

from course_work.evaluation.metrics import (
    MAPE_METRIC_VERSION,
    MAPE_STATUS_DEFINED,
    MAPE_STATUS_ZERO_TARGET,
    PRIMARY_SELECTION_METRIC,
    build_mape_metric_contract,
    compute_mape_pct,
)


def test_mape_matches_hand_calculation() -> None:
    result = compute_mape_pct([10.0, 20.0, 40.0], [12.0, 18.0, 44.0])
    assert result.mape_pct == pytest.approx((20.0 + 10.0 + 10.0) / 3.0)
    assert result.mape_status == MAPE_STATUS_DEFINED
    assert result.zero_target_count == 0


def test_mape_matches_sklearn_percentage_scale() -> None:
    y_true = np.asarray([10.0, 20.0, 40.0], dtype=np.float64)
    y_pred = np.asarray([12.0, 18.0, 44.0], dtype=np.float64)
    expected = 100.0 * mean_absolute_percentage_error(y_true, y_pred)
    assert compute_mape_pct(y_true, y_pred).mape_pct == pytest.approx(expected)


def test_mape_perfect_prediction_is_zero() -> None:
    result = compute_mape_pct([10.0, 20.0], [10.0, 20.0])
    assert result.mape_pct == 0.0


def test_mape_accepts_supported_shapes_and_tensor() -> None:
    expected = compute_mape_pct([10.0, 20.0], [12.0, 18.0]).mape_pct
    column = compute_mape_pct(np.asarray([[10.0], [20.0]]), np.asarray([[12.0], [18.0]])).mape_pct
    tensor = compute_mape_pct(torch.tensor([10.0, 20.0]), torch.tensor([12.0, 18.0])).mape_pct
    assert column == pytest.approx(expected)
    assert tensor == pytest.approx(expected)


def test_mape_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        compute_mape_pct([10.0, 20.0], [10.0])
    with pytest.raises(ValueError):
        compute_mape_pct([10.0, np.nan], [10.0, 20.0])
    with pytest.raises(ValueError):
        compute_mape_pct([10.0, 20.0], [10.0, np.inf])
    with pytest.raises(ValueError):
        compute_mape_pct(np.ones((2, 2)), np.ones((2, 2)))


def test_mape_zero_target_is_explicitly_undefined() -> None:
    result = compute_mape_pct([0.0, 20.0], [1.0, 18.0])
    assert math.isnan(result.mape_pct)
    assert result.mape_status == MAPE_STATUS_ZERO_TARGET
    assert result.zero_target_count == 1


def test_mape_contract_is_supplementary_and_deterministic() -> None:
    first = build_mape_metric_contract()
    second = build_mape_metric_contract()
    assert first == second
    assert first["metric_version"] == MAPE_METRIC_VERSION
    assert first["primary_selection"] is False
    assert first["epsilon_policy"] == "PROHIBITED"
    assert first["zero_target_filtering"] == "PROHIBITED"
    assert PRIMARY_SELECTION_METRIC == "rmse_wh"
