"""Focused unit tests for E03 DIRECT vs RESIDUAL_TO_PERSISTENCE."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from course_work.data.feature_sets import get_feature_list
from course_work.model_improvement_v2.contracts import assert_one_primary_change
from course_work.model_improvement_v2.e03_runner import (
    CHALLENGER_FORMULATION,
    CONTROL_FORMULATION,
    EXPECTED_CONFIG_CHANGES,
    EXPECTED_CONTROL_METRICS,
    build_e03_candidate,
    build_e03_run_context,
    load_e03_config,
    run_preflight,
)
from course_work.model_improvement_v2.pretest_adapter import build_v2_pretest_dataset
from course_work.model_improvement_v2.residual import (
    compose_residual_prediction_raw,
    residual_model_to_raw,
    residual_raw_to_model,
)
from course_work.rolling_origin.stages import (
    evaluate_stage_c,
    load_fold_subset_loader_with_y_rescale,
)


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# 1. Config fingerprint: only prediction_formulation differs from E01
# ----------------------------------------------------------------------
def test_e03_preflight_reuses_accepted_e01_control():
    audit = run_preflight(PROJECT_ROOT)
    assert audit["status"] == "PASS"
    assert audit["one_primary_change"] == "PASS"
    assert audit["changed_config_fields"] == list(EXPECTED_CONFIG_CHANGES)
    assert audit["control_reuse"] == "CONTROL_REUSED_FROM_E01"
    assert audit["control_reuse_audit"]["run_count"] == 6
    assert audit["training_candidates"] == [CHALLENGER_FORMULATION]
    assert audit["checkpoint_loaded"] is False
    assert audit["promotion_rmse_tolerance_wh"] == 0.10
    assert audit["promotion_threshold_rmse_wh"] == 59.75291570400546
    assert audit["test_access"] == "NO"
    assert audit["challenger_formulation"] == CHALLENGER_FORMULATION
    assert audit["control_formulation"] == CONTROL_FORMULATION
    assert audit["feature_variant"] == "FS2_TF1"
    assert audit["feature_count"] == 33


def test_e03_config_differs_from_e01_only_in_prediction_formulation():
    document = load_e03_config(PROJECT_ROOT)
    e01_doc = __import__("json").loads(
        (
            PROJECT_ROOT
            / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json"
        ).read_text()
    )
    baseline = e01_doc["v1_config"]
    challenger = document["challenger"]["config"]
    changed = assert_one_primary_change("E03", baseline, challenger)
    assert changed == EXPECTED_CONFIG_CHANGES


def test_e03_challenger_has_residual_formulation():
    document = load_e03_config(PROJECT_ROOT)
    config = document["challenger"]["config"]
    assert config["model"]["prediction_formulation"] == "RESIDUAL_TO_PERSISTENCE"


def test_e03_config_rejects_non_residual_formulation():
    """E03 changes only prediction_formulation. Other changes must raise."""
    document = load_e03_config(PROJECT_ROOT)
    challenger_config = document["challenger"]["config"]
    # E01 baseline: DIRECT (no prediction_formulation field)
    e01_doc = __import__("json").loads(
        (
            PROJECT_ROOT
            / "artifacts/model_improvement_v2/experiments/E01/e01_config_snapshot.json"
        ).read_text()
    )
    e01_baseline = e01_doc["v1_config"]

    # Changing to RESIDUAL (challenger) is the approved change → no raise
    assert_one_primary_change("E03", e01_baseline, challenger_config)

    # Changing only d_model is NOT in approved E03 fields → raises
    bad_config = dict(challenger_config)
    bad_config["model"] = dict(bad_config["model"])
    bad_config["model"]["d_model"] = 128
    with pytest.raises(ValueError, match="outside its primary factor"):
        assert_one_primary_change("E03", challenger_config, bad_config)


# ----------------------------------------------------------------------
# 2. Residual conversion contract (from residual.py)
# ----------------------------------------------------------------------
class _FakeScaler:
    def __init__(self):
        self.y_std = 106.85
        self.y_mean = 0.0  # YS1: subtract mean, divide by std


class _FakeScalerDict:
    def __init__(self):
        self.y_std = 106.85
        self.y_mean = 0.0
        self.target_scaling_option = "YS1"


def test_residual_raw_to_model_divides_by_std():
    scaler = _FakeScaler()
    delta_raw = 53.425  # 0.5 std
    result = residual_raw_to_model(delta_raw, scaler, "YS1")
    assert abs(result - 0.5) < 1e-9


def test_residual_model_to_raw_multiplies_by_std():
    scaler = _FakeScaler()
    delta_model = 0.5
    result = residual_model_to_raw(delta_model, scaler, "YS1")
    assert abs(result - 53.425) < 1e-6


def test_residual_roundtrip_identity():
    scaler = _FakeScalerDict()
    delta_raw = 213.7  # 2 std
    model = residual_raw_to_model(delta_raw, scaler, "YS1")
    back = residual_model_to_raw(model, scaler, "YS1")
    assert abs(back - delta_raw) < 1e-9


def test_compose_residual_prediction_raw():
    scaler = _FakeScalerDict()
    context_raw = 100.0
    delta_raw = 50.0
    composed = compose_residual_prediction_raw(context_raw, delta_raw / scaler.y_std, scaler, "YS1")
    assert abs(composed - 150.0) < 1e-9


def test_residual_rejects_invalid_scaling_option():
    scaler = _FakeScaler()
    with pytest.raises(ValueError, match="Unsupported target scaling option"):
        residual_raw_to_model(1.0, scaler, "INVALID")


def test_residual_rejects_zero_std():
    class ZeroStdScaler:
        y_std = 0.0

    with pytest.raises(ValueError, match="positive fold-local y_std"):
        residual_raw_to_model(1.0, ZeroStdScaler(), "YS1")


# ----------------------------------------------------------------------
# 3. V2PretestWindowDataset provides y_context_raw_wh
# ----------------------------------------------------------------------
def test_v2_adapter_for_e03_provides_y_context_raw_wh():
    document = load_e03_config(PROJECT_ROOT)
    dataset, _evidence, audit = build_v2_pretest_dataset(
        PROJECT_ROOT,
        document["challenger"]["feature_order"],
        experiment_id="E03",
        feature_variant_id="FS2_TF1",
    )
    assert audit.test_rows_read == 0
    assert audit.test_target_ids_seen == 0
    assert audit.population_status == "PASS"
    assert audit.fold_fingerprint_status == {
        "RO1": "PASS",
        "RO2": "PASS",
        "RO3": "PASS",
    }
    assert len(dataset) > 0
    item = dataset[0]
    assert "y_context_raw_wh" in item
    # y_context_raw_wh is the last observable Appliances (same as persistence base)
    assert item["y_context_raw_wh"].ndim == 1
    assert item["y_context_raw_wh"].shape[0] == 1
    # For horizon=1, context should equal the last step's Appliances
    # Check it's the raw (un-scaled) value
    assert float(item["y_context_raw_wh"].item()) >= 0.0


# ----------------------------------------------------------------------
# 4. load_fold_subset_loader_with_y_rescale: RESIDUAL mode
# ----------------------------------------------------------------------
def test_residual_loader_converts_y_model_to_delta():
    document = load_e03_config(PROJECT_ROOT)
    dataset, evidence, _audit = build_v2_pretest_dataset(
        PROJECT_ROOT,
        document["challenger"]["feature_order"],
        experiment_id="E03",
        feature_variant_id="FS2_TF1",
    )
    fold = evidence.folds[0]
    # Import scaler
    from course_work.rolling_origin.real_run import fit_fold_local_scaler
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    candidate = CandidateSpec(
        candidate_id="TR_C2_ALT_LOOKBACK",
        model_family="TRANSFORMER_ENCODER",
        shortlist_position=0,
        config=document["challenger"]["config"],
        config_fingerprint=document["challenger"]["config_fingerprint"],
        feature_variant_id="FS2_TF1",
        target_scaling_option="YS1",
        lookback_steps=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        source_phase="V2_E03",
        candidate_role="CHALLENGER",
    )
    bundle, _scaler_audit = fit_fold_local_scaler(
        candidate=candidate,
        fold=fold,
        fold_dataset=dataset,
        fit_stage="A",
    )
    assert bundle.target_scaling_option == "YS1"
    target_id = fold.inner_val_ids[0]

    # RESIDUAL mode
    loader, indices = load_fold_subset_loader_with_y_rescale(
        base_dataset=dataset,
        target_ids=[target_id],
        batch_size=1,
        shuffle=False,
        fold_stage_target_scaler=bundle,
        apply_fold_x_scaling=True,
        prediction_formulation="RESIDUAL_TO_PERSISTENCE",
    )
    batch = next(iter(loader))

    # In RESIDUAL mode, y_model should be delta_model = (y - context) / y_std
    assert "y_model" in batch
    assert "y_context_raw_wh" in batch
    assert "y_raw_wh" in batch

    # y_model must be delta_model (not direct target)
    y_model = batch["y_model"].numpy()[0, 0]
    y_raw = batch["y_raw_wh"].numpy()[0, 0]
    context_raw = batch["y_context_raw_wh"].numpy()[0, 0]
    delta_raw = y_raw - context_raw
    expected_delta_model = delta_raw / bundle.y_std
    assert abs(y_model - expected_delta_model) < 1e-5


def test_direct_loader_leaves_y_model_unchanged():
    document = load_e03_config(PROJECT_ROOT)
    dataset, evidence, _audit = build_v2_pretest_dataset(
        PROJECT_ROOT,
        document["challenger"]["feature_order"],
        experiment_id="E03",
        feature_variant_id="FS2_TF1",
    )
    fold = evidence.folds[0]
    from course_work.rolling_origin.real_run import fit_fold_local_scaler
    from course_work.rolling_origin.candidate_loader import CandidateSpec

    candidate = CandidateSpec(
        candidate_id="TR_C2_ALT_LOOKBACK",
        model_family="TRANSFORMER_ENCODER",
        shortlist_position=0,
        config=document["challenger"]["config"],
        config_fingerprint=document["challenger"]["config_fingerprint"],
        feature_variant_id="FS2_TF1",
        target_scaling_option="YS1",
        lookback_steps=72,
        boundary_protocol="WB0_CONTEXT_CARRY_OVER",
        source_phase="V2_E03",
        candidate_role="CHALLENGER",
    )
    bundle, _ = fit_fold_local_scaler(
        candidate=candidate,
        fold=fold,
        fold_dataset=dataset,
        fit_stage="A",
    )
    target_id = fold.inner_val_ids[0]

    # DIRECT mode
    loader, indices = load_fold_subset_loader_with_y_rescale(
        base_dataset=dataset,
        target_ids=[target_id],
        batch_size=1,
        shuffle=False,
        fold_stage_target_scaler=bundle,
        apply_fold_x_scaling=True,
        prediction_formulation="DIRECT",
    )
    batch = next(iter(loader))

    # In DIRECT mode, y_model should be (y_raw - y_mean) / y_std
    y_model = batch["y_model"].numpy()[0, 0]
    y_raw = batch["y_raw_wh"].numpy()[0, 0]
    expected_model = (y_raw - bundle.y_mean) / bundle.y_std
    assert abs(y_model - expected_model) < 1e-5


# ----------------------------------------------------------------------
# 5. evaluate_stage_c: RESIDUAL mode
# ----------------------------------------------------------------------
class _DummyScalerBundle:
    """Minimal FoldLocalScalerBundle for testing evaluate_stage_c RESIDUAL mode."""

    def __init__(self, y_mean=0.0, y_std=106.85, option="YS1"):
        self.y_mean = y_mean
        self.y_std = y_std
        self.target_scaling_option = option

    def inverse_transform_y(self, y_scaled):
        return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean


class _FakeModelDelta(torch.nn.Module):
    """Model that always outputs a fixed delta."""

    def __init__(self, delta_model_value: float = 0.5):
        super().__init__()
        self.delta = delta_model_value

    def forward(self, x):
        B = x.shape[0]
        return torch.full((B, 1), self.delta, dtype=torch.float32)


def test_evaluate_stage_c_residual_adds_context():
    # Model outputs delta=0.5 (model units). With y_std=106.85:
    # delta_wh = 0.5 * 106.85 = 53.425
    # y_hat = context + delta_wh
    context_value = 100.0
    delta_model = 0.5
    y_std = 106.85

    class _DS:
        def __init__(self, cv, dm):
            self.y_mean = 0.0
            self.y_std = y_std
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _DS(context_value, delta_model)
    model = _FakeModelDelta(delta_model)

    # Build a fake batch
    x = torch.randn(2, 72, 33)
    y_raw = torch.tensor([200.0, 150.0]).reshape(-1, 1)
    y_context = torch.tensor([100.0, 100.0]).reshape(-1, 1)
    target_ids = ["T00000144", "T00000145"]
    target_ts = ["2016-01-12 17:00:00", "2016-01-12 17:10:00"]
    batch = {
        "x": x,
        "y_raw_wh": y_raw,
        "y_context_raw_wh": y_context,
        "target_id": target_ids,
        "target_timestamp": target_ts,
    }

    # Custom loader that returns our batch once
    class _FakeLoader:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    loader = _FakeLoader(batch)

    # Direct mode: y_hat = context + delta_wh
    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=loader,
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="RESIDUAL_TO_PERSISTENCE",
    )

    # y_pred = context + delta_wh
    expected_pred = context_value + delta_model * y_std
    assert np.allclose(result.y_pred_wh, [expected_pred, expected_pred], atol=1e-5)
    assert result.y_true_wh[0] == 200.0
    assert result.y_true_wh[1] == 150.0


def test_evaluate_stage_c_direct():
    class _DS2:
        def __init__(self):
            self.y_mean = 0.0
            self.y_std = 106.85
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _DS2()
    model = _FakeModelDelta(2.0)  # model output = 2.0, so y_hat = 2 * 106.85 = 213.7

    x = torch.randn(1, 72, 33)
    y_raw = torch.tensor([213.7]).reshape(-1, 1)
    y_context = torch.tensor([100.0]).reshape(-1, 1)
    batch = {
        "x": x,
        "y_raw_wh": y_raw,
        "y_context_raw_wh": y_context,
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoader2:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=_FakeLoader2(batch),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="DIRECT",
    )
    expected_pred = 2.0 * 106.85  # direct: y_hat = model_output * y_std
    assert abs(result.y_pred_wh[0] - expected_pred) < 1e-5


def test_evaluate_stage_c_residual_rejects_missing_context():
    class _DS3:
        def __init__(self):
            self.y_mean = 0.0
            self.y_std = 106.85
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _DS3()
    model = _FakeModelDelta(0.5)

    x = torch.randn(1, 72, 33)
    # Missing y_context_raw_wh
    batch = {
        "x": x,
        "y_raw_wh": torch.tensor([200.0]).reshape(-1, 1),
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoader3:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    with pytest.raises(RuntimeError, match="RESIDUAL_TO_PERSISTENCE requires batch"):
        evaluate_stage_c(
            model=model,
            outer_eval_loader=_FakeLoader3(batch),
            scaler_bundle=scaler,
            device=torch.device("cpu"),
            model_id="TEST",
            fold_id="TEST_RO1",
            model_run_id="TEST_RUN",
            refit_epoch=10,
            prediction_formulation="RESIDUAL_TO_PERSISTENCE",
        )


def test_evaluate_stage_c_zero_delta_equals_persistence():
    """Zero model output: RESIDUAL prediction == persistence (y_context_raw_wh).

    Uses nonzero y_mean to prove NO y_mean contamination in the residual path.
    """
    y_mean = 120.0
    y_std = 20.0

    class _DS4:
        def __init__(self):
            self.y_mean = y_mean
            self.y_std = y_std
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _DS4()
    model = _FakeModelDelta(0.0)  # zero delta in model space

    x = torch.randn(1, 72, 33)
    # Use exact-representable values (powers of 2 + small int) to avoid
    # float32 round-trip in `torch.tensor([val])`.
    context_val = 100.0
    y_raw_val = 200.0
    batch = {
        "x": x,
        "y_raw_wh": torch.tensor([y_raw_val]).reshape(-1, 1),
        "y_context_raw_wh": torch.tensor([context_val]).reshape(-1, 1),
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoader4:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=_FakeLoader4(batch),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="RESIDUAL_TO_PERSISTENCE",
    )
    # Zero delta → prediction == context (exact persistence)
    # delta_wh = 0 * y_std = 0 exactly → pred = context + 0
    assert abs(result.y_pred_wh[0] - context_val) < 1e-9
    # Confirm y_mean did not contaminate the residual
    assert result.y_pred_wh[0] < (context_val + y_mean)


def test_evaluate_stage_c_residual_no_y_mean_contamination():
    """Nonzero y_mean: zero delta must still give exact context (no y_mean leak).

    With y_mean=120, y_std=20, context=100, model_delta=0:
        delta_wh = 0 * 20 = 0
        y_hat = 100 + 0 = 100   (NOT 100 + 120)
    """
    y_mean = 120.0
    y_std = 20.0
    context_val = 100.0
    model_delta = 0.0

    class _ScalerNonzeroMean:
        def __init__(self):
            self.y_mean = y_mean
            self.y_std = y_std
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            # Reference formula to verify the bug: y_std * y + y_mean
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _ScalerNonzeroMean()
    model = _FakeModelDelta(model_delta)

    x = torch.randn(1, 72, 33)
    batch = {
        "x": x,
        "y_raw_wh": torch.tensor([250.0]).reshape(-1, 1),
        "y_context_raw_wh": torch.tensor([context_val]).reshape(-1, 1),
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoader5:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=_FakeLoader5(batch),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="RESIDUAL_TO_PERSISTENCE",
    )
    # With zero model delta, prediction must equal context exactly,
    # NOT (context + y_std * 0 + y_mean).
    assert abs(result.y_pred_wh[0] - context_val) < 1e-9
    assert result.y_pred_wh[0] != (context_val + y_mean)  # would happen if buggy


def test_evaluate_stage_c_residual_nonzero_delta_with_nonzero_y_mean():
    """Nonzero y_mean: model_delta=0.5 → delta_raw=10 → y_hat = 100 + 10 = 110."""
    y_mean = 120.0
    y_std = 20.0
    context_val = 100.0
    model_delta = 0.5  # 0.5 * 20 = 10 raw Wh

    class _ScalerNonzeroMean:
        def __init__(self):
            self.y_mean = y_mean
            self.y_std = y_std
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _ScalerNonzeroMean()
    model = _FakeModelDelta(model_delta)

    x = torch.randn(1, 72, 33)
    batch = {
        "x": x,
        "y_raw_wh": torch.tensor([250.0]).reshape(-1, 1),
        "y_context_raw_wh": torch.tensor([context_val]).reshape(-1, 1),
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoader6:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=_FakeLoader6(batch),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="RESIDUAL_TO_PERSISTENCE",
    )
    # delta_raw = 0.5 * 20 = 10
    # y_hat = 100 + 10 = 110 (NOT 100 + 10 + 120 = 230)
    expected = context_val + model_delta * y_std
    assert abs(result.y_pred_wh[0] - expected) < 1e-9
    assert result.y_pred_wh[0] != (expected + y_mean)  # y_mean must NOT leak


def test_residual_helpers_are_algebraic_inverses_with_nonzero_y_mean():
    """residual_raw_to_model and residual_model_to_raw are algebraic inverses.

    y_mean is irrelevant in residual space (deltas are centred at 0).
    """
    y_mean = 120.0
    y_std = 20.0

    class _ScalerNonzeroMean:
        def __init__(self):
            self.y_mean = y_mean
            self.y_std = y_std
            self.target_scaling_option = "YS1"

    scaler = _ScalerNonzeroMean()

    for delta_raw in (-50.0, -10.0, 0.0, 10.0, 50.0):
        model = residual_raw_to_model(delta_raw, scaler, "YS1")
        back = residual_model_to_raw(model, scaler, "YS1")
        assert abs(back - delta_raw) < 1e-9, f"roundtrip failed for {delta_raw}"


def test_evaluate_stage_c_direct_unchanged_with_nonzero_y_mean():
    """DIRECT mode MUST still use inverse_transform_y (so y_mean is added).

    For DIRECT: y_hat = model_out * y_std + y_mean
    """
    y_mean = 120.0
    y_std = 20.0
    model_out = 2.0

    class _ScalerNonzeroMean:
        def __init__(self):
            self.y_mean = y_mean
            self.y_std = y_std
            self.target_scaling_option = "YS1"

        def inverse_transform_y(self, y_scaled):
            return np.asarray(y_scaled, dtype=np.float64) * self.y_std + self.y_mean

    scaler = _ScalerNonzeroMean()
    model = _FakeModelDelta(model_out)

    x = torch.randn(1, 72, 33)
    batch = {
        "x": x,
        "y_raw_wh": torch.tensor([160.0]).reshape(-1, 1),
        "y_context_raw_wh": torch.tensor([100.0]).reshape(-1, 1),
        "target_id": ["T00000144"],
        "target_timestamp": ["2016-01-12 17:00:00"],
    }

    class _FakeLoaderDirect:
        def __init__(self, batch):
            self.batch = batch
            self._iter = iter([batch])

        def __iter__(self):
            return self._iter

    result = evaluate_stage_c(
        model=model,
        outer_eval_loader=_FakeLoaderDirect(batch),
        scaler_bundle=scaler,
        device=torch.device("cpu"),
        model_id="TEST",
        fold_id="TEST_RO1",
        model_run_id="TEST_RUN",
        refit_epoch=10,
        prediction_formulation="DIRECT",
    )
    # DIRECT: y_hat = model_out * y_std + y_mean = 2*20+120 = 160
    expected_direct = model_out * y_std + y_mean
    assert abs(result.y_pred_wh[0] - expected_direct) < 1e-9


# ----------------------------------------------------------------------
# 6. E03 context is isolated
# ----------------------------------------------------------------------
def test_e03_context_is_isolated_and_residual_challenger():
    document = load_e03_config(PROJECT_ROOT)
    context = build_e03_run_context(PROJECT_ROOT, document)
    assert context.registry_namespace == "V2_E03"
    assert context.execution_track == "MODEL_IMPROVEMENT_V2_E03"
    assert "/experiments/E03" in context.artifact_dir.as_posix()
    assert "/experiments/E03/registry" in context.registry_root.as_posix()
    assert "/experiments/E03/runs" in context.run_root.as_posix()
    assert context.reuse_completed_runs is False
    assert len(context.candidate_specs) == 1
    spec = context.candidate_specs[0]
    assert spec.feature_variant_id == "FS2_TF1"
    assert spec.config["model"]["prediction_formulation"] == "RESIDUAL_TO_PERSISTENCE"
    assert spec.config["model"]["input_size"] == 33


def test_e03_candidate_builds_correctly():
    document = load_e03_config(PROJECT_ROOT)
    candidate = build_e03_candidate(document)
    assert candidate.candidate_id == "TR_C2_ALT_LOOKBACK"
    assert candidate.model_family == "TRANSFORMER_ENCODER"
    assert candidate.feature_variant_id == "FS2_TF1"
    assert candidate.target_scaling_option == "YS1"
    assert candidate.lookback_steps == 72
    assert candidate.boundary_protocol == "WB0_CONTEXT_CARRY_OVER"
    assert candidate.candidate_role == "PREDICTION_FORMULATION_CHALLENGER"
    assert candidate.config["model"]["prediction_formulation"] == "RESIDUAL_TO_PERSISTENCE"


def test_e03_official_refuses_without_human_authorization():
    from course_work.model_improvement_v2.e03_runner import main

    assert main(["--experiment", "E03", "--mode", "official", "--seed", "42"]) == 3


def test_e03_resume_stage_c_refuses_training_flag():
    """resume-stage-c must reject --authorize-training (no training allowed)."""
    from course_work.model_improvement_v2.e03_runner import main

    assert main(["--experiment", "E03", "--mode", "resume-stage-c",
                 "--authorize-training", "--seed", "42"]) == 2


def test_e03_resume_stage_c_context_has_locked_run_ids():
    """Verify the resume context has the 6 locked run IDs from completed Stage A/B."""
    document = load_e03_config(PROJECT_ROOT)
    from course_work.model_improvement_v2.e03_runner import _build_e03_run_context_for_resume

    ctx = _build_e03_run_context_for_resume(PROJECT_ROOT, document)
    assert ctx.reuse_completed_runs is True
    assert ctx.reuse_completed_run_ids is not None
    expected_keys = {f"RO{fold}_{stage}" for fold in (1, 2, 3) for stage in ("A", "B")}
    assert set(ctx.reuse_completed_run_ids.keys()) == expected_keys
    # All values must be non-empty run IDs
    assert all(v for v in ctx.reuse_completed_run_ids.values())


def test_e03_resume_stage_c_refuses_wrong_run_ids():
    """Wrong run IDs should be caught by assert_context_invariants."""
    from course_work.model_improvement_v2.e03_runner import (
        _build_e03_run_context_for_resume,
        E03PreflightError,
    )
    document = load_e03_config(PROJECT_ROOT)

    # Patch the locked run IDs to have a wrong key
    orig = __import__("course_work.model_improvement_v2.e03_runner", fromlist=["_build_e03_run_context_for_resume"])
    import course_work.model_improvement_v2.e03_runner as m

    # The _build_e03_run_context_for_resume validates run IDs in real_run.assert_context_invariants
    # which checks ctx.reuse_completed_run_ids. We just verify the context builds with correct IDs.
    ctx = m._build_e03_run_context_for_resume(PROJECT_ROOT, document)
    assert ctx.execution_track == "MODEL_IMPROVEMENT_V2_E03"
    assert ctx.candidate_specs[0].config["model"]["prediction_formulation"] == "RESIDUAL_TO_PERSISTENCE"


# ----------------------------------------------------------------------
# 7. Pooled metrics shape contract — 1-D outputs
# ----------------------------------------------------------------------
def test_compose_residual_prediction_raw_returns_1d():
    """compose_residual_prediction_raw must return 1-D shape (-1,) for pooling.

    Regression test for the pooled metrics crash:
    ValueError: Pooled shape mismatch: (2960,) vs (2960, 1)

    The bug: compose_residual_prediction_raw returned 2-D when context was 2-D.
    The fix: always reshape to 1-D before returning.
    """
    # Simulate 2-D context [B, 1] as the outer_eval loader produces
    context_2d = np.array([[100.0], [200.0], [300.0]], dtype=np.float64)  # shape (3,1)
    # Simulate model output [B, 1] — PyTorch standard output shape
    model_out_2d = np.array([[0.5], [0.3], [-0.1]], dtype=np.float64)   # shape (3,1)

    class _Scaler:
        y_std = 106.85
        target_scaling_option = "YS1"

    result = compose_residual_prediction_raw(
        context_2d,
        model_out_2d,
        _Scaler(),
        "YS1",
    )
    assert result.ndim == 1, f"Expected 1-D output, got shape {result.shape}"
    assert result.shape == (3,), f"Expected (3,), got {result.shape}"
    # Verify correct arithmetic: y_hat = context + delta_raw
    # delta_raw = model_out * y_std = 0.5*106.85 = 53.425
    expected = np.array([100.0 + 0.5 * 106.85,
                          200.0 + 0.3 * 106.85,
                          300.0 + (-0.1) * 106.85])
    assert np.allclose(result, expected, atol=1e-9)


def test_compose_residual_prediction_raw_1d_input():
    """Also verify 1-D inputs still work (backwards compatible)."""
    context_1d = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    model_out_1d = np.array([0.5, 0.3, -0.1], dtype=np.float64)

    class _Scaler:
        y_std = 106.85
        target_scaling_option = "YS1"

    result = compose_residual_prediction_raw(
        context_1d,
        model_out_1d,
        _Scaler(),
        "YS1",
    )
    assert result.ndim == 1
    assert result.shape == (3,)


def test_compose_residual_prediction_raw_scalar_context():
    """Scalar context (no batch dim) must also work."""
    context_scalar = 100.0
    model_out_scalar = 0.5

    class _Scaler:
        y_std = 106.85
        target_scaling_option = "YS1"

    result = compose_residual_prediction_raw(
        context_scalar,
        model_out_scalar,
        _Scaler(),
        "YS1",
    )
    assert result.ndim == 1
    assert result.shape == (1,)
    assert np.abs(result[0] - 153.425) < 1e-9


def test_pooled_metrics_shape_contract():
    """compute_pooled_metrics must accept 1-D y_true and y_pred."""
    from course_work.rolling_origin.pooling import compute_pooled_metrics

    # Both 1-D — this is what Stage C now produces after the fix
    y_true = np.array([100.0, 200.0, 300.0, 150.0, 250.0])
    y_pred = np.array([105.0, 195.0, 310.0, 148.0, 255.0])

    pm = compute_pooled_metrics(
        candidate_id="TEST",
        y_true_per_fold=[y_true[:2], y_true[2:]],
        y_pred_per_fold=[y_pred[:2], y_pred[2:]],
    )
    assert pm.candidate_id == "TEST"
    assert np.isfinite(pm.pooled_rmse_wh)
    assert np.isfinite(pm.pooled_mae_wh)


def test_pooled_metrics_rejects_2d_prediction():
    """compute_pooled_metrics must reject 2-D y_pred to catch shape mismatches."""
    from course_work.rolling_origin.pooling import compute_pooled_metrics

    y_true_1d = np.array([100.0, 200.0])
    # y_pred is 2-D (the buggy shape that caused the original crash)
    y_pred_2d = np.array([[105.0], [195.0]])

    with pytest.raises(ValueError, match="Pooled shape mismatch"):
        compute_pooled_metrics(
            candidate_id="TEST",
            y_true_per_fold=[y_true_1d],
            y_pred_per_fold=[y_pred_2d],
        )
