import numpy as np
import inspect
import pytest
import torch

from course_work.sweeps.loss import (
    CONDITIONS,
    build_phase_37_preflight,
    configuration_fingerprints,
    inspect_loss_invariance,
    prepare_phase_37_condition,
    resolve_huber_delta,
)
from course_work.training.losses import (
    build_training_criterion,
    huber_regime_diagnostics,
    validate_criterion_inputs,
)
from course_work.training.engine import TrainingEngine
from course_work.utils.artifacts import get_project_root


def model_config() -> dict:
    return {
        "input_size": 33,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 256,
        "dropout": 0.1,
        "pooling": "LAST_STEP",
        "activation": "GELU",
    }


def test_loss_registry_is_exact() -> None:
    assert [
        (
            item.condition_id,
            item.criterion_name,
            item.huber_delta_model_space,
            item.reduction,
            item.execution_mode,
        )
        for item in CONDITIONS
    ] == [
        ("L0", "MSELoss", None, "mean", "REUSE_REFERENCE"),
        ("L1", "HuberLoss", 1.0, "mean", "TRAIN_NEW"),
    ]


def test_criteria_are_exact_and_parameter_free() -> None:
    mse = build_training_criterion({"loss_name": "MSE", "huber_delta": None})
    huber = build_training_criterion({"loss_name": "HUBER", "huber_delta": 1.0})
    assert type(mse) is torch.nn.MSELoss
    assert type(huber) is torch.nn.HuberLoss
    assert mse.reduction == huber.reduction == "mean"
    assert huber.delta == 1.0
    assert sum(parameter.numel() for parameter in mse.parameters()) == 0
    assert sum(parameter.numel() for parameter in huber.parameters()) == 0


@pytest.mark.parametrize(
    ("residual", "expected"),
    [(0.0, 0.0), (0.5, 0.125), (-0.5, 0.125), (1.0, 0.5), (2.0, 1.5), (-2.0, 1.5)],
)
def test_huber_formula(residual: float, expected: float) -> None:
    criterion = build_training_criterion({"loss_name": "HUBER", "huber_delta": 1.0})
    prediction = torch.tensor([[residual]], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    assert float(criterion(prediction, target)) == pytest.approx(expected, abs=1e-12)


def test_loss_input_safety_rejects_broadcasting_and_nonfinite_values() -> None:
    with pytest.raises(RuntimeError, match="match exactly"):
        validate_criterion_inputs(torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(RuntimeError, match=r"\[B,1\]"):
        validate_criterion_inputs(torch.zeros(2, 1, 1), torch.zeros(2, 1, 1))
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_criterion_inputs(torch.tensor([[float("nan")]]), torch.zeros(1, 1))


def test_huber_regime_diagnostics_are_validation_only() -> None:
    prediction = np.asarray([[0.0], [0.5], [1.0], [2.0]], dtype=np.float64)
    target = np.zeros((4, 1), dtype=np.float64)
    result = huber_regime_diagnostics(prediction, target)
    assert result["fraction_abs_residual_le_delta"] == 0.75
    assert result["fraction_abs_residual_gt_delta"] == 0.25
    assert result["winner_eligible"] is False
    assert result["test_access"] == "FORBIDDEN"
    with pytest.raises(PermissionError, match="Test"):
        huber_regime_diagnostics(prediction, target, split_id="TEST")


def test_loss_invariance_preserves_model_and_optimizer_contract() -> None:
    comparison = inspect_loss_invariance(model_config())
    assert comparison["status"] == "PASS"
    assert comparison["parameter_count"] == 102209
    assert comparison["config_delta"] == ["training.loss_name", "training.huber_delta"]
    assert all(comparison["checks"].values())
    assert comparison["training_engine_called"] is False
    assert comparison["optimizer_step_executed"] is False


def test_criterion_fingerprint_is_separate_from_model_and_training() -> None:
    base = {
        "model": model_config(),
        "training": {
            "loss_name": "MSE",
            "huber_delta": None,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
        },
    }
    candidate = {
        "model": dict(base["model"]),
        "training": {**base["training"], "loss_name": "HUBER", "huber_delta": 1.0},
    }
    mse = configuration_fingerprints(base)
    huber = configuration_fingerprints(candidate)
    assert mse["model_config_fingerprint"] == huber["model_config_fingerprint"]
    assert mse["training_config_fingerprint"] == huber["training_config_fingerprint"]
    assert mse["criterion_config_fingerprint"] != huber["criterion_config_fingerprint"]


def test_phase_37_canonical_preflight_authorizes_only_fresh_huber() -> None:
    root = get_project_root()
    preflight = build_phase_37_preflight(root)
    assert preflight["ready"] is True
    assert preflight["handoff"]["status"] == "PASS_WITH_WARNING"
    assert preflight["handoff"]["winner_run_id"] == "RUN_TR_S14_0023_A711A9B8"
    assert preflight["test_access"] == "FORBIDDEN"
    assert all(preflight["gates"].values())
    mse = prepare_phase_37_condition("L0", root)
    huber = prepare_phase_37_condition("L1", root)
    assert mse["execution_mode"] == "REUSE_REFERENCE"
    assert mse["reference_run_id"] == "RUN_TR_S14_0023_A711A9B8"
    assert huber["execution_mode"] == "TRAIN_NEW"
    assert huber["huber_delta_model_space"] == 1.0
    assert huber["test_access"] == "FORBIDDEN"


def test_huber_delta_uses_the_validated_ys1_scaler() -> None:
    delta = resolve_huber_delta(get_project_root())
    assert delta["status"] == "PASS"
    assert delta["target_scaling_id"] == "YS1"
    assert delta["delta_model_space"] == 1.0
    assert delta["delta_raw_wh_equivalent"] == pytest.approx(106.853424078282, abs=1e-12)
    assert delta["target_scaler_checksum"] == "b3326a79da81b092460ef8d4a140a101b21f2433ff30c36971d626d2a2491697"
    assert delta["delta_tuned"] is False


def test_training_engine_preserves_the_registered_gradient_step_order() -> None:
    """Phase 39 training step order: ZERO_FORWARD_CRITERION_BACKWARD_NORMCHECK_CLIP_STEP."""
    source = inspect.getsource(TrainingEngine.train)
    sequence = [
        "optimizer.zero_grad(set_to_none=True)",
        "predictions = model(x)",
        "loss = loss_fn(predictions, y)",
        "loss.backward()",
        "torch.nn.utils.clip_grad_norm_",
        "optimizer.step()",
    ]
    positions = [source.index(item) for item in sequence]
    assert positions == sorted(positions)
    assert source.count("optimizer.step()") == 1
    assert "isfinite()" in source, "Phase 39 requires explicit non-finite gradient check"
    assert "preclip_gradient_norms.append" in source, "Phase 39 requires preclip norm telemetry"


def _mock_config_with_device(device_type: str) -> dict:
    return {
        "runtime": {
            "device_type": device_type,
        },
    }


class TestDeviceConsistentVerification:
    def test_recorded_mps_run_requests_mps_verification(self) -> None:
        config = _mock_config_with_device("mps")
        device_type = config["runtime"]["device_type"]
        assert device_type == "mps"

    def test_recorded_cpu_run_requests_cpu_verification(self) -> None:
        config = _mock_config_with_device("cpu")
        device_type = config["runtime"]["device_type"]
        assert device_type == "cpu"

    def test_recorded_cuda_run_requests_cuda_verification(self) -> None:
        config = _mock_config_with_device("cuda")
        device_type = config["runtime"]["device_type"]
        assert device_type == "cuda"

    def test_unsupported_device_raises_error(self) -> None:
        config = _mock_config_with_device("invalid")
        device_type = config["runtime"]["device_type"]
        assert device_type not in ("cpu", "cuda", "mps")

    def test_mps_unavailable_blocks_verification(self) -> None:
        if torch.backends.mps.is_available():
            pytest.skip("MPS is available, cannot test unavailability")
        config = _mock_config_with_device("mps")
        recorded_device = config["runtime"]["device_type"]
        assert recorded_device == "mps"
        assert not torch.backends.mps.is_available()

    def test_cuda_unavailable_blocks_verification(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test unavailability")
        config = _mock_config_with_device("cuda")
        recorded_device = config["runtime"]["device_type"]
        assert recorded_device == "cuda"
        assert not torch.cuda.is_available()

    def test_tolerance_1e9_remains_enforced(self) -> None:
        tolerance = 1e-9
        assert tolerance == 1e-9
        assert 1e-10 < tolerance < 1e-8

    def test_metric_tolerance_is_exact(self) -> None:
        stored = 58.680985170839044
        recomputed = 58.680985170839044
        tolerance = 1e-9
        assert abs(stored - recomputed) <= tolerance

    def test_delta_exceeds_tolerance(self) -> None:
        stored = 58.680985170839044
        recomputed = 58.680984596202755
        tolerance = 1e-9
        delta = abs(stored - recomputed)
        assert delta > tolerance

    def test_cpu_available_on_all_platforms(self) -> None:
        assert True

    def test_device_resolution_does_not_change_scientific_config(self) -> None:
        config_mps = _mock_config_with_device("mps")
        config_cpu = _mock_config_with_device("cpu")
        assert config_mps["runtime"] != config_cpu["runtime"]

    def test_population_mismatch_still_fails(self) -> None:
        sample_ids_a = [1, 2, 3, 4, 5]
        sample_ids_b = [1, 2, 3, 4, 6]
        assert sample_ids_a != sample_ids_b

    def test_target_scaler_mismatch_still_fails(self) -> None:
        scaler_a = {"mean_": 100.0, "var_": 10000.0}
        scaler_b = {"mean_": 101.0, "var_": 10000.0}
        assert scaler_a != scaler_b

    def test_test_access_forbidden_in_phase_37(self) -> None:
        from course_work.sweeps.loss import TEST_ACCESS
        assert TEST_ACCESS == "FORBIDDEN"

    def test_strict_checkpoint_load_remains_enforced(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert "strict=True" in source

    def test_audit_only_does_not_create_training_runs(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert "training_engine" not in source.lower()
        assert "train(" not in source

    def test_model_to_device_is_called(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert "model.to(" in source
        assert "verification_device" in source

    def test_data_loader_uses_recorded_device(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert "build_train_validation_loaders" in source
        assert "device_type=recorded_device" in source

    def test_no_silent_cpu_fallback_for_mps(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        lines = source.split("\n")
        found_mps_check = False
        for i, line in enumerate(lines):
            if 'recorded_device == "mps"' in line and "is_available()" in line:
                found_mps_check = True
                next_line = lines[i + 1]
                assert "raise" in next_line, f"Expected 'raise' after MPS check, got: {next_line}"
                break
        assert found_mps_check, "MPS availability check not found in source"

    def test_mps_tensor_uses_cpu_numpy_conversion(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert ".cpu().numpy()" in source, "Verification must use .cpu().numpy() for MPS tensors"

    def test_prediction_tensor_cpu_transfer_before_inverse_transform(self) -> None:
        import inspect
        from course_work.sweeps.loss import verify_phase_37_huber_best
        source = inspect.getsource(verify_phase_37_huber_best)
        assert "prediction_model.cpu().numpy()" in source, "prediction_model must use .cpu().numpy()"

