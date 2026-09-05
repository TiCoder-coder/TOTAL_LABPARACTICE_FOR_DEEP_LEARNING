import pytest

from scripts.run_single_condition import (
    _prepared_condition_base,
    _validate_phase_34_config_delta,
    _validate_phase_35_config_delta,
    _validate_phase_36_config_delta,
    _validate_phase_37_config_delta,
)


def frozen_configuration() -> dict:
    return {
        "feature_variant_id": "FS2_TF1",
        "lookback_id": "L36",
        "target_scaling_id": "YS1",
        "pooling_id": "LAST_STEP",
        "activation_id": "GELU",
        "batch_id": "B32",
        "learning_rate": 0.0003,
        "dropout": 0.1,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
    }


def test_phase_31_condition_owns_weight_decay() -> None:
    payload = {
        "weight_decay": 0.0,
        "frozen_configuration": frozen_configuration(),
    }
    base = _prepared_condition_base(payload)
    assert base["weight_decay"] == 0.0
    assert base["dropout"] == 0.1


def test_phase_32_condition_owns_dropout() -> None:
    frozen = frozen_configuration()
    frozen["weight_decay"] = 0.001
    payload = {
        "dropout_probability": 0.3,
        "frozen_configuration": frozen,
    }
    base = _prepared_condition_base(payload)
    assert base["weight_decay"] == 0.001
    assert base["dropout"] == 0.3


def test_phase_33_condition_owns_d_model() -> None:
    frozen = frozen_configuration()
    frozen["weight_decay"] = 0.001
    payload = {
        "d_model": 32,
        "frozen_configuration": frozen,
    }
    base = _prepared_condition_base(payload)
    assert base["weight_decay"] == 0.001
    assert base["dropout"] == 0.1
    assert base["d_model"] == 32


def test_phase_34_condition_owns_num_heads() -> None:
    frozen = frozen_configuration()
    frozen["weight_decay"] = 0.001
    payload = {
        "num_heads": 2,
        "frozen_configuration": frozen,
    }
    base = _prepared_condition_base(payload)
    assert base["d_model"] == 64
    assert base["num_heads"] == 2


def test_phase_34_config_delta_allows_only_num_heads() -> None:
    source = {"model": {"d_model": 64, "num_heads": 4}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"d_model": 64, "num_heads": 2}, "training": {"weight_decay": 0.001}}
    _validate_phase_34_config_delta(source, candidate)


def test_phase_34_config_delta_rejects_frozen_drift() -> None:
    source = {"model": {"d_model": 64, "num_heads": 4}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"d_model": 64, "num_heads": 2}, "training": {"weight_decay": 0.01}}
    with pytest.raises(RuntimeError, match="training.weight_decay"):
        _validate_phase_34_config_delta(source, candidate)


def test_phase_35_condition_owns_num_layers() -> None:
    frozen = frozen_configuration()
    frozen["weight_decay"] = 0.001
    payload = {"num_layers": 1, "frozen_configuration": frozen}
    base = _prepared_condition_base(payload)
    assert base["num_heads"] == 4
    assert base["num_layers"] == 1


def test_phase_35_config_delta_allows_only_num_layers() -> None:
    source = {"model": {"num_heads": 4, "num_layers": 2}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"num_heads": 4, "num_layers": 1}, "training": {"weight_decay": 0.001}}
    _validate_phase_35_config_delta(source, candidate)


def test_phase_35_config_delta_rejects_frozen_drift() -> None:
    source = {"model": {"num_heads": 4, "num_layers": 2}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"num_heads": 2, "num_layers": 1}, "training": {"weight_decay": 0.001}}
    try:
        _validate_phase_35_config_delta(source, candidate)
    except RuntimeError as error:
        assert "model.num_heads" in str(error)
    else:
        raise AssertionError("Phase 35 frozen drift was not rejected")


def test_phase_36_condition_owns_ffn_dim() -> None:
    frozen = frozen_configuration()
    frozen["weight_decay"] = 0.001
    payload = {"ffn_dim": 64, "frozen_configuration": frozen}
    base = _prepared_condition_base(payload)
    assert base["d_model"] == 64
    assert base["num_heads"] == 4
    assert base["num_layers"] == 2
    assert base["ffn_dim"] == 64


@pytest.mark.parametrize("ffn_dim", [64, 256])
def test_phase_36_config_delta_allows_only_ffn_dim(ffn_dim: int) -> None:
    source = {"model": {"d_model": 64, "ffn_dim": 128}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"d_model": 64, "ffn_dim": ffn_dim}, "training": {"weight_decay": 0.001}}
    _validate_phase_36_config_delta(source, candidate)


def test_phase_36_config_delta_rejects_frozen_drift() -> None:
    source = {"model": {"d_model": 64, "ffn_dim": 128}, "training": {"weight_decay": 0.001}}
    candidate = {"model": {"d_model": 32, "ffn_dim": 64}, "training": {"weight_decay": 0.001}}
    with pytest.raises(RuntimeError, match="model.d_model"):
        _validate_phase_36_config_delta(source, candidate)


def test_phase_37_condition_owns_only_training_loss() -> None:
    frozen = frozen_configuration()
    frozen.update({"weight_decay": 0.001, "loss": "MSE"})
    payload = {
        "registry_loss_name": "HUBER",
        "huber_delta_model_space": 1.0,
        "frozen_configuration": frozen,
    }
    base = _prepared_condition_base(payload)
    assert base["loss_name"] == "HUBER"
    assert base["huber_delta"] == 1.0
    assert base["ffn_dim"] == 128


def test_phase_37_config_delta_allows_only_registered_loss_fields() -> None:
    source = {"model": {"ffn_dim": 256}, "training": {"loss_name": "MSE", "huber_delta": None}}
    candidate = {"model": {"ffn_dim": 256}, "training": {"loss_name": "HUBER", "huber_delta": 1.0}}
    _validate_phase_37_config_delta(source, candidate)


def test_phase_37_config_delta_rejects_architecture_drift() -> None:
    source = {"model": {"ffn_dim": 256}, "training": {"loss_name": "MSE", "huber_delta": None}}
    candidate = {"model": {"ffn_dim": 128}, "training": {"loss_name": "HUBER", "huber_delta": 1.0}}
    with pytest.raises(RuntimeError, match="model.ffn_dim"):
        _validate_phase_37_config_delta(source, candidate)
