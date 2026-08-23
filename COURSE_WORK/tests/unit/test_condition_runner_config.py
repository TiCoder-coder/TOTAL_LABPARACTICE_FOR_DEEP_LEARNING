from scripts.run_single_condition import _prepared_condition_base


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
